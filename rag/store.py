"""sqlite-vec 向量库：文档/分块元数据 + 向量/FTS5 双通道混合检索（M11）。

索引库为单文件 SQLite（index.db），符合"本地优先"定位；
向量表使用 sqlite-vec 扩展（cosine），全文通道使用 SQLite 内置 FTS5（BM25），
两路候选用 RRF（Reciprocal Rank Fusion）融合。
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import sqlite_vec

from rag.chunking import Chunk
from rag.errors import RAGError
from rag.parser import Document

logger = logging.getLogger(__name__)

# CJK 字符范围：CJK 统一表意 / 扩展 A / 兼容表意 / 假名（FTS 分词用）
_CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]")
_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def fts_normalize(text: str) -> str:
    """FTS5 索引文本归一化：在每个 CJK 字符间插入空格（字符级分词）。

    SQLite unicode61 分词器不切分连续中文（整句变一个 token），
    预先按字切开后即可用短语查询匹配中文子串。
    """
    return _CJK_CHAR_RE.sub(lambda match: f" {match.group(0)} ", text or "")


def build_fts_query(query: str) -> str:
    """把用户查询转成 FTS5 MATCH 表达式：中文按二元词组、英文按词，OR 连接保召回。"""
    terms: list[str] = []
    for run in _CJK_RUN_RE.findall(query or ""):
        if len(run) == 1:
            terms.append(f'"{run}"')
        else:
            # 二元词组（bigram）：单字短语过散、整句短语过严，bigram 兼顾召回与精度
            terms.extend(
                '"' + " ".join(run[i : i + 2]) + '"' for i in range(len(run) - 1)
            )
    for word in _ASCII_WORD_RE.findall(query or ""):
        terms.append(f'"{word}"')
    # 去重（保持顺序）后 OR 连接：任一词命中即可进入候选，靠 BM25/RRF 排序
    unique = list(dict.fromkeys(terms))
    return " OR ".join(unique)


@dataclass
class SearchHit:
    """一条检索命中：来源分块 + 归一化分数 + 命中通道（可观测）。"""

    path: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    score: float                     # 0-1 归一化分数（向量相似 / RRF 融合 / rerank）
    sources: list[str] = field(default_factory=list)  # vector / fts / rerank


class VectorStore:
    """向量库封装：建表、增量元数据、top-k 检索与按文档删除。"""

    def __init__(self, db_path: str | Path, *, model_id: str):
        self.db_path = Path(db_path)
        self.model_id = model_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_base_schema()
        self._verify_model()

    # ---- 建表与元数据 ----

    def _ensure_base_schema(self) -> None:
        """创建元数据 / 文档 / 分块表（向量表在首次写入时按实际维度创建）。"""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rag_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                mtime_ms INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, chunk_id UNINDEXED);
            """
        )
        self._conn.commit()

    def _verify_model(self) -> None:
        """校验索引库的 embedding 模型与当前配置一致（不同模型的向量不可混用）。"""
        stored = self._meta_get("embedding_model")
        if stored is None:
            self._meta_set("embedding_model", self.model_id)
        elif stored != self.model_id:
            raise RAGError(
                "EMBEDDING_MODEL_MISMATCH",
                f"索引由模型 {stored} 创建，当前配置为 {self.model_id}；"
                "请用 index_docs(rebuild=true) 全量重建索引",
            )

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM rag_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO rag_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def _ensure_vec_table(self, dimension: int) -> None:
        """按实际向量维度创建虚拟表；维度变化说明模型已更换，拒绝写入。"""
        stored = self._meta_get("embedding_dimension")
        if stored is not None and int(stored) != dimension:
            raise RAGError(
                "EMBEDDING_DIMENSION_MISMATCH",
                f"索引向量维度为 {stored}，当前模型输出 {dimension} 维；"
                "请用 index_docs(rebuild=true) 全量重建索引",
            )
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding float[{dimension}] distance_metric=cosine)"
        )
        if stored is None:
            self._meta_set("embedding_dimension", str(dimension))

    def _vec_table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()
        return row is not None

    # ---- 写入 ----

    def upsert_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """替换式写入一个文档：先删旧分块/向量，再插入新内容（单事务保证一致）。"""
        if len(chunks) != len(vectors):
            raise RAGError(
                "EMBEDDING_FAILED",
                f"分块数 {len(chunks)} 与向量数 {len(vectors)} 不一致",
            )
        if vectors:
            self._ensure_vec_table(len(vectors[0]))
        try:
            with self._conn:
                self._delete_document_rows(document.path)
                self._conn.execute(
                    "INSERT INTO documents(path, content_hash, size_bytes, mtime_ms, chunk_count)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        document.path,
                        document.content_hash,
                        document.size_bytes,
                        document.mtime_ms,
                        len(chunks),
                    ),
                )
                for chunk, vector in zip(chunks, vectors, strict=True):
                    cursor = self._conn.execute(
                        "INSERT INTO chunks(path, chunk_index, content, char_start, char_end)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            document.path,
                            chunk.index,
                            chunk.content,
                            chunk.char_start,
                            chunk.char_end,
                        ),
                    )
                    self._conn.execute(
                        "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                        (cursor.lastrowid, sqlite_vec.serialize_float32(vector)),
                    )
                    # FTS5 通道：写入按字切分后的中文，支持 BM25 全文召回
                    self._conn.execute(
                        "INSERT INTO chunks_fts(content, chunk_id) VALUES (?, ?)",
                        (fts_normalize(chunk.content), cursor.lastrowid),
                    )
        except sqlite3.Error as exc:
            raise RAGError("STORE_ERROR", f"索引写入失败: {exc}") from exc

    # ---- 查询 ----

    def get_document(self, path: str) -> dict | None:
        """返回文档的索引元数据（含内容哈希），未索引时返回 None。"""
        row = self._conn.execute(
            "SELECT path, content_hash, size_bytes, mtime_ms, chunk_count, indexed_at"
            " FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        if row is None:
            return None
        keys = ("path", "content_hash", "size_bytes", "mtime_ms", "chunk_count", "indexed_at")
        return dict(zip(keys, row, strict=True))

    def list_documents(self) -> list[dict]:
        """列出全部已索引文档（调试与演示用）。"""
        rows = self._conn.execute(
            "SELECT path, content_hash, size_bytes, mtime_ms, chunk_count, indexed_at"
            " FROM documents ORDER BY path"
        ).fetchall()
        keys = ("path", "content_hash", "size_bytes", "mtime_ms", "chunk_count", "indexed_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def search(
        self,
        vector: list[float],
        top_k: int,
        *,
        query: str = "",
        mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
    ) -> list[SearchHit]:
        """混合检索：向量 cosine 与 FTS5 BM25 候选按 RRF 融合后取 top_k。

        mode 可选 hybrid（默认，双通道融合）/ vector（纯向量）/ fts（纯全文）。
        混合模式先召回 top_k * candidate_multiplier 个候选，融合后再截断，
        给上层 rerank 留出足够的候选池。
        """
        if not self._vec_table_exists():
            raise RAGError("INDEX_EMPTY", "索引为空，请先调用 index_docs 工具建立索引")
        if mode in ("hybrid", "vector"):
            stored = self._meta_get("embedding_dimension")
            if stored is not None and int(stored) != len(vector):
                raise RAGError(
                    "EMBEDDING_DIMENSION_MISMATCH",
                    f"索引向量维度为 {stored}，查询向量为 {len(vector)} 维；"
                    "请用 index_docs(rebuild=true) 全量重建索引",
                )
        candidate_limit = max(top_k, top_k * max(1, candidate_multiplier))
        vector_rows = (
            self._vector_search(vector, candidate_limit)
            if mode in ("hybrid", "vector")
            else []
        )
        fts_rows = self._fts_search(query, candidate_limit) if mode in ("hybrid", "fts") else []
        fused = self._fuse_rankings(vector_rows, fts_rows, mode=mode, rrf_k=rrf_k)
        return self._build_hits(fused[:top_k])

    def _vector_search(self, vector: list[float], limit: int) -> list[tuple[int, float]]:
        """向量通道：sqlite-vec KNN，返回 (chunk_id, cosine_distance)。"""
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks"
                " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(vector), limit),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RAGError("STORE_ERROR", f"向量检索失败: {exc}") from exc
        # cosine 对零向量未定义（sqlite-vec 返回 NULL）：跳过该行，避免崩溃
        return [(int(row[0]), float(row[1])) for row in rows if row[1] is not None]

    def _fts_search(self, query: str, limit: int) -> list[tuple[int, int]]:
        """全文通道：FTS5 BM25 排序，返回 (chunk_id, 名次)；异常时降级为空结果。"""
        match = build_fts_query(query)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # 查询语法异常 / 旧库无 FTS 数据：不中断混合检索，向量通道兜底
            logger.warning("FTS5 检索降级为空结果: %s", exc)
            return []
        return [(int(row[0]), rank) for rank, row in enumerate(rows)]

    @staticmethod
    def _fuse_rankings(
        vector_rows: list[tuple[int, float]],
        fts_rows: list[tuple[int, int]],
        *,
        mode: str,
        rrf_k: int,
    ) -> list[tuple[int, float, list[str]]]:
        """融合双通道排名，返回 (chunk_id, 归一化分数, 命中通道列表)。"""
        if mode == "vector":
            ranked = [(chunk_id, 1.0 - distance) for chunk_id, distance in vector_rows]
            sources = {chunk_id: ["vector"] for chunk_id, _ in vector_rows}
        elif mode == "fts":
            ranked = [
                (chunk_id, 1.0 / (rrf_k + rank + 1)) for chunk_id, rank in fts_rows
            ]
            sources = {chunk_id: ["fts"] for chunk_id, _ in fts_rows}
        else:
            # RRF：每通道按名次贡献 1/(k+rank+1)，双通道同时命中者得分累加
            scores: dict[int, float] = defaultdict(float)
            sources: dict[int, list[str]] = defaultdict(list)
            for channel, rows in (("vector", vector_rows), ("fts", fts_rows)):
                for rank, (chunk_id, _) in enumerate(rows):
                    scores[chunk_id] += 1.0 / (rrf_k + rank + 1)
                    if channel not in sources[chunk_id]:
                        sources[chunk_id].append(channel)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            return []
        max_score = ranked[0][1] or 1.0
        return [
            (chunk_id, score / max_score, sources.get(chunk_id, []))
            for chunk_id, score in ranked
        ]

    def _build_hits(self, fused: list[tuple[int, float, list[str]]]) -> list[SearchHit]:
        """按融合后的 chunk_id 顺序取回分块内容，构造命中列表。"""
        if not fused:
            return []
        placeholders = ",".join("?" for _ in fused)
        chunk_rows = self._conn.execute(
            "SELECT chunk_id, path, chunk_index, content, char_start, char_end"
            f" FROM chunks WHERE chunk_id IN ({placeholders})",
            [item[0] for item in fused],
        ).fetchall()
        by_id = {row[0]: row for row in chunk_rows}
        hits: list[SearchHit] = []
        for chunk_id, score, sources in fused:
            row = by_id.get(chunk_id)
            if row is None:
                continue
            hits.append(
                SearchHit(
                    path=row[1],
                    chunk_index=row[2],
                    content=row[3],
                    char_start=row[4],
                    char_end=row[5],
                    score=round(score, 4),
                    sources=sources,
                )
            )
        return hits

    # ---- 删除与重建 ----

    def delete_document(self, path: str) -> bool:
        """删除一个文档及其全部分块与向量；返回是否确实删除了内容。"""
        existed = self.get_document(path) is not None
        if existed:
            with self._conn:
                self._delete_document_rows(path)
        return existed

    def reset(self) -> None:
        """清空全部索引数据（rebuild=true 时调用），保留当前模型元数据。"""
        with self._conn:
            self._conn.execute("DROP TABLE IF EXISTS vec_chunks")
            self._conn.execute("DROP TABLE IF EXISTS chunks_fts")
            self._conn.execute("DROP TABLE IF EXISTS chunks")
            self._conn.execute("DROP TABLE IF EXISTS documents")
            self._conn.execute("DELETE FROM rag_meta WHERE key != 'embedding_model'")
        self._ensure_base_schema()
        self._meta_set("embedding_model", self.model_id)

    def _delete_document_rows(self, path: str) -> None:
        """删除文档的向量行与分块行（documents 级联删除 chunks）。"""
        self._conn.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN"
            " (SELECT chunk_id FROM chunks WHERE path = ?)",
            (path,),
        )
        # FTS 行必须先于 documents 删除（级联会清掉 chunks，导致子查询查不到）
        self._conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN"
            " (SELECT chunk_id FROM chunks WHERE path = ?)",
            (path,),
        )
        self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
