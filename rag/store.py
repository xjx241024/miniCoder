"""sqlite-vec 向量库：文档/分块元数据 + 向量虚拟表，余弦相似检索（M11）。

索引库为单文件 SQLite（index.db），符合"本地优先"定位；
向量表使用 sqlite-vec 扩展，距离度量固定为 cosine。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from rag.chunking import Chunk
from rag.errors import RAGError
from rag.parser import Document


@dataclass
class SearchHit:
    """一条检索命中：来源分块 + 余弦距离（distance 越小越相似）。"""

    path: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    distance: float

    @property
    def score(self) -> float:
        """把余弦距离换算成相似度分数（1.0 完全相同，0.0 无关）。"""
        return 1.0 - self.distance


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

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        """向量 top-k 检索：返回带来源路径与原文片段的命中列表。"""
        if not self._vec_table_exists():
            raise RAGError("INDEX_EMPTY", "索引为空，请先调用 index_docs 工具建立索引")
        stored = self._meta_get("embedding_dimension")
        if stored is not None and int(stored) != len(vector):
            raise RAGError(
                "EMBEDDING_DIMENSION_MISMATCH",
                f"索引向量维度为 {stored}，查询向量为 {len(vector)} 维；"
                "请用 index_docs(rebuild=true) 全量重建索引",
            )
        try:
            rows = self._conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks"
                " WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (sqlite_vec.serialize_float32(vector), top_k),
            ).fetchall()
        except sqlite3.Error as exc:
            raise RAGError("STORE_ERROR", f"向量检索失败: {exc}") from exc
        if not rows:
            return []
        placeholders = ",".join("?" for _ in rows)
        chunk_rows = self._conn.execute(
            "SELECT chunk_id, path, chunk_index, content, char_start, char_end"
            f" FROM chunks WHERE chunk_id IN ({placeholders})",
            [row[0] for row in rows],
        ).fetchall()
        by_id = {row[0]: row for row in chunk_rows}
        hits: list[SearchHit] = []
        for chunk_id, distance in rows:
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
                    distance=float(distance),
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
        self._conn.execute("DELETE FROM documents WHERE path = ?", (path,))

    def close(self) -> None:
        """关闭数据库连接。"""
        self._conn.close()

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
