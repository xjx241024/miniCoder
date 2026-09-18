"""索引编排：收集 → 解析 → 分块 → 增量比对 → 向量化 → 入库（M11）。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config import RAGConfig
from rag.chunking import Chunker
from rag.errors import RAGError
from rag.parser import collect_files, parse_file
from rag.store import VectorStore
from tools.workspace import Workspace


@dataclass
class IndexReport:
    """一次索引任务的可读报告（同时作为工具返回 data 落入 trace）。"""

    total_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    chunk_count: int = 0
    failed_files: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典。"""
        return asdict(self)


class DocumentIndexer:
    """把工作区内的文本文档构建为可语义检索的本地索引。"""

    def __init__(
        self,
        workspace: Workspace,
        embedder,
        store: VectorStore,
        config: RAGConfig | None = None,
    ):
        self.workspace = workspace
        self.embedder = embedder
        self.store = store
        self.config = config or RAGConfig()
        self.chunker = Chunker(self.config.chunk_size, self.config.chunk_overlap)

    def index_path(
        self,
        requested: str | Path,
        *,
        pattern: str = "*",
        rebuild: bool = False,
    ) -> IndexReport:
        """索引一个目录或文件；默认增量（内容哈希未变则跳过）。"""
        files = collect_files(self.workspace, requested, pattern=pattern)
        report = IndexReport(total_files=len(files))
        for abs_path in files:
            try:
                document = parse_file(
                    self.workspace, abs_path, max_chars=self.config.index_max_chars
                )
                stored = self.store.get_document(document.path)
                if stored and not rebuild and stored["content_hash"] == document.content_hash:
                    report.skipped_files += 1
                    continue
                chunks = self.chunker.split(document.content)
                vectors = (
                    self.embedder.embed_texts([chunk.content for chunk in chunks])
                    if chunks
                    else []
                )
                self.store.upsert_document(document, chunks, vectors)
            except RAGError as exc:
                # 单个文件失败不影响其余文件继续索引
                report.failed_files.append(
                    {
                        "path": self._relative_or_raw(abs_path),
                        "code": exc.code,
                        "message": exc.message,
                    }
                )
                continue
            report.indexed_files += 1
            report.chunk_count += len(chunks)
        return report

    def _relative_or_raw(self, path: Path) -> str:
        """把绝对路径转为工作区相对路径（失败时原样返回，用于错误报告）。"""
        try:
            return self.workspace.relative(path)
        except Exception:
            return str(path)
