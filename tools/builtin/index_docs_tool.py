"""index_docs 工具：为工作区内文本文档建立/更新语义检索索引（M11 RAG）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.config import RAGConfig, load_rag_config
from rag.backend import RAGBackend, open_rag_backend
from rag.errors import RAGError
from rag.indexer import DocumentIndexer
from tools.base import BaseTool, ToolResult
from tools.workspace import Workspace


class IndexDocsTool(BaseTool):
    """RAG 索引入口：目录递归或单文件，默认增量、可强制全量重建。"""

    name = "index_docs"
    description = (
        "为工作区内的文本文档建立或更新语义检索索引（RAG 知识库）。"
        "path 支持目录（递归索引 Markdown/文本/代码/配置文件）或单个文件；"
        "默认增量索引：内容未变的文件自动跳过；rebuild=true 强制全量重建"
        "（更换 embedding 模型后必须重建）。索引完成后用 search_docs 检索。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要索引的目录或文件（工作区内路径）",
            },
            "pattern": {
                "type": "string",
                "description": "文件名过滤（fnmatch 语法，如 *.md），默认 * 全部支持文件",
            },
            "rebuild": {
                "type": "boolean",
                "description": "是否强制全量重建索引（默认 false 增量更新）",
            },
        },
        "required": ["path"],
    }

    def __init__(
        self,
        workspace: Workspace,
        backend_factory: Callable | None = None,
        rag_config: RAGConfig | None = None,
        data_dir: str | Path | None = None,
    ):
        self.workspace = workspace
        self._backend_factory = backend_factory  # 测试注入（假 embedder + 临时库）
        self.rag_config = rag_config or load_rag_config()
        self.data_dir = data_dir

    def _run(self, arguments: dict) -> ToolResult:
        backend: RAGBackend | None = None
        try:
            backend = self._open_backend()
            rebuild = bool(arguments.get("rebuild", False))
            if rebuild:
                backend.store.reset()
            indexer = DocumentIndexer(
                self.workspace, backend.embedder, backend.store, self.rag_config
            )
            report = indexer.index_path(
                arguments.get("path", ""),
                pattern=arguments.get("pattern", "*"),
                rebuild=rebuild,
            )
            summary = (
                f"索引完成：共扫描 {report.total_files} 个支持文件，"
                f"新建/更新 {report.indexed_files} 个，跳过未变化 {report.skipped_files} 个，"
                f"分块总数 {report.chunk_count}。"
            )
            if report.failed_files:
                summary += f"（{len(report.failed_files)} 个文件失败，详见 data 字段）"
            return ToolResult.success(data=report.to_dict(), text=summary)
        except RAGError as exc:
            return ToolResult.failure(exc.code, exc.message)
        finally:
            # 每次调用独立开关后端：避免长期持有 SQLite 连接与 HTTP 连接
            if backend is not None:
                backend.close()

    def _open_backend(self) -> RAGBackend:
        """打开 RAG 后端：测试注入工厂优先，否则按 .env 配置构建真实后端。"""
        if self._backend_factory is not None:
            return self._backend_factory()
        return open_rag_backend(self.workspace, data_dir=self.data_dir)
