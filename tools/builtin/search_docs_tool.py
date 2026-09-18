"""search_docs 工具：在已索引文档库中语义检索，返回带来源引用的片段（M11 RAG）。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag.backend import open_rag_backend
from rag.errors import RAGError
from rag.retriever import DocRetriever, format_hits
from tools.base import BaseTool, ToolResult
from tools.workspace import Workspace


class SearchDocsTool(BaseTool):
    """RAG 检索入口：查询向量化 → top-k 召回 → 带来源路径的结果文本。"""

    name = "search_docs"
    description = (
        "在已索引的文档知识库中做语义检索，返回带来源路径、分块位置与相似度的原文片段。"
        "回答文档/知识库问题前，先确保已用 index_docs 建立索引，"
        "再调用本工具检索，并基于返回内容回答（建议引用来源路径）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索问题或关键词（会向量化后做语义匹配）",
            },
            "top_k": {
                "type": "integer",
                "description": "返回最相关的前 K 条（默认 5，范围 1-20）",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        workspace: Workspace,
        backend_factory: Callable | None = None,
        data_dir: str | Path | None = None,
    ):
        self.workspace = workspace
        self._backend_factory = backend_factory
        self.data_dir = data_dir

    def _run(self, arguments: dict) -> ToolResult:
        embedder = None
        store = None
        try:
            embedder, store = self._open_backend()
            retriever = DocRetriever(embedder, store)
            top_k = self._clamp_top_k(arguments.get("top_k", 5))
            hits = retriever.search(arguments.get("query", ""), top_k=top_k)
            data = {
                "query": arguments.get("query", ""),
                "top_k": top_k,
                "hits": [
                    {
                        "path": hit.path,
                        "chunk_index": hit.chunk_index,
                        "content": hit.content,
                        "score": round(hit.score, 4),
                    }
                    for hit in hits
                ],
            }
            return ToolResult.success(data=data, text=format_hits(hits))
        except RAGError as exc:
            return ToolResult.failure(exc.code, exc.message)
        finally:
            for backend in (store, embedder):
                close = getattr(backend, "close", None)
                if close is not None:
                    close()

    def _open_backend(self):
        """打开 RAG 后端：测试注入工厂优先，否则按 .env 配置构建真实后端。"""
        if self._backend_factory is not None:
            return self._backend_factory()
        return open_rag_backend(self.workspace, data_dir=self.data_dir)

    @staticmethod
    def _clamp_top_k(value) -> int:
        """把 top_k 归一为 1-20 的整数，避免模型传入异常值。"""
        try:
            top_k = int(value)
        except (TypeError, ValueError):
            top_k = 5
        return max(1, min(20, top_k))
