"""RAG 后端工厂：按配置组装 Embedding / 向量库 / 可选 Rerank（M11 增强）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.config import EmbeddingConfig, load_embedding_config, load_rerank_config
from memory.paths import default_rag_dir
from rag.embeddings import EmbeddingClient
from rag.errors import RAGError
from rag.rerank import RerankClient
from rag.store import VectorStore
from tools.workspace import Workspace


@dataclass
class RAGBackend:
    """一次检索后端的完整句柄：统一打开与关闭，便于工具层生命周期管理。"""

    embedder: Any
    store: VectorStore
    reranker: RerankClient | None = None

    def close(self) -> None:
        """按依赖顺序关闭（rerank → store → embedder）。"""
        for backend in (self.reranker, self.store, self.embedder):
            close = getattr(backend, "close", None)
            if close is not None:
                close()


def open_rag_backend(
    workspace: Workspace, data_dir: str | Path | None = None
) -> RAGBackend:
    """构建真实 RAG 后端；未配置 API Key 时返回可诊断错误（不抛网络异常）。"""
    config = load_embedding_config()
    if not config.api_key:
        raise RAGError(
            "EMBEDDING_NOT_CONFIGURED",
            "未配置 EMBEDDING_API_KEY：请在 .env 中填写 EMBEDDING_* 配置"
            "（默认 SiliconFlow BAAI/bge-m3），保存后重新运行 minicoder",
        )
    db_path = default_rag_dir(workspace_root=workspace.root, data_dir=data_dir) / "index.db"
    embedder = EmbeddingClient(config)
    try:
        store = VectorStore(db_path, model_id=config.model_id)
        return RAGBackend(
            embedder=embedder,
            store=store,
            reranker=_build_reranker(config),
        )
    except Exception:
        embedder.close()
        raise


def _build_reranker(embedding_config: EmbeddingConfig) -> RerankClient | None:
    """按配置构建可选 rerank 客户端；启用但缺 Key 时给出可诊断错误。"""
    rerank_config = load_rerank_config()
    if not rerank_config.enabled:
        return None
    api_key = rerank_config.api_key or embedding_config.api_key
    if not api_key:
        raise RAGError(
            "RERANK_NOT_CONFIGURED",
            "已启用 RERANK_ENABLED=1，但缺少 API Key："
            "请配置 RERANK_API_KEY 或 EMBEDDING_API_KEY",
        )
    # base_url 未单独配置时复用 Embedding 服务（SiliconFlow 同 Key 可用）
    base_url = rerank_config.base_url or embedding_config.base_url
    return RerankClient(
        rerank_config.model_copy(update={"api_key": api_key, "base_url": base_url})
    )
