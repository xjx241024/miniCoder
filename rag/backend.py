"""RAG 后端工厂：按配置组装 Embedding 客户端与向量库连接（M11）。"""

from __future__ import annotations

from pathlib import Path

from core.config import load_embedding_config
from memory.paths import default_rag_dir
from rag.embeddings import EmbeddingClient
from rag.errors import RAGError
from rag.store import VectorStore
from tools.workspace import Workspace


def open_rag_backend(
    workspace: Workspace, data_dir: str | Path | None = None
) -> tuple[EmbeddingClient, VectorStore]:
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
    except Exception:
        embedder.close()
        raise
    return embedder, store
