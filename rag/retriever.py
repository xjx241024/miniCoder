"""检索编排：混合召回 → 可选 rerank 精排 → 组装带来源的结果（M11 增强）。"""

from __future__ import annotations

from core.config import RAGConfig
from rag.errors import RAGError
from rag.store import SearchHit


class DocRetriever:
    """在已索引文档库中做混合检索（向量 + FTS5 + RRF，可选 rerank）。"""

    def __init__(self, embedder, store, reranker=None, config: RAGConfig | None = None):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.config = config or RAGConfig()

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """混合召回 → rerank 精排（可选）→ 截断 top_k。"""
        if not query or not query.strip():
            raise RAGError("EMPTY_QUERY", "检索查询不能为空")
        query = query.strip()
        vector = self.embedder.embed_texts([query])[0]
        # 先放大候选池（top_k * multiplier），融合/rerank 后再截断
        candidates = self.store.search(
            vector,
            top_k,
            query=query,
            mode=self.config.search_mode,
            rrf_k=self.config.rrf_k,
            candidate_multiplier=self.config.candidate_multiplier,
        )
        if self.reranker is not None and candidates:
            return self.reranker.rerank(query, candidates, top_n=top_k)
        return candidates[:top_k]


def format_hits(hits: list[SearchHit]) -> str:
    """把命中列表渲染成带来源引用的文本（工具的 text 字段，供模型阅读）。"""
    if not hits:
        return "未检索到相关内容。索引中可能没有对应知识，可考虑用 index_docs 补充文档。"
    lines: list[str] = []
    for number, hit in enumerate(hits, start=1):
        channels = "/".join(hit.sources) if hit.sources else "unknown"
        lines.append(
            f"[{number}] 来源: {hit.path}（第 {hit.chunk_index + 1} 块，"
            f"分数 {hit.score:.2f}，通道 {channels}）"
        )
        lines.append(hit.content.strip())
        lines.append("")
    return "\n".join(lines).rstrip()
