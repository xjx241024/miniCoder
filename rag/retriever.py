"""检索编排：查询向量化 → 向量 top-k 召回 → 组装带来源的结果（M11）。"""

from __future__ import annotations

from rag.errors import RAGError
from rag.store import SearchHit


class DocRetriever:
    """在已索引文档库中做语义检索。"""

    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        """把查询向量化并召回最相关的分块；空查询直接拒绝。"""
        if not query or not query.strip():
            raise RAGError("EMPTY_QUERY", "检索查询不能为空")
        vector = self.embedder.embed_texts([query.strip()])[0]
        return self.store.search(vector, top_k)


def format_hits(hits: list[SearchHit]) -> str:
    """把命中列表渲染成带来源引用的文本（工具的 text 字段，供模型阅读）。"""
    if not hits:
        return "未检索到相关内容。索引中可能没有对应知识，可考虑用 index_docs 补充文档。"
    lines: list[str] = []
    for number, hit in enumerate(hits, start=1):
        lines.append(
            f"[{number}] 来源: {hit.path}（第 {hit.chunk_index + 1} 块，相似度 {hit.score:.2f}）"
        )
        lines.append(hit.content.strip())
        lines.append("")
    return "\n".join(lines).rstrip()
