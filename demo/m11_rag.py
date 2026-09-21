"""M11 演示：RAG 子系统（索引 → 语义检索 → 带来源引用的结果）。

不调用真实模型：在临时目录里创建示例文档，用确定性假向量
完整走一遍"解析 → 分块 → 向量化 → 入库 → 检索"链路。
真实使用请配置 .env 的 EMBEDDING_* 后运行 minicoder，
让模型自主调用 index_docs / search_docs 工具。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.config import RAGConfig
from rag.indexer import DocumentIndexer
from rag.retriever import DocRetriever, format_hits
from rag.store import VectorStore
from tools.workspace import Workspace


class DemoEmbedder:
    """演示用假向量：按关键词计数构造 2 维向量，模拟语义相关性。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1 + text.count("苹果"), 0.1 + text.count("香蕉")]
            for text in texts
        ]


DEMO_DOCS = {
    "apple.md": "苹果是一种水果，苹果富含膳食纤维，苹果很好吃。",
    "banana.md": "香蕉是黄色的水果，香蕉富含钾元素。",
    "notes.md": "会议记录：下周发布新版本，重点优化检索质量。",
}


def main() -> None:
    """在隔离目录里演示索引与检索全链路。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in DEMO_DOCS.items():
            (root / name).write_text(content, encoding="utf-8")

        workspace = Workspace(root)
        store = VectorStore(root / "rag-data" / "index.db", model_id="demo-model")
        embedder = DemoEmbedder()
        try:
            indexer = DocumentIndexer(
                workspace, embedder, store, RAGConfig(chunk_size=50, chunk_overlap=10)
            )
            report = indexer.index_path(".")
            print("==== 索引 ====")
            print(
                f"扫描 {report.total_files} 个文件，"
                f"索引 {report.indexed_files} 个，分块 {report.chunk_count} 个"
            )

            retriever = DocRetriever(embedder, store)
            print("\n==== 检索：苹果 ====")
            hits = retriever.search("苹果有什么特点？", top_k=2)
            print(format_hits(hits))

            print("\n==== 检索：香蕉 ====")
            hits = retriever.search("香蕉的营养成分", top_k=2)
            print(format_hits(hits))

            print("\n==== 增量索引 ====")
            again = indexer.index_path(".")
            print(
                f"第二次索引：跳过未变化 {again.skipped_files} 个文件"
                f"（不重复调用 Embedding）"
            )
        finally:
            store.close()


if __name__ == "__main__":
    main()
