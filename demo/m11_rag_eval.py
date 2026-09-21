"""M11 增强演示：检索质量评测（Hit@K / MRR），对比 vector / fts / hybrid 三种模式。

不调用真实模型：用确定性假向量（关键词计数）+ 合成语料，
展示混合检索如何补足纯向量在"词面不同但相关"查询上的短板。
真实评测可在配置好 Embedding 后，用同样的 evaluate_retrieval 对真实语料运行。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.config import RAGConfig
from rag.evaluation import EvalCase, evaluate_retrieval
from rag.indexer import DocumentIndexer
from rag.retriever import DocRetriever
from rag.store import VectorStore
from tools.workspace import Workspace


class DemoEmbedder:
    """演示用假向量：只统计 苹果/香蕉 两个关键词（故意制造向量盲区）。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1 + text.count("苹果"), 0.1 + text.count("香蕉")] for text in texts]


CORPUS = {
    "apple.md": "苹果是一种水果，苹果富含膳食纤维，苹果很好吃。",
    "banana.md": "香蕉是黄色的水果，香蕉富含钾元素。",
    "fiber.md": "膳食纤维有助于肠道健康，常见于苹果与全谷物中。",
    "potassium.md": "钾元素是人体必需的矿物质，香蕉与豆类含量丰富。",
    "notes.md": "会议记录：下周发布新版本，重点优化检索质量。",
}

CASES = [
    EvalCase(query="苹果有什么特点？", relevant_paths=["apple.md", "fiber.md"]),
    EvalCase(query="香蕉的营养成分", relevant_paths=["banana.md", "potassium.md"]),
    EvalCase(query="膳食纤维对健康的作用", relevant_paths=["fiber.md"]),
    EvalCase(query="哪种水果含钾多", relevant_paths=["banana.md", "potassium.md"]),
    EvalCase(query="钾元素的作用", relevant_paths=["potassium.md"]),
]


def main() -> None:
    """离线对比三种检索模式的 Hit@K 与 MRR。"""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, content in CORPUS.items():
            (root / name).write_text(content, encoding="utf-8")

        workspace = Workspace(root)
        store = VectorStore(root / "rag-data" / "index.db", model_id="demo-model")
        embedder = DemoEmbedder()
        try:
            indexer = DocumentIndexer(
                workspace, embedder, store, RAGConfig(chunk_size=50, chunk_overlap=10)
            )
            report = indexer.index_path(".")
            print(f"索引完成：{report.indexed_files} 个文件 / {report.chunk_count} 个分块")

            print(f"\n评测集：{len(CASES)} 条查询，top_k=3")
            print(f"{'模式':<10}{'Hit@K':>8}{'MRR':>8}")
            for mode in ("vector", "fts", "hybrid"):
                retriever = DocRetriever(
                    embedder, store, config=RAGConfig(search_mode=mode)
                )
                result = evaluate_retrieval(retriever.search, CASES, top_k=3)
                print(f"{mode:<10}{result.hit_rate:>8.2%}{result.mrr:>8.3f}")

            print("\n说明：假向量只识别 苹果/香蕉 两个词，")
            print("因此 膳食纤维/钾元素 类查询在纯向量模式下会失手，")
            print("FTS5 关键词通道 + RRF 融合（hybrid）能显著提升召回。")
        finally:
            store.close()


if __name__ == "__main__":
    main()
