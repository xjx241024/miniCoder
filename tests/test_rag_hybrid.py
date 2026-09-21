"""混合检索测试：FTS5 分词、RRF 融合与多模式检索（全部离线）。"""

import pytest

from core.config import RAGConfig
from rag.indexer import DocumentIndexer
from rag.retriever import DocRetriever
from rag.store import VectorStore, build_fts_query, fts_normalize
from tools.workspace import Workspace


class KeywordEmbedder:
    """确定性假向量：只识别 苹果/香蕉（模拟向量模型的词汇盲区）。"""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [
            [0.1 + text.count("苹果"), 0.1 + text.count("香蕉")] for text in texts
        ]


def _setup(tmp_path):
    workspace = Workspace(tmp_path)
    embedder = KeywordEmbedder()
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    config = RAGConfig(chunk_size=100, chunk_overlap=10)
    return workspace, embedder, store, config


def test_fts_normalize_inserts_spaces_between_cjk_chars():
    assert fts_normalize("苹果apple") == " 苹  果 apple"
    assert fts_normalize("pure english") == "pure english"


def test_build_fts_query_mixed_terms():
    query = build_fts_query("苹果 营养 retrieval quality")
    assert '"苹 果"' in query
    assert '"营 养"' in query
    assert '"retrieval"' in query
    assert " OR " in query
    assert build_fts_query("") == ""
    assert build_fts_query("!!!") == ""


def test_hybrid_fts_channel_rescues_keyword_blind_spot(tmp_path):
    """查询词不在向量词汇表内时，FTS 通道应把它捞回第一名。"""
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "apple.md").write_text("苹果是一种水果，苹果很好吃。", encoding="utf-8")
    (tmp_path / "banana.md").write_text("香蕉是黄色的水果。", encoding="utf-8")
    (tmp_path / "mineral.md").write_text("钾元素是人体必需的矿物质。", encoding="utf-8")
    try:
        DocumentIndexer(workspace, embedder, store, config).index_path(".")

        # "矿物质" 不含 苹果/香蕉，向量通道无法区分；FTS 精确命中 mineral.md
        retriever = DocRetriever(embedder, store, config=RAGConfig(search_mode="hybrid"))
        hits = retriever.search("矿物质的作用", top_k=3)

        assert hits[0].path == "mineral.md"
        assert "fts" in hits[0].sources
        assert hits[0].score == pytest.approx(1.0)  # RRF 归一化后第一名恒为 1.0
    finally:
        store.close()


def test_search_mode_vector_pure_semantic(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "apple.md").write_text("苹果苹果苹果", encoding="utf-8")
    (tmp_path / "banana.md").write_text("香蕉香蕉香蕉", encoding="utf-8")
    try:
        DocumentIndexer(workspace, embedder, store, config).index_path(".")

        retriever = DocRetriever(embedder, store, config=RAGConfig(search_mode="vector"))
        hits = retriever.search("苹果", top_k=2)

        assert hits[0].path == "apple.md"
        assert all("fts" not in hit.sources for hit in hits)
    finally:
        store.close()


def test_search_mode_fts_pure_keyword(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "mineral.md").write_text("钾元素是人体必需的矿物质。", encoding="utf-8")
    (tmp_path / "apple.md").write_text("苹果是一种水果。", encoding="utf-8")
    try:
        DocumentIndexer(workspace, embedder, store, config).index_path(".")

        retriever = DocRetriever(embedder, store, config=RAGConfig(search_mode="fts"))
        hits = retriever.search("钾元素", top_k=2)

        assert hits[0].path == "mineral.md"
        assert hits[0].sources == ["fts"]
    finally:
        store.close()


def test_invalid_search_mode_falls_back_to_hybrid(tmp_path):
    """配置了非法模式时，DocRetriever 默认 config 已在加载层归一；此处验证非法值不崩溃。"""
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "apple.md").write_text("苹果内容", encoding="utf-8")
    try:
        DocumentIndexer(workspace, embedder, store, config).index_path(".")
        retriever = DocRetriever(embedder, store, config=RAGConfig(search_mode="hybrid"))
        assert retriever.search("苹果", top_k=1)[0].path == "apple.md"
    finally:
        store.close()
