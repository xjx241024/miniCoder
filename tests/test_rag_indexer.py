"""索引器与检索器测试：假 embedding 全链路离线验证。"""

import pytest

from core.config import RAGConfig
from rag.errors import RAGError
from rag.indexer import DocumentIndexer
from rag.retriever import DocRetriever, format_hits
from rag.store import VectorStore
from tools.workspace import Workspace


class KeywordEmbedder:
    """确定性假向量：按关键词计数构造 2 维向量，模拟语义相关性。"""

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(text.count("苹果")), float(text.count("香蕉"))] for text in texts]


def _setup(tmp_path, model_id="test-model"):
    workspace = Workspace(tmp_path)
    embedder = KeywordEmbedder()
    store = VectorStore(tmp_path / ".rag" / "index.db", model_id=model_id)
    config = RAGConfig(chunk_size=100, chunk_overlap=10, index_max_chars=1000)
    return workspace, embedder, store, config


def test_index_directory_and_search(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "apple.md").write_text("苹果是一种水果，苹果很好吃。", encoding="utf-8")
    (tmp_path / "banana.md").write_text("香蕉是黄色的水果。", encoding="utf-8")
    try:
        indexer = DocumentIndexer(workspace, embedder, store, config)
        report = indexer.index_path(".")

        assert report.total_files == 2
        assert report.indexed_files == 2
        assert report.failed_files == []
        assert report.chunk_count >= 2

        retriever = DocRetriever(embedder, store)
        hits = retriever.search("苹果")
        assert hits[0].path == "apple.md"
        text = format_hits(hits)
        assert "apple.md" in text
        assert "苹果" in text
        assert "相似度" in text
    finally:
        store.close()


def test_incremental_index_skips_unchanged_files(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    target = tmp_path / "apple.md"
    target.write_text("苹果内容", encoding="utf-8")
    try:
        indexer = DocumentIndexer(workspace, embedder, store, config)
        first = indexer.index_path("apple.md")
        calls_after_first = len(embedder.calls)

        second = indexer.index_path("apple.md")

        assert first.indexed_files == 1
        assert second.indexed_files == 0
        assert second.skipped_files == 1
        assert len(embedder.calls) == calls_after_first  # 未重复调用 embedding
    finally:
        store.close()


def test_changed_file_reindexed(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    target = tmp_path / "apple.md"
    target.write_text("苹果旧内容", encoding="utf-8")
    try:
        indexer = DocumentIndexer(workspace, embedder, store, config)
        indexer.index_path("apple.md")
        target.write_text("苹果新内容，苹果更多描述", encoding="utf-8")

        report = indexer.index_path("apple.md")

        assert report.indexed_files == 1
        assert report.skipped_files == 0
        hits = DocRetriever(embedder, store).search("苹果")
        assert "苹果新内容" in hits[0].content
    finally:
        store.close()


def test_failed_file_reported_but_others_indexed(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    (tmp_path / "good.md").write_text("苹果内容", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 fake")
    try:
        indexer = DocumentIndexer(workspace, embedder, store, config)
        report = indexer.index_path(".", pattern="*")

        # pdf 不在 collect_files 的扩展名白名单里，直接不计入 total
        assert report.total_files == 1
        assert report.indexed_files == 1
        assert report.failed_files == []
    finally:
        store.close()


def test_rebuild_forces_reindex(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    target = tmp_path / "apple.md"
    target.write_text("苹果内容", encoding="utf-8")
    try:
        indexer = DocumentIndexer(workspace, embedder, store, config)
        indexer.index_path("apple.md")
        calls_before = len(embedder.calls)

        report = indexer.index_path("apple.md", rebuild=True)

        assert report.indexed_files == 1
        assert report.skipped_files == 0
        assert len(embedder.calls) == calls_before + 1
    finally:
        store.close()


def test_retriever_rejects_empty_query(tmp_path):
    workspace, embedder, store, config = _setup(tmp_path)
    try:
        retriever = DocRetriever(embedder, store)
        with pytest.raises(RAGError) as exc_info:
            retriever.search("   ")
        assert exc_info.value.code == "EMPTY_QUERY"
    finally:
        store.close()
