"""向量库测试：用确定性假向量离线验证写入、检索与增量元数据。"""

import pytest

from rag.errors import RAGError
from rag.parser import Document
from rag.store import VectorStore


def _document(path: str, content: str, tmp_path) -> Document:
    """构造测试用文档（指纹字段用占位值，store 不校验其真实性）。"""
    abs_path = tmp_path / path
    return Document(
        path=path,
        abs_path=abs_path,
        content=content,
        size_bytes=len(content.encode("utf-8")),
        mtime_ms=1000,
        content_hash="hash-" + path,
    )


def _chunk(index: int, content: str, start: int, end: int):
    from rag.chunking import Chunk

    return Chunk(index=index, content=content, char_start=start, char_end=end)


def test_upsert_and_get_document(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        doc = _document("a.md", "内容", tmp_path)
        store.upsert_document(doc, [_chunk(0, "内容", 0, 2)], [[1.0, 0.0]])

        stored = store.get_document("a.md")
        assert stored is not None
        assert stored["content_hash"] == "hash-a.md"
        assert stored["chunk_count"] == 1
    finally:
        store.close()


def test_search_returns_topk_ordered_by_similarity(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        doc_a = _document("apple.md", "苹果", tmp_path)
        doc_b = _document("banana.md", "香蕉", tmp_path)
        store.upsert_document(doc_a, [_chunk(0, "苹果", 0, 2)], [[1.0, 0.0]])
        store.upsert_document(doc_b, [_chunk(0, "香蕉", 0, 2)], [[0.0, 1.0]])

        hits = store.search([1.0, 0.0], top_k=2)

        assert [hit.path for hit in hits] == ["apple.md", "banana.md"]
        assert hits[0].distance == pytest.approx(0.0)
        assert hits[0].score == pytest.approx(1.0)
        assert hits[0].content == "苹果"
        assert hits[1].distance == pytest.approx(1.0)
    finally:
        store.close()


def test_upsert_replaces_old_chunks_without_duplicates(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        doc = _document("a.md", "旧内容", tmp_path)
        store.upsert_document(doc, [_chunk(0, "旧内容", 0, 3)], [[1.0, 0.0]])
        doc.content_hash = "hash-v2"
        store.upsert_document(doc, [_chunk(0, "新内容", 0, 3)], [[0.9, 0.1]])

        hits = store.search([1.0, 0.0], top_k=10)
        assert len(hits) == 1
        assert hits[0].content == "新内容"
        assert store.get_document("a.md")["chunk_count"] == 1
    finally:
        store.close()


def test_delete_document_removes_chunks_and_vectors(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        doc = _document("a.md", "内容", tmp_path)
        store.upsert_document(doc, [_chunk(0, "内容", 0, 2)], [[1.0, 0.0]])

        assert store.delete_document("a.md") is True
        assert store.get_document("a.md") is None
        assert store.search([1.0, 0.0], top_k=5) == []
        assert store.delete_document("a.md") is False
    finally:
        store.close()


def test_store_reopen_persists_data(tmp_path):
    db_path = tmp_path / "index.db"
    store = VectorStore(db_path, model_id="test-model")
    doc = _document("a.md", "内容", tmp_path)
    store.upsert_document(doc, [_chunk(0, "内容", 0, 2)], [[1.0, 0.0]])
    store.close()

    reopened = VectorStore(db_path, model_id="test-model")
    try:
        assert reopened.get_document("a.md") is not None
        assert len(reopened.search([1.0, 0.0], top_k=1)) == 1
    finally:
        reopened.close()


def test_model_mismatch_rejected(tmp_path):
    db_path = tmp_path / "index.db"
    store = VectorStore(db_path, model_id="model-a")
    store.close()

    with pytest.raises(RAGError) as exc_info:
        VectorStore(db_path, model_id="model-b")
    assert exc_info.value.code == "EMBEDDING_MODEL_MISMATCH"


def test_search_empty_index_raises(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        with pytest.raises(RAGError) as exc_info:
            store.search([1.0, 0.0], top_k=5)
        assert exc_info.value.code == "INDEX_EMPTY"
    finally:
        store.close()


def test_dimension_mismatch_on_search_rejected(tmp_path):
    store = VectorStore(tmp_path / "index.db", model_id="test-model")
    try:
        doc = _document("a.md", "内容", tmp_path)
        store.upsert_document(doc, [_chunk(0, "内容", 0, 2)], [[1.0, 0.0]])
        with pytest.raises(RAGError) as exc_info:
            store.search([1.0, 0.0, 0.0], top_k=5)
        assert exc_info.value.code == "EMBEDDING_DIMENSION_MISMATCH"
    finally:
        store.close()
