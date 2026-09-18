"""Embedding 客户端测试：MockTransport 离线验证批量、排序与重试。"""

import httpx
import pytest

from core.config import EmbeddingConfig
from rag.embeddings import EmbeddingClient
from rag.errors import RAGError


def _make_client(handler, **config_overrides) -> EmbeddingClient:
    """构造注入 MockTransport 的客户端（可覆盖配置便于测批量/重试）。"""
    config = EmbeddingConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        retry_backoff_seconds=0.0,
        **config_overrides,
    )
    return EmbeddingClient(config, transport=httpx.MockTransport(handler))


def _response(vectors: list[list[float]]) -> dict:
    """构造 OpenAI 兼容 /embeddings 响应（index 乱序，验证排序逻辑）。"""
    items = [
        {"index": i, "embedding": vector} for i, vector in reversed(list(enumerate(vectors)))
    ]
    return {"data": items, "model": "mock"}


def test_embed_texts_success_and_order():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = request.read()
        assert b"BAAI/bge-m3" in payload
        return httpx.Response(200, json=_response([[1.0, 0.0], [0.0, 1.0]]))

    client = _make_client(handler)
    try:
        vectors = client.embed_texts(["你好", "世界"])
    finally:
        client.close()
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_texts_empty_input_returns_empty():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    try:
        assert client.embed_texts([]) == []
    finally:
        client.close()


def test_embed_texts_batches_requests():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        # 解析 body 记录每次批量大小
        import json
        body = json.loads(payload)
        calls.append(len(body["input"]))
        return httpx.Response(200, json=_response([[0.1, 0.2]] * len(body["input"])))

    client = _make_client(handler, max_batch=2)
    try:
        vectors = client.embed_texts(["a", "b", "c", "d", "e"])
    finally:
        client.close()
    assert calls == [2, 2, 1]
    assert len(vectors) == 5


def test_embed_texts_retries_on_429_then_succeeds():
    attempts = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        if attempts[0] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json=_response([[1.0]]))

    client = _make_client(handler, max_retries=1)
    try:
        vectors = client.embed_texts(["hello"])
    finally:
        client.close()
    assert attempts == [2]
    assert vectors == [[1.0]]


def test_embed_texts_does_not_retry_on_400():
    attempts = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        return httpx.Response(400, text="bad request")

    client = _make_client(handler, max_retries=2)
    try:
        with pytest.raises(RAGError) as exc_info:
            client.embed_texts(["hello"])
    finally:
        client.close()
    assert attempts == [1]  # 4xx 不重试
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "EMBEDDING_FAILED"


def test_embed_texts_rejects_inconsistent_dimensions():
    def handler(request: httpx.Request) -> httpx.Response:
        # 第一次返回 2 维，第二次返回 3 维，应触发维度不一致错误
        return httpx.Response(200, json=_response([[1.0, 2.0], [1.0, 2.0, 3.0]]))

    client = _make_client(handler, max_batch=2)
    try:
        with pytest.raises(RAGError) as exc_info:
            client.embed_texts(["a", "b"])
    finally:
        client.close()
    assert "维度不一致" in exc_info.value.message


def test_embed_texts_rejects_count_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_response([[1.0]]))

    client = _make_client(handler)
    try:
        with pytest.raises(RAGError) as exc_info:
            client.embed_texts(["a", "b"])
    finally:
        client.close()
    assert "数量与输入不一致" in exc_info.value.message
