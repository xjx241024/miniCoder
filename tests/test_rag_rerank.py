"""Rerank 客户端测试：重排、降级、重试与补齐逻辑（MockTransport 离线）。"""

import httpx
import pytest

from core.config import RerankConfig
from rag.rerank import RerankClient
from rag.store import SearchHit


def _config(**overrides) -> RerankConfig:
    return RerankConfig(
        api_key="test-key",
        base_url="https://example.com/v1",
        retry_backoff_seconds=0.0,
        **overrides,
    )


def _hits() -> list[SearchHit]:
    return [
        SearchHit("a.md", 0, "内容A", 0, 3, 0.9, ["vector"]),
        SearchHit("b.md", 0, "内容B", 0, 3, 0.8, ["vector", "fts"]),
        SearchHit("c.md", 0, "内容C", 0, 3, 0.7, ["fts"]),
    ]


def _client(handler, **overrides) -> RerankClient:
    return RerankClient(_config(**overrides), transport=httpx.MockTransport(handler))


def test_rerank_reorders_by_relevance():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = request.read()
        assert b"bge-reranker-v2-m3" in payload
        return httpx.Response(200, json={"results": [
            {"index": 1, "relevance_score": 0.98},
            {"index": 0, "relevance_score": 0.75},
        ]})

    client = _client(handler)
    try:
        hits = client.rerank("查询", _hits(), top_n=2)
    finally:
        client.close()
    assert [hit.path for hit in hits] == ["b.md", "a.md"]
    assert hits[0].score == pytest.approx(0.98)
    assert "rerank" in hits[0].sources


def test_rerank_falls_back_to_original_order_on_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    client = _client(handler, max_retries=0)
    try:
        hits = client.rerank("查询", _hits(), top_n=2)
    finally:
        client.close()
    assert [hit.path for hit in hits] == ["a.md", "b.md"]  # 原序降级，不中断检索


def test_rerank_retries_on_429_then_succeeds():
    attempts = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        attempts[0] += 1
        if attempts[0] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": 0.9}]})

    client = _client(handler, max_retries=1)
    try:
        hits = client.rerank("查询", _hits(), top_n=1)
    finally:
        client.close()
    assert attempts == [2]
    assert hits[0].path == "c.md"


def test_rerank_pads_results_when_server_returns_fewer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 2, "relevance_score": 0.9}]})

    client = _client(handler)
    try:
        hits = client.rerank("查询", _hits(), top_n=3)
    finally:
        client.close()
    assert [hit.path for hit in hits] == ["c.md", "a.md", "b.md"]
