"""Rerank 精排客户端：Cohere 兼容 /rerank 接口（M11 增强）。

SiliconFlow 等服务商提供 Cohere 风格的 /rerank 端点
（默认模型 BAAI/bge-reranker-v2-m3）：输入 query + 候选文档列表，
返回按相关度重排的 index 列表。失败时调用方应降级回混合检索排序。
"""

from __future__ import annotations

import logging
import time

import httpx

from core.config import RerankConfig
from rag.errors import RAGError
from rag.store import SearchHit

logger = logging.getLogger(__name__)


class RerankClient:
    """/rerank 客户端，支持注入 transport 以便离线测试。"""

    def __init__(self, config: RerankConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """释放连接资源。"""
        self._client.close()

    def rerank(self, query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
        """对混合检索候选做精排；接口异常时降级为原序（不中断检索）。"""
        if not hits:
            return []
        try:
            results = self._request(query, [hit.content for hit in hits], top_n)
        except RAGError as exc:
            logger.warning("rerank 失败，降级为混合检索排序: %s", exc.message)
            return hits[:top_n]
        reranked: list[SearchHit] = []
        for item in results:
            index = int(item.get("index", -1))
            if not 0 <= index < len(hits):
                continue
            hit = hits[index]
            score = item.get("relevance_score", item.get("relevanceScore"))
            reranked.append(
                SearchHit(
                    path=hit.path,
                    chunk_index=hit.chunk_index,
                    content=hit.content,
                    char_start=hit.char_start,
                    char_end=hit.char_end,
                    score=round(float(score or 0.0), 4),
                    sources=[*hit.sources, "rerank"],
                )
            )
        # 服务端返回数量可能少于 top_n：不足部分按原序补齐，避免丢结果
        used_indexes = {int(item["index"]) for item in results}
        remaining = [hit for i, hit in enumerate(hits) if i not in used_indexes]
        return (reranked + remaining)[:top_n]

    def _request(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        """发送一次 /rerank 请求并解析 results；带指数退避重试。"""
        payload = {
            "model": self.config.model_id,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post("/rerank", json=payload)
                if resp.status_code != 200:
                    raise RAGError(
                        "RERANK_FAILED",
                        f"Rerank 调用失败: HTTP {resp.status_code}: {resp.text[:300]}",
                        status_code=resp.status_code,
                    )
                data = resp.json()
                results = data.get("results") or []
                if len(results) > len(documents):
                    raise RAGError("RERANK_FAILED", "Rerank 返回数量超过候选数")
                return results
            except (httpx.TransportError, RAGError, ValueError) as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt == self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        if isinstance(last_error, RAGError):
            raise last_error
        raise RAGError("RERANK_FAILED", f"Rerank 请求失败: {last_error}") from last_error

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """网络错误 / 429 / 5xx 可重试；4xx 业务错误不重试。"""
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, RAGError):
            code = exc.status_code
            return code is not None and (code == 429 or code >= 500)
        return False

    def __enter__(self) -> RerankClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
