"""OpenAI 兼容 /embeddings 客户端：批量向量化、指数退避重试（M11）。

与 LLM 客户端相互独立：Embedding 服务商可与对话模型不同
（如对话用 DeepSeek、向量用 SiliconFlow），更换服务商只需改 .env 配置。
"""

from __future__ import annotations

import time

import httpx

from core.config import EmbeddingConfig
from rag.errors import RAGError


class EmbeddingClient:
    """/embeddings 客户端，支持注入 transport 以便离线测试。"""

    def __init__(self, config: EmbeddingConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        # 已见向量维度；用于校验同一模型内维度一致（跨模型校验由向量库负责）
        self._dimension: int | None = None
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """释放连接资源。"""
        self._client.close()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """把一批文本向量化，按 max_batch 分批请求；返回顺序与输入一致。"""
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, self.config.max_batch)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """请求一次 /embeddings 并解析结果（按 index 排序保证与输入对齐）。"""
        payload = {"model": self.config.model_id, "input": batch}
        resp = self._post_with_retries(payload)
        try:
            data = resp.json()
        except ValueError as exc:
            raise RAGError("EMBEDDING_FAILED", f"Embedding 响应不是合法 JSON: {exc}") from exc
        items = data.get("data") or []
        if len(items) != len(batch):
            raise RAGError(
                "EMBEDDING_FAILED",
                f"Embedding 返回数量与输入不一致: 输入 {len(batch)}，返回 {len(items)}",
            )
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        vectors: list[list[float]] = []
        for item in ordered:
            raw = item.get("embedding")
            if not isinstance(raw, list) or not raw:
                raise RAGError("EMBEDDING_FAILED", "Embedding 返回了空向量")
            vector = [float(value) for value in raw]
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise RAGError(
                    "EMBEDDING_FAILED",
                    f"向量维度不一致: 期望 {self._dimension}，实际 {len(vector)}",
                )
            vectors.append(vector)
        return vectors

    def _post_with_retries(self, payload: dict) -> httpx.Response:
        """带指数退避重试地发送请求；仅网络错误 / 429 / 5xx 重试。"""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._request_once(payload)
            except (httpx.TransportError, RAGError) as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt == self.config.max_retries:
                    break
                delay = self.config.retry_backoff_seconds * (2**attempt)
                time.sleep(delay)
        assert last_error is not None
        if isinstance(last_error, RAGError):
            raise last_error
        raise RAGError("EMBEDDING_FAILED", f"Embedding 请求失败: {last_error}") from last_error

    def _request_once(self, payload: dict) -> httpx.Response:
        """发一次请求；非 2xx 读取错误体并转为 RAGError。"""
        resp = self._client.post("/embeddings", json=payload)
        if resp.status_code != 200:
            body = resp.text
            raise RAGError(
                "EMBEDDING_FAILED",
                f"Embedding 调用失败: HTTP {resp.status_code}: {body[:500]}",
                status_code=resp.status_code,
            )
        return resp

    def _is_retryable(self, exc: Exception) -> bool:
        """网络错误 / 429 / 5xx 可重试；4xx 业务错误不重试。"""
        if isinstance(exc, httpx.TransportError):
            return True
        if isinstance(exc, RAGError):
            code = exc.status_code
            return code is not None and (code == 429 or code >= 500)
        return False

    def __enter__(self) -> EmbeddingClient:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
