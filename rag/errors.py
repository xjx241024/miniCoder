"""RAG 子系统统一错误：携带错误码，便于工具层转换为 ToolResult。"""

from __future__ import annotations


class RAGError(Exception):
    """RAG 内部错误，code 为大写错误码（如 EMBEDDING_NOT_CONFIGURED）。"""

    def __init__(self, code: str, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.code = code          # 稳定错误码，模型可据此纠错
        self.message = message
        self.status_code = status_code  # HTTP 状态码；None 表示非网络错误
