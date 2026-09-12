"""工具输出治理：超阈值全文落盘 artifact，给模型头部预览 + 精读提示（M9）。

解决"超大输出头部硬截断导致信息永久丢失"：正文完整写入数据目录 artifacts/，
模型收到"前 N 行 + 文件路径 + 用 read/grep 精读"的提示，既不撑爆上下文也不丢信息。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from memory.paths import default_artifact_dir
from tools.base import ToolResult

# 触发治理的字符上限与预览行数（构造时可覆盖）
MAX_CHARS = 4000
PREVIEW_LINES = 200


class OutputGuard:
    """对所有工具输出做统一的"截断 + 落盘"治理。

    只处理 text 字段；未超阈值时原样返回，零副作用（目录延迟到真正落盘才创建）。
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        max_chars: int = MAX_CHARS,
        preview_lines: int = PREVIEW_LINES,
    ):
        self.max_chars = max_chars
        self.preview_lines = preview_lines
        # 与 trace/transcript 同属一个数据目录（~/.minicoder/<项目哈希>/artifacts/）
        self.artifact_dir = Path(
            default_artifact_dir(workspace_root=workspace_root, data_dir=data_dir)
        )

    def guard(self, tool_name: str, result: ToolResult, *, session_id: str = "") -> ToolResult:
        """若输出超阈值：全文落盘，返回"预览 + 路径 + 精读提示"的新结果。"""
        text = result.text or ""
        if not self._exceeds(text):
            return result
        path = self._write_artifact(tool_name, session_id, text)
        hint = (
            f"\n（输出共 {len(text.splitlines())} 行 / {len(text)} 字符，已超阈值。"
            f"完整内容已保存到: {path}\n"
            "如需精读可用 read 工具读取该文件，或用 grep 在该文件中搜索关键词。）"
        )
        return ToolResult(
            status=result.status,
            data=result.data,
            text=self._preview(text) + hint,
            error=result.error,
        )

    def _exceeds(self, text: str) -> bool:
        """超过字符上限或超过预览行数时触发治理。"""
        if len(text) > self.max_chars:
            return True
        return text.count("\n") + 1 > self.preview_lines

    def _preview(self, text: str) -> str:
        """生成头部预览：取前 preview_lines 行，超字符上限再截断。"""
        lines = text.splitlines()[: self.preview_lines]
        preview = "\n".join(lines)
        if len(preview) > self.max_chars:
            preview = preview[: self.max_chars] + "…"
        return preview

    def _write_artifact(self, tool_name: str, session_id: str, text: str) -> Path:
        """把完整输出写入 artifacts/ 并返回路径（目录按需创建）。"""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        sid = session_id or uuid.uuid4().hex[:6]
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifact_dir / f"{stamp}-{sid}-{tool_name}.txt"
        path.write_text(text, encoding="utf-8")
        return path