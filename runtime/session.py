"""交互式会话：跨轮次保持上下文、transcript 持久化、支持继续会话（M8）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.message import Message
from memory.paths import default_trace_path, default_transcript_path
from memory.trace import new_session_id
from memory.transcript import load_messages
from runtime.loop import AgentLoop
from runtime.state import AgentRunResult


class AgentSession:
    """一个持续可对话的会话：把每轮新增消息累积进 history，供下一轮继续。

    复用同一个 AgentLoop 实例（不再每次任务重建），会话内 trace / transcript
    持续追加到 ~/.minicoder/<项目哈希>/ 下；--resume 可从既有 transcript 恢复。
    """

    def __init__(
        self,
        llm: Any,
        registry,
        context_builder,
        *,
        max_steps: int = 20,
        session_id: str | None = None,
        workspace_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        streaming: bool = False,
        on_text_delta=None,
        on_tool_event=None,
        resume: str | Path | None = None,
    ):
        self.context_builder = context_builder
        self.workspace_root = Path(workspace_root) if workspace_root else context_builder.cwd
        self.data_dir = data_dir
        self.session_id = session_id or new_session_id()
        # resume 时复用既有 transcript 并载入历史，否则新建会话文件
        self.transcript_path = (
            Path(resume)
            if resume
            else default_transcript_path(
                self.session_id, workspace_root=self.workspace_root, data_dir=data_dir
            )
        )
        self.trace_path = default_trace_path(
            self.session_id, workspace_root=self.workspace_root, data_dir=data_dir
        )
        self.history: list[Message] = load_messages(self.transcript_path) if resume else []
        # 复用同一个 loop：会话内不再反复重建
        self.loop = AgentLoop(
            llm,
            registry,
            max_steps=max_steps,
            session_id=self.session_id,
            trace_path=self.trace_path,
            transcript_path=self.transcript_path,
            context_builder=context_builder,
            streaming=streaming,
            on_text_delta=on_text_delta,
            on_tool_event=on_tool_event,
            data_dir=data_dir,
        )

    def ask(self, task: str) -> AgentRunResult:
        """执行一轮任务，并把本轮新增消息并入会话历史。"""
        result = self.loop.run(task, history=self.history or None)
        # result.messages 含 system + 历史 + 本轮新增，去掉 system 后即完整会话历史
        self.history = [m for m in result.messages if m.role != "system"]
        return result

    def reset(self) -> None:
        """清空历史（配合 /new 命令，新会话请直接创建新实例）。"""
        self.history = []