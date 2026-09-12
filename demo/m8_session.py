"""M8 演示：单会话持续对话 + 自动持久化 + 恢复继续（离线，用假模型）。

不调用真实模型，展示：同一会话内跨轮次历史累积、trace/transcript 落到
~/.minicoder（演示用临时目录），以及从既有 transcript 恢复继续对话。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.llm import ChatResponse
from core.message import FunctionCall, ToolCall
from memory.transcript import load_messages
from runtime.session import AgentSession
from tools.builtin.glob_tool import GlobTool
from tools.registry import ToolRegistry
from tools.workspace import Workspace


class ScriptedLLM:
    """离线假模型：按脚本顺序回应，并记录每次请求的消息数。"""

    def __init__(self, script: list[ChatResponse]) -> None:
        self.script = list(script)
        self.seen_lengths: list[int] = []

    def chat(self, messages, tools=None):
        self.seen_lengths.append(len(messages))
        return self.script.pop(0)


def _tool_call(name: str, arguments: str) -> ChatResponse:
    """构造一个"发起工具调用"的模型响应。"""
    call = ToolCall(id="call_1", function=FunctionCall(name=name, arguments=arguments))
    return ChatResponse(content="", tool_calls=[call], finish_reason="tool_calls")


def main() -> None:
    # 用临时目录当工作空间与数据目录，演示结束后自动清理，不污染真实数据
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "workspace"
        root.mkdir()
        (root / "a.py").write_text("print('a')\n", encoding="utf-8")
        (root / "b.py").write_text("print('b')\n", encoding="utf-8")

        script = [
            _tool_call("glob", f'{{"pattern": "*.py", "path": "{root.as_posix()}"}}'),
            ChatResponse(content="找到 2 个 Python 文件：a.py、b.py"),
            ChatResponse(content="不客气，随时可以继续问我。"),
        ]
        llm = ScriptedLLM(script)
        registry = ToolRegistry()
        registry.register(GlobTool(Workspace(root)))
        data_dir = Path(tmp) / "data"

        print("==== 单会话持续对话（历史累积）====")
        session = AgentSession(
            llm,
            registry,
            None,
            workspace_root=root,
            data_dir=data_dir,
            streaming=True,  # 演示流式回退：假模型无流式接口时自动回退普通 chat
        )
        first = session.ask("这个工作区有哪些 Python 文件？")
        second = session.ask("谢谢！")
        print(f"第 1 轮回答: {first.answer}")
        print(f"第 2 轮回答: {second.answer}")
        print(f"模型收到的消息数变化：{llm.seen_lengths}（第 2 轮 > 第 1 轮 = 带上了历史）")
        print(f"会话历史条数（不含 system）：{len(session.history)}")
        print(f"transcript 落盘：{session.transcript_path}")
        transcript_path = session.transcript_path

        print("\n==== 从 transcript 恢复会话 ====")
        restored = AgentSession(
            llm,
            registry,
            None,
            workspace_root=root,
            data_dir=data_dir,
            resume=transcript_path,
        )
        print(f"恢复出的历史条数：{len(restored.history)}")
        print("恢复后的完整对话记录：")
        for message in load_messages(transcript_path):
            print(f"  [{message.role}] {message.content[:60]}")


if __name__ == "__main__":
    main()