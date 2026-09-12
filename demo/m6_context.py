"""M6 演示：上下文工程（L1 系统 / L2 项目 / L3 会话）与水位 compact 的离线演示。

不调用真实模型，展示 ContextBuilder 的拼装结果、token 估算、水位检测与 compact。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.config import ContextConfig
from core.message import FunctionCall, Message, ToolCall, system
from runtime.context import ContextBuilder

# 用 miniCoder 自身作为演示项目
MINICODER_ROOT = Path(__file__).resolve().parents[1]


def _demo_l1_l2() -> None:
    """展示 L1 环境信息 + L2 项目规则/文件地图的拼装。"""
    builder = ContextBuilder(MINICODER_ROOT)
    print("==== system_block（L1 + L2 拼装，前 30 行） ====")
    for line in builder.system_block().splitlines()[:30]:
        print(line)
    print("…（共", len(builder.system_block().splitlines()), "行）")


def _demo_project_rules() -> None:
    """在临时项目里放 AGENTS.md / .minicoder/context.md，展示 L2 自动发现。"""
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "demo_project"
        (project / ".minicoder").mkdir(parents=True)
        agents = "本仓库使用 Python 3.12。\n所有改动需通过 ruff。"
        (project / "AGENTS.md").write_text(agents, encoding="utf-8")
        (project / ".minicoder" / "context.md").write_text("本地构建：uv sync。", encoding="utf-8")
        (project / "main.py").write_text("print('hi')\n", encoding="utf-8")
        builder = ContextBuilder(project)
        print("\n==== L2 项目规则（AGENTS.md 优先，context.md 合并） ====")
        print(builder.project_block() or "(无)")


def _demo_budget_and_compact() -> None:
    """用极小预算演示水位检测与 compact 的折叠效果。"""
    builder = ContextBuilder(MINICODER_ROOT)
    # 模拟 8 轮工具调用单元，每轮含 assistant(工具调用) + tool 结果
    messages = [system(builder.system_block())]
    for i in range(8):
        call = ToolCall(
            id=f"c{i}",
            function=FunctionCall(name="glob", arguments='{"pattern": "**/*.py"}'),
        )
        messages.append(Message(role="assistant", content=f"第{i}轮思考" * 20, tool_calls=[call]))
        messages.append(
            Message(
                role="tool",
                content=f"第{i}轮结果" * 20,
                tool_call_id=f"c{i}",
                name="glob",
            )
        )

    tight = ContextBuilder(MINICODER_ROOT, ContextConfig(max_tokens=2000, keep_turns=2))
    before = len(messages)
    print("\n==== 水位与 compact ====")
    tokens = tight.estimate_tokens(messages)
    need = tight.needs_compact(messages)
    print(f"消息数={before}  估算token={tokens}  需compact={need}")
    compacted = tight.compact(messages)
    print(f"compact 后消息数={len(compacted)}（折叠掉 {before - len(compacted)} 条旧记录）")
    print("首条消息角色:", compacted[0].role)
    print("折叠占位:", compacted[1].content[:20], "…")


def main() -> None:
    _demo_l1_l2()
    _demo_project_rules()
    _demo_budget_and_compact()


if __name__ == "__main__":
    main()
