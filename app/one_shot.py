"""单轮任务入口：`python -m app.one_shot -p "任务"` 执行一次并退出。

支持保存 trace / 会话记录，也可从已有 transcript 继续会话（--resume）。
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from app.cli import DEFAULT_MAX_STEPS, build_registry
from core.config import load_context_config, load_llm_config
from core.llm import LLMClient
from memory.trace import default_trace_path, new_session_id
from memory.transcript import default_transcript_path, load_messages
from runtime.context import ContextBuilder
from runtime.loop import AgentLoop
from runtime.state import AgentRunResult
from tools.permissions import PermissionGateway
from tools.registry import ToolRegistry
from tools.workspace import Workspace


def on_tool_event(kind: str, name: str, payload) -> None:
    """把工具调用过程打印到终端（简单文本，便于脚本调用）。"""
    if kind == "tool_start":
        print(f"→ 调用工具: {name} {payload}")
    else:
        print(f"← {name} 状态: {getattr(payload, 'status', '?')}")


def execute_once(
    llm,
    registry: ToolRegistry,
    task: str,
    *,
    session_id: str | None = None,
    trace_path=None,
    transcript_path=None,
    resume: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    context_builder: ContextBuilder | None = None,
) -> AgentRunResult:
    """执行一次单轮任务；llm 可注入假模型以便离线测试。"""
    history = load_messages(resume) if resume else None
    loop = AgentLoop(
        llm,
        registry,
        max_steps=max_steps,
        session_id=session_id,
        trace_path=trace_path,
        transcript_path=transcript_path,
        on_tool_event=on_tool_event,
        context_builder=context_builder,
    )
    return loop.run(task, history=history)


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行一次单轮任务，返回退出码（0 成功 / 2 未完成 / 1 配置缺失）。"""
    parser = argparse.ArgumentParser(description="miniCoder 单轮任务执行器")
    parser.add_argument("-p", "--prompt", required=True, help="要执行的任务")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="最大工具调用轮数")
    parser.add_argument("--trace", default=None, help="trace 输出路径（默认 memory/traces/）")
    parser.add_argument("--transcript", default=None, help="会话记录路径（默认自动生成）")
    parser.add_argument("--resume", default=None, help="从指定 transcript 继续会话")
    parser.add_argument(
        "--permission", choices=["ask", "allow", "deny"], default=None,
        help="Bash 审批策略；非交互默认 deny（只放行只读，fail-closed）",
    )
    parser.add_argument(
        "--cwd", default=None, help="先切换到该目录再执行任务（与 CLI 的 --cwd 一致）"
    )
    args = parser.parse_args(argv)

    if args.cwd:
        target = Path(args.cwd).expanduser()
        if not target.is_dir():
            print(f"--cwd 目录不存在: {target}")
            return 1
        os.chdir(target)  # 切换后再取 Path.cwd()，让工作空间指向目标目录

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_llm_config()
    if not config.api_key:
        print("未配置 LLM_API_KEY，请复制 .env.example 为 .env 并填写。")
        return 1

    session_id = new_session_id()
    # 数据落盘到 ~/.minicoder/<项目哈希>/ 下（与 CLI 一致）
    workspace_root = Path.cwd()
    trace_path = args.trace or default_trace_path(session_id, workspace_root=workspace_root)
    # 继续会话时，默认把新消息继续写回原 transcript 文件
    transcript_path = args.transcript or (
        args.resume or default_transcript_path(session_id, workspace_root=workspace_root)
    )

    # 非交互场景默认 deny（fail-closed）：没有用户在场，未审命令一律不放行
    policy = args.permission or "deny"
    workspace = Workspace(Path.cwd())
    gateway = PermissionGateway(ask_policy=policy, ask_handler=None, remember=False)
    registry = build_registry(workspace, gateway)
    context_builder = ContextBuilder(Path.cwd(), load_context_config())
    with LLMClient(config) as client:
        result = execute_once(
            client,
            registry,
            args.prompt,
            session_id=session_id,
            trace_path=trace_path,
            transcript_path=transcript_path,
            resume=args.resume,
            max_steps=args.max_steps,
            context_builder=context_builder,
        )

    if result.success:
        status_label = "完成"
    elif result.partial:
        status_label = "未完成（已达步数上限，以下为阶段性总结）"
    else:
        status_label = f"未完成: {result.reason}"
    print(f"[{status_label}] {result.answer}")
    if result.trace_path:
        print(f"trace: {result.trace_path}")
    if result.transcript_path:
        print(f"transcript: {result.transcript_path}")
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
