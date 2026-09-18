"""交互式命令行：单会话持续对话，跨轮次保持上下文，支持流式输出（M8）。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from core.config import (
    load_context_config,
    load_llm_config,
    load_retention_config,
    load_security_config,
)
from core.llm import LLMClient
from memory import retention
from memory.paths import default_data_dir
from runtime.context import ContextBuilder
from runtime.session import AgentSession
from tools.builtin.bash_tool import BashTool
from tools.builtin.edit_tool import EditTool
from tools.builtin.glob_tool import GlobTool
from tools.builtin.grep_tool import GrepTool
from tools.builtin.index_docs_tool import IndexDocsTool
from tools.builtin.read_tool import ReadTool
from tools.builtin.search_docs_tool import SearchDocsTool
from tools.builtin.write_tool import WriteTool
from tools.permissions import PermissionDecision, PermissionGateway
from tools.registry import ToolRegistry
from tools.workspace import Workspace

console = Console()

# 默认最大工具调用轮数；配合超限"强制总结"，任务未完成也能给出阶段性结果
DEFAULT_MAX_STEPS = 20


class _StreamSink:
    """接收流式文本并实时打印；记录是否输出过内容，用于结束后回退面板。"""

    def __init__(self) -> None:
        self.any_text = False

    def on_delta(self, text: str) -> None:
        """把增量文本原样打印（不解析 rich 标记，避免内容被误渲染）。"""
        self.any_text = True
        console.print(text, end="", markup=False)


def build_registry(workspace=None, bash_permission=None) -> ToolRegistry:
    """组装并注册内置工具；工作空间与 Bash 审批网关可注入。"""
    workspace = workspace or Workspace(Path.cwd())
    registry = ToolRegistry(workspace=workspace)
    registry.register(GlobTool(workspace))
    registry.register(GrepTool(workspace))
    registry.register(ReadTool(workspace))
    registry.register(EditTool(workspace))
    registry.register(WriteTool(workspace))
    registry.register(BashTool(workspace, bash_permission))
    registry.register(IndexDocsTool(workspace))
    registry.register(SearchDocsTool(workspace))
    return registry


def _apply_startup_cwd(argv: list[str] | None = None) -> None:
    """解析 --cwd <目录> 并切换工作目录（否则以当前目录为工作空间）。

    让 minicoder 可以在任意目录运行：`minicoder --cwd E:/some/project`。
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--cwd" in args:
        index = args.index("--cwd")
        if index + 1 >= len(args):
            console.print("[yellow]用法: minicoder --cwd <目录>[/]")
            raise SystemExit(1)
        target = Path(args[index + 1]).expanduser()
        if not target.is_dir():
            console.print(f"[red]--cwd 目录不存在: {target}[/]")
            raise SystemExit(1)
        os.chdir(target)
        console.print(f"[dim]工作目录已切换: {Path.cwd()}[/]")


def _ask_user(command: str, decision: PermissionDecision) -> bool:
    """向用户确认是否允许执行命令；输入 y/yes 放行，其余拒绝。"""
    answer = console.input(
        f"[yellow]是否允许执行命令？(y-允许，n-拒绝)[/]\n  {command}\n"
        f"  [dim]风险: {decision.risk.value} - {decision.reason}[/]\n[y/N] "
    )
    return answer.strip().lower() in ("y", "yes")


def on_tool_event(kind: str, name: str, payload) -> None:
    """把工具调用过程实时打印到终端（观察者回调）。"""
    if kind == "tool_start":
        console.print(f"[cyan]→ 调用工具[/] [bold]{name}[/] {payload}")
    else:
        status = getattr(payload, "status", "?")
        console.print(f"[green]← {name}[/] 状态: {status}")


def _result_panel(result) -> Panel:
    """按结果类型渲染面板：完成 / 未完成（阶段性总结）/ 出错。"""
    if result.reason == "completed":
        return Panel(result.answer, title="最终回答", border_style="green")
    if result.partial:
        return Panel(
            result.answer,
            title="未完成（已达步数上限）—— 阶段性总结",
            border_style="yellow",
        )
    return Panel(result.answer, title=f"未完成: {result.reason}", border_style="red")


def _result_summary(result) -> str:
    """流式输出后的一行状态摘要（回答已实时展示，无需再重复面板）。"""
    if result.reason == "completed":
        label = f"[green]✓ 完成（{result.steps_used} 步）[/]"
    elif result.partial:
        label = "[yellow]⚠ 未完成（已达步数上限，以上为阶段性总结）[/]"
    else:
        label = f"[red]✗ 未完成: {result.reason}[/]"
    lines = [label]
    if result.trace_path:
        lines.append(f"[dim]trace: {result.trace_path}[/]")
    if result.transcript_path:
        lines.append(f"[dim]transcript: {result.transcript_path}[/]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    _apply_startup_cwd(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_llm_config()
    if not config.api_key:
        console.print("[red]未配置 LLM_API_KEY，请复制 .env.example 为 .env 并填写。[/]")
        return

    # 安全边界：工作空间固定为当前目录；Bash 中高危命令询问用户
    security = load_security_config()
    workspace = Workspace(Path.cwd())
    gateway = PermissionGateway(
        ask_policy=security.ask_policy,
        ask_handler=_ask_user,
        remember=security.remember_choices,
    )
    registry = build_registry(workspace, gateway)
    # 上下文构建器：负责 L1 系统规则 / L2 项目规则 / L3 会话动态的拼装与水位 compact
    context_builder = ContextBuilder(Path.cwd(), load_context_config())

    # 启动时清理过期 trace / transcript / artifact，避免长期堆积占空间
    data_dir = default_data_dir()
    retention_cfg = load_retention_config()
    removed = retention.prune(
        data_dir,
        keep_sessions=retention_cfg.keep_sessions,
        max_age_days=retention_cfg.max_age_days,
        workspace_root=Path.cwd(),
    )
    if removed:
        console.print(f"[dim]已清理 {removed} 个过期运行文件[/]")

    console.print(
        Panel(
            "输入任务执行；/new 新会话  /resume <路径> 继续  /clean 清理  /exit 退出",
            title="miniCoder CLI",
        )
    )

    # 流式渲染：实时展示最终回答；会话内复用同一个 AgentSession
    sink = _StreamSink()

    with LLMClient(config) as client:
        def new_session(resume: str | None = None) -> AgentSession:
            """创建（或从 transcript 恢复）一个持续会话，并接上流式回调。"""
            return AgentSession(
                client,
                registry,
                context_builder,
                max_steps=DEFAULT_MAX_STEPS,
                workspace_root=Path.cwd(),
                streaming=True,
                on_text_delta=sink.on_delta,
                on_tool_event=on_tool_event,
                resume=resume,
            )

        session = new_session()
        while True:
            try:
                task = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]已退出。[/]")
                return
            if not task:
                continue
            if task in ("/exit", "/quit"):
                return
            if task == "/help":
                console.print(
                    "/new 新会话；/resume <transcript 路径> 继续旧会话；"
                    "/clean 清理过期数据；/exit 退出"
                )
                continue
            if task == "/new":
                session = new_session()
                console.print("[dim]已开始新会话。[/]")
                continue
            if task.startswith("/resume"):
                path = task[len("/resume"):].strip()
                if not path:
                    console.print("[yellow]用法: /resume <transcript 文件路径>[/]")
                    continue
                session = new_session(resume=path)
                console.print(f"[dim]已从 {path} 继续会话（{len(session.history)} 条历史）。[/]")
                continue
            if task == "/clean":
                removed = retention.prune(
                    data_dir,
                    keep_sessions=retention_cfg.keep_sessions,
                    max_age_days=retention_cfg.max_age_days,
                    workspace_root=Path.cwd(),
                )
                console.print(f"[dim]已清理 {removed} 个过期运行文件。[/]")
                continue

            sink.any_text = False  # 每轮重置，判断本轮是否真的有流式输出
            result = session.ask(task)
            console.print()  # 结束流式行
            if sink.any_text:
                console.print(_result_summary(result))
            else:
                # 未走流式（模型不支持或出错兜底）：用面板完整展示结果
                console.print(_result_panel(result))
                if result.trace_path:
                    console.print(f"[dim]trace: {result.trace_path}[/]")
                if result.transcript_path:
                    console.print(f"[dim]transcript: {result.transcript_path}[/]")


if __name__ == "__main__":
    main()
