"""上下文工程测试：L1/L2/L3 拼装、token 估算、水位检测与 compact。"""

from core.config import ContextConfig
from core.llm import ChatResponse
from core.message import FunctionCall, Message, ToolCall, assistant, system, user
from runtime.context import ContextBuilder
from runtime.context.budget import estimate_messages_tokens, estimate_text_tokens
from runtime.loop import AgentLoop
from tools.builtin.glob_tool import GlobTool
from tools.registry import ToolRegistry


def _tool_unit(i: int) -> list[Message]:
    """构造一对"assistant(工具调用) + tool 结果"的原子单元。"""
    call = ToolCall(
        id=f"c{i}",
        function=FunctionCall(name="glob", arguments='{"pattern": "**/*.py"}'),
    )
    return [
        Message(role="assistant", content=f"第{i}轮思考", tool_calls=[call]),
        Message(role="tool", content=f"第{i}轮结果", tool_call_id=f"c{i}", name="glob"),
    ]


def test_estimate_text_tokens_basic():
    """空文本为 0；CJK 一字约 1 token，ASCII 约 4 字符 1 token。"""
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("你好") >= 2
    assert estimate_text_tokens("hello world") >= 1


def test_estimate_messages_tokens_counts_tool_calls():
    """带工具调用的消息比普通消息估算更多 token。"""
    plain = [Message(role="assistant", content="思考")]
    call = ToolCall(id="c1", function=FunctionCall(name="glob", arguments='{"pattern": "*.py"}'))
    with_tools = [Message(role="assistant", content="思考", tool_calls=[call])]
    assert estimate_messages_tokens(with_tools) > estimate_messages_tokens(plain)


def test_env_block_contains_cwd_and_platform(tmp_path):
    """L1 环境信息包含工作目录、平台与 Python 版本。"""
    block = ContextBuilder(tmp_path).env_block()
    assert str(tmp_path) in block
    assert "Python 版本" in block
    assert "平台" in block
    # shell 提示（M9 增强）：无论哪个平台都应有 shell 说明
    assert "shell" in block


def test_project_rules_discovers_agents_md(tmp_path):
    """向上自动发现 AGENTS.md 并作为 L2 项目规则。"""
    (tmp_path / "AGENTS.md").write_text("用 Python 3.12\n所有代码过 ruff", encoding="utf-8")
    block = ContextBuilder(tmp_path).project_block()
    assert block is not None
    assert "AGENTS.md" in block
    assert "Python 3.12" in block


def test_project_rules_merges_custom_context(tmp_path):
    """AGENTS.md 与 .minicoder/context.md 并存时都注入，且 AGENTS 在前。"""
    (tmp_path / "AGENTS.md").write_text("AGENTS 内容", encoding="utf-8")
    custom_dir = tmp_path / ".minicoder"
    custom_dir.mkdir()
    (custom_dir / "context.md").write_text("自定义内容", encoding="utf-8")
    block = ContextBuilder(tmp_path).project_block()
    assert block is not None
    assert "AGENTS 内容" in block
    assert "自定义内容" in block
    assert block.index("AGENTS 内容") < block.index("自定义内容")


def test_project_rules_none_without_files(tmp_path):
    """没有任何规则文件时 project_block 返回 None。

    在 tmp_path 里放 .git 标记"这是一个仓库根"：向上扫描到此停止，
    避免命中真实仓库（miniCoder 根目录）的 AGENTS.md。
    """
    (tmp_path / ".git").mkdir()
    assert ContextBuilder(tmp_path).project_block() is None


def test_project_rules_disabled_by_config(tmp_path):
    """配置关闭 project_files 时不读取任何规则文件。"""
    (tmp_path / "AGENTS.md").write_text("x", encoding="utf-8")
    builder = ContextBuilder(tmp_path, ContextConfig(project_files=False))
    assert builder.project_block() is None


def test_repo_map_excludes_noise_dirs(tmp_path):
    """文件地图过滤 .venv / __pycache__ / 隐藏目录，保留可见文件。"""
    (tmp_path / ".venv").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / ".hidden_dir").mkdir()
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x")
    text = ContextBuilder(tmp_path).repo_map_block()
    assert text is not None
    assert ".venv" not in text
    assert "__pycache__" not in text
    assert ".hidden_dir" not in text
    assert "app/" in text and "main.py" in text


def test_repo_map_caps_lines(tmp_path):
    """文件地图超过上限行数时截断并给出提示。"""
    for i in range(40):
        (tmp_path / f"f{i}.txt").write_text("x")
    builder = ContextBuilder(tmp_path, ContextConfig(repo_map_max_lines=10))
    text = builder.repo_map_block()
    assert text is not None
    assert len(text.splitlines()) <= 11  # 10 行 + 1 行截断提示
    assert "已截断" in text


def test_system_block_contains_l1_and_l2(tmp_path):
    """system 块同时包含 L1 系统规则 + L2 项目规则 + 文件地图。"""
    (tmp_path / "AGENTS.md").write_text("仓库规范：先看 README", encoding="utf-8")
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "x.py").write_text("x")
    block = ContextBuilder(tmp_path).system_block()
    assert "环境信息" in block
    assert "仓库规范" in block
    assert "文件地图" in block and "core/" in block


def test_build_starts_with_system_and_appends_task(tmp_path):
    """全新会话：system 开头，任务作为最后一条 user 消息。"""
    msgs = ContextBuilder(tmp_path).build("帮我改代码")
    assert msgs[0].role == "system"
    assert msgs[-1].role == "user"
    assert msgs[-1].content == "帮我改代码"


def test_build_with_history_prepends_system(tmp_path):
    """继续会话：system 置顶，历史在中间，新任务在最后。"""
    history = [user("你好"), assistant("你好！")]
    msgs = ContextBuilder(tmp_path).build("继续", history=history)
    assert msgs[0].role == "system"
    assert msgs[1] is history[0]
    assert msgs[-1].content == "继续"


def test_needs_compact_threshold(tmp_path):
    """水位阈值：估算 token 达预算 * compact_ratio 才触发。"""
    messages = [system("s" * 500), user("任务")]  # 500 字符 ≈ 125 token
    small = ContextBuilder(tmp_path, ContextConfig(max_tokens=100))
    assert small.needs_compact(messages) is True
    big = ContextBuilder(tmp_path, ContextConfig(max_tokens=10000))
    assert big.needs_compact(messages) is False


def test_compact_keeps_system_and_recent_units(tmp_path):
    """compact 保留 system 与最近单元，旧的折叠为占位，不拆开工具组。"""
    builder = ContextBuilder(tmp_path, ContextConfig(keep_turns=2))
    messages = [system("系统")]
    for i in range(6):
        messages.extend(_tool_unit(i))
    out = builder.compact(messages)
    assert out is not messages
    # 1 系统 + 1 占位 + 最近 2 单元（每单元 2 条）= 6 条
    assert len(out) == 6
    assert out[0].role == "system"
    assert out[1].role == "user" and "已折叠" in out[1].content
    assert out[-2].role == "assistant" and out[-1].role == "tool"
    assert out[-1].content == "第5轮结果"


def test_compact_no_drop_returns_same_list(tmp_path):
    """没有可折叠内容时原样返回原列表（供调用方判断是否发生压缩）。"""
    builder = ContextBuilder(tmp_path, ContextConfig(keep_turns=10))
    messages = [system("系统"), user("你好")]
    assert builder.compact(messages) is messages


def test_loop_logs_context_record_when_over_budget(tmp_path):
    """AgentLoop 接入上下文构建器后，超水位压缩会在 trace 里留下 context 记录。"""

    class FakeLLM:
        """前 3 轮调用工具，第 4 轮给出最终回答。"""

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls >= 4:
                return ChatResponse(content="完成")
            call = ToolCall(
                id=f"c{self.calls}",
                function=FunctionCall(
                    name="glob",
                    arguments=f'{{"pattern": "*.py", "path": "{tmp_path.as_posix()}"}}',
                ),
            )
            return ChatResponse(content="", tool_calls=[call], finish_reason="tool_calls")

    registry = ToolRegistry()
    registry.register(GlobTool())
    builder = ContextBuilder(tmp_path, ContextConfig(max_tokens=200, keep_turns=2))
    result = AgentLoop(FakeLLM(), registry, max_steps=10, context_builder=builder).run("找文件")
    assert result.success is True
    assert any(step.kind == "context" for step in result.trace)


def test_note_usage_lifts_watermark(tmp_path):
    """记录实测 token 后，水位预测 = 上一轮实测 + 新增估算（比纯估算更敏感）。"""
    builder = ContextBuilder(tmp_path, ContextConfig(max_tokens=200))
    messages = [system("s" * 100), user("任务")]  # 估算很小，不触发
    assert builder.needs_compact(messages) is False
    # 模型实测上一轮输入已达 180 token → 下一轮预测超水位
    builder.note_usage({"prompt_tokens": 180}, messages)
    assert builder.needs_compact(messages) is True


def test_note_usage_ignores_missing_usage(tmp_path):
    """无 usage（假模型/旧接口）时保持纯估算，不改变水位行为。"""
    builder = ContextBuilder(tmp_path, ContextConfig(max_tokens=10000))
    messages = [system("s" * 100), user("任务")]
    builder.note_usage({}, messages)
    builder.note_usage(None, messages)
    assert builder.needs_compact(messages) is False


def test_note_usage_after_compact_falls_back(tmp_path):
    """compact 折叠后消息数变少，新增无法定位时回退 max(估算, 实测)。"""
    builder = ContextBuilder(tmp_path, ContextConfig(max_tokens=200, keep_turns=1))
    messages = [system("s" * 100)] + [m for i in range(4) for m in _tool_unit(i)]
    builder.note_usage({"prompt_tokens": 50}, messages)
    compacted = builder.compact(messages)
    assert compacted is not messages
    assert len(compacted) < len(messages)  # 4 个工具组被折叠为 1 组
    # _last_seen_len 已大于压缩后消息数，水位判断回退保守值且不崩溃
    assert isinstance(builder.needs_compact(compacted), bool)
