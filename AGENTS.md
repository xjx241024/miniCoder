# Repository Guidelines

miniCoder 是本地优先的编程 Agent 运行时（ReAct 循环 + 工具系统 + 上下文工程）。
本文件是面向人类与 AI 贡献者的开发约定。

## Project Structure

- `core/`：配置、LLM 封装、消息模型
- `runtime/`：ReAct 主循环、会话、上下文工程、输出治理
- `tools/`：工具基类、注册中心、工作空间约束、Bash 审批、内置工具
- `memory/`：trace/transcript 落盘与数据目录
- `app/`：CLI 与单轮入口
- `demo/`：任务演示与基准测量
- `prompts/`：系统提示词；`tests/`：pytest 测试

## Build, Test & Dev Commands

```powershell
uv sync --extra dev            # 安装依赖
uv run ruff check .            # 代码检查（E/F/I/UP，行宽 100）
uv run python -m pytest -q     # 运行全部测试
uv run python -m app.cli       # 交互 CLI
uv run python -m app.one_shot -p "任务"   # 单轮执行
```

## Coding Style

- Python 3.10+，ruff 检查，行宽 100；文件开头写一两句用途，关键逻辑加中文注释。
- 函数/变量 `snake_case`，类 `PascalCase`；内置工具文件 `<name>_tool.py`，类名 `<Name>Tool`。
- 工具统一返回 `ToolResult`，错误码大写（如 `NOT_FOUND`）。

## Testing Guidelines

- 框架 pytest；文件 `tests/test_*.py`，命名 `test_<对象>_<行为>`。
- 用 MockTransport / 假模型注入，不触网、不耗 API；新增逻辑必须配测试。

## Commit & Pull Request

- 提交按里程碑：`M<n>: <中文摘要>`；增强/修复用 `M<n> 增强: ...`，文档用 `docs: ...`。
- 每阶段开 `milestone/m<n>` 分支保留成果，`main` 存最新。
- PR 说明需包含：改动内容、为什么改、测试结果；涉及工具或主循环的改动附运行示例。

## Security & Configuration

- 从 `.env.example` 复制 `.env` 配置 API Key，密钥不入 git。
- 工具只能访问工作空间内路径，越界一律拒绝；Bash 按 ask/allow/deny 分级审批，非交互默认 deny。
- 运行数据落 `~/.minicoder`（可用 `MINICODER_DATA_DIR` 覆盖）；`files_for_test/` 不入 git。

## Agent-Specific Notes

- 别破坏 `ContextBuilder.system_block()` 的会话内缓存（前缀缓存依赖请求体字节稳定）。
- 新工具三步：在 `tools/builtin/` 建文件 → 在 `app/cli.py` 的 `build_registry` 注册 → 补测试。
- 超长工具输出交由 `runtime/output_guard.py` 全文落盘 + 预览提示，不要自行头部截断。
