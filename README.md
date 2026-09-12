# miniCoder

本地优先的编程 Agent 运行时（单 Agent、单进程、最小闭环），用于学习与实习展示。
通过 OpenAI 兼容接口接入大模型，用 ReAct 循环驱动工具完成任务，并把每次运行落盘为可回放的 trace。

## 快速开始

```powershell
# 1. 安装依赖（含测试）
uv sync --extra dev

# 2. 配置模型：复制 .env.example 为 .env，填写 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL

# 3. 跑测试
.venv\Scripts\python.exe -m pytest -q

# 4. 交互式 CLI（单会话：跨轮次保持上下文，最终回答流式输出）
#    安装后可简化成 .venv\Scripts\minicoder.exe；--cwd 可指定工作目录
#    或用该命令启动(两条命令本质相同)： uv run minicoder --cwd 工作目录
.venv\Scripts\python.exe -m app.cli
.venv\Scripts\python.exe -m app.cli --cwd E:/some/project

# 5. 单轮任务（执行一次即退出）
.venv\Scripts\python.exe -m app.one_shot -p "找出 tools/builtin 下所有 .py 文件"
.venv\Scripts\python.exe -m app.one_shot --cwd E:/some/project -p "统计该目录下 .py 文件数"
```

## 交互式 CLI 命令

```
输入任务即可执行；会话内自动累积历史，可连续追问。

/new              开始一个新会话（生成新的 trace / transcript）
/resume <路径>    从某个 transcript 文件恢复历史，继续对话
/clean            立即清理过期运行数据（按保留策略）
/help             查看帮助
/exit             退出
```

## 运行期数据

- 数据统一存放在用户主目录 `~/.minicoder/<项目哈希>/` 下（可用环境变量 `MINICODER_DATA_DIR` 覆盖），不污染被操作的仓库：
  - `traces/<会话id>.jsonl`：逐步运行记录，可回放排查
  - `transcripts/<会话id>.jsonl`：对话消息，可用于 `/resume` 继续会话
  - `artifacts/`：工具大输出落盘（后续里程碑使用）
- 保留策略（`.env` 可调）：保留最近 `RETENTION_KEEP_SESSIONS` 个会话，超过 `RETENTION_MAX_AGE_DAYS` 天的运行文件在启动或 `/clean` 时自动清理。

## 里程碑

- M1 能对话：配置 + 消息 + 模型封装
- M2 能调工具：工具注册中心 + Glob / Read
- M3 闭环跑通：ReAct 主循环 + Grep / Edit + 交互 CLI
- M4 变得可靠：读后写三件套 + JSONL trace / transcript + Bash 兜底 + 单轮入口
- M5 提升一轮能力（已完成）：内容哈希指纹（解决 ABA）+ 超限强制总结（partial 标记）+ 工具提示词工作流 + 默认 20 轮
- M6 上下文工程（已完成）：L1 系统规则 / L2 项目规则（AGENTS.md + 文件地图）/ L3 会话动态拼装 + 水位检测与 compact
- M7 安全边界（已完成）：工作空间约束（越界路径一律拒绝）+ Bash 风险分级与用户审批 + 注册中心参数清洗
- M8 会话连续与流式（已完成）：单会话复用（跨轮次历史累积）+ `/new /resume /clean` + 流式最终回答（SSE 聚合）+ 打转检测 + 数据迁移 `~/.minicoder` 与保留清理
- M9 输出治理与预算（已完成）：超长工具输出全文落盘 `artifacts/`（模型收预览 + 精读提示，不再头部硬截断）+ 用 `usage.prompt_tokens` 实测校准上下文水位（预测 = 上一轮实测 + 新增估算）
- M9 增强：新增 `write` 工具（新建/整文件覆盖，读后写保护 + 原子写 + 内容上限）+ L1 环境块注入 shell 类型与限制（按平台动态生成）+ `python -c` 拒绝消息给出替代路径 + `glob` 增加 `include_hidden`/`include_ignored`（默认排除噪声目录，可开关）
- M10 启动与缓存（已完成）：`minicoder` / `minicoder-run` 命令一键启动（仿 claude / codex）+ `--cwd` 任意目录运行 + 流式请求携带 `stream_options.include_usage` 解析用量 + 前缀缓存命中逐轮观测（usage 打日志）

## 目录

- `core/`：基础层，配置 / 消息 / 模型封装（含流式聚合 `chat_stream_response`、前缀缓存用量解析）
- `runtime/`：ReAct 主循环与运行状态、上下文工程
  - `runtime/context/`：L1/L2/L3 上下文拼装、水位检测与 compact（M9 起用实测 usage 校准）
  - `runtime/session.py`：单会话封装（跨轮次历史 + 自动持久化 + 恢复）
  - `runtime/output_guard.py`：超长工具输出治理（全文落盘 + 预览提示）
- `prompts/`：系统提示词（行为规则 + 工具工作流程）
- `tools/`：Glob / Grep / Read / Edit / Bash + 工作空间约束（workspace.py）与 Bash 审批（permissions.py）
- `memory/`：运行期数据（JSONL trace / transcript）+ 数据目录（paths.py）与保留清理（retention.py）
- `app/`：交互 CLI 与单轮入口
- `demo/`：比单元测试更完整的任务演示
- `tests/`：pytest 单元测试
