# miniCoder 项目骨架说明（最小闭环）

> 状态：规划文档，代码已从 M1 搭建到 M9（输出治理与预算）。
> 参考：`../KamaClaude`（架构与学习地图）、`../MyCodeAgent`（工程纪律）、`../YYHDBL-HelloCodeAgentCli`（最小起点）、`../Extra09-Agent应用开发实践踩坑与经验分享.md`（设计原则来源）。

## 一、项目定位与目标

一句话定位：本地优先的编程 Agent 运行时（单 Agent、单进程、最小闭环）。

最小闭环的验收目标：

```
用户输入一个自然语言目标
  → Agent 拆解为多步
  → 调用工具（搜索 / 读取 / 修改 / 低频命令）
  → 观察工具结果，继续下一步
  → 完成一个真实小任务
  → 输出可读结果，并留下可排查的 trace
```

第一个真实任务示例：让 Agent 在某个仓库里“找到 process_data 函数的定义，读取它，并修改其中一行”。

## 二、第一版范围（做什么 / 不做什么）

第一版只做这些：
- OpenAI 兼容的模型接入（DeepSeek / 通义 / 智谱 / OpenAI 可切换）
- 一个 ReAct 主循环（思考 → 工具调用 → 观察 → 再思考，带 max_steps 上限）
- 4 个原子工具（Glob / Grep / Read / Edit）+ 1 个低频兜底工具 Bash
- 交互式 CLI（rich）与单轮执行模式（-p 任务）
- JSONL trace（每步输入输出可回放）与会话记录（可继续会话）

第一版明确不做（避免第一版就堆复杂架构）：
- 不做 daemon / TUI / Web
- 不做 MCP、Skills、子代理、多智能体
- 不做 RAG / 向量记忆
- 不做 LLM 摘要式 compact（M6 已实现截断式，摘要式留接口）

## 三、核心设计原则

1. 先跑通最短链路，再谈优化。
2. 工具分层：高频原子层（Glob / Grep / Read）→ 中频受控层（Edit）→ 低频兜底层（Bash，明确禁区）。
3. 统一工具响应协议：status / data / text / error（含错误码）。
4. 可诊断性优先：每个失败返回具体错误码，模型才能针对性纠错。
5. Edit 必须“读过才能改”：读后写 + 乐观锁（mtime / size 校验）+ 原子写入。
6. 上下文按 L1 系统规则 / L2 项目规则 / L3 动态会话顺序拼接，预留水位检测接口。
7. 所有工具调用与结果写进 trace，出错可回放；消息写进 transcript，可继续会话。
8. 安全边界：所有路径必须落在工作空间内；Bash 中高危命令需用户审批（策略 ask/allow/deny）。
9. 会话连续：同一会话复用单个 loop 并累积 history（跨轮次上下文）；trace/transcript 落到 ~/.minicoder，支持 /resume 恢复。
10. 打转检测：同一工具同参数连续失败达到阈值时注入"换策略"提示，避免无效轮次。
11. 输出治理：工具输出超阈值不再头部硬截断，全文写入 artifacts/，模型收到"预览 + 路径 + 精读提示"。
12. 预算校准：上下文水位以模型实测 usage.prompt_tokens 为准（下一轮预测 = 上一轮实测 + 新增估算），而非纯启发式。

## 四、文件结构

```
JobAgent/    # 本机磁盘目录名（git 仓库目录）；产品/包名已改为 miniCoder
├── SKELETON.md              # 本文件：规划与结构说明
├── README.md                # 项目简介与快速开始
├── pyproject.toml           # uv 工程配置，Python 3.10+
├── .env.example             # LLM_PROVIDER / LLM_MODEL_ID / LLM_API_KEY
├── .gitignore
├── core/                    # 基础层
│   ├── __init__.py
│   ├── config.py            # 配置加载（.env → 类型化配置）
│   ├── llm.py               # OpenAI-compatible 模型封装（流式 / 非流式 / 重试）
│   └── message.py           # 消息结构：system / user / assistant / tool
├── runtime/                 # 运行时层
│   ├── loop.py              # ReAct AgentLoop（max_steps、流式、打转检测、输出治理、落盘）
│   ├── session.py           # 单会话封装（跨轮次历史 + 自动持久化 + 恢复）
│   ├── output_guard.py      # 超长工具输出治理（全文落盘 artifacts/ + 预览提示）
│   ├── state.py             # 会话状态与步骤记录
│   └── context/             # 上下文工程（L1/L2/L3 + 水位 compact）
├── tools/                   # 工具层
│   ├── base.py              # 工具基类 + 统一响应协议
│   ├── registry.py          # 注册、JSON Schema 生成、调用分发、读后写保护
│   └── builtin/
│       ├── glob_tool.py     # 按模式找文件
│       ├── grep_tool.py     # 按内容搜索（带行号）
│       ├── read_tool.py     # 读取文件（带行号、大小 / mtime）
│       ├── edit_tool.py     # 读后写 + 乐观锁 + 原子替换
│       ├── write_tool.py    # 新建/整文件覆盖（读后写保护 + 内容上限 + 原子写）
│       └── bash_tool.py     # 低频兜底命令（黑白名单 + 审批 + 平台动态描述）
├── app/
│   ├── cli.py               # 交互式命令行入口
│   └── one_shot.py          # 单轮任务入口（-p / --resume）
├── prompts/
│   └── system.md            # 系统提示词（行为规则 + 工具使用示例）
├── memory/
│   ├── paths.py             # 运行期数据目录（~/.minicoder/<项目哈希>/）
│   ├── retention.py         # 按会话数 / 天数清理过期运行数据
│   ├── trace.py             # JSONL trace 写入与读取（可回放）
│   └── transcript.py        # append-only 会话记录（可继续会话）
├── demo/                    # 较完整的任务演示 / 集成验证脚本
│   ├── __init__.py
│   ├── m1_chat.py           # M1 演示：一次对话闭环
│   ├── m2_tool_call.py      # M2 演示：模型自主调用工具
│   ├── m3_task.py           # M3 演示：搜索 → 读取 → 总结
│   ├── m6_context.py        # M6 演示：上下文拼装 / 水位 compact
│   ├── m7_security.py       # M7 演示：工作空间约束 + Bash 审批
│   └── m8_session.py        # M8 演示：单会话持续对话 + 恢复
└── tests/
    ├── test_message.py / test_llm.py   # M1 基础层（含流式聚合）
    ├── test_tools.py / test_grep_tool.py / test_edit_tool.py  # M2/M3 工具
    ├── test_loop.py / test_loop_trace.py   # M3/M4 循环与落盘
    ├── test_memory.py       # trace / transcript
    ├── test_session.py      # M8 单会话历史 / 恢复
    ├── test_retention.py    # M8 数据保留清理
    ├── test_oneshot.py      # 单轮入口
    └── test_bash_tool.py    # Bash 兜底工具
```

## 五、模块职责

- core/config.py：把 .env 与环境变量集中成类型化配置，换模型只改配置不改代码。
- core/message.py：统一消息结构（四种角色），tool 字段在 M2/M3 使用。
- core/llm.py：把不同模型的 API 统一成同一个接口（chat / chat_stream），支持注入 transport 便于测试。
- runtime/loop.py：核心循环；拿到模型返回后判断是“继续思考”还是“调用工具”，超过 max_steps 强制结束；M4 起可把每一步写进 trace、把消息写进 transcript。
- tools/registry.py：工具注册 + 生成 JSON Schema 给模型 + 调用分发 + 统一包装返回 + 读后写保护。
- tools/builtin/edit_tool.py：唯一会改文件的工具，必须“先 Read 再 Edit”，写入时校验文件未被外部修改，用临时文件原子替换。
- tools/builtin/bash_tool.py：低频兜底命令，命中禁止模式直接拒绝，超时 / 非零退出按错误码返回。
- memory/trace.py：每轮记录时间、会话、消息、工具名、参数、结果，用于排查和面试演示“可观测性”。
- memory/transcript.py：append-only 记录消息，配合 history 参数实现“读档继续”。
- runtime/session.py：持有单个 AgentLoop 与 history，每次 ask 把新增消息并入历史并写回 transcript，支持 resume 从既有 transcript 恢复。
- memory/paths.py：把运行期数据统一放到 ~/.minicoder/<项目哈希>/，避免污染被操作的仓库；MINICODER_DATA_DIR 可覆盖。
- memory/retention.py：按会话数量与保存天数清理 trace / transcript / artifact，配合 CLI 启动 / /clean 自动执行。
- runtime/output_guard.py：对超长工具输出做"截断 + 落盘"，返回"前 N 行预览 + artifact 路径 + 用 read/grep 精读"的提示，避免头部信息永久丢失。
- app/one_shot.py：`-p "任务"` 单轮执行并退出，`--resume` 可继续会话。
- demo/：比单元测试更完整的任务演示脚本，可跑通真实链路。

## 六、里程碑（最小闭环路线）

- M1 能对话（已完成）：core/config.py + core/message.py + core/llm.py 打通“发消息 → 收到回复”，demo/m1_chat.py 可演示。
- M2 能调工具（已完成）：tools/registry.py + Glob / Read 两个工具，模型能根据任务选择工具。
- M3 闭环跑通（已完成）：runtime/loop.py + Grep/Edit 工具 + app/cli.py，注册中心内置读后写保护（read 缓存 + 乐观锁）。
- M4 变得可靠（已完成）：Edit 读后写 + 乐观锁 + 原子写入；memory/ 的 JSONL trace / transcript；app/one_shot.py 的 -p 单轮模式；Bash 低频兜底工具。
- M5 提升一轮能力（已完成）：内容哈希指纹解决 ABA；工具 schema / 系统提示词工作流；grep 支持单文件；超限强制总结 + partial 标记；默认 max_steps=20。
- M6 上下文工程（已完成）：runtime/context/ 实现 L1 系统规则 / L2 项目规则（AGENTS.md + 文件地图）/ L3 会话动态拼装；token 估算水位检测 + 截断式 compact；LLM 摘要式 compact 预留。
- M7 安全边界（已完成）：tools/workspace.py 工作空间约束（越界/逃逸拒绝）；tools/permissions.py Bash 风险分级 + 审批网关（ask/allow/deny + 会话记忆）；文件工具接入 workspace；注册中心参数清洗；非交互默认 deny（fail-closed）。
- M8 会话连续与流式（已完成）：runtime/session.py 单会话复用 + 历史累积；memory/paths.py 迁移 ~/.minicoder、memory/retention.py 保留清理；core/llm.py chat_stream_response 流式聚合（含工具调用分片）；loop 打转检测；CLI 增加 /new /resume /clean 与流式输出。
- M9 输出治理与预算（已完成）：runtime/output_guard.py 超长输出全文落盘 artifacts/ + 预览提示（Bash 去掉头部硬截断）；ContextBuilder.note_usage 用实测 usage 校准水位；error 工具结果把错误码/原因回填给模型。
- M9 增强：tools/builtin/write_tool.py 新建/覆盖文件（读后写保护 + 原子写 + 内容上限）；L1 环境块按平台注入 shell 类型与限制；Bash 工具描述动态生成；python -c 拒绝消息附带替代路径。

## 七、验收清单（M4 结束时）

- [x] `uv run python -m app.cli` 能进入交互界面并完成任务
- [x] Agent 能自主完成“找到函数 → 读取 → 修改一行”的演示任务
- [x] 修改文件前必须 Read，未读直接 Edit 会被拒绝
- [x] 文件被外部修改时 Edit 返回 CONFLICT，Agent 能重新 Read 后重试
- [x] 每次运行都在 memory/traces/ 留下 JSONL，可逐条回放
- [x] pytest 通过，ruff 无错误

## 八、下一步

- 真实模型联调：配置 .env 后，用 `demo/m3_task.py` 或 `app.one_shot -p "任务"` 跑真实链路。
- 规划（已确认优先级）：
  - 核心（M6 已完成）：上下文工程（runtime/context/：L1/L2/L3 拼装 + 水位 compact）。
  - 安全（M7 已完成）：工作空间约束 + Bash 审批 + 参数清洗。
  - 中等（M8 起）：会话连续性与流式交互（单会话复用 loop、~/.minicoder 数据目录、清理机制）、打转/重复调用检测、LLM 摘要式 compact、read 分段读取。
  - 后期：Skills / MCP / 子代理（优先级低，之后再实现）。
  - 求职场景：简历 / 岗位搜索 demo。
