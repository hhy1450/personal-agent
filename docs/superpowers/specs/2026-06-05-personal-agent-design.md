# 多 Agent 工作流自动化应用 — 设计文档

> 日期：2026-06-05  
> 状态：待实现  
> 作者：黄海亦

---

## 1. 项目概述

构建一个基于 LangGraph 的多 Agent 工作流自动化应用。用户通过 CLI 或 Web 界面提交任务，系统自动完成：任务拆解（Planner）→ 路由分派（Router）→ 各 Agent 调用工具执行（Executor）→ 质量审核（Reviewer）→ 结果汇总输出。

核心用例：
- **信息调研**：用户输入调研主题 → 多 Agent 协同搜索、分析、生成结构化报告
- **定时任务**：配置 cron 表达式 → 每日自动执行（如早报抓取、代码仓库监控）
- **代码辅助**：代码审查、Bug 分析、自动化脚本生成

---

## 2. 技术选型

| 领域 | 选型 | 理由 |
|------|------|------|
| 编程语言 | Python 3.11+ | AI/LLM 生态最完善 |
| 工作流引擎 | **LangGraph** | 状态图建模，天然支持条件路由、错误恢复、状态持久化 |
| LLM 后端 | **DeepSeek API**（可插拔） | 性价比高，兼容 OpenAI SDK，支持 Function Calling |
| Web 框架 | **FastAPI** | 异步支持好，自动生成 API 文档 |
| Web 界面 | **Streamlit** | 纯 Python，快速搭建，不需要前端技术栈 |
| CLI 框架 | **Click** | 轻量，Python CLI 标准选择 |
| 结构化存储 | **SQLite** | 零配置，单文件，足够个人使用 |
| 向量存储 | **ChromaDB** | 轻量，Python 原生，适合 Agent 长期记忆 |
| 任务调度 | **APScheduler** | 支持 cron，任务可持久化 |
| 包管理 | **uv + pyproject.toml** | 快速依赖解析，现代化 Python 工程 |

---

## 3. 系统架构

```
┌──────────────────────────────────────────────────┐
│                  用户入口                          │
│           CLI (Click)  │  Web UI (Streamlit)       │
└──────────────┬──────────────┘─────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              API 层 (FastAPI)                      │
│        /tasks  /workflows  /agents  /logs          │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│           工作流引擎 (LangGraph)                    │
│  ┌──────────────────────────────────────────┐    │
│  │  Planner → Router → Executor → Reviewer  │    │
│  │     ↑         ↓          ↓          ↓      │    │
│  │     └─────── Tool ──── Agent ──── Memory  │    │
│  └──────────────────────────────────────────┘    │
└──────────────┬──────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────┐
│              基础设施层                             │
│  LLM Provider  │  Storage  │  Vector DB  │  Cron  │
│  (可插拔)       │ (SQLite)  │ (ChromaDB)  │ Timer  │
└──────────────────────────────────────────────────┘
```

### 分层职责

| 层 | 职责 | 不负责 |
|------|------|--------|
| **用户入口** | 接收用户输入，展示结果 | 不处理业务逻辑 |
| **API 层** | 参数校验、任务调度、提供 REST 接口 | 不直接调用 LLM |
| **工作流引擎** | 任务拆解、Agent 路由、工具调用、质量审核 | 不关心 UI 和存储细节 |
| **基础设施** | LLM 调用、数据持久化、向量检索、定时触发 | 不包含业务逻辑 |

---

## 4. LangGraph 工作流引擎

### 4.1 状态定义

所有节点通过类型化状态通信：

```python
class WorkflowState(TypedDict):
    task: str               # 用户原始输入
    plan: list[SubTask]     # Planner 拆解的子任务列表
    current_step: int       # 当前执行到第几步
    results: dict           # 各子任务的执行结果
    final_output: str       # 汇总后的最终输出
    errors: list[dict]      # 错误日志（步骤、类型、详情）
    next_action: str        # 路由决策: "continue" | "review" | "retry" | "finish"
```

### 4.2 图结构

```
START → Planner → Router → [Agent Nodes] → Tool Executor → Reviewer
                       ↑        ↑                               ↓
                       │        └───── Retry ←───── reject ──────┤
                       │                                          │
                       └── Aggregator ←── accept ←───────────────┘
                              ↓
                            END
```

### 4.3 节点定义

| 节点 | 输入 | 输出 | 描述 |
|------|------|------|------|
| **Planner** | `task: str` | `plan: list[SubTask]` | LLM 将模糊任务拆解为有顺序、有类型的子任务列表 |
| **Router** | `plan[current_step]` | `next_agent: str` | 根据子任务类型（research/code/write）路由到对应 Agent |
| **Coder Agent** | 子任务描述 + 上下文 | 工具调用 | 写代码、改 Bug、运行脚本 |
| **Researcher Agent** | 子任务描述 + 上下文 | 工具调用 | 搜索信息、抓取网页、整理资料 |
| **Writer Agent** | 子任务描述 + 上下文 | 工具调用 | 生成报告、写邮件、翻译 |
| **Tool Executor** | 工具调用请求 | 工具执行结果 | 实际执行工具（搜索、读写文件、运行代码） |
| **Reviewer** | 子任务结果 + 原始需求 | `next_action` | 检查结果质量，决定通过/重试/失败 |
| **Aggregator** | 所有子任务结果 | `final_output` | 合并结果，生成最终输出 |

### 4.4 错误处理

每个节点挂 `on_error` 边：

```
执行失败 → 错误分类 →
  ├─ Retryable（超时/限流/临时故障）→ 指数退避（1s, 2s, 4s）→ 重试，最多 3 次
  ├─ Fixable（格式错误/输出不合规）→ LLM 自动修正 → 重新提交
  └─ Fatal（权限/不存在/超出能力）→ 记录日志 → 跳过该子任务 → 通知用户
```

工作流不会因单个子任务失败而整体崩溃。

---

## 5. Agent 与 Tool 系统

### 5.1 Agent 设计

所有 Agent 共享基类，通过配置区分行为：

```python
class AgentConfig:
    name: str             # 标识符
    system_prompt: str    # 角色定义
    tools: list[str]      # 可用工具列表
    model: str            # 默认 "deepseek-chat"
    max_retries: int      # 最大重试次数，默认 3
    temperature: float    # 默认 0.1（Agent 需要确定性）
```

| Agent | 角色定位 | 可用工具 |
|-------|----------|----------|
| **Planner** | 将模糊的用户意图转化为结构化、可执行的子任务列表 | 无需工具 |
| **Researcher** | 搜索信息、抓取网页、对比分析、事实核查 | `web_search`, `fetch_url`, `save_to_memory` |
| **Coder** | 编写/修改/运行代码，代码审查 | `read_file`, `write_file`, `run_code`, `search_code` |
| **Writer** | 撰写结构化文档（报告/邮件/方案），翻译 | `read_file`, `write_file`, `save_to_memory` |
| **Reviewer** | 检查输出是否符合用户意图、质量是否过关 | `read_file`, `web_search`（事实核查用） |

### 5.2 Tool 系统

工具即函数，通过装饰器自动注册并生成 OpenAI Function Calling 格式的 JSON Schema：

```python
@tool(
    name="web_search",
    description="搜索互联网内容，返回相关结果列表"
)
def web_search(query: str, max_results: int = 10) -> list[dict]:
    """搜索并返回结果列表"""
    ...

@tool(
    name="run_code",
    description="在隔离沙箱中执行 Python 代码"
)
def run_code(code: str, timeout: int = 30) -> dict:
    """返回 {stdout, stderr, exit_code}"""
    ...
```

安全约束：
- 文件操作限制在 `workspace/` 目录下（沙箱）
- 代码执行使用 `subprocess` + `timeout` 限制
- 网络请求限制白名单域名
- 所有工具调用记录审计日志

---

## 6. 存储设计

### 6.1 SQLite（结构化数据）

```sql
-- 任务表
tasks (id, title, description, status, created_at, updated_at)

-- 工作流执行日志
workflow_runs (id, task_id, state_json, started_at, finished_at, status)

-- Agent 配置
agent_configs (id, name, system_prompt, tools_json, model)

-- 触发器
triggers (id, name, cron_expr, task_template, enabled, last_run_at)
```

### 6.2 ChromaDB（向量记忆）

- 存储 Agent 长期记忆的 embedding
- 支持语义检索：「上次那个 XX 是怎么做的？」
- collection 分类：`conversations`, `knowledge_base`, `task_results`

### 6.3 文件系统

```
workspace/
├── reports/     # 生成的报告
├── code/        # 生成/修改的代码
├── downloads/   # 下载的文件
└── logs/        # 运行日志
```

---

## 7. 用户界面

### 7.1 CLI（Click）

```bash
# 单次任务
pa run "帮我调研 DeepSeek V3 和 Qwen3 的对比"

# 定时任务
pa schedule --cron "0 9 * * *" "每天早上抓取 AI 新闻并总结"

# 查看历史
pa history --limit 10

# 查看某个任务的执行过程
pa inspect <task_id>
```

### 7.2 Web 界面（Streamlit）

单页应用，涵盖核心操作：

- **左侧栏**：任务列表（历史 + 当前运行中），按状态筛选
- **主面板**：选中任务的详情 —— 子任务进度、执行日志流、最终输出
- **顶部**：新建任务输入框，支持 Markdown 预览结果
- **底部**：定时任务管理

---

## 8. 测试策略

| 层级 | 范围 | 工具 | 目标覆盖率 |
|------|------|------|------------|
| **单元测试** | Tool 函数、状态转换、路由逻辑 | `pytest` | 80%+ |
| **集成测试** | Agent + Tool 协作、完整工作流 | `pytest` + Mock LLM | 关键路径 100% |
| **E2E** | CLI 命令、API 端点 | 手动 + 关键路径自动化 | 核心场景 |

Mock LLM 策略：测试中不实际调用 DeepSeek API，而是 mock `openai.ChatCompletion.create` 返回固定 JSON。这保证测试快速、稳定、免费。

---

## 9. 项目目录结构

```
personal-agent/
├── src/
│   ├── engine/              # LangGraph 工作流引擎
│   │   ├── __init__.py
│   │   ├── graph.py         # StateGraph 构建
│   │   ├── state.py         # WorkflowState 定义
│   │   └── nodes/           # 图节点实现
│   │       ├── planner.py
│   │       ├── router.py
│   │       ├── executor.py
│   │       └── reviewer.py
│   ├── agents/              # Agent 定义
│   │   ├── __init__.py
│   │   ├── base.py          # Agent 基类
│   │   ├── config.py        # AgentConfig
│   │   └── prompts/         # System prompts 模板
│   ├── tools/               # Tool 系统
│   │   ├── __init__.py
│   │   ├── registry.py      # Tool 注册中心
│   │   ├── web_search.py
│   │   ├── file_ops.py
│   │   ├── code_exec.py
│   │   └── sandbox.py       # 沙箱安全
│   ├── memory/              # 记忆系统
│   │   ├── __init__.py
│   │   ├── vector_store.py  # ChromaDB 封装
│   │   └── conversation.py  # 对话历史管理
│   ├── llm/                 # LLM Provider 抽象
│   │   ├── __init__.py
│   │   ├── base.py          # 抽象基类
│   │   ├── deepseek.py      # DeepSeek 实现
│   │   └── factory.py       # Provider 工厂（可插拔）
│   ├── triggers/            # 调度系统
│   │   ├── __init__.py
│   │   ├── scheduler.py     # APScheduler 封装
│   │   └── models.py        # 触发器数据模型
│   ├── api/                 # FastAPI 接口
│   │   ├── __init__.py
│   │   ├── app.py           # FastAPI 应用入口
│   │   └── routes/          # 路由模块
│   │       ├── tasks.py
│   │       ├── workflows.py
│   │       └── triggers.py
│   ├── cli/                 # CLI 入口
│   │   ├── __init__.py
│   │   └── main.py          # Click 命令定义
│   ├── storage/             # 持久化层
│   │   ├── __init__.py
│   │   ├── database.py      # SQLite 操作
│   │   └── models.py        # ORM 模型
│   └── config/              # 配置管理
│       ├── __init__.py
│       └── settings.py      # 环境变量 + 默认配置
├── web/                     # Streamlit 界面
│   ├── app.py               # Streamlit 入口
│   └── components/          # 可复用 UI 组件
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py          # Fixtures + Mock
├── workspace/               # 运行时工作目录（gitignore）
├── pyproject.toml
├── .env.example             # API key 等配置模板
├── .gitignore
└── README.md
```

---

## 10. MVP 范围与后续迭代

### MVP（v0.1.0）

- [x] LangGraph 工作流引擎核心（Planner → Router → Executor → Reviewer）
- [x] 3 个 Agent：Researcher、Writer、Reviewer（Coder Agent 延后）
- [x] 3 个 Tool：web_search、read_file、write_file
- [x] CLI 单次任务
- [x] DeepSeek API 集成
- [x] SQLite 任务记录

### v0.2.0

- [ ] Coder Agent + run_code Tool
- [ ] 定时任务（APScheduler）
- [ ] Streamlit Web 界面

### v0.3.0

- [ ] ChromaDB 向量记忆
- [ ] LLM Provider 可插拔（Claude / OpenAI）
- [ ] 更完善的错误恢复

---

## 11. 关键设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 框架 vs 自研 | **LangGraph** | 大二学生找日常实习，用成熟框架更务实；LangGraph 的图/状态/条件路由概念足够深入，面试有得聊 |
| Web 框架 | **FastAPI + Streamlit** | FastAPI 异步性能好、自动文档；Streamlit 免去前端学习成本 |
| 数据库 | **SQLite + ChromaDB** | 个人项目无需 MySQL/PostgreSQL，SQLite 零运维 |
| Agent 间通信 | **共享 State（TypedDict）** | LangGraph 原生模式，比消息队列简单，数据流可追踪 |
| LLM 调用 | **DeepSeek + OpenAI SDK** | 兼容 OpenAI Function Calling 格式，成本低，中文强 |
