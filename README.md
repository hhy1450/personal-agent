# Personal Agent — 多模态多策略 Agent 工作流自动化

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)]()

基于 LangGraph 的多模态多 Agent 工作流自动化系统。支持文本/图像多模态感知、三种编排策略、动态上下文管理、结构化动作映射和 WebSocket 实时通信。

## 快速开始

### 1. 安装

```bash
git clone https://github.com/hhy1450/personal-agent.git
cd personal-agent
pip install -e .
```

### 2. 配置

```bash
cp .env.example .env
# 编辑 .env，填入 API Key：
# - DEEPSEEK_API_KEY：文本推理（必填）
# - DASHSCOPE_API_KEY：视觉理解，阿里云 DashScope（可选，用于多模态）
```

### 3. 使用

```bash
# 命令行模式
pa run "帮我调研 DeepSeek V3 的特点并写一份报告"
pa history          # 查看历史
pa inspect 1        # 查看任务详情

# Web 模式（FastAPI + WebSocket + Streamlit）
pa web              # API: http://localhost:8000, UI: http://localhost:8501
```

## 架构

```
用户输入 → Planner（拆解任务 + 选择策略）
                ↓
        StrategyRouter（编排路由）
       ↙         ↓          ↘
  Sequential   Parallel     Loop
       ↓         ↓          ↓
       Router（Agent 路由）
    ↙    ↓    ↘
Researcher Writer Reviewer
    ↘    ↓    ↙
    QualityGate（质量审核 + 结构化校验）
         ↓
    Aggregator（汇总）
         ↓
    最终输出
         ↓ (WebSocket)
       实时推送至前端
```

## v0.2.0 新特性

| 模块 | 说明 |
|------|------|
| **多模态感知** | DeepSeek(文本) + Qwen-VL-Max(视觉) 双 Provider，Agent 可理解图片 |
| **三种编排策略** | Sequential / Parallel / Loop，Planner 智能选择 |
| **动态上下文管理** | Token 预算粗裁剪 + LLM 摘要精压缩，支撑 20+ 步长任务 |
| **结构化动作映射** | LLM 输出可审计的 JSON Schema 动作，Reviewer 结构化校验 |
| **WebSocket 实时通信** | FastAPI + WebSocket 实时推送每步执行进度 |

## 技术栈

- **工作流引擎**: LangGraph
- **大模型**: DeepSeek API + Qwen-VL-Max (DashScope)
- **API 服务**: FastAPI + WebSocket
- **Web 界面**: Streamlit
- **命令行**: Click
- **存储**: MySQL
- **工具**: web_search（DuckDuckGo）、read_file、write_file

## 项目结构

```
src/
  engine/         LangGraph 工作流引擎（图、节点、状态）
  agents/         Agent 定义、ReAct 循环、提示词、配置
  actions/        结构化动作映射（Schema、Mapper、Registry）
  tools/          工具系统（搜索、文件操作、沙箱、注册中心）
  memory/         动态上下文管理（Token 预算 + LLM 摘要）
  llm/            LLM Provider 抽象 + DeepSeek/Qwen-VL 适配器
  events/         事件总线（WebSocket 推送）
  api/            FastAPI 应用 + REST/WebSocket 路由
  storage/        MySQL 持久化（任务、工作流运行记录）
  cli/            Click 命令行（run、history、inspect、web）
  config/         环境配置
web/
  app.py           Streamlit Web 界面
tests/
  unit/           各模块单元测试
  integration/    端到端工作流测试（Mock LLM）
```

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v      # 75 passed

# 仅单元测试
python -m pytest tests/unit/ -v

# 仅集成测试
python -m pytest tests/integration/ -v
```

## License

MIT
