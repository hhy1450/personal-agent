# Personal Agent — 多 Agent 工作流自动化

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

基于 LangGraph 的多 Agent 工作流自动化工具。输入一个复杂任务，AI 自动拆解、分派、执行、审核，最终输出结构化结果。

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
# 编辑 .env，填入你的 DeepSeek API Key
# 去 https://platform.deepseek.com 注册即可获取
```

### 3. 使用

```bash
# 单次任务
pa run "帮我调研 DeepSeek V3 的特点并写一份报告"

# 查看历史
pa history

# 查看任务详情
pa inspect 1
```

## 架构

```
用户输入 → Planner（拆解任务）
                ↓
         Router（路由分发）
         ↙    ↓    ↘
   Researcher  Writer  Reviewer
         ↘    ↓    ↙
         Aggregator（汇总）
                ↓
            最终输出
```

Agent 采用 ReAct 循环模式运行：LLM 决定调用哪个工具 → 工具执行 → 结果反馈给 LLM → 循环直到 Agent 输出最终文本。

## 技术栈

- **工作流引擎**: LangGraph
- **大模型**: DeepSeek API（兼容 OpenAI SDK）
- **命令行**: Click
- **存储**: SQLite
- **工具**: web_search（DuckDuckGo）、read_file、write_file

## 项目结构

```
src/
  engine/         LangGraph 工作流引擎（图、节点、状态）
  agents/         Agent 定义、ReAct 循环、提示词、配置
  tools/          工具系统（搜索、文件操作、沙箱、注册中心）
  llm/            LLM Provider 抽象 + DeepSeek 适配器
  storage/        SQLite 持久化（任务、工作流运行记录）
  cli/            Click 命令行（run、history、inspect）
  config/         环境配置
tests/
  unit/           各模块单元测试
  integration/    端到端工作流测试（Mock LLM）
```

## MVP 功能

- [x] Planner → Router → Executor → Reviewer 工作流
- [x] Researcher + Writer + Reviewer 三个 Agent
- [x] web_search + read_file + write_file 三个工具
- [x] CLI 单次任务执行
- [x] DeepSeek API 集成
- [x] SQLite 任务记录
- [x] Mock LLM 集成测试

## 运行测试

```bash
# 全部测试
python -m pytest tests/ -v

# 仅单元测试
python -m pytest tests/unit/ -v

# 仅集成测试
python -m pytest tests/integration/ -v
```

## License

MIT
