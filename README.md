# Personal Agent -- Multi-Agent Workflow Automation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

A LangGraph-based multi-agent workflow automation tool. Input a complex task, and AI agents automatically decompose, assign, execute, review, and deliver structured results.

## Quick Start

### 1. Install

```bash
git clone <repo-url>
cd personal-agent
pip install -e .
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your DeepSeek API key
```

### 3. Use

```bash
# Single task
pa run "Research DeepSeek V3 features and write a report"

# View history
pa history

# Inspect a task
pa inspect 1
```

## Architecture

```
User Input -> Planner (decompose task)
                 |
          Router (dispatch)
          /    |    \
   Researcher  Writer  Reviewer
          \    |    /
          Aggregator (combine)
                 |
            Final Output
```

Agents execute in a ReAct loop: the LLM decides which tool to call, the tool executes, results feed back to the LLM, and the cycle repeats until the agent produces a final text response.

## Tech Stack

- **Workflow Engine**: LangGraph
- **LLM**: DeepSeek API (OpenAI SDK compatible)
- **CLI**: Click
- **Storage**: SQLite
- **Tools**: web_search (DuckDuckGo), read_file, write_file

## Project Structure

```
src/
  engine/         LangGraph workflow engine (graph, nodes, state)
  agents/         Agent definitions, ReAct loop, prompts, config
  tools/          Tool system (search, file ops, sandbox, registry)
  llm/            LLM Provider abstraction + DeepSeek adapter
  storage/        SQLite persistence (tasks, workflow runs)
  cli/            Click CLI (run, history, inspect)
  config/         Environment configuration (.env)
tests/
  unit/           Unit tests for each module
  integration/    End-to-end workflow tests with mocked LLM
```

## MVP Features

- [x] Planner -> Router -> Executor -> Reviewer workflow
- [x] Researcher + Writer + Reviewer agents
- [x] web_search + read_file + write_file tools
- [x] CLI single-task execution
- [x] DeepSeek API integration
- [x] SQLite task recording
- [x] Integration tests with mocked LLM

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only
python -m pytest tests/integration/ -v
```

## License

MIT
