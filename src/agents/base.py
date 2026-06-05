"""Base agent implementation using LLM provider + tool execution loop."""
import json
from typing import Any

from src.agents.config import AgentConfig
from src.llm.base import LLMProvider
from src.tools.registry import get_tools_as_openai_schemas, execute_tool


class Agent:
    """An AI agent that can use tools to accomplish tasks.

    Implements a ReAct-style loop:
    1. Send task + context + tool schemas to LLM
    2. If LLM responds with tool calls, execute them
    3. Feed tool results back to LLM
    4. Repeat until LLM responds with text (no tool calls) or max retries
    """

    def __init__(self, config: AgentConfig, llm_provider: LLMProvider):
        self.config = config
        self.llm = llm_provider
        self._messages: list[dict] = []

    def run(self, task: str, context: str = "") -> str:
        """Execute the agent's task.

        Args:
            task: The task description.
            context: Previous context/results from other agents.

        Returns:
            The agent's final text response.
        """
        system_prompt = self.config.system_prompt.format(
            task=task, context=context or "None"
        )

        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        tool_schemas = get_tools_as_openai_schemas(self.config.tools) if self.config.tools else None

        for attempt in range(self.config.max_retries):
            response = self.llm.chat_completion(
                messages=self._messages,
                tools=tool_schemas,
                temperature=self.config.temperature,
            )

            choice = response["choices"][0]
            message = choice["message"]

            # If no tool calls, agent is done
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return message.get("content") or ""

            # Add assistant message to history
            self._messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            # Execute each tool call and add results
            for tc in tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError:
                    args = {}

                try:
                    result = execute_tool(tool_name, args)
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", tool_name),
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Max retries reached — return last content or error
        return self._messages[-1].get("content", "Max retries exceeded, agent stopped.")

    def run_streaming(self, task: str, context: str = "") -> Any:
        """Run the agent and yield intermediate results.

        Yields tuples of (event_type, data) where event_type is:
        - "thinking": agent is deciding what to do
        - "tool_call": agent called a tool (tool_name, args, result)
        - "response": agent's final text response
        - "error": something went wrong
        """
        system_prompt = self.config.system_prompt.format(
            task=task, context=context or "None"
        )

        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        tool_schemas = get_tools_as_openai_schemas(self.config.tools) if self.config.tools else None

        for attempt in range(self.config.max_retries):
            yield ("thinking", f"Agent {self.config.name}: thinking (attempt {attempt + 1})...")

            try:
                response = self.llm.chat_completion(
                    messages=self._messages,
                    tools=tool_schemas,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                yield ("error", f"LLM error: {str(e)}")
                return

            choice = response["choices"][0]
            message = choice["message"]

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                yield ("response", message.get("content") or "")
                return

            self._messages.append({
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                func = tc["function"]
                tool_name = func["name"]
                try:
                    args = json.loads(func["arguments"])
                except json.JSONDecodeError:
                    args = {}

                yield ("tool_call", {"tool": tool_name, "args": args})

                try:
                    result = execute_tool(tool_name, args)
                except Exception as e:
                    result = f"Tool execution error: {str(e)}"

                yield ("tool_result", {"tool": tool_name, "result": str(result)[:500]})

                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", tool_name),
                    "content": json.dumps(result, ensure_ascii=False),
                })

        yield ("error", "Max retries exceeded")
