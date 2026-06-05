"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract interface for LLM backends.

    All providers must implement chat_completion with the OpenAI-compatible
    messages + tools format.
    """

    @abstractmethod
    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of OpenAI-format messages.
            tools: Optional list of tool definitions (JSON Schema format).
            tool_choice: "auto", "none", or "required".
            temperature: Sampling temperature.
            max_tokens: Max tokens in response.

        Returns:
            Raw API response dict with at least:
            {"choices": [{"message": {"content": ..., "tool_calls": [...]}}]}
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier for this provider."""
        ...
