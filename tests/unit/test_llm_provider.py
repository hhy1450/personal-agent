"""Tests for LLM provider layer."""

from unittest.mock import MagicMock, patch

from src.llm.base import LLMProvider
from src.llm.deepseek import DeepSeekProvider


class TestLLMProvider:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """LLMProvider should not be directly instantiable."""
        try:
            LLMProvider()  # type: ignore
            instantiable = True
        except TypeError:
            instantiable = False
        assert not instantiable, "Abstract class should raise TypeError"


class TestDeepSeekProvider:
    """Tests for DeepSeek provider."""

    def test_default_construction(self):
        """Provider should construct with defaults (without real API key in tests)."""
        provider = DeepSeekProvider(
            api_key="test-key",
            base_url="https://test.api.com",
            model="test-model",
        )
        assert provider.model_name == "test-model"

    @patch("src.llm.deepseek.OpenAI")
    def test_chat_completion_no_tools(self, mock_openai_class):
        """chat_completion should pass messages to OpenAI SDK."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {
            "choices": [{"message": {"content": "Hello back"}}]
        }
        mock_client.chat.completions.create.return_value = mock_completion

        provider = DeepSeekProvider(
            api_key="test-key",
            base_url="https://test.api.com",
            model="test-model",
        )
        result = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert result["choices"][0]["message"]["content"] == "Hello back"
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.llm.deepseek.OpenAI")
    def test_chat_completion_with_tools(self, mock_openai_class):
        """chat_completion should pass tools to OpenAI SDK."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_completion = MagicMock()
        mock_completion.model_dump.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "1",
                                "function": {
                                    "name": "search",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_client.chat.completions.create.return_value = mock_completion

        provider = DeepSeekProvider(
            api_key="test-key",
            base_url="https://test.api.com",
            model="test-model",
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {},
                },
            }
        ]
        result = provider.chat_completion(
            messages=[{"role": "user", "content": "Search for X"}],
            tools=tools,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert (
            result["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            == "search"
        )
