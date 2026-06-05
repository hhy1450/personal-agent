"""DeepSeek API provider using OpenAI-compatible SDK."""

from openai import OpenAI

from src.llm.base import LLMProvider
from src.config.settings import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


class DeepSeekProvider(LLMProvider):
    """LLM provider backed by DeepSeek API.

    DeepSeek's API is OpenAI-compatible, so we use the OpenAI SDK
    with a custom base_url.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or DEEPSEEK_API_KEY
        self._base_url = base_url or DEEPSEEK_BASE_URL
        self._model = model or LLM_MODEL
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=30.0,
            max_retries=1,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict:
        kwargs = dict(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        response = self._client.chat.completions.create(**kwargs)
        return response.model_dump()
