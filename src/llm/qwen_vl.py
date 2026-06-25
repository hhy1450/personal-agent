"""Qwen-VL vision-language provider via DashScope OpenAI-compatible API."""
import logging

from openai import OpenAI

from src.llm.base import LLMProvider
from src.config.settings import DASHSCOPE_API_KEY, VISION_MODEL

logger = logging.getLogger(__name__)

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenVLProvider(LLMProvider):
    """LLM provider backed by Alibaba Qwen-VL-Max via DashScope.

    DashScope provides an OpenAI-compatible endpoint, so we reuse
    the same OpenAI SDK pattern as DeepSeekProvider.

    Supports vision: accepts image_url content blocks alongside text.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or DASHSCOPE_API_KEY
        self._model = model or VISION_MODEL
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=QWEN_BASE_URL,
            timeout=60.0,
            max_retries=1,
        )
        logger.info("QwenVLProvider initialised — model=%s", self._model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_vision(self) -> bool:
        return True

    def chat_completion(
        self,
        messages: list[dict],
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
