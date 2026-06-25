"""LLM Provider factory — automatic selection based on Agent requirements."""
import logging

from src.llm.base import LLMProvider
from src.llm.deepseek import DeepSeekProvider
from src.llm.qwen_vl import QwenVLProvider

logger = logging.getLogger(__name__)


class LLMFactory:
    """Create and cache LLM providers.

    Automatic selection rules:
    - If the agent requires vision → return QwenVLProvider
    - Otherwise → return DeepSeekProvider (default text model)

    Providers are cached so multiple agents share the same client instance.
    """

    def __init__(self):
        self._providers: dict[str, LLMProvider] = {}

    def get_provider(self, requires_vision: bool = False) -> LLMProvider:
        """Get or create an LLM provider.

        Args:
            requires_vision: True if the agent needs image understanding.

        Returns:
            An LLMProvider instance (cached).
        """
        key = "vision" if requires_vision else "text"

        if key not in self._providers:
            if requires_vision:
                self._providers[key] = QwenVLProvider()
                logger.info("LLMFactory created QwenVLProvider for vision tasks")
            else:
                self._providers[key] = DeepSeekProvider()
                logger.info("LLMFactory created DeepSeekProvider for text tasks")

        return self._providers[key]

    @property
    def text_provider(self) -> LLMProvider:
        """Shortcut for the default text provider."""
        return self.get_provider(requires_vision=False)

    @property
    def vision_provider(self) -> LLMProvider:
        """Shortcut for the vision-capable provider."""
        return self.get_provider(requires_vision=True)


# Global factory singleton
_factory: LLMFactory | None = None


def get_llm_factory() -> LLMFactory:
    """Get the global LLM factory (creates on first call)."""
    global _factory
    if _factory is None:
        _factory = LLMFactory()
    return _factory
