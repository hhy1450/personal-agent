"""Dynamic context manager — two-layer compression for long-running tasks.

Implements the Mobile-OwnClaw-inspired context window management:
1. Layer 1 (Coarse): Token budget trimming — preserve user intent + recent N steps,
   truncate old steps to configurable max chars.
2. Layer 2 (Fine): LLM summarization — when coarse trim isn't enough, compress
   older history into a concise summary via a lightweight LLM call.
"""
import logging
from dataclasses import dataclass

from src.llm.base import LLMProvider
from src.memory.models import ContextConfig, HistoryStep

logger = logging.getLogger(__name__)

# Token estimation: rough heuristic (~4 chars per token for Chinese, ~4 for English)
CHARS_PER_TOKEN = 4


class ContextManager:
    """Manages context window for agent execution.

    Responsibility chain:
        raw history → coarse trim → token check → LLM summary (if needed)
    Each stage reduces token count until within budget.

    Usage:
        mgr = ContextManager(config, llm_provider)
        context = mgr.build_context(task="...", history=[...], current_step=2)
    """

    def __init__(self, config: ContextConfig | None = None, llm: LLMProvider | None = None):
        self.config = config or ContextConfig()
        self._llm = llm

    def build_context(
        self,
        task: str,
        history: list[HistoryStep],
        current_step: int,
    ) -> str:
        """Build a token-budget-aware context string for the current agent.

        Args:
            task: The user's original task description.
            history: All completed HistorySteps so far.
            current_step: The step index about to be executed.

        Returns:
            A context string suitable for the agent's system/user prompt.
        """
        if not history:
            return "None"

        # Split: recent steps always kept in full, older steps subject to trimming
        keep_count = self.config.keep_recent_steps
        old_steps = history[:-keep_count] if len(history) > keep_count else []
        recent_steps = history[-keep_count:] if len(history) > keep_count else history

        # Build context
        parts = []
        parts.append(f"# Original Task\n{task}")

        # --- Layer 1: Coarse trim on old steps ---
        if old_steps:
            parts.append("\n# Earlier Steps (trimmed)")
            for step in old_steps:
                result_preview = step.result[:self.config.max_chars_per_old_step]
                if len(step.result) > self.config.max_chars_per_old_step:
                    result_preview += "..."
                parts.append(
                    f"Step {step.step_index} [{step.agent_name}]: {step.description}\n"
                    f"Result: {result_preview}"
                )

        # --- Recent steps kept in full ---
        if recent_steps and (old_steps or len(recent_steps) < len(history)):
            parts.append("\n# Recent Steps (full)")
        for step in recent_steps:
            parts.append(
                f"Step {step.step_index} [{step.agent_name}]: {step.description}\n"
                f"Result: {step.result[:2000]}"
            )

        context = "\n\n".join(parts)

        # --- Token budget check ---
        estimated_tokens = self._estimate_tokens(context)
        logger.debug(
            "Context build: %d steps → ~%d tokens (budget: %d)",
            len(history), estimated_tokens, self.config.max_tokens,
        )

        # --- Layer 2: LLM summary if over threshold ---
        if estimated_tokens > self.config.summary_threshold and self._llm is not None:
            logger.info(
                "Context ~%d tokens exceeds summary threshold %d — triggering LLM compression",
                estimated_tokens, self.config.summary_threshold,
            )
            context = self._llm_summarize(context, task)
            compressed_tokens = self._estimate_tokens(context)
            logger.info(
                "LLM compression: %d → ~%d tokens",
                estimated_tokens, compressed_tokens,
            )

        return context

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character count.

        Uses a heuristic (chars / 4) which is accurate enough for
        budget management without adding a tiktoken dependency.
        """
        return max(1, len(text) // CHARS_PER_TOKEN)

    def _llm_summarize(self, full_context: str, task: str) -> str:
        """Compress context via LLM summarization (Layer 2).

        Asks the LLM to distill the execution history into a concise
        summary that preserves key findings, decisions, and errors.
        """
        prompt = (
            "You are a context compression assistant. Given the full execution "
            "history below, produce a concise summary (max 500 words) that "
            "preserves:\n"
            "1. Key findings and facts discovered\n"
            "2. Important decisions made\n"
            "3. Any errors or issues encountered\n"
            "4. What remains to be done\n\n"
            f"Original task: {task}\n\n"
            f"Execution history:\n{full_context[:6000]}\n\n"
            "Summary:"
        )

        try:
            response = self._llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1024,
            )
            summary = response["choices"][0]["message"]["content"]
            return (
                f"# Original Task\n{task}\n\n"
                f"# Execution Summary (compressed from earlier steps)\n{summary}"
            )
        except Exception as e:
            logger.warning("LLM summarization failed: %s — falling back to truncated context", e)
            return full_context  # Graceful fallback
