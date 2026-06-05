"""Agent configuration."""
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for an AI agent.

    Defines the agent's identity (system prompt), capabilities (tools),
    and behavior (model, temperature, retries).
    """
    name: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = "deepseek-chat"
    max_retries: int = 3
    temperature: float = 0.1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "model": self.model,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
        }
