"""Base provider module."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ProviderResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelProvider(ABC):
    """Unified interface across Anthropic, Azure OpenAI, Ollama, and OpenRouter.

    All agents (except Orchestrator) call this — they never touch provider SDKs directly.
    Tools parameter uses Anthropic tool schema format; each provider translates internally.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        ...
