import anthropic
from providers.base import ModelProvider, ProviderResponse, ToolCall


class AnthropicProvider(ModelProvider):
    """Anthropic provider for specialist agents (no adaptive thinking — use direct SDK for Orchestrator)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        system_block = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "system": system_block,
            "messages": messages,
        }
        if tools:
            create_kwargs["tools"] = tools

        response = self.client.messages.create(**create_kwargs)

        text_content = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return ProviderResponse(content=text_content, tool_calls=tool_calls)
