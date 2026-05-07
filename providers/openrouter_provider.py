from openai import OpenAI
from providers.base import ModelProvider, ProviderResponse


class OpenRouterProvider(ModelProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        openai_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return ProviderResponse(content=content, tool_calls=[])
