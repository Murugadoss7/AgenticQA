import json
from openai import AsyncAzureOpenAI
from providers.base import ModelProvider, ProviderResponse, ToolCall


class AzureOpenAIProvider(ModelProvider):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str) -> None:
        self.client = AsyncAzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        openai_messages = [{"role": "system", "content": system}] + messages

        create_kwargs: dict = {
            "model": self.model,
            "messages": openai_messages,
            "response_format": {"type": "json_object"},
        }
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)
            create_kwargs.pop("response_format")

        response = await self.client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message

        content = message.content or ""
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return ProviderResponse(content=content, tool_calls=tool_calls)

    @staticmethod
    def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in anthropic_tools
        ]
