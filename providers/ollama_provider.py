import httpx
from providers.base import ModelProvider, ProviderResponse


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        if tools:
            raise ValueError(
                "OllamaProvider does not support tool_use — assign Orchestrator to anthropic or azure_openai"
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return ProviderResponse(content=data["message"]["content"], tool_calls=[])
