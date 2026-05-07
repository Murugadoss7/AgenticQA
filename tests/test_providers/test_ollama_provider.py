import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from providers.ollama_provider import OllamaProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return OllamaProvider(base_url="http://localhost:11434", model="llama3.1:8b")


async def test_complete_returns_text(provider):
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": '{"strengths": ["good"]}'}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        result = await provider.complete(
            messages=[{"role": "user", "content": "analyze"}],
            system="You are analyzer.",
        )

    assert result.content == '{"strengths": ["good"]}'
    assert result.tool_calls == []


async def test_tools_raises(provider):
    with pytest.raises(ValueError, match="tool"):
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=[{"name": "some_tool", "input_schema": {}}],
        )
