import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from providers.anthropic_provider import AnthropicProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key", model="claude-haiku-4-5")


async def test_complete_returns_text(provider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text='{"score": 85}')]

    with patch.object(provider.client.messages, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "evaluate this"}],
            system="You are an evaluator.",
        )

    assert isinstance(result, ProviderResponse)
    assert result.content == '{"score": 85}'
    assert result.tool_calls == []


async def test_complete_with_cache_control(provider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="ok")]

    with patch.object(provider.client.messages, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
        await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            system="stable system prompt",
        )
        call_kwargs = mock_create.call_args.kwargs
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
