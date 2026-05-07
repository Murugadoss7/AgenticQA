import pytest
from unittest.mock import patch, MagicMock
from providers.openrouter_provider import OpenRouterProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3.1-70b-instruct")


async def test_complete_returns_text(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"next_topics": ["Math"]}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "advise"}],
            system="You are advisor.",
        )

    assert '{"next_topics"' in result.content
    assert result.tool_calls == []
