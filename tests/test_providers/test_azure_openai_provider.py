import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from providers.azure_openai_provider import AzureOpenAIProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return AzureOpenAIProvider(
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        model="gpt-4o-mini",
    )


async def test_complete_returns_text(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"score": 70}'
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "evaluate"}],
            system="You are evaluator.",
        )

    assert result.content == '{"score": 70}'
    assert result.tool_calls == []


async def test_system_prepended_as_message(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response) as mock_create:
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
        )
        sent_messages = mock_create.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "Be helpful."}
