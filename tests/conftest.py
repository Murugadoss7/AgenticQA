import pytest
from unittest.mock import AsyncMock, MagicMock
from providers.base import ProviderResponse, ToolCall


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=ProviderResponse(content='{"ok": true}', tool_calls=[])
    )
    return provider


@pytest.fixture
def mock_tool_provider():
    """Provider that returns a single tool call."""
    provider = MagicMock()

    def _make_tool_response(name, input_data, tool_id="tc_001"):
        return ProviderResponse(
            content="",
            tool_calls=[ToolCall(id=tool_id, name=name, input=input_data)],
        )

    provider.make_tool_response = _make_tool_response
    provider.complete = AsyncMock(return_value=ProviderResponse(content="", tool_calls=[]))
    return provider
