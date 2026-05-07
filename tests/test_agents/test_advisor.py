# tests/test_agents/test_advisor.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.advisor import AdvisorAgent
from models.session import Analysis, Recommendation


@pytest.fixture
def agent(mock_provider):
    return AdvisorAgent(provider=mock_provider)


@pytest.fixture
def analysis():
    return Analysis(strengths=["concept recall"], weaknesses=["mechanism detail"], overall_score=68.0)


async def test_returns_recommendation(agent, mock_provider, analysis):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "next_topics": ["Chemistry"],
                "focus_areas": ["reaction mechanisms", "molecular bonding"],
                "message": "You have a solid foundation. Focus on mechanisms next.",
            }),
            tool_calls=[],
        )
    )
    rec = await agent.run(
        analysis=analysis, current_topic="Science", all_topics=["Science", "Chemistry", "Math"]
    )
    assert isinstance(rec, Recommendation)
    assert "Chemistry" in rec.next_topics
    assert len(rec.focus_areas) == 2


async def test_all_topics_appear_in_prompt(agent, mock_provider, analysis):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"next_topics": [], "focus_areas": [], "message": ""}),
            tool_calls=[],
        )
    )
    all_topics = ["Science", "Math", "History"]
    await agent.run(analysis=analysis, current_topic="Science", all_topics=all_topics)
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    for topic in all_topics:
        assert topic in user_msg
