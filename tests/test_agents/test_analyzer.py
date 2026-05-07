# tests/test_agents/test_analyzer.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.analyzer import AnalyzerAgent
from models.session import Question, Evaluation, Analysis


@pytest.fixture
def agent(mock_provider):
    return AnalyzerAgent(provider=mock_provider)


@pytest.fixture
def session_data():
    questions = [
        Question(text="Q1?", difficulty="easy", expected_outline="o1"),
        Question(text="Q2?", difficulty="medium", expected_outline="o2"),
    ]
    evaluations = [
        Evaluation(question_index=0, score=90, feedback="Great.", correct_points=["A"], missing_points=[]),
        Evaluation(question_index=1, score=55, feedback="Weak.", correct_points=[], missing_points=["B", "C"]),
    ]
    return questions, evaluations


async def test_returns_analysis(agent, mock_provider, session_data):
    questions, evaluations = session_data
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "strengths": ["Core concept recall"],
                "weaknesses": ["Mechanism detail"],
                "overall_score": 72.5,
            }),
            tool_calls=[],
        )
    )
    analysis = await agent.run(questions=questions, evaluations=evaluations)
    assert isinstance(analysis, Analysis)
    assert analysis.overall_score == 72.5
    assert "Core concept recall" in analysis.strengths


async def test_prompt_contains_all_questions(agent, mock_provider, session_data):
    questions, evaluations = session_data
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"strengths": [], "weaknesses": [], "overall_score": 0.0}),
            tool_calls=[],
        )
    )
    await agent.run(questions=questions, evaluations=evaluations)
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    assert "Q1?" in user_msg
    assert "Q2?" in user_msg
    assert "90" in user_msg
