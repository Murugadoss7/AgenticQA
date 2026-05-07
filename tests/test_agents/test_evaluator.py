# tests/test_agents/test_evaluator.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.evaluator import EvaluatorAgent
from models.session import Question, Evaluation


@pytest.fixture
def agent(mock_provider):
    return EvaluatorAgent(provider=mock_provider)


@pytest.fixture
def question():
    return Question(text="What is photosynthesis?", difficulty="medium", expected_outline="light, CO2, glucose, chlorophyll")


async def test_returns_evaluation(agent, mock_provider, question):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "score": 78,
                "feedback": "Good coverage of light reactions, missed glucose detail.",
                "correct_points": ["light absorption", "CO2 intake"],
                "missing_points": ["glucose synthesis"],
            }),
            tool_calls=[],
        )
    )
    ev = await agent.run(question=question, question_index=2, answer_text="Plants use sunlight and CO2.")
    assert isinstance(ev, Evaluation)
    assert ev.question_index == 2
    assert ev.score == 78
    assert "glucose" in ev.missing_points[0]


async def test_prompt_includes_expected_outline(agent, mock_provider, question):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"score": 50, "feedback": "ok", "correct_points": [], "missing_points": []}),
            tool_calls=[],
        )
    )
    await agent.run(question=question, question_index=0, answer_text="some answer")
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    assert "chlorophyll" in user_msg   # expected_outline appears in prompt
    assert "some answer" in user_msg
