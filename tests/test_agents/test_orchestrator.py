# tests/test_agents/test_orchestrator.py
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.orchestrator import OrchestratorAgent
from agents.question_generator import QuestionGeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.analyzer import AnalyzerAgent
from agents.advisor import AdvisorAgent
from models.session import SessionState, Question, Answer, Evaluation, Analysis, Recommendation


def _make_agent():
    return OrchestratorAgent(
        api_key="test-key",
        endpoint="https://test.openai.azure.com",
        api_version="2024-02-01",
        model="gpt-4o",
        question_generator=MagicMock(spec=QuestionGeneratorAgent),
        evaluator=MagicMock(spec=EvaluatorAgent),
        analyzer=MagicMock(spec=AnalyzerAgent),
        advisor=MagicMock(spec=AdvisorAgent),
        all_topics=["Science", "Math"],
    )


def _selecting_state():
    return SessionState(
        session_id="s1", topic="Science", question_count=3,
        status="selecting", created_at="t", updated_at="t",
    )


def _make_tool_response(name: str, args: dict, tool_id: str = "tc1") -> MagicMock:
    """Build a mock OpenAI chat completion response with one function call."""
    tc = MagicMock()
    tc.id = tool_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.tool_calls = [tc]

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_stop_response() -> MagicMock:
    """Build a mock OpenAI chat completion response with no tool calls (stop)."""
    msg = MagicMock()
    msg.tool_calls = None

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


async def test_dispatches_to_question_generator():
    agent = _make_agent()
    state = _selecting_state()

    generated_questions = [
        Question(text="Q1?", difficulty="easy", expected_outline="o1"),
    ]
    agent.question_generator.run = AsyncMock(return_value=generated_questions)

    with patch.object(agent.client.chat.completions, "create", side_effect=[
        _make_tool_response("generate_questions", {"topic": "Science", "count": 3}),
        _make_stop_response(),
    ]):
        updated = await agent.run(state)

    agent.question_generator.run.assert_called_once_with(topic="Science", count=3)
    assert updated.questions == generated_questions
    assert updated.status == "questioning"


async def test_dispatches_to_evaluator():
    agent = _make_agent()
    state = _selecting_state()
    state.status = "questioning"
    state.questions = [Question(text="Q1?", difficulty="easy", expected_outline="o1")]
    state.answers = [Answer(question_index=0, text="my answer", submitted_at="t")]

    ev = Evaluation(question_index=0, score=80, feedback="Good.", correct_points=["A"], missing_points=[])
    agent.evaluator.run = AsyncMock(return_value=ev)

    with patch.object(agent.client.chat.completions, "create", side_effect=[
        _make_tool_response("evaluate_answer", {"question_index": 0}, tool_id="tc2"),
        _make_stop_response(),
    ]):
        updated = await agent.run(state)

    agent.evaluator.run.assert_called_once()
    assert len(updated.evaluations) == 1
    assert updated.evaluations[0].score == 80
