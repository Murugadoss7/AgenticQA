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
        model="claude-opus-4-7",
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


async def test_dispatches_to_question_generator():
    agent = _make_agent()
    state = _selecting_state()

    generated_questions = [
        Question(text="Q1?", difficulty="easy", expected_outline="o1"),
    ]
    agent.question_generator.run = AsyncMock(return_value=generated_questions)

    # Mock Claude to return generate_questions tool call, then stop
    tool_block = MagicMock(type="tool_use", id="tc1", input={"topic": "Science", "count": 3})
    tool_block.name = "generate_questions"
    mock_response_tool = MagicMock()
    mock_response_tool.content = [tool_block]
    mock_response_stop = MagicMock()
    mock_response_stop.content = [MagicMock(type="text", text="Done.")]

    with patch.object(agent.client.messages, "create", side_effect=[mock_response_tool, mock_response_stop]):
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

    tool_block2 = MagicMock(type="tool_use", id="tc2", input={"question_index": 0})
    tool_block2.name = "evaluate_answer"
    mock_response_tool = MagicMock()
    mock_response_tool.content = [tool_block2]
    mock_response_stop = MagicMock()
    mock_response_stop.content = [MagicMock(type="text", text="Done.")]

    with patch.object(agent.client.messages, "create", side_effect=[mock_response_tool, mock_response_stop]):
        updated = await agent.run(state)

    agent.evaluator.run.assert_called_once()
    assert len(updated.evaluations) == 1
    assert updated.evaluations[0].score == 80
