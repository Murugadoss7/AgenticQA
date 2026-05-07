# tests/test_agents/test_question_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from providers.base import ProviderResponse
from agents.question_generator import QuestionGeneratorAgent
from models.session import Question


@pytest.fixture
def agent(mock_provider):
    return QuestionGeneratorAgent(provider=mock_provider)


async def test_returns_correct_number_of_questions(agent, mock_provider):
    questions_data = [
        {"text": "Q1?", "difficulty": "easy", "expected_outline": "outline1"},
        {"text": "Q2?", "difficulty": "medium", "expected_outline": "outline2"},
        {"text": "Q3?", "difficulty": "hard", "expected_outline": "outline3"},
    ]
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": questions_data}), tool_calls=[]
        )
    )
    questions = await agent.run(topic="Science", count=3)
    assert len(questions) == 3
    assert all(isinstance(q, Question) for q in questions)


async def test_question_fields_mapped_correctly(agent, mock_provider):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": [
                {"text": "What is DNA?", "difficulty": "easy", "expected_outline": "double helix, genetics"}
            ]}),
            tool_calls=[],
        )
    )
    questions = await agent.run(topic="Biology", count=1)
    q = questions[0]
    assert q.text == "What is DNA?"
    assert q.difficulty == "easy"
    assert q.expected_outline == "double helix, genetics"


async def test_prompt_includes_topic_and_count(agent, mock_provider):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": [
                {"text": "Q?", "difficulty": "easy", "expected_outline": "o"}
            ]}),
            tool_calls=[],
        )
    )
    await agent.run(topic="History", count=5)
    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["messages"][0]["content"]
    assert "History" in user_message
    assert "5" in user_message
