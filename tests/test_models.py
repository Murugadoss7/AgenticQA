import dataclasses
from models.session import (
    SessionState, Question, Answer, Evaluation, Analysis, Recommendation,
    state_to_dict, state_from_dict,
)


def _sample_state() -> SessionState:
    return SessionState(
        session_id="abc123",
        topic="Science",
        question_count=3,
        questions=[Question(text="Q1?", difficulty="easy", expected_outline="A1")],
        current_index=0,
        answers=[],
        evaluations=[],
        analysis=None,
        recommendation=None,
        status="questioning",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


def test_session_state_roundtrip():
    state = _sample_state()
    d = state_to_dict(state)
    restored = state_from_dict(d)
    assert restored.session_id == state.session_id
    assert restored.topic == state.topic
    assert len(restored.questions) == 1
    assert restored.questions[0].text == "Q1?"


def test_roundtrip_with_evaluation():
    state = _sample_state()
    state.evaluations = [
        Evaluation(question_index=0, score=85, feedback="Good.", correct_points=["A"], missing_points=["B"])
    ]
    restored = state_from_dict(state_to_dict(state))
    assert restored.evaluations[0].score == 85
    assert restored.evaluations[0].correct_points == ["A"]


def test_roundtrip_with_analysis_and_recommendation():
    state = _sample_state()
    state.analysis = Analysis(strengths=["strong"], weaknesses=["weak"], overall_score=72.5)
    state.recommendation = Recommendation(
        next_topics=["Math"], focus_areas=["algebra"], message="Keep going!"
    )
    restored = state_from_dict(state_to_dict(state))
    assert restored.analysis.overall_score == 72.5
    assert restored.recommendation.next_topics == ["Math"]
