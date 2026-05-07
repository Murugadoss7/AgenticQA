from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    text: str
    difficulty: str          # "easy" | "medium" | "hard"
    expected_outline: str


@dataclass
class Answer:
    question_index: int
    text: str
    submitted_at: str


@dataclass
class Evaluation:
    question_index: int
    score: int               # 0–100
    feedback: str
    correct_points: list[str]
    missing_points: list[str]


@dataclass
class Analysis:
    strengths: list[str]
    weaknesses: list[str]
    overall_score: float


@dataclass
class Recommendation:
    next_topics: list[str]
    focus_areas: list[str]
    message: str


@dataclass
class SessionState:
    session_id: str
    topic: str
    question_count: int
    questions: list[Question] = field(default_factory=list)
    current_index: int = 0
    answers: list[Answer] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    analysis: Optional[Analysis] = None
    recommendation: Optional[Recommendation] = None
    status: str = "selecting"   # "selecting"|"questioning"|"evaluating"|"complete"
    created_at: str = ""
    updated_at: str = ""


def state_to_dict(state: SessionState) -> dict:
    return dataclasses.asdict(state)


def state_from_dict(data: dict) -> SessionState:
    questions = [Question(**q) for q in data.get("questions", [])]
    answers = [Answer(**a) for a in data.get("answers", [])]
    evaluations = [Evaluation(**e) for e in data.get("evaluations", [])]
    analysis = Analysis(**data["analysis"]) if data.get("analysis") else None
    recommendation = (
        Recommendation(**data["recommendation"]) if data.get("recommendation") else None
    )
    return SessionState(
        session_id=data["session_id"],
        topic=data["topic"],
        question_count=data["question_count"],
        questions=questions,
        current_index=data["current_index"],
        answers=answers,
        evaluations=evaluations,
        analysis=analysis,
        recommendation=recommendation,
        status=data["status"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )
