import json
from providers.base import ModelProvider
from models.session import Question, Evaluation, Analysis

_SYSTEM = """You are a learning coach reviewing a complete Q&A session. Identify recurring patterns,
not just per-question scores. Always respond with valid JSON only."""


class AnalyzerAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(self, questions: list[Question], evaluations: list[Evaluation]) -> Analysis:
        rows = []
        for q, e in zip(questions, evaluations):
            correct = ", ".join(e.correct_points) or "none"
            missing = ", ".join(e.missing_points) or "none"
            rows.append(f"Q: {q.text} | Score: {e.score}/100 | Correct: {correct} | Missing: {missing}")

        session_table = "\n".join(rows)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Review this Q&A session:\n\n{session_table}\n\n"
                    f"Identify recurring strengths and knowledge gaps. "
                    f'Return JSON: {{"strengths": ["..."], "weaknesses": ["..."], "overall_score": float}}'
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return Analysis(
            strengths=data["strengths"],
            weaknesses=data["weaknesses"],
            overall_score=float(data["overall_score"]),
        )
