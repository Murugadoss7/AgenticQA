import json
from providers.base import ModelProvider
from models.session import Question, Evaluation

_SYSTEM = """You are an objective academic evaluator. Score answers 0–100 based on accuracy and completeness.
Ground every judgment in the expected outline. Be specific. Always respond with valid JSON only."""


class EvaluatorAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(self, question: Question, question_index: int, answer_text: str) -> Evaluation:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: {question.text}\n\n"
                    f"Expected outline: {question.expected_outline}\n\n"
                    f"User's answer: {answer_text}\n\n"
                    f"Return JSON: {{\"score\": 0-100, \"feedback\": \"2-3 sentences\", "
                    f"\"correct_points\": [\"...\"], \"missing_points\": [\"...\"]}}"
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return Evaluation(
            question_index=question_index,
            score=data["score"],
            feedback=data["feedback"],
            correct_points=data["correct_points"],
            missing_points=data["missing_points"],
        )
