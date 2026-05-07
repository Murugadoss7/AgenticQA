import json
from providers.base import ModelProvider
from models.session import Question

_SYSTEM = """You are an expert educational question writer. Generate questions that test genuine understanding.
Always respond with valid JSON only — no other text."""


class QuestionGeneratorAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(self, topic: str, count: int) -> list[Question]:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Generate exactly {count} questions about '{topic}'. "
                    f"Order them easy → medium → hard. "
                    f'Return JSON: {{"questions": [{{"text": "...", "difficulty": "easy|medium|hard", "expected_outline": "key points for grading"}}]}}'
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return [
            Question(
                text=q["text"],
                difficulty=q["difficulty"],
                expected_outline=q["expected_outline"],
            )
            for q in data["questions"]
        ]
