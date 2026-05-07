import json
from providers.base import ModelProvider
from models.session import Analysis, Recommendation

_SYSTEM = """You are a personalized study advisor. Recommend next steps based on demonstrated strengths and weaknesses.
Only recommend topics from the provided list. Always respond with valid JSON only."""


class AdvisorAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(
        self, analysis: Analysis, current_topic: str, all_topics: list[str]
    ) -> Recommendation:
        topics_list = ", ".join(all_topics)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Current topic: {current_topic}\n"
                    f"Overall score: {analysis.overall_score}\n"
                    f"Strengths: {', '.join(analysis.strengths)}\n"
                    f"Weaknesses: {', '.join(analysis.weaknesses)}\n\n"
                    f"Available topics: {topics_list}\n\n"
                    f"Recommend 1–2 next topics (only from the available list) and 2–3 specific focus areas. "
                    f'Return JSON: {{"next_topics": ["..."], "focus_areas": ["..."], "message": "narrative"}}'
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return Recommendation(
            next_topics=data["next_topics"],
            focus_areas=data["focus_areas"],
            message=data["message"],
        )
