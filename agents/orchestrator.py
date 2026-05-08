from __future__ import annotations
import json
from typing import Callable, Awaitable
from openai import AzureOpenAI
from models.session import SessionState, Question, Answer, Evaluation
from agents.question_generator import QuestionGeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.analyzer import AnalyzerAgent
from agents.advisor import AdvisorAgent

_SYSTEM = """You are the orchestrator of a Q&A learning platform. Inspect the session state and
decide which action to take next by calling exactly one function. Never generate questions or evaluations
yourself — always delegate to the appropriate function."""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_questions",
            "description": "Generate questions for the topic at the start of a session (status=selecting).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The session topic"},
                    "count": {"type": "integer", "description": "Number of questions to generate"},
                },
                "required": ["topic", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_answer",
            "description": "Evaluate the most recently submitted answer (status=questioning).",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_index": {"type": "integer", "description": "Index of the question being answered"},
                },
                "required": ["question_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_performance",
            "description": "Analyze overall session performance after all answers are submitted.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_next",
            "description": "Recommend next topics and focus areas based on analysis results.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

Emitter = Callable[[str, dict], Awaitable[None]]


class OrchestratorAgent:
    def __init__(
        self,
        api_key: str,
        endpoint: str,
        api_version: str,
        model: str,
        question_generator: QuestionGeneratorAgent,
        evaluator: EvaluatorAgent,
        analyzer: AnalyzerAgent,
        advisor: AdvisorAgent,
        all_topics: list[str],
    ) -> None:
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self.model = model
        self.question_generator = question_generator
        self.evaluator = evaluator
        self.analyzer = analyzer
        self.advisor = advisor
        self.all_topics = all_topics

    async def run(self, state: SessionState, emit: Emitter | None = None) -> SessionState:
        async def _emit(event: str, data: dict) -> None:
            if emit:
                await emit(event, data)

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": self._state_summary(state)},
        ]

        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            tool_calls = msg.tool_calls or []

            if not tool_calls:
                break

            messages.append(msg)

            for tc in tool_calls:
                tool_input = json.loads(tc.function.arguments)
                result = await self._dispatch(tc.function.name, tool_input, state, _emit)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                })

        return state

    async def _dispatch(self, name: str, tool_input: dict, state: SessionState, emit: Emitter) -> dict:
        if name == "generate_questions":
            await emit("agent_status", {"message": "Generating questions…"})
            questions = await self.question_generator.run(
                topic=tool_input["topic"], count=tool_input["count"]
            )
            state.questions = questions
            state.status = "questioning"
            first_q = questions[0]
            await emit("question_ready", {"question": {"text": first_q.text, "difficulty": first_q.difficulty}, "index": 0})
            return {"questions_generated": len(questions)}

        if name == "evaluate_answer":
            q_idx = tool_input["question_index"]
            await emit("agent_status", {"message": "Evaluating your answer…"})
            ev = await self.evaluator.run(
                question=state.questions[q_idx],
                question_index=q_idx,
                answer_text=state.answers[q_idx].text,
            )
            state.evaluations.append(ev)
            state.current_index = q_idx + 1
            await emit("evaluation_ready", {
                "evaluation": {
                    "score": ev.score, "feedback": ev.feedback,
                    "correct_points": ev.correct_points, "missing_points": ev.missing_points,
                }
            })
            next_idx = state.current_index
            if next_idx < len(state.questions):
                next_q = state.questions[next_idx]
                await emit("question_ready", {"question": {"text": next_q.text, "difficulty": next_q.difficulty}, "index": next_idx})
            return {"evaluation_score": ev.score, "questions_remaining": len(state.questions) - state.current_index}

        if name == "analyze_performance":
            await emit("agent_status", {"message": "Analyzing your performance…"})
            analysis = await self.analyzer.run(
                questions=state.questions, evaluations=state.evaluations
            )
            state.analysis = analysis
            state.status = "evaluating"
            await emit("analysis_ready", {
                "analysis": {
                    "strengths": analysis.strengths,
                    "weaknesses": analysis.weaknesses,
                    "overall_score": analysis.overall_score,
                }
            })
            return {"overall_score": analysis.overall_score}

        if name == "recommend_next":
            await emit("agent_status", {"message": "Preparing recommendations…"})
            rec = await self.advisor.run(
                analysis=state.analysis,
                current_topic=state.topic,
                all_topics=self.all_topics,
            )
            state.recommendation = rec
            state.status = "complete"
            await emit("complete", {
                "recommendation": {
                    "next_topics": rec.next_topics,
                    "focus_areas": rec.focus_areas,
                    "message": rec.message,
                }
            })
            return {"recommendations_ready": True}

        return {"error": f"Unknown tool: {name}"}

    def _state_summary(self, state: SessionState) -> str:
        lines = [
            f"Session ID: {state.session_id}",
            f"Topic: {state.topic}",
            f"Status: {state.status}",
            f"Question count: {state.question_count}",
            f"Questions generated: {len(state.questions)}",
            f"Answers submitted: {len(state.answers)}",
            f"Evaluations done: {len(state.evaluations)}",
            f"Analysis complete: {state.analysis is not None}",
            f"Recommendation ready: {state.recommendation is not None}",
        ]
        return "\n".join(lines)
