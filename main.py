import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from providers.factory import load_providers
from agents.question_generator import QuestionGeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.analyzer import AnalyzerAgent
from agents.advisor import AdvisorAgent
from agents.orchestrator import OrchestratorAgent
from api.sessions import router as sessions_router
from api.topics import router as topics_router

load_dotenv()

app = FastAPI(title="Q&A Platform")
app.include_router(sessions_router)
app.include_router(topics_router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
def startup() -> None:
    Path("sessions").mkdir(exist_ok=True)

    providers = load_providers("model_config.json")  # raises on invalid config

    with open("data/topics.json") as f:
        all_topics: list[str] = json.load(f)

    question_generator = QuestionGeneratorAgent(provider=providers["question_generator"])
    evaluator = EvaluatorAgent(provider=providers["evaluator"])
    analyzer = AnalyzerAgent(provider=providers["analyzer"])
    advisor = AdvisorAgent(provider=providers["advisor"])

    with open("model_config.json") as f:
        orch_cfg = json.load(f)["agents"]["orchestrator"]
    anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]

    orchestrator = OrchestratorAgent(
        api_key=anthropic_api_key,
        model=orch_cfg["model"],
        question_generator=question_generator,
        evaluator=evaluator,
        analyzer=analyzer,
        advisor=advisor,
        all_topics=all_topics,
    )

    app.state.orchestrator = orchestrator
