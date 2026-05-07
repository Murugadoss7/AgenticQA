# tests/test_api/test_sessions.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from models.session import SessionState, Question
from api.sessions import router, _session_queues


def _make_app():
    app = FastAPI()
    app.include_router(router)
    # Inject mock orchestrator
    from agents.orchestrator import OrchestratorAgent
    mock_orch = MagicMock(spec=OrchestratorAgent)
    mock_orch.run = AsyncMock(return_value=MagicMock(status="questioning"))
    app.state.orchestrator = mock_orch
    return app


def test_create_session_returns_session_id():
    with patch("api.sessions.save_session", new_callable=AsyncMock):
        app = _make_app()
        client = TestClient(app)
        response = client.post("/api/sessions", json={"topic": "Science", "question_count": 3})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["topic"] == "Science"


def test_get_session_not_found():
    with patch("api.sessions.load_session", new_callable=AsyncMock, return_value=None):
        app = _make_app()
        client = TestClient(app)
        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404
