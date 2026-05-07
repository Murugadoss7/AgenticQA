import pytest
import os
from pathlib import Path
from models.session import SessionState, state_to_dict
from storage.json_store import save_session, load_session


@pytest.fixture(autouse=True)
def tmp_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr("storage.json_store.SESSIONS_DIR", tmp_path)
    return tmp_path


def _make_state(session_id="test-id") -> SessionState:
    return SessionState(
        session_id=session_id,
        topic="Science",
        question_count=3,
        status="selecting",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


async def test_save_and_load():
    state = _make_state()
    await save_session(state)
    loaded = await load_session("test-id")
    assert loaded is not None
    assert loaded.session_id == "test-id"
    assert loaded.topic == "Science"


async def test_load_missing_returns_none():
    result = await load_session("nonexistent")
    assert result is None


async def test_save_overwrites():
    state = _make_state()
    await save_session(state)
    state.status = "questioning"
    await save_session(state)
    loaded = await load_session("test-id")
    assert loaded.status == "questioning"
