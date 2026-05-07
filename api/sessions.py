from __future__ import annotations
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.session import SessionState, Answer, state_to_dict
from storage.json_store import save_session, load_session

router = APIRouter()

# Per-session event queues for SSE
_session_queues: dict[str, asyncio.Queue] = {}


class CreateSessionRequest(BaseModel):
    topic: str
    question_count: int


class AnswerRequest(BaseModel):
    answer: str


@router.post("/api/sessions")
async def create_session(
    body: CreateSessionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    state = SessionState(
        session_id=session_id,
        topic=body.topic,
        question_count=body.question_count,
        status="selecting",
        created_at=now,
        updated_at=now,
    )
    _session_queues[session_id] = asyncio.Queue()
    await save_session(state)
    background_tasks.add_task(_run_orchestrator, session_id, state, request.app.state.orchestrator)
    return {"session_id": session_id, "topic": body.topic, "status": "selecting"}


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    state = await load_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state_to_dict(state)


@router.post("/api/sessions/{session_id}/answer")
async def submit_answer(
    session_id: str,
    body: AnswerRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict:
    state = await load_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")

    now = datetime.now(timezone.utc).isoformat()
    answer = Answer(
        question_index=state.current_index,
        text=body.answer,
        submitted_at=now,
    )
    state.answers.append(answer)
    state.updated_at = now
    await save_session(state)

    if session_id not in _session_queues:
        _session_queues[session_id] = asyncio.Queue()

    background_tasks.add_task(_run_orchestrator, session_id, state, request.app.state.orchestrator)
    return {"status": "processing"}


@router.get("/api/sessions/{session_id}/stream")
async def stream_session(session_id: str) -> StreamingResponse:
    if session_id not in _session_queues:
        _session_queues[session_id] = asyncio.Queue()

    queue = _session_queues[session_id]

    async def event_generator():
        try:
            while True:
                try:
                    event_type, data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
                    if event_type in ("complete", "error"):
                        break
                except asyncio.TimeoutError:
                    yield "event: ping\ndata: {}\n\n"
        finally:
            _session_queues.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _run_orchestrator(session_id: str, state: SessionState, orchestrator: Any) -> None:
    async def emit(event_type: str, data: dict) -> None:
        q = _session_queues.get(session_id)
        if q:
            await q.put((event_type, data))

    try:
        updated = await orchestrator.run(state, emit=emit)
        updated.updated_at = datetime.now(timezone.utc).isoformat()
        await save_session(updated)
    except Exception as exc:
        await emit("error", {"message": str(exc)})
