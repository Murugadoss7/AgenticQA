import json
from pathlib import Path
from models.session import SessionState, state_to_dict, state_from_dict

SESSIONS_DIR = Path("sessions")


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)


async def save_session(state: SessionState) -> None:
    _ensure_dir()
    path = SESSIONS_DIR / f"{state.session_id}.json"
    path.write_text(json.dumps(state_to_dict(state), indent=2))


async def load_session(session_id: str) -> SessionState | None:
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    return state_from_dict(json.loads(path.read_text()))
