import json
from fastapi import APIRouter

router = APIRouter()
TOPICS_FILE = "data/topics.json"


@router.get("/api/topics")
def get_topics() -> dict:
    with open(TOPICS_FILE) as f:
        topics: list[str] = json.load(f)
    return {"topics": topics, "custom": True}
