# tests/test_api/test_topics.py
import json
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.topics import router


@pytest.fixture
def client(tmp_path):
    app = FastAPI()
    app.include_router(router)
    # Point data file to temp location
    topics_file = tmp_path / "topics.json"
    topics_file.write_text(json.dumps(["Science", "Math"]))
    import api.topics as topics_module
    topics_module.TOPICS_FILE = str(topics_file)
    return TestClient(app)


def test_get_topics_returns_list(client):
    response = client.get("/api/topics")
    assert response.status_code == 200
    data = response.json()
    assert "topics" in data
    assert "Science" in data["topics"]


def test_get_topics_includes_custom_option(client):
    response = client.get("/api/topics")
    assert "custom" in response.json()
