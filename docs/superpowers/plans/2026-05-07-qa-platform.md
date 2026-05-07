# Q&A Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based Q&A platform where a 5-agent AI pipeline generates questions, evaluates answers, analyzes performance, and recommends next steps — using only the Anthropic Python SDK, FastAPI, and Vanilla JS.

**Architecture:** The OrchestratorAgent uses `anthropic.Anthropic()` directly with tool_use + adaptive thinking to decide which specialist to invoke. The four specialist agents (QuestionGenerator, Evaluator, Analyzer, Advisor) each use the ModelProvider abstraction, letting any provider (Anthropic, Azure OpenAI, Ollama, OpenRouter) be assigned per-agent via `model_config.json`. FastAPI exposes REST endpoints for session actions and an SSE stream for live agent status updates to the browser.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Anthropic SDK (`anthropic`), OpenAI SDK (`openai` — for Azure and OpenRouter), HTTPX (for Ollama), python-dotenv, pytest + pytest-asyncio

---

## File Map

```
ClaudeSDK/
├── main.py                        # FastAPI app, agent wiring, startup validation
├── model_config.json              # Provider + agent-to-model mapping
├── .env.example                   # API key placeholders
├── requirements.txt
├── pytest.ini
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py            # tool-use agentic loop — Anthropic SDK direct
│   ├── question_generator.py
│   ├── evaluator.py
│   ├── analyzer.py
│   └── advisor.py
├── providers/
│   ├── __init__.py
│   ├── base.py                    # ModelProvider ABC, ProviderResponse, ToolCall
│   ├── anthropic_provider.py      # specialist calls via Anthropic SDK
│   ├── azure_openai_provider.py
│   ├── ollama_provider.py
│   ├── openrouter_provider.py
│   └── factory.py                 # reads model_config.json, validates, instantiates
├── models/
│   ├── __init__.py
│   └── session.py                 # all dataclasses + to_dict/from_dict
├── api/
│   ├── __init__.py
│   ├── sessions.py                # session CRUD + background agent tasks + SSE stream
│   └── topics.py
├── storage/
│   ├── __init__.py
│   └── json_store.py              # async read/write sessions/*.json
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── topics.json
├── sessions/                      # auto-created at runtime
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    ├── test_json_store.py
    ├── test_factory.py
    ├── test_providers/
    │   ├── __init__.py
    │   ├── test_anthropic_provider.py
    │   ├── test_azure_openai_provider.py
    │   ├── test_ollama_provider.py
    │   └── test_openrouter_provider.py
    ├── test_agents/
    │   ├── __init__.py
    │   ├── test_question_generator.py
    │   ├── test_evaluator.py
    │   ├── test_analyzer.py
    │   ├── test_advisor.py
    │   └── test_orchestrator.py
    └── test_api/
        ├── __init__.py
        ├── test_topics.py
        └── test_sessions.py
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.env.example`
- Create: `model_config.json`
- Create: `data/topics.json`
- Create: `tests/conftest.py`
- Create all `__init__.py` files

- [ ] **Step 1: Create requirements.txt**

```text
anthropic>=0.50.0
openai>=1.0.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-dotenv>=1.0.0
httpx>=0.27.0
aiofiles>=23.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 3: Create .env.example**

```text
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
```

- [ ] **Step 4: Create model_config.json**

```json
{
  "providers": {
    "anthropic":    { "api_key": "${ANTHROPIC_API_KEY}" },
    "azure_openai": { "api_key": "${AZURE_OPENAI_API_KEY}", "endpoint": "${AZURE_OPENAI_ENDPOINT}", "api_version": "2024-02-01" },
    "ollama":       { "base_url": "http://localhost:11434" },
    "openrouter":   { "api_key": "${OPENROUTER_API_KEY}" }
  },
  "agents": {
    "orchestrator":       { "provider": "anthropic",    "model": "claude-opus-4-7" },
    "question_generator": { "provider": "anthropic",    "model": "claude-sonnet-4-6" },
    "evaluator":          { "provider": "anthropic",    "model": "claude-haiku-4-5" },
    "analyzer":           { "provider": "anthropic",    "model": "claude-haiku-4-5" },
    "advisor":            { "provider": "anthropic",    "model": "claude-haiku-4-5" }
  }
}
```

> Note: All agents default to Anthropic for easy first-run. Edit to assign other providers once those credentials are set up.

- [ ] **Step 5: Create data/topics.json**

```json
["Science", "History", "Math", "Technology", "Literature", "Geography", "Philosophy", "Programming"]
```

- [ ] **Step 6: Create tests/conftest.py**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from providers.base import ProviderResponse, ToolCall


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.complete = AsyncMock(
        return_value=ProviderResponse(content='{"ok": true}', tool_calls=[])
    )
    return provider


@pytest.fixture
def mock_tool_provider():
    """Provider that returns a single tool call."""
    provider = MagicMock()

    def _make_tool_response(name, input_data, tool_id="tc_001"):
        return ProviderResponse(
            content="",
            tool_calls=[ToolCall(id=tool_id, name=name, input=input_data)],
        )

    provider.make_tool_response = _make_tool_response
    provider.complete = AsyncMock(return_value=ProviderResponse(content="", tool_calls=[]))
    return provider
```

- [ ] **Step 7: Create all __init__.py files**

Create empty `__init__.py` in: `agents/`, `providers/`, `models/`, `api/`, `storage/`, `tests/`, `tests/test_providers/`, `tests/test_agents/`, `tests/test_api/`

- [ ] **Step 8: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "feat: project scaffolding — requirements, config, test infra"
```

---

## Task 2: Data Models

**Files:**
- Create: `models/session.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_models.py
import dataclasses
from models.session import (
    SessionState, Question, Answer, Evaluation, Analysis, Recommendation,
    state_to_dict, state_from_dict,
)


def _sample_state() -> SessionState:
    return SessionState(
        session_id="abc123",
        topic="Science",
        question_count=3,
        questions=[Question(text="Q1?", difficulty="easy", expected_outline="A1")],
        current_index=0,
        answers=[],
        evaluations=[],
        analysis=None,
        recommendation=None,
        status="questioning",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )


def test_session_state_roundtrip():
    state = _sample_state()
    d = state_to_dict(state)
    restored = state_from_dict(d)
    assert restored.session_id == state.session_id
    assert restored.topic == state.topic
    assert len(restored.questions) == 1
    assert restored.questions[0].text == "Q1?"


def test_roundtrip_with_evaluation():
    state = _sample_state()
    state.evaluations = [
        Evaluation(question_index=0, score=85, feedback="Good.", correct_points=["A"], missing_points=["B"])
    ]
    restored = state_from_dict(state_to_dict(state))
    assert restored.evaluations[0].score == 85
    assert restored.evaluations[0].correct_points == ["A"]


def test_roundtrip_with_analysis_and_recommendation():
    state = _sample_state()
    state.analysis = Analysis(strengths=["strong"], weaknesses=["weak"], overall_score=72.5)
    state.recommendation = Recommendation(
        next_topics=["Math"], focus_areas=["algebra"], message="Keep going!"
    )
    restored = state_from_dict(state_to_dict(state))
    assert restored.analysis.overall_score == 72.5
    assert restored.recommendation.next_topics == ["Math"]
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'models.session'`

- [ ] **Step 3: Implement models/session.py**

```python
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Question:
    text: str
    difficulty: str          # "easy" | "medium" | "hard"
    expected_outline: str


@dataclass
class Answer:
    question_index: int
    text: str
    submitted_at: str


@dataclass
class Evaluation:
    question_index: int
    score: int               # 0–100
    feedback: str
    correct_points: list[str]
    missing_points: list[str]


@dataclass
class Analysis:
    strengths: list[str]
    weaknesses: list[str]
    overall_score: float


@dataclass
class Recommendation:
    next_topics: list[str]
    focus_areas: list[str]
    message: str


@dataclass
class SessionState:
    session_id: str
    topic: str
    question_count: int
    questions: list[Question] = field(default_factory=list)
    current_index: int = 0
    answers: list[Answer] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    analysis: Optional[Analysis] = None
    recommendation: Optional[Recommendation] = None
    status: str = "selecting"   # "selecting"|"questioning"|"evaluating"|"complete"
    created_at: str = ""
    updated_at: str = ""


def state_to_dict(state: SessionState) -> dict:
    return dataclasses.asdict(state)


def state_from_dict(data: dict) -> SessionState:
    questions = [Question(**q) for q in data.get("questions", [])]
    answers = [Answer(**a) for a in data.get("answers", [])]
    evaluations = [Evaluation(**e) for e in data.get("evaluations", [])]
    analysis = Analysis(**data["analysis"]) if data.get("analysis") else None
    recommendation = (
        Recommendation(**data["recommendation"]) if data.get("recommendation") else None
    )
    return SessionState(
        session_id=data["session_id"],
        topic=data["topic"],
        question_count=data["question_count"],
        questions=questions,
        current_index=data["current_index"],
        answers=answers,
        evaluations=evaluations,
        analysis=analysis,
        recommendation=recommendation,
        status=data["status"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_models.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add models/ tests/test_models.py
git commit -m "feat: data models — SessionState and all supporting dataclasses"
```

---

## Task 3: Storage Layer

**Files:**
- Create: `storage/json_store.py`
- Create: `tests/test_json_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_json_store.py
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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_json_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'storage.json_store'`

- [ ] **Step 3: Implement storage/json_store.py**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_json_store.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add storage/ tests/test_json_store.py
git commit -m "feat: storage layer — async JSON session persistence"
```

---

## Task 4: Provider Base

**Files:**
- Create: `providers/base.py`

- [ ] **Step 1: Implement providers/base.py** (no test needed — pure interface)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ProviderResponse:
    content: str                        # text response (JSON string for specialists)
    tool_calls: list[ToolCall] = field(default_factory=list)


class ModelProvider(ABC):
    """Unified interface across Anthropic, Azure OpenAI, Ollama, and OpenRouter.

    All agents (except Orchestrator) call this — they never touch provider SDKs directly.
    Tools parameter uses Anthropic tool schema format; each provider translates internally.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        ...
```

- [ ] **Step 2: Commit**

```bash
git add providers/base.py
git commit -m "feat: ModelProvider ABC — unified provider interface"
```

---

## Task 5: AnthropicProvider (for Specialists)

**Files:**
- Create: `providers/anthropic_provider.py`
- Create: `tests/test_providers/test_anthropic_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers/test_anthropic_provider.py
import pytest
from unittest.mock import patch, MagicMock
from providers.anthropic_provider import AnthropicProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key", model="claude-haiku-4-5")


async def test_complete_returns_text(provider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text='{"score": 85}')]

    with patch.object(provider.client.messages, "create", return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "evaluate this"}],
            system="You are an evaluator.",
        )

    assert isinstance(result, ProviderResponse)
    assert result.content == '{"score": 85}'
    assert result.tool_calls == []


async def test_complete_with_cache_control(provider):
    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="ok")]

    with patch.object(provider.client.messages, "create", return_value=mock_response) as mock_create:
        await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            system="stable system prompt",
        )
        call_kwargs = mock_create.call_args.kwargs
        # System should be a list with cache_control block
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_providers/test_anthropic_provider.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement providers/anthropic_provider.py**

```python
import anthropic
from providers.base import ModelProvider, ProviderResponse, ToolCall


class AnthropicProvider(ModelProvider):
    """Anthropic provider for specialist agents (no adaptive thinking — use direct SDK for Orchestrator)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        system_block = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]

        create_kwargs: dict = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "system": system_block,
            "messages": messages,
        }
        if tools:
            create_kwargs["tools"] = tools

        response = self.client.messages.create(**create_kwargs)

        text_content = ""
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return ProviderResponse(content=text_content, tool_calls=tool_calls)
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_providers/test_anthropic_provider.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add providers/anthropic_provider.py tests/test_providers/test_anthropic_provider.py
git commit -m "feat: AnthropicProvider — specialist calls with prompt caching"
```

---

## Task 6: AzureOpenAIProvider

**Files:**
- Create: `providers/azure_openai_provider.py`
- Create: `tests/test_providers/test_azure_openai_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers/test_azure_openai_provider.py
import pytest
from unittest.mock import patch, MagicMock
from providers.azure_openai_provider import AzureOpenAIProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return AzureOpenAIProvider(
        api_key="test-key",
        endpoint="https://test.openai.azure.com/",
        api_version="2024-02-01",
        model="gpt-4o-mini",
    )


async def test_complete_returns_text(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"score": 70}'
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "evaluate"}],
            system="You are evaluator.",
        )

    assert result.content == '{"score": 70}'
    assert result.tool_calls == []


async def test_system_prepended_as_message(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_choice.message.tool_calls = None
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", return_value=mock_response) as mock_create:
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="Be helpful.",
        )
        sent_messages = mock_create.call_args.kwargs["messages"]
        assert sent_messages[0] == {"role": "system", "content": "Be helpful."}
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_providers/test_azure_openai_provider.py -v
```

- [ ] **Step 3: Implement providers/azure_openai_provider.py**

```python
import json
from openai import AzureOpenAI
from providers.base import ModelProvider, ProviderResponse, ToolCall


class AzureOpenAIProvider(ModelProvider):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str) -> None:
        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        openai_messages = [{"role": "system", "content": system}] + messages

        create_kwargs: dict = {
            "model": self.model,
            "messages": openai_messages,
            "response_format": {"type": "json_object"},
        }
        if tools:
            create_kwargs["tools"] = self._convert_tools(tools)
            create_kwargs.pop("response_format")  # can't combine with tool_choice

        response = self.client.chat.completions.create(**create_kwargs)
        message = response.choices[0].message

        content = message.content or ""
        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
                )

        return ProviderResponse(content=content, tool_calls=tool_calls)

    @staticmethod
    def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in anthropic_tools
        ]
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_providers/test_azure_openai_provider.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add providers/azure_openai_provider.py tests/test_providers/test_azure_openai_provider.py
git commit -m "feat: AzureOpenAIProvider with tool conversion"
```

---

## Task 7: OllamaProvider

**Files:**
- Create: `providers/ollama_provider.py`
- Create: `tests/test_providers/test_ollama_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers/test_ollama_provider.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from providers.ollama_provider import OllamaProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return OllamaProvider(base_url="http://localhost:11434", model="llama3.1:8b")


async def test_complete_returns_text(provider):
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": '{"strengths": ["good"]}'}}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("providers.ollama_provider.httpx.AsyncClient", return_value=mock_client):
        result = await provider.complete(
            messages=[{"role": "user", "content": "analyze"}],
            system="You are analyzer.",
        )

    assert result.content == '{"strengths": ["good"]}'
    assert result.tool_calls == []


async def test_tools_raises(provider):
    with pytest.raises(ValueError, match="tool"):
        await provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=[{"name": "some_tool", "input_schema": {}}],
        )
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_providers/test_ollama_provider.py -v
```

- [ ] **Step 3: Implement providers/ollama_provider.py**

```python
import httpx
from providers.base import ModelProvider, ProviderResponse


class OllamaProvider(ModelProvider):
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        if tools:
            raise ValueError(
                "OllamaProvider does not support tool_use — assign Orchestrator to anthropic or azure_openai"
            )

        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": False,
            "format": "json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

        return ProviderResponse(content=data["message"]["content"], tool_calls=[])
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_providers/test_ollama_provider.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add providers/ollama_provider.py tests/test_providers/test_ollama_provider.py
git commit -m "feat: OllamaProvider — local model support via Ollama API"
```

---

## Task 8: OpenRouterProvider

**Files:**
- Create: `providers/openrouter_provider.py`
- Create: `tests/test_providers/test_openrouter_provider.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers/test_openrouter_provider.py
import pytest
from unittest.mock import patch, MagicMock
from providers.openrouter_provider import OpenRouterProvider
from providers.base import ProviderResponse


@pytest.fixture
def provider():
    return OpenRouterProvider(api_key="test-key", model="meta-llama/llama-3.1-70b-instruct")


async def test_complete_returns_text(provider):
    mock_choice = MagicMock()
    mock_choice.message.content = '{"next_topics": ["Math"]}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch.object(provider.client.chat.completions, "create", return_value=mock_response):
        result = await provider.complete(
            messages=[{"role": "user", "content": "advise"}],
            system="You are advisor.",
        )

    assert '{"next_topics"' in result.content
    assert result.tool_calls == []
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_providers/test_openrouter_provider.py -v
```

- [ ] **Step 3: Implement providers/openrouter_provider.py**

```python
from openai import OpenAI
from providers.base import ModelProvider, ProviderResponse


class OpenRouterProvider(ModelProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model

    async def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> ProviderResponse:
        openai_messages = [{"role": "system", "content": system}] + messages
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        return ProviderResponse(content=content, tool_calls=[])
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_providers/test_openrouter_provider.py -v
```
Expected: 1 test PASSED

- [ ] **Step 5: Commit**

```bash
git add providers/openrouter_provider.py tests/test_providers/test_openrouter_provider.py
git commit -m "feat: OpenRouterProvider — cost-efficient routing via OpenRouter API"
```

---

## Task 9: Provider Factory + Startup Validation

**Files:**
- Create: `providers/factory.py`
- Create: `tests/test_factory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_factory.py
import json
import pytest
from pathlib import Path
from providers.factory import load_providers, TOOL_USE_PROVIDERS
from providers.anthropic_provider import AnthropicProvider
from providers.ollama_provider import OllamaProvider


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "providers": {
            "anthropic": {"api_key": "sk-test"},
            "ollama": {"base_url": "http://localhost:11434"},
        },
        "agents": {
            "orchestrator":       {"provider": "anthropic", "model": "claude-opus-4-7"},
            "question_generator": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
            "evaluator":          {"provider": "anthropic", "model": "claude-haiku-4-5"},
            "analyzer":           {"provider": "ollama",    "model": "llama3.1:8b"},
            "advisor":            {"provider": "ollama",    "model": "llama3.1:8b"},
        },
    }
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_load_providers_returns_all_agents(config_file):
    providers = load_providers(config_file)
    assert set(providers.keys()) == {"orchestrator", "question_generator", "evaluator", "analyzer", "advisor"}


def test_orchestrator_is_anthropic_provider(config_file):
    providers = load_providers(config_file)
    assert isinstance(providers["orchestrator"], AnthropicProvider)


def test_analyzer_is_ollama_provider(config_file):
    providers = load_providers(config_file)
    assert isinstance(providers["analyzer"], OllamaProvider)


def test_invalid_orchestrator_provider_raises(tmp_path):
    cfg = {
        "providers": {"ollama": {"base_url": "http://localhost:11434"}},
        "agents": {
            "orchestrator":       {"provider": "ollama", "model": "llama3.1:8b"},
            "question_generator": {"provider": "ollama", "model": "llama3.1:8b"},
            "evaluator":          {"provider": "ollama", "model": "llama3.1:8b"},
            "analyzer":           {"provider": "ollama", "model": "llama3.1:8b"},
            "advisor":            {"provider": "ollama", "model": "llama3.1:8b"},
        },
    }
    p = tmp_path / "model_config.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="tool_use"):
        load_providers(str(p))
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_factory.py -v
```

- [ ] **Step 3: Implement providers/factory.py**

```python
import json
import os
from dotenv import load_dotenv
from providers.base import ModelProvider
from providers.anthropic_provider import AnthropicProvider
from providers.azure_openai_provider import AzureOpenAIProvider
from providers.ollama_provider import OllamaProvider
from providers.openrouter_provider import OpenRouterProvider

TOOL_USE_PROVIDERS = {"anthropic", "azure_openai"}


def _resolve(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        resolved = os.environ.get(env_var, "")
        if not resolved:
            raise ValueError(f"Required environment variable '{env_var}' is not set")
        return resolved
    return value


def load_providers(config_path: str = "model_config.json") -> dict[str, ModelProvider]:
    load_dotenv()

    with open(config_path) as f:
        config = json.load(f)

    orch_provider = config["agents"]["orchestrator"]["provider"]
    if orch_provider not in TOOL_USE_PROVIDERS:
        raise ValueError(
            f"Orchestrator is configured to use '{orch_provider}' which does not support tool_use. "
            f"Allowed: {sorted(TOOL_USE_PROVIDERS)}"
        )

    providers: dict[str, ModelProvider] = {}
    for agent_name, agent_cfg in config["agents"].items():
        provider_name = agent_cfg["provider"]
        model = agent_cfg["model"]
        pcfg = config["providers"][provider_name]

        if provider_name == "anthropic":
            providers[agent_name] = AnthropicProvider(
                api_key=_resolve(pcfg["api_key"]), model=model
            )
        elif provider_name == "azure_openai":
            providers[agent_name] = AzureOpenAIProvider(
                api_key=_resolve(pcfg["api_key"]),
                endpoint=_resolve(pcfg["endpoint"]),
                api_version=pcfg["api_version"],
                model=model,
            )
        elif provider_name == "ollama":
            providers[agent_name] = OllamaProvider(base_url=pcfg["base_url"], model=model)
        elif provider_name == "openrouter":
            providers[agent_name] = OpenRouterProvider(
                api_key=_resolve(pcfg["api_key"]), model=model
            )
        else:
            raise ValueError(f"Unknown provider '{provider_name}' for agent '{agent_name}'")

    return providers
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_factory.py -v
```
Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add providers/factory.py tests/test_factory.py
git commit -m "feat: provider factory — config-driven instantiation with startup validation"
```

---

## Task 10: QuestionGeneratorAgent

**Files:**
- Create: `agents/question_generator.py`
- Create: `tests/test_agents/test_question_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_question_generator.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from providers.base import ProviderResponse
from agents.question_generator import QuestionGeneratorAgent
from models.session import Question


@pytest.fixture
def agent(mock_provider):
    return QuestionGeneratorAgent(provider=mock_provider)


async def test_returns_correct_number_of_questions(agent, mock_provider):
    questions_data = [
        {"text": "Q1?", "difficulty": "easy", "expected_outline": "outline1"},
        {"text": "Q2?", "difficulty": "medium", "expected_outline": "outline2"},
        {"text": "Q3?", "difficulty": "hard", "expected_outline": "outline3"},
    ]
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": questions_data}), tool_calls=[]
        )
    )
    questions = await agent.run(topic="Science", count=3)
    assert len(questions) == 3
    assert all(isinstance(q, Question) for q in questions)


async def test_question_fields_mapped_correctly(agent, mock_provider):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": [
                {"text": "What is DNA?", "difficulty": "easy", "expected_outline": "double helix, genetics"}
            ]}),
            tool_calls=[],
        )
    )
    questions = await agent.run(topic="Biology", count=1)
    q = questions[0]
    assert q.text == "What is DNA?"
    assert q.difficulty == "easy"
    assert q.expected_outline == "double helix, genetics"


async def test_prompt_includes_topic_and_count(agent, mock_provider):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"questions": [
                {"text": "Q?", "difficulty": "easy", "expected_outline": "o"}
            ]}),
            tool_calls=[],
        )
    )
    await agent.run(topic="History", count=5)
    call_args = mock_provider.complete.call_args
    user_message = call_args.kwargs["messages"][0]["content"]
    assert "History" in user_message
    assert "5" in user_message
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agents/test_question_generator.py -v
```

- [ ] **Step 3: Implement agents/question_generator.py**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_agents/test_question_generator.py -v
```
Expected: 3 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/question_generator.py tests/test_agents/test_question_generator.py
git commit -m "feat: QuestionGeneratorAgent — topic-aware question generation"
```

---

## Task 11: EvaluatorAgent

**Files:**
- Create: `agents/evaluator.py`
- Create: `tests/test_agents/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_evaluator.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.evaluator import EvaluatorAgent
from models.session import Question, Evaluation


@pytest.fixture
def agent(mock_provider):
    return EvaluatorAgent(provider=mock_provider)


@pytest.fixture
def question():
    return Question(text="What is photosynthesis?", difficulty="medium", expected_outline="light, CO2, glucose, chlorophyll")


async def test_returns_evaluation(agent, mock_provider, question):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "score": 78,
                "feedback": "Good coverage of light reactions, missed glucose detail.",
                "correct_points": ["light absorption", "CO2 intake"],
                "missing_points": ["glucose synthesis"],
            }),
            tool_calls=[],
        )
    )
    ev = await agent.run(question=question, question_index=2, answer_text="Plants use sunlight and CO2.")
    assert isinstance(ev, Evaluation)
    assert ev.question_index == 2
    assert ev.score == 78
    assert "glucose" in ev.missing_points[0]


async def test_prompt_includes_expected_outline(agent, mock_provider, question):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"score": 50, "feedback": "ok", "correct_points": [], "missing_points": []}),
            tool_calls=[],
        )
    )
    await agent.run(question=question, question_index=0, answer_text="some answer")
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    assert "chlorophyll" in user_msg   # expected_outline appears in prompt
    assert "some answer" in user_msg
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agents/test_evaluator.py -v
```

- [ ] **Step 3: Implement agents/evaluator.py**

```python
import json
from providers.base import ModelProvider
from models.session import Question, Evaluation

_SYSTEM = """You are an objective academic evaluator. Score answers 0–100 based on accuracy and completeness.
Ground every judgment in the expected outline. Be specific. Always respond with valid JSON only."""


class EvaluatorAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(self, question: Question, question_index: int, answer_text: str) -> Evaluation:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: {question.text}\n\n"
                    f"Expected outline: {question.expected_outline}\n\n"
                    f"User's answer: {answer_text}\n\n"
                    f"Return JSON: {{\"score\": 0-100, \"feedback\": \"2-3 sentences\", "
                    f"\"correct_points\": [\"...\"], \"missing_points\": [\"...\"]}}"
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return Evaluation(
            question_index=question_index,
            score=data["score"],
            feedback=data["feedback"],
            correct_points=data["correct_points"],
            missing_points=data["missing_points"],
        )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_agents/test_evaluator.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/evaluator.py tests/test_agents/test_evaluator.py
git commit -m "feat: EvaluatorAgent — per-answer scoring grounded in expected outline"
```

---

## Task 12: AnalyzerAgent

**Files:**
- Create: `agents/analyzer.py`
- Create: `tests/test_agents/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_analyzer.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.analyzer import AnalyzerAgent
from models.session import Question, Evaluation, Analysis


@pytest.fixture
def agent(mock_provider):
    return AnalyzerAgent(provider=mock_provider)


@pytest.fixture
def session_data():
    questions = [
        Question(text="Q1?", difficulty="easy", expected_outline="o1"),
        Question(text="Q2?", difficulty="medium", expected_outline="o2"),
    ]
    evaluations = [
        Evaluation(question_index=0, score=90, feedback="Great.", correct_points=["A"], missing_points=[]),
        Evaluation(question_index=1, score=55, feedback="Weak.", correct_points=[], missing_points=["B", "C"]),
    ]
    return questions, evaluations


async def test_returns_analysis(agent, mock_provider, session_data):
    questions, evaluations = session_data
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "strengths": ["Core concept recall"],
                "weaknesses": ["Mechanism detail"],
                "overall_score": 72.5,
            }),
            tool_calls=[],
        )
    )
    analysis = await agent.run(questions=questions, evaluations=evaluations)
    assert isinstance(analysis, Analysis)
    assert analysis.overall_score == 72.5
    assert "Core concept recall" in analysis.strengths


async def test_prompt_contains_all_questions(agent, mock_provider, session_data):
    questions, evaluations = session_data
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"strengths": [], "weaknesses": [], "overall_score": 0.0}),
            tool_calls=[],
        )
    )
    await agent.run(questions=questions, evaluations=evaluations)
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    assert "Q1?" in user_msg
    assert "Q2?" in user_msg
    assert "90" in user_msg
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agents/test_analyzer.py -v
```

- [ ] **Step 3: Implement agents/analyzer.py**

```python
import json
from providers.base import ModelProvider
from models.session import Question, Evaluation, Analysis

_SYSTEM = """You are a learning coach reviewing a complete Q&A session. Identify recurring patterns,
not just per-question scores. Always respond with valid JSON only."""


class AnalyzerAgent:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    async def run(self, questions: list[Question], evaluations: list[Evaluation]) -> Analysis:
        rows = []
        for q, e in zip(questions, evaluations):
            correct = ", ".join(e.correct_points) or "none"
            missing = ", ".join(e.missing_points) or "none"
            rows.append(f"Q: {q.text} | Score: {e.score}/100 | Correct: {correct} | Missing: {missing}")

        session_table = "\n".join(rows)
        messages = [
            {
                "role": "user",
                "content": (
                    f"Review this Q&A session:\n\n{session_table}\n\n"
                    f"Identify recurring strengths and knowledge gaps. "
                    f'Return JSON: {{"strengths": ["..."], "weaknesses": ["..."], "overall_score": float}}'
                ),
            }
        ]
        response = await self.provider.complete(messages=messages, system=_SYSTEM)
        data = json.loads(response.content)
        return Analysis(
            strengths=data["strengths"],
            weaknesses=data["weaknesses"],
            overall_score=float(data["overall_score"]),
        )
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_agents/test_analyzer.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/analyzer.py tests/test_agents/test_analyzer.py
git commit -m "feat: AnalyzerAgent — session-wide performance pattern analysis"
```

---

## Task 13: AdvisorAgent

**Files:**
- Create: `agents/advisor.py`
- Create: `tests/test_agents/test_advisor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_advisor.py
import json
import pytest
from unittest.mock import AsyncMock
from providers.base import ProviderResponse
from agents.advisor import AdvisorAgent
from models.session import Analysis, Recommendation


@pytest.fixture
def agent(mock_provider):
    return AdvisorAgent(provider=mock_provider)


@pytest.fixture
def analysis():
    return Analysis(strengths=["concept recall"], weaknesses=["mechanism detail"], overall_score=68.0)


async def test_returns_recommendation(agent, mock_provider, analysis):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({
                "next_topics": ["Chemistry"],
                "focus_areas": ["reaction mechanisms", "molecular bonding"],
                "message": "You have a solid foundation. Focus on mechanisms next.",
            }),
            tool_calls=[],
        )
    )
    rec = await agent.run(
        analysis=analysis, current_topic="Science", all_topics=["Science", "Chemistry", "Math"]
    )
    assert isinstance(rec, Recommendation)
    assert "Chemistry" in rec.next_topics
    assert len(rec.focus_areas) == 2


async def test_all_topics_appear_in_prompt(agent, mock_provider, analysis):
    mock_provider.complete = AsyncMock(
        return_value=ProviderResponse(
            content=json.dumps({"next_topics": [], "focus_areas": [], "message": ""}),
            tool_calls=[],
        )
    )
    all_topics = ["Science", "Math", "History"]
    await agent.run(analysis=analysis, current_topic="Science", all_topics=all_topics)
    user_msg = mock_provider.complete.call_args.kwargs["messages"][0]["content"]
    for topic in all_topics:
        assert topic in user_msg
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agents/test_advisor.py -v
```

- [ ] **Step 3: Implement agents/advisor.py**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_agents/test_advisor.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/advisor.py tests/test_agents/test_advisor.py
git commit -m "feat: AdvisorAgent — constrained next-topic recommendation"
```

---

## Task 14: OrchestratorAgent

**Files:**
- Create: `agents/orchestrator.py`
- Create: `tests/test_agents/test_orchestrator.py`

The Orchestrator uses `anthropic.Anthropic()` directly (not ModelProvider) because it requires adaptive thinking and the Anthropic multi-turn tool_use message format. It loops until Claude stops calling tools.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agents/test_orchestrator.py
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agents.orchestrator import OrchestratorAgent
from agents.question_generator import QuestionGeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.analyzer import AnalyzerAgent
from agents.advisor import AdvisorAgent
from models.session import SessionState, Question, Answer, Evaluation, Analysis, Recommendation


def _make_agent(mock_provider):
    return OrchestratorAgent(
        api_key="test-key",
        model="claude-opus-4-7",
        question_generator=MagicMock(spec=QuestionGeneratorAgent),
        evaluator=MagicMock(spec=EvaluatorAgent),
        analyzer=MagicMock(spec=AnalyzerAgent),
        advisor=MagicMock(spec=AdvisorAgent),
        all_topics=["Science", "Math"],
    )


def _selecting_state():
    return SessionState(
        session_id="s1", topic="Science", question_count=3,
        status="selecting", created_at="t", updated_at="t",
    )


async def test_dispatches_to_question_generator():
    agent = _make_agent(None)
    state = _selecting_state()

    generated_questions = [
        Question(text="Q1?", difficulty="easy", expected_outline="o1"),
    ]
    agent.question_generator.run = AsyncMock(return_value=generated_questions)

    # Mock Claude to return generate_questions tool call, then stop
    mock_response_tool = MagicMock()
    mock_response_tool.content = [
        MagicMock(type="tool_use", id="tc1", name="generate_questions", input={"topic": "Science", "count": 3}),
    ]
    mock_response_stop = MagicMock()
    mock_response_stop.content = [MagicMock(type="text", text="Done.")]

    with patch.object(agent.client.messages, "create", side_effect=[mock_response_tool, mock_response_stop]):
        updated = await agent.run(state)

    agent.question_generator.run.assert_called_once_with(topic="Science", count=3)
    assert updated.questions == generated_questions
    assert updated.status == "questioning"


async def test_dispatches_to_evaluator():
    agent = _make_agent(None)
    state = _selecting_state()
    state.status = "questioning"
    state.questions = [Question(text="Q1?", difficulty="easy", expected_outline="o1")]
    state.answers = [Answer(question_index=0, text="my answer", submitted_at="t")]

    ev = Evaluation(question_index=0, score=80, feedback="Good.", correct_points=["A"], missing_points=[])
    agent.evaluator.run = AsyncMock(return_value=ev)

    mock_response_tool = MagicMock()
    mock_response_tool.content = [
        MagicMock(type="tool_use", id="tc2", name="evaluate_answer", input={"question_index": 0}),
    ]
    mock_response_stop = MagicMock()
    mock_response_stop.content = [MagicMock(type="text", text="Done.")]

    with patch.object(agent.client.messages, "create", side_effect=[mock_response_tool, mock_response_stop]):
        updated = await agent.run(state)

    agent.evaluator.run.assert_called_once()
    assert len(updated.evaluations) == 1
    assert updated.evaluations[0].score == 80
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_agents/test_orchestrator.py -v
```

- [ ] **Step 3: Implement agents/orchestrator.py**

```python
from __future__ import annotations
import json
from typing import Callable, Awaitable
import anthropic
from models.session import SessionState, Question, Answer, Evaluation
from agents.question_generator import QuestionGeneratorAgent
from agents.evaluator import EvaluatorAgent
from agents.analyzer import AnalyzerAgent
from agents.advisor import AdvisorAgent

_SYSTEM = """You are the orchestrator of a Q&A learning platform. Inspect the session state and
decide which action to take next by calling exactly one tool. Never generate questions or evaluations
yourself — always delegate to the appropriate tool."""

_TOOLS = [
    {
        "name": "generate_questions",
        "description": "Generate questions for the topic at the start of a session (status=selecting).",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The session topic"},
                "count": {"type": "integer", "description": "Number of questions to generate"},
            },
            "required": ["topic", "count"],
        },
    },
    {
        "name": "evaluate_answer",
        "description": "Evaluate the most recently submitted answer (status=questioning).",
        "input_schema": {
            "type": "object",
            "properties": {
                "question_index": {"type": "integer", "description": "Index of the question being answered"},
            },
            "required": ["question_index"],
        },
    },
    {
        "name": "analyze_performance",
        "description": "Analyze overall session performance after all answers are submitted.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "recommend_next",
        "description": "Recommend next topics and focus areas based on analysis results.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

Emitter = Callable[[str, dict], Awaitable[None]]


class OrchestratorAgent:
    def __init__(
        self,
        api_key: str,
        model: str,
        question_generator: QuestionGeneratorAgent,
        evaluator: EvaluatorAgent,
        analyzer: AnalyzerAgent,
        advisor: AdvisorAgent,
        all_topics: list[str],
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key)
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

        messages: list[dict] = [{"role": "user", "content": self._state_summary(state)}]
        system_block = [{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}]

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=system_block,
                tools=_TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": "high"},
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break  # Claude finished without calling a tool

            # Build assistant message preserving all content blocks
            assistant_content = []
            for block in response.content:
                if block.type == "tool_use":
                    assistant_content.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    )
                elif block.type == "text" and block.text:
                    assistant_content.append({"type": "text", "text": block.text})
            messages.append({"role": "assistant", "content": assistant_content})

            # Dispatch each tool and collect results
            tool_results = []
            for block in tool_use_blocks:
                result = await self._dispatch(block.name, block.input, state, _emit)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                )

            messages.append({"role": "user", "content": tool_results})

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
            # If more questions remain, tell Claude the next question index
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_agents/test_orchestrator.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add agents/orchestrator.py tests/test_agents/test_orchestrator.py
git commit -m "feat: OrchestratorAgent — adaptive thinking tool-use routing loop"
```

---

## Task 15: Topics API

**Files:**
- Create: `api/topics.py`
- Create: `tests/test_api/test_topics.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_api/test_topics.py -v
```

- [ ] **Step 3: Implement api/topics.py**

```python
import json
from fastapi import APIRouter

router = APIRouter()
TOPICS_FILE = "data/topics.json"


@router.get("/api/topics")
def get_topics() -> dict:
    with open(TOPICS_FILE) as f:
        topics: list[str] = json.load(f)
    return {"topics": topics, "custom": True}
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_api/test_topics.py -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add api/topics.py tests/test_api/test_topics.py
git commit -m "feat: topics API endpoint"
```

---

## Task 16: Sessions API — Create + Get

**Files:**
- Create: `api/sessions.py` (partial — create and get)
- Create: `tests/test_api/test_sessions.py` (partial)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_sessions.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.sessions import router, _session_queues


def _make_app():
    app = FastAPI()
    app.include_router(router)
    # Inject mock agents
    from agents.orchestrator import OrchestratorAgent
    mock_orch = MagicMock(spec=OrchestratorAgent)
    mock_orch.run = AsyncMock(return_value=MagicMock(status="questioning"))
    app.state.orchestrator = mock_orch
    return app


@pytest.fixture
def client(tmp_path):
    with patch("api.sessions.save_session", new_callable=AsyncMock):
        with patch("api.sessions.load_session", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = None
            app = _make_app()
            yield TestClient(app)


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
```

- [ ] **Step 2: Run tests — expect failure**

```bash
pytest tests/test_api/test_sessions.py::test_create_session_returns_session_id tests/test_api/test_sessions.py::test_get_session_not_found -v
```

- [ ] **Step 3: Implement api/sessions.py (create + get)**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
pytest tests/test_api/test_sessions.py::test_create_session_returns_session_id tests/test_api/test_sessions.py::test_get_session_not_found -v
```
Expected: 2 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add api/sessions.py tests/test_api/test_sessions.py
git commit -m "feat: sessions API — create and get endpoints"
```

---

## Task 17: Sessions API — Submit Answer + SSE Stream

**Files:**
- Modify: `api/sessions.py` (add answer + stream endpoints)
- Modify: `tests/test_api/test_sessions.py` (add tests)

- [ ] **Step 1: Write failing tests — add to test_sessions.py**

```python
# Add these to tests/test_api/test_sessions.py

def test_submit_answer_returns_processing():
    sample_state = SessionState(
        session_id="s1", topic="Science", question_count=3,
        questions=[Question(text="Q1?", difficulty="easy", expected_outline="o1")],
        status="questioning", current_index=0,
        created_at="t", updated_at="t",
    )
    with patch("api.sessions.load_session", new_callable=AsyncMock, return_value=sample_state):
        with patch("api.sessions.save_session", new_callable=AsyncMock):
            app = _make_app()
            # Pre-create the queue so background task has somewhere to emit
            _session_queues["s1"] = asyncio.Queue()
            client = TestClient(app)
            response = client.post("/api/sessions/s1/answer", json={"answer": "my answer"})
            assert response.status_code == 200
            assert response.json()["status"] == "processing"


def test_submit_answer_404_on_missing_session():
    with patch("api.sessions.load_session", new_callable=AsyncMock, return_value=None):
        app = _make_app()
        client = TestClient(app)
        response = client.post("/api/sessions/missing/answer", json={"answer": "hi"})
        assert response.status_code == 404
```

(Add the missing import at the top of test_sessions.py: `from models.session import SessionState, Question`)

- [ ] **Step 2: Run new tests — expect failure**

```bash
pytest tests/test_api/test_sessions.py::test_submit_answer_returns_processing tests/test_api/test_sessions.py::test_submit_answer_404_on_missing_session -v
```

- [ ] **Step 3: Add answer + stream to api/sessions.py**

Append this to `api/sessions.py` after the existing `get_session` function:

```python
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
```

- [ ] **Step 4: Run all session tests — expect pass**

```bash
pytest tests/test_api/test_sessions.py -v
```
Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add api/sessions.py tests/test_api/test_sessions.py
git commit -m "feat: sessions API — answer submission and SSE stream"
```

---

## Task 18: FastAPI Main App

**Files:**
- Create: `main.py`

- [ ] **Step 1: Implement main.py**

```python
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

    orch_cfg = json.load(open("model_config.json"))["agents"]["orchestrator"]
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
```

- [ ] **Step 2: Verify the app starts**

Create a `.env` file with at minimum `ANTHROPIC_API_KEY=sk-ant-your-key-here`, then:

```bash
uvicorn main:app --reload --port 8000
```

Expected: Server starts without errors, no "Required environment variable" exceptions.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: FastAPI main — agent wiring, startup validation, static serving"
```

---

## Task 19: Frontend HTML + CSS

**Files:**
- Create: `static/index.html`
- Create: `static/style.css`

- [ ] **Step 1: Create static/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>QuizMind</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <!-- Screen 1: Topic Selection -->
  <div id="screen-select" class="screen active">
    <nav class="nav">🧠 QuizMind</nav>
    <div class="container">
      <h1>Choose a Topic</h1>
      <div id="topic-chips" class="chips"></div>
      <div class="divider">— or type your own —</div>
      <input id="custom-topic" class="input" placeholder="Custom topic…" type="text">
      <div class="label">Questions per session:</div>
      <div class="count-picker">
        <button class="count-btn" data-count="3">3</button>
        <button class="count-btn active" data-count="5">5</button>
        <button class="count-btn" data-count="10">10</button>
      </div>
      <button id="start-btn" class="btn primary" disabled>Start Session →</button>
    </div>
  </div>

  <!-- Screen 2: Q&A Session -->
  <div id="screen-qa" class="screen">
    <nav class="nav">
      <span id="qa-topic-label"></span>
      <span id="qa-progress"></span>
    </nav>
    <div class="container">
      <div id="prev-result" class="result-card hidden"></div>
      <div id="question-card" class="question-card">
        <div id="question-text"></div>
        <div id="difficulty-badge" class="badge"></div>
      </div>
      <div id="agent-status" class="status hidden"></div>
      <textarea id="answer-input" class="textarea" placeholder="Type your answer here…" rows="5"></textarea>
      <button id="submit-btn" class="btn primary">Submit Answer →</button>
    </div>
  </div>

  <!-- Screen 3: Results -->
  <div id="screen-results" class="screen">
    <nav class="nav">🧠 QuizMind — Results</nav>
    <div class="container">
      <div class="score-display">
        <div id="overall-score" class="score-number"></div>
        <div class="score-label">Overall Score</div>
      </div>
      <div class="section-title success">✓ Strengths</div>
      <ul id="strengths-list" class="bullet-list"></ul>
      <div class="section-title danger">✗ Weaknesses</div>
      <ul id="weaknesses-list" class="bullet-list"></ul>
      <div class="advisor-card">
        <div class="advisor-title">🗺️ Next Steps</div>
        <div id="advisor-message" class="advisor-message"></div>
        <div class="label">Try next:</div>
        <div id="next-topics" class="chips small"></div>
        <div class="label">Focus on:</div>
        <ul id="focus-areas" class="bullet-list small"></ul>
      </div>
      <button id="new-session-btn" class="btn primary">Start New Session →</button>
    </div>
  </div>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create static/style.css**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0f172a;
  --surface: #1e293b;
  --surface2: #1a0f2e;
  --border: #334155;
  --border-accent: #2563eb;
  --purple: #7c3aed;
  --text: #e2e8f0;
  --muted: #94a3b8;
  --faint: #64748b;
  --blue: #60a5fa;
  --green: #34d399;
  --yellow: #fbbf24;
  --red: #f87171;
  --radius: 8px;
}

body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; min-height: 100vh; }

.screen { display: none; }
.screen.active { display: block; }

.nav { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 24px; font-weight: bold; display: flex; justify-content: space-between; align-items: center; color: var(--muted); }

.container { max-width: 640px; margin: 0 auto; padding: 32px 20px; }

h1 { font-size: 1.5rem; margin-bottom: 20px; color: var(--text); }

.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
.chip { background: var(--surface); border: 1px solid var(--border-accent); border-radius: 20px; padding: 6px 14px; color: var(--blue); cursor: pointer; font-size: 14px; transition: all 0.15s; }
.chip:hover, .chip.selected { background: #1e3a5f; border-color: var(--blue); }
.chips.small .chip { font-size: 12px; padding: 4px 10px; }

.divider { text-align: center; color: var(--faint); font-size: 13px; margin: 12px 0; }

.input { width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 10px 14px; font-size: 14px; outline: none; }
.input:focus { border-color: var(--border-accent); }

.label { color: var(--faint); font-size: 13px; margin: 16px 0 8px; }

.count-picker { display: flex; gap: 8px; margin-bottom: 24px; }
.count-btn { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 8px 20px; cursor: pointer; font-size: 14px; }
.count-btn.active { background: #1e3a5f; border-color: var(--blue); color: var(--blue); font-weight: bold; }

.btn { display: block; width: 100%; padding: 12px; border: none; border-radius: var(--radius); font-size: 15px; cursor: pointer; font-weight: bold; }
.btn.primary { background: var(--purple); color: #fff; }
.btn.primary:hover { background: #6d28d9; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

.question-card { background: var(--surface); border-left: 3px solid var(--purple); border-radius: var(--radius); padding: 16px; margin-bottom: 16px; }
#question-text { font-size: 16px; line-height: 1.6; margin-bottom: 8px; }
.badge { display: inline-block; font-size: 12px; padding: 2px 8px; border-radius: 4px; background: #2d1b69; color: #c4b5fd; }

.result-card { background: var(--surface); border: 1px solid #065f46; border-radius: var(--radius); padding: 12px; margin-bottom: 16px; font-size: 13px; }
.result-card .score { color: var(--green); font-weight: bold; }

.textarea { width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); padding: 12px; font-size: 14px; resize: vertical; outline: none; margin-bottom: 12px; font-family: inherit; }
.textarea:focus { border-color: var(--border-accent); }

.status { text-align: center; color: var(--faint); font-size: 13px; padding: 8px 0; }
.hidden { display: none !important; }

.score-display { text-align: center; margin: 24px 0; }
.score-number { font-size: 64px; font-weight: bold; color: var(--yellow); }
.score-label { color: var(--muted); font-size: 14px; }

.section-title { font-size: 13px; font-weight: bold; margin: 20px 0 8px; }
.section-title.success { color: var(--green); }
.section-title.danger { color: var(--red); }

.bullet-list { list-style: none; padding-left: 4px; }
.bullet-list li { color: var(--muted); font-size: 14px; padding: 3px 0; }
.bullet-list li::before { content: "• "; color: var(--faint); }
.bullet-list.small li { font-size: 13px; }

.advisor-card { background: var(--surface); border: 1px solid var(--border-accent); border-radius: var(--radius); padding: 16px; margin: 20px 0; }
.advisor-title { color: var(--blue); font-weight: bold; margin-bottom: 8px; }
.advisor-message { color: var(--muted); font-size: 14px; line-height: 1.6; margin-bottom: 12px; }
```

- [ ] **Step 3: Verify static files load**

With the server running (`uvicorn main:app --reload`), visit http://localhost:8000. You should see the Topic Selection screen with no errors in browser console.

- [ ] **Step 4: Commit**

```bash
git add static/index.html static/style.css
git commit -m "feat: frontend HTML and CSS — 3-screen layout with dark theme"
```

---

## Task 20: Frontend JavaScript

**Files:**
- Create: `static/app.js`

- [ ] **Step 1: Implement static/app.js**

```javascript
// ── State ────────────────────────────────────────────────────────────────────
const state = {
  selectedTopic: null,
  customTopic: '',
  questionCount: 5,
  sessionId: null,
  currentIndex: 0,
  totalQuestions: 0,
  lastEvaluation: null,
  eventSource: null,
};

// ── Screen management ────────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── Screen 1: Topic Selection ────────────────────────────────────────────────
async function initTopicSelection() {
  const res = await fetch('/api/topics');
  const { topics } = await res.json();

  const container = document.getElementById('topic-chips');
  topics.forEach(topic => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      container.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      state.selectedTopic = topic;
      state.customTopic = '';
      document.getElementById('custom-topic').value = '';
      updateStartButton();
    });
    container.appendChild(chip);
  });

  document.getElementById('custom-topic').addEventListener('input', e => {
    state.customTopic = e.target.value.trim();
    if (state.customTopic) {
      container.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
      state.selectedTopic = null;
    }
    updateStartButton();
  });

  document.querySelectorAll('.count-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.count-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.questionCount = parseInt(btn.dataset.count, 10);
    });
  });

  document.getElementById('start-btn').addEventListener('click', startSession);
}

function updateStartButton() {
  const topic = state.selectedTopic || state.customTopic;
  document.getElementById('start-btn').disabled = !topic;
}

async function startSession() {
  const topic = state.selectedTopic || state.customTopic;
  document.getElementById('start-btn').disabled = true;
  document.getElementById('start-btn').textContent = 'Starting…';

  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, question_count: state.questionCount }),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.totalQuestions = state.questionCount;
  state.currentIndex = 0;

  openSSEStream();

  document.getElementById('start-btn').textContent = 'Start Session →';
  document.getElementById('start-btn').disabled = false;
}

// ── SSE Stream ───────────────────────────────────────────────────────────────
function openSSEStream() {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/sessions/${state.sessionId}/stream`);

  state.eventSource.addEventListener('agent_status', e => {
    const { message } = JSON.parse(e.data);
    showAgentStatus(message);
  });

  state.eventSource.addEventListener('question_ready', e => {
    const { question, index } = JSON.parse(e.data);
    state.currentIndex = index;
    showQAScreen(question, index);
  });

  state.eventSource.addEventListener('evaluation_ready', e => {
    const { evaluation } = JSON.parse(e.data);
    state.lastEvaluation = evaluation;
    hideAgentStatus();
    document.getElementById('submit-btn').disabled = false;
    document.getElementById('answer-input').disabled = false;
    document.getElementById('answer-input').value = '';
    // Show previous result on next question render (handled by question_ready)
  });

  state.eventSource.addEventListener('analysis_ready', e => {
    showAgentStatus('Building recommendations…');
  });

  state.eventSource.addEventListener('complete', e => {
    const { recommendation } = JSON.parse(e.data);
    state.eventSource.close();
    showResultsScreen(recommendation);
  });

  state.eventSource.addEventListener('error', e => {
    hideAgentStatus();
    alert('An error occurred. Please refresh and try again.');
  });
}

// ── Screen 2: Q&A Session ────────────────────────────────────────────────────
function showQAScreen(question, index) {
  showScreen('screen-qa');
  document.getElementById('qa-topic-label').textContent =
    `Topic: ${state.selectedTopic || state.customTopic}`;
  document.getElementById('qa-progress').textContent =
    `Q ${index + 1} of ${state.totalQuestions}`;
  document.getElementById('question-text').textContent = question.text;
  document.getElementById('difficulty-badge').textContent =
    `${difficultyIcon(question.difficulty)} ${capitalize(question.difficulty)} difficulty`;

  // Show previous evaluation if exists
  const prevCard = document.getElementById('prev-result');
  if (state.lastEvaluation && index > 0) {
    const ev = state.lastEvaluation;
    prevCard.innerHTML =
      `<span class="score">✓ Previous: ${ev.score}/100</span> — ${ev.feedback}`;
    prevCard.classList.remove('hidden');
  } else {
    prevCard.classList.add('hidden');
  }

  document.getElementById('answer-input').disabled = false;
  document.getElementById('answer-input').value = '';
  document.getElementById('submit-btn').disabled = false;
  hideAgentStatus();
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('submit-btn').addEventListener('click', submitAnswer);
  document.getElementById('new-session-btn').addEventListener('click', () => {
    state.sessionId = null;
    state.selectedTopic = null;
    state.customTopic = '';
    state.lastEvaluation = null;
    state.currentIndex = 0;
    document.getElementById('custom-topic').value = '';
    document.getElementById('topic-chips').querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
    updateStartButton();
    showScreen('screen-select');
  });
});

async function submitAnswer() {
  const text = document.getElementById('answer-input').value.trim();
  if (!text) return;

  document.getElementById('submit-btn').disabled = true;
  document.getElementById('answer-input').disabled = true;
  showAgentStatus('🤖 Submitting…');

  await fetch(`/api/sessions/${state.sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer: text }),
  });
}

function showAgentStatus(msg) {
  const el = document.getElementById('agent-status');
  el.textContent = `🤖 ${msg}`;
  el.classList.remove('hidden');
}

function hideAgentStatus() {
  document.getElementById('agent-status').classList.add('hidden');
}

// ── Screen 3: Results ────────────────────────────────────────────────────────
async function showResultsScreen(recommendation) {
  // Load final session state for overall score
  const res = await fetch(`/api/sessions/${state.sessionId}`);
  const session = await res.json();
  const overallScore = session.analysis ? Math.round(session.analysis.overall_score) : '—';

  document.getElementById('overall-score').textContent = overallScore;

  const strengths = session.analysis?.strengths ?? [];
  const weaknesses = session.analysis?.weaknesses ?? [];

  renderList('strengths-list', strengths);
  renderList('weaknesses-list', weaknesses);

  document.getElementById('advisor-message').textContent = recommendation.message;

  const nextTopicsEl = document.getElementById('next-topics');
  nextTopicsEl.innerHTML = '';
  recommendation.next_topics.forEach(topic => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      state.selectedTopic = topic;
      state.customTopic = '';
      showScreen('screen-select');
      updateStartButton();
    });
    nextTopicsEl.appendChild(chip);
  });

  renderList('focus-areas', recommendation.focus_areas);
  showScreen('screen-results');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function renderList(id, items) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(item => `<li>${item}</li>`).join('');
}

function difficultyIcon(d) {
  return d === 'easy' ? '🟢' : d === 'medium' ? '⚡' : '🔴';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ── Init ──────────────────────────────────────────────────────────────────────
initTopicSelection();
```

- [ ] **Step 2: Start the server and test Screen 1**

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000. Verify:
- Topic chips render from `/api/topics`
- Selecting a chip enables Start button
- Custom topic input enables Start button
- Count picker toggles correctly

- [ ] **Step 3: Test Screen 2 (Q&A Session)**

Click "Start Session". Verify:
- SSE stream connects (Network tab shows open EventStream connection)
- Agent status appears ("Generating questions…")
- Question card renders with difficulty badge when `question_ready` fires
- Submit button works; agent status shows "Evaluating your answer…"
- After evaluation, next question renders with previous result card
- Progress indicator increments correctly (Q 1 of 5, Q 2 of 5…)

- [ ] **Step 4: Test Screen 3 (Results)**

After answering all questions, verify:
- Results screen shows overall score
- Strengths and weaknesses bullet lists populate
- Advisor message and focus areas render
- "Try next" topic chips start a new session with that topic pre-selected when clicked
- "Start New Session" button returns to Screen 1 cleanly

- [ ] **Step 5: Commit**

```bash
git add static/app.js
git commit -m "feat: frontend JavaScript — 3-screen Q&A flow with SSE integration"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| SessionState + all dataclasses | Task 2 |
| JSON persistence | Task 3 |
| ModelProvider ABC | Task 4 |
| 4 concrete providers | Tasks 5–8 |
| model_config.json + startup validation | Task 9 |
| QuestionGeneratorAgent | Task 10 |
| EvaluatorAgent | Task 11 |
| AnalyzerAgent | Task 12 |
| AdvisorAgent | Task 13 |
| OrchestratorAgent (tool-use loop) | Task 14 |
| GET /api/topics | Task 15 |
| POST /api/sessions + GET /api/sessions/{id} | Task 16 |
| POST /api/sessions/{id}/answer + SSE stream | Task 17 |
| FastAPI app + static file serving | Task 18 |
| Frontend 3-screen layout | Task 19 |
| Frontend JS (topic select, Q&A, results) | Task 20 |
| SSE events: agent_status, question_ready, evaluation_ready, analysis_ready, complete, error | Tasks 14, 17, 20 |
| prompt caching on system prompts | Task 5 (AnthropicProvider), Task 14 (Orchestrator) |
| Adaptive thinking on Orchestrator | Task 14 |

**Placeholder scan:** No TBD, TODO, or incomplete steps found.

**Type consistency:** `SessionState`, `Question`, `Answer`, `Evaluation`, `Analysis`, `Recommendation` defined in Task 2 and used consistently by name across all agent tasks. `ProviderResponse` and `ToolCall` defined in Task 4, used in Tasks 5–9 and 14. `emit` callable signature (`str, dict → Awaitable[None]`) matches between Task 14 (OrchestratorAgent) and Task 17 (API sessions).
