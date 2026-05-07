# Q&A Platform — Design Spec

**Date:** 2026-05-07  
**Status:** Approved  
**Stack:** Python · FastAPI · Anthropic SDK (`claude-opus-4-7`) · Vanilla JS

---

## Overview

A browser-based Q&A platform where an agentic AI system generates questions on any topic, evaluates user answers, analyzes performance, and recommends next steps. The system is built as a multi-agent pipeline using the Anthropic Python SDK — replicating LangChain/LangGraph patterns natively without those frameworks.

---

## Decisions Made

| Question | Decision |
|---|---|
| Interface | Web app (browser) |
| Language | Python + FastAPI + Vanilla JS |
| Persistence | Light — local JSON file per session |
| Speech input | Text only (v1) |
| Topics | Predefined list + custom free-text |
| Questions per session | User chooses: 3, 5, or 10 |
| Agent architecture | Orchestrator + Specialists (Hybrid) |
| Agent communication | Tool-use Orchestrator + Class Specialists |
| Claude model | `claude-opus-4-7` with adaptive thinking |

---

## Architecture

### Layer Overview

```
Browser (HTML + Vanilla JS)
  ├── Topic Selection Screen
  ├── Q&A Session Screen
  └── Results & Recommendations Screen
       ↕  REST (actions) + SSE (live agent updates)
FastAPI Backend
  ├── POST /api/sessions          — create session
  ├── POST /api/sessions/{id}/answer — submit answer → triggers agent chain
  ├── GET  /api/sessions/{id}/stream — SSE event stream
  ├── GET  /api/sessions/{id}        — load existing session
  └── GET  /api/topics               — list predefined topics
       ↕  Python calls
Agent Layer (5 agents, all using Anthropic SDK)
  ├── OrchestratorAgent     — tool-use Claude call; routes to specialists
  ├── QuestionGeneratorAgent — own Claude call; generates questions
  ├── EvaluatorAgent        — own Claude call; scores + feedback per answer
  ├── AnalyzerAgent         — own Claude call; identifies strengths/weaknesses
  └── AdvisorAgent          — own Claude call; recommends next topics/focus
       ↕  read / write
Storage
  ├── sessions/{id}.json    — SessionState (light persistence)
  └── data/topics.json      — predefined topics list
```

### LangGraph Analogy

| LangGraph concept | Our implementation |
|---|---|
| StateGraph | `SessionState` dataclass passed through all agents |
| Nodes | Specialist agent classes (`QuestionGeneratorAgent`, etc.) |
| Conditional edges | Orchestrator's tool-use — Claude decides which specialist to call |
| State reducers | Each specialist reads state, returns enriched state |

---

## Data Model

### SessionState

```python
@dataclass
class SessionState:
    session_id: str
    topic: str
    question_count: int           # 3 | 5 | 10
    questions: List[Question]
    current_index: int
    answers: List[Answer]
    evaluations: List[Evaluation]
    analysis: Optional[Analysis]
    recommendation: Optional[Recommendation]
    status: str                   # "selecting" | "questioning" | "evaluating" | "complete"
    created_at: str
    updated_at: str
```

### Supporting Types

```python
@dataclass
class Question:
    text: str
    difficulty: str              # "easy" | "medium" | "hard"
    expected_outline: str        # used by EvaluatorAgent for grounding

@dataclass
class Answer:
    question_index: int
    text: str
    submitted_at: str

@dataclass
class Evaluation:
    question_index: int
    score: int                   # 0–100
    feedback: str                # 2–3 sentences
    correct_points: List[str]
    missing_points: List[str]

@dataclass
class Analysis:
    strengths: List[str]
    weaknesses: List[str]
    overall_score: float

@dataclass
class Recommendation:
    next_topics: List[str]       # 1–2 topics from predefined list
    focus_areas: List[str]       # 2–3 specific areas within current topic
    message: str                 # narrative summary
```

---

## Agent Design

### OrchestratorAgent

**Role:** Routes to the correct specialist based on `SessionState.status`. Uses Claude `tool_use` — Claude autonomously decides which tool (specialist) to invoke next.

**How it works:**
- Receives `SessionState`
- Makes a Claude API call with 4 tools defined
- Claude reads the state and decides: generate questions, evaluate an answer, analyze session, or recommend next steps
- Dispatches to the correct specialist class
- Returns updated `SessionState`

**Tools exposed to Claude:**
```python
tools = [
    generate_questions(topic: str, count: int)   # always easy→medium→hard
    evaluate_answer(question: str, expected_outline: str, user_answer: str)
    analyze_performance(questions: list, evaluations: list)
    recommend_next(analysis: dict, topic: str, all_topics: list)
]
```

**SDK config:**
```python
model = "claude-opus-4-7"
thinking = {"type": "adaptive"}
output_config = {"effort": "high"}
```

---

### QuestionGeneratorAgent

**Role:** Called once at session start. Generates N questions for the topic with escalating difficulty.

| | |
|---|---|
| **In** | `topic: str`, `count: int` |
| **Out** | `List[Question]` — each with `text`, `difficulty`, `expected_outline` |
| **Called when** | `status == "selecting"` → Orchestrator dispatches |
| **Prompt strategy** | System: expert question writer. User: topic + count + instruction to vary difficulty easy→medium→hard. Uses structured output (JSON schema) to guarantee parseable response. |

---

### EvaluatorAgent

**Role:** Called after each user answer. Scores 0–100 and provides targeted feedback grounded in `expected_outline`.

| | |
|---|---|
| **In** | `question: Question`, `answer: str` |
| **Out** | `Evaluation` — score, feedback, correct_points, missing_points |
| **Called when** | `status == "questioning"` and new answer submitted |
| **Prompt strategy** | System: objective academic evaluator. User: question + expected outline + user answer. Uses structured output for guaranteed JSON. |

---

### AnalyzerAgent

**Role:** Called once after all answers submitted. Reviews full session to identify recurring patterns, not just per-answer scores.

| | |
|---|---|
| **In** | `questions: List[Question]`, `evaluations: List[Evaluation]` |
| **Out** | `Analysis` — strengths, weaknesses, overall_score |
| **Called when** | `current_index == question_count` (all answered) |
| **Prompt strategy** | System: learning coach. User: full session history formatted as a table. Produces narrative analysis, not a score list. |

---

### AdvisorAgent

**Role:** Called once after analysis. Picks next topics from the predefined list and names specific focus areas.

| | |
|---|---|
| **In** | `analysis: Analysis`, `topic: str`, `all_topics: List[str]` |
| **Out** | `Recommendation` — next_topics, focus_areas, message |
| **Called when** | After `AnalyzerAgent` completes |
| **Prompt strategy** | System: study advisor. User: analysis + current topic + full topics list. Constrained to recommend only topics from `all_topics`. |

---

## Session Lifecycle (agent dispatch order)

```
User picks topic + count
    → POST /api/sessions
    → Orchestrator (tool_use) → QuestionGeneratorAgent
    → questions saved to SessionState, status = "questioning"
    → first question returned to browser

User submits answer
    → POST /api/sessions/{id}/answer
    → Orchestrator (tool_use) → EvaluatorAgent
    → evaluation saved, feedback returned to browser
    → if more questions: next question shown (loop)
    → if last answer: status = "evaluating"

All answers submitted
    → Orchestrator (tool_use) → AnalyzerAgent → AdvisorAgent
    → status = "complete", full results returned
```

Throughout: `GET /api/sessions/{id}/stream` SSE pushes agent status updates to the frontend. SSE event types:

| Event type | Payload | When |
|---|---|---|
| `agent_status` | `{"message": "Generating questions…"}` | Agent starts running |
| `question_ready` | `{"question": {...}, "index": N}` | QuestionGenerator done |
| `evaluation_ready` | `{"evaluation": {...}}` | Evaluator done |
| `analysis_ready` | `{"analysis": {...}}` | Analyzer done |
| `complete` | `{"recommendation": {...}}` | Advisor done, session complete |
| `error` | `{"message": "..."}` | Any agent error |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/topics` | Returns predefined topics list + custom option |
| `POST` | `/api/sessions` | Creates session, triggers QuestionGenerator, returns session |
| `GET` | `/api/sessions/{id}` | Loads existing session from JSON file |
| `POST` | `/api/sessions/{id}/answer` | Submits answer, triggers Evaluator (and Analyzer+Advisor if last) |
| `GET` | `/api/sessions/{id}/stream` | SSE stream — agent status updates |

---

## Frontend — 3 Screens

### Screen 1: Topic Selection
- Grid of predefined topic chips (Science, History, Math, Technology, Literature, Geography, Philosophy, Programming)
- "Custom topic" free-text input
- Question count picker: 3 / 5 / 10
- "Start Session" button → POST /api/sessions

### Screen 2: Q&A Session
- Topic label + progress indicator (Q 2 of 5)
- Current question card with difficulty badge
- Previous answer result (score + brief feedback, collapsed)
- Text area for answer input
- Submit button → POST /api/sessions/{id}/answer
- Live agent status strip (SSE): "Evaluating your answer…"

### Screen 3: Results & Recommendations
- Overall score (large number)
- Strength bullets
- Weakness bullets
- Advisor message (narrative)
- Recommended next topics (clickable → starts new session)
- "Start New Session" button

---

## Multi-Provider Configuration

Agents are decoupled from any specific AI provider. A `ModelProvider` abstraction normalizes calls across Anthropic, Azure OpenAI, Ollama, and OpenRouter. Each agent is mapped to a provider + model in `model_config.json`, which is loaded at startup.

### ModelProvider Abstraction

```python
class ModelProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list,
        system: str,
        tools: list | None = None,
        **kwargs,
    ) -> ProviderResponse:
        """Unified interface — all agents call this, never touching the SDK directly."""
```

Concrete implementations: `AnthropicProvider`, `AzureOpenAIProvider`, `OllamaProvider`, `OpenRouterProvider`. Each translates the common `complete()` call into its provider's native API format (tool_use vs. function calling vs. JSON output).

### Provider Capabilities

| Provider | Tool use / function calling | Notes |
|---|---|---|
| `anthropic` | Native `tool_use` | Required for Orchestrator |
| `azure_openai` | Native function calling | Translated to Anthropic tool_use interface |
| `ollama` | No tool_use | Specialist-only; uses JSON output mode |
| `openrouter` | Model-dependent | Verify per model before assigning Orchestrator |

**Startup validation:** If `model_config.json` assigns the Orchestrator to `ollama`, the app refuses to start with a clear error. Orchestrator must use a provider with tool_use support (`anthropic` or `azure_openai`).

### model_config.json

```json
{
  "providers": {
    "anthropic":    { "api_key": "${ANTHROPIC_API_KEY}" },
    "azure_openai": { "api_key": "...", "endpoint": "...", "api_version": "2024-02-01" },
    "ollama":       { "base_url": "http://localhost:11434" },
    "openrouter":   { "api_key": "${OPENROUTER_API_KEY}" }
  },
  "agents": {
    "orchestrator":      { "provider": "anthropic",    "model": "claude-opus-4-7" },
    "question_generator":{ "provider": "anthropic",    "model": "claude-sonnet-4-6" },
    "evaluator":         { "provider": "azure_openai", "model": "gpt-4o-mini" },
    "analyzer":          { "provider": "ollama",       "model": "llama3.1:8b" },
    "advisor":           { "provider": "openrouter",   "model": "meta-llama/llama-3.1-70b-instruct" }
  }
}
```

### Default Agent → Model Rationale

| Agent | Provider / Model | Reason |
|---|---|---|
| Orchestrator | `anthropic / claude-opus-4-7` | Needs tool_use + strongest reasoning; drives entire flow |
| QuestionGenerator | `anthropic / claude-sonnet-4-6` | Creative + structured output; moderate capacity |
| Evaluator | `azure_openai / gpt-4o-mini` | Called once per answer — high volume; fast + cheap |
| Analyzer | `ollama / llama3.1:8b` | Local, zero API cost; no tool_use needed |
| Advisor | `openrouter / llama-3.1-70b` | Cost-efficient for recommendation task |

Any agent mapping can be changed by editing `model_config.json` — no code changes required.

---

## File Structure

```
ClaudeSDK/
├── main.py                      # FastAPI app, static file serving
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py          # OrchestratorAgent — tool-use routing
│   ├── question_generator.py    # QuestionGeneratorAgent
│   ├── evaluator.py             # EvaluatorAgent
│   ├── analyzer.py              # AnalyzerAgent
│   └── advisor.py               # AdvisorAgent
├── providers/
│   ├── __init__.py
│   ├── base.py                  # ModelProvider ABC + ProviderResponse
│   ├── anthropic_provider.py    # AnthropicProvider
│   ├── azure_openai_provider.py # AzureOpenAIProvider
│   ├── ollama_provider.py       # OllamaProvider
│   └── openrouter_provider.py   # OpenRouterProvider
├── models/
│   ├── __init__.py
│   └── session.py               # SessionState + all dataclasses
├── api/
│   ├── __init__.py
│   ├── sessions.py              # Session endpoints + SSE stream
│   └── topics.py                # Topics endpoint
├── storage/
│   ├── __init__.py
│   └── json_store.py            # Read/write sessions/*.json
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── topics.json              # Predefined topic list
├── sessions/                    # Auto-created; one JSON file per session
├── model_config.json            # Provider + agent-to-model mapping
├── requirements.txt
└── .env.example                 # ANTHROPIC_API_KEY, OPENROUTER_API_KEY placeholders
```

---

## SDK & Prompt Caching Strategy

- Orchestrator always uses Anthropic SDK (`anthropic.Anthropic()`) with `claude-opus-4-7`, `thinking: {"type": "adaptive"}`, `output_config: {"effort": "high"}`
- Specialist agents call their configured provider via the `ModelProvider` abstraction
- Specialists use structured output (JSON schema via `output_config.format` for Anthropic; equivalent for other providers) to guarantee parseable responses
- System prompts for each agent are stable and cacheable — `cache_control: {"type": "ephemeral"}` applied where the provider supports prompt caching (Anthropic only)
- Streaming via `messages.stream()` (Anthropic) or equivalent → FastAPI SSE → browser `EventSource`

---

## Error Handling

- Provider errors caught per agent → SSE pushes error status to frontend → user sees friendly message
- Startup validation: missing required API keys or invalid Orchestrator provider → clear error, app does not start
- Session JSON read/write wrapped with file lock (no concurrent writes)
- Invalid session ID → 404 from API

---

## Out of Scope (v1)

- Speech input (text only)
- User accounts / authentication
- Database (JSON file persistence only)
- Multi-language support
- Leaderboard / history across sessions
