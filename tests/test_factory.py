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
