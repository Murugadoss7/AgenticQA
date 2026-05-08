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
    elif value.startswith("$"):
        env_var = value[1:]
    else:
        return value
    resolved = os.environ.get(env_var, "")
    if not resolved:
        raise ValueError(f"Required environment variable '{env_var}' is not set")
    return resolved


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
                api_version=_resolve(pcfg["api_version"]),
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
