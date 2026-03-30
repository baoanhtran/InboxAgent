"""Model-agnostic LLM factory.

Each agent can use a different model from a different provider.
Configure per-agent models in .env using the format:

    <AGENT>_MODEL=<provider>/<model-name>

Examples:
    COORDINATOR_MODEL=openai/gpt-4o
    COMPOSER_MODEL=anthropic/claude-opus-4-6
    REVIEWER_MODEL=openai/gpt-4o-mini
    INBOX_SCANNER_MODEL=google/gemini-2.0-flash
    DEFAULT_MODEL=openai/gpt-4o          # fallback for any unset agent

Supported providers:
    openai     → langchain-openai       (OPENAI_API_KEY)
    anthropic  → langchain-anthropic    (ANTHROPIC_API_KEY)
    google     → langchain-google-genai (GOOGLE_API_KEY)
    mistral    → langchain-mistralai    (MISTRAL_API_KEY)
    ollama     → langchain-ollama       (local, no key needed)
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

# Map agent name → env var that holds its model string
_AGENT_ENV_KEYS: dict[str, str] = {
    "coordinator":       "COORDINATOR_MODEL",
    "inbox_scanner":     "INBOX_SCANNER_MODEL",
    "thread_researcher": "THREAD_RESEARCHER_MODEL",
    "sender_profiler":   "SENDER_PROFILER_MODEL",
    "composer":          "COMPOSER_MODEL",
    "reviewer":          "REVIEWER_MODEL",
}

_DEFAULT_MODEL_KEY = "DEFAULT_MODEL"
_FALLBACK = "openai/gpt-4o"


def _resolve_model_string(agent_name: str) -> str:
    """Return the 'provider/model' string for a given agent, with fallback chain."""
    env_key = _AGENT_ENV_KEYS.get(agent_name)
    if env_key:
        value = os.getenv(env_key)
        if value:
            return value
    return os.getenv(_DEFAULT_MODEL_KEY) or _FALLBACK


def _build_llm(model_string: str, temperature: float) -> BaseChatModel:
    """Instantiate the right LangChain chat model from a 'provider/model' string."""
    if "/" not in model_string:
        raise ValueError(
            f"Invalid model string: '{model_string}'. "
            "Expected format: 'provider/model-name' (e.g. 'openai/gpt-4o')."
        )

    provider, model_name = model_string.split("/", 1)
    provider = provider.lower().strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model_name, temperature=temperature)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name, temperature=temperature)

    raise ValueError(
        f"Unknown provider: '{provider}'. "
        "Supported: openai, anthropic, google, mistral, ollama."
    )


def get_llm(agent_name: str, temperature: float = 0) -> BaseChatModel:
    """Return the configured LLM for the given agent.

    Reads from env vars, falls back to DEFAULT_MODEL, then to openai/gpt-4o.
    """
    model_string = _resolve_model_string(agent_name)
    return _build_llm(model_string, temperature)
