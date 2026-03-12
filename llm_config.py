# LLM Configuration Module
# Supports multiple LLM providers: DeepSeek, MiniMax, OpenAI, etc.

import os
from langchain_openai import ChatOpenAI

# Default to DeepSeek if not specified
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").lower()

# LLM Model Configurations
LLM_CONFIGS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0
    },
    "minimax": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.5-highspeed"),
        "temperature": 0
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": "https://api.openai.com/v1",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "temperature": 0
    },
    "azure_openai": {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
        "base_url": os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        "model": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
        "temperature": 0,
        "api_version": "2024-02-01"
    }
}


def get_llm(provider: str = None, **kwargs):
    """Get LLM instance based on provider configuration.

    Args:
        provider: LLM provider name. If None, uses LLM_PROVIDER env var or default.
        **kwargs: Override any default config values.

    Returns:
        ChatOpenAI instance

    Examples:
        # Use default provider from env
        llm = get_llm()

        # Explicitly specify provider
        llm = get_llm("minimax")

        # Override model or other params
        llm = get_llm("minimax", model="abab6.5g-chat", temperature=0.5)
    """
    provider = (provider or LLM_PROVIDER).lower()

    if provider not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(LLM_CONFIGS.keys())}")

    config = LLM_CONFIGS[provider].copy()
    config.update(kwargs)

    # Validate required config
    if not config.get("api_key"):
        raise ValueError(f"API key not configured for {provider}. Set {provider.upper()}_API_KEY environment variable.")

    return ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        temperature=config.get("temperature", 0)
    )


# Default LLM instance
llm = get_llm()
