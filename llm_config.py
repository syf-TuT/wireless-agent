# LLM Configuration Module
# Supports multiple LLM providers: DeepSeek, MiniMax, OpenAI, etc.

import os
from langchain_openai import ChatOpenAI

# Default to DeepSeek if not specified
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "glm-5").lower()

# LLM Model Configurations
LLM_CONFIGS = {
    # "deepseek": {
    #     "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    #     "base_url": "https://api.deepseek.com",
    #     "model": "deepseek-chat",
    #     "temperature": 0
    # },
    "minimax-m2": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2"),
        "temperature": 0
    },
    "minimax-m2.1": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.1"),
        "temperature": 0
    },
    "minimax-m2.5": {
            "api_key": os.getenv("MINIMAX_API_KEY", ""),
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.7-highspeed",
            "temperature": 0
    },
    # "glm-4.7": {
    #     "api_key": "sk-sp-3e78c46552b54cc79e87686fb28a0475",
    #     "base_url": "https://coding.dashscope.aliyuncs.com/v1",
    #     "model": "glm-4.7",
    #     "temperature": 0
    # },
    "glm-4.7": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "temperature": 0
    },
    "glm-5": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "temperature": 0
    },
    # "glm-5": {
    #     "api_key": "sk-sp-3e78c46552b54cc79e87686fb28a0475",
    #     "base_url": "https://coding.dashscope.aliyuncs.com/v1",
    #     "model": "glm-5",
    #     "temperature": 0
    # },
    "qwen3-coder-next": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "temperature": 0
    },
    "qwen3-coder-plus": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "temperature": 0
    },
    # "kimi-k2.5": {
    #     "api_key": "ms-2a1c196e-a453-4e1c-90c1-cc228772cfe4",
    #     "base_url": "https://api-inference.modelscope.cn/v1",
    #     "model": "moonshotai/Kimi-K2.5",
    #     "temperature": 0
    # },
    "kimi-k2.5": {
        "api_key": os.getenv("MINIMAX_API_KEY", ""),
        "base_url": "https://api.minimaxi.com/v1",
        "model": os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        "temperature": 0
    },
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
