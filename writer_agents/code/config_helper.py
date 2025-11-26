"""Helper functions for configuring local vs OpenAI models."""

from typing import Optional
import os
from agents import ModelConfig
from sk_config import SKConfig


def create_hybrid_config(
    use_local: bool = True,
    local_model: str = "qwen2.5:14b",
    review_use_openai: bool = True  # Use OpenAI for critical review tasks
) -> tuple[ModelConfig, SKConfig]:
    """
    Create hybrid model configuration.

    Args:
        use_local: Use local models for drafting/bulk tasks
        local_model: Ollama model name to use
        review_use_openai: Use OpenAI for final review (recommended)

    Returns:
        Tuple of (ModelConfig, SKConfig) configured for hybrid use
    """
    # Environment override
    env_toggle = os.environ.get("MATRIX_USE_LOCAL")
    if env_toggle is not None:
        use_local = env_toggle.strip().lower() in {"1", "true", "yes", "on"}
    local_model = os.environ.get("MATRIX_LOCAL_MODEL", local_model)

    # AutoGen config - use local for bulk tasks
    autogen_config = ModelConfig(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=4096,
        use_local=use_local,
        local_model=local_model,
        local_base_url="http://localhost:11434"
    )

    # SK config - use local for drafting
    sk_config = SKConfig(
        model_name="gpt-4o",
        temperature=0.3,
        max_tokens=4000,
        use_local=use_local,
        local_model=local_model,
        local_base_url="http://localhost:11434"
    )

    return autogen_config, sk_config


def create_local_only_config(local_model: str = "qwen2.5:14b") -> tuple[ModelConfig, SKConfig]:
    """Create configuration using only local models (100% free)."""
    return create_hybrid_config(
        use_local=True,
        local_model=local_model,
        review_use_openai=False
    )


def create_openai_only_config() -> tuple[ModelConfig, SKConfig]:
    """Create configuration using only OpenAI (highest quality)."""
    return create_hybrid_config(
        use_local=False,
        review_use_openai=True
    )


def check_local_model_available(model: str = "qwen2.5:14b") -> bool:
    """Check if specified local model is available."""
    try:
        from ollama_client import find_available_model
        available = find_available_model(model)
        return available is not None
    except Exception:
        return False
