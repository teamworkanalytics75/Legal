"""Semantic Kernel configuration and initialization for hybrid orchestration."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_plugins import TextPlugin, TimePlugin
from semantic_kernel.memory import VolatileMemoryStore

# Try to import Ollama connector
try:
    from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
    OLLAMA_AVAILABLE = True
except ImportError:
    OllamaChatCompletion = None  # type: ignore
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _load_openai_api_key() -> str:
    """Load OpenAI API key from file or environment."""
    # Try file first
    key_file = Path(".openai_api_key.txt")
    if key_file.exists():
        try:
            api_key = key_file.read_text(encoding="utf-8").strip()
            if api_key and len(api_key) > 20:
                logger.info("Loaded OpenAI API key from .openai_api_key.txt")
                return api_key
        except Exception as e:
            logger.warning(f"Could not read API key from file: {e}")

    # Try environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.info("Using OpenAI API key from environment variable")
        return api_key

    # Don't raise error if we're using local models
    # Return empty string instead - will be ignored if use_local=True
    logger.warning(
        "OpenAI API key not found. Set OPENAI_API_KEY or create .openai_api_key.txt. "
        "Or enable local models with use_local=True"
    )
    return ""


class SKConfig:
    """Configuration for Semantic Kernel setup."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        api_key: Optional[str] = None,
        enable_memory: bool = True,
        memory_store: Optional[VolatileMemoryStore] = None,
        use_local: bool = True,  # Default to local LLM (Ollama) instead of OpenAI
        local_model: str = "qwen2.5:14b",  # Default: Qwen2.5 14B (better reasoning for legal writing)
        local_base_url: str = "http://localhost:11434"  # Ollama server URL
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Environment override: MATRIX_USE_LOCAL, MATRIX_LOCAL_MODEL, MATRIX_LOCAL_URL
        env_toggle = os.environ.get("MATRIX_USE_LOCAL")
        if env_toggle is not None:
            use_local = env_toggle.strip().lower() in {"1", "true", "yes", "on"}
        self.use_local = use_local
        self.local_model = os.environ.get("MATRIX_LOCAL_MODEL", local_model)
        self.local_base_url = os.environ.get("MATRIX_LOCAL_URL", local_base_url)

        # Only load OpenAI API key if not using local
        if not use_local:
            self.api_key = api_key or _load_openai_api_key()
        else:
            self.api_key = None

        self.enable_memory = enable_memory
        self.memory_store = memory_store


def create_sk_kernel(config: Optional[SKConfig] = None) -> Kernel:
    """
    Create and configure a Semantic Kernel instance.

    Args:
        config: SK configuration. If None, uses default settings.

    Returns:
        Configured Kernel instance ready for plugin registration.

    Example:
        >>> kernel = create_sk_kernel()
        >>> result = await kernel.invoke_function(
        ...     plugin_name="DraftingPlugin",
        ...     function_name="PrivacyHarmFunction",
        ...     variables={"evidence": {...}}
        ... )
    """
    config = config or SKConfig()

    # Initialize kernel
    kernel = Kernel()

    # Add chat service (OpenAI or Ollama)
    if config.use_local:
        logger.info(f"Creating Semantic Kernel with Ollama model: {config.local_model}")
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama connector not available; falling back to OpenAI for SK chat service")
            api_key = config.api_key or _load_openai_api_key()
            if not api_key:
                raise RuntimeError("Neither local Ollama nor OpenAI API key available for SK")
            chat_service = OpenAIChatCompletion(
                ai_model_id=config.model_name,
                api_key=api_key,
            )
        else:
            try:
                # Use minimal constructor to avoid version-specific kwargs
                chat_service = OllamaChatCompletion(
                    ai_model_id=config.local_model,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama ({e}); falling back to OpenAI for SK chat service")
                api_key = config.api_key or _load_openai_api_key()
                if not api_key:
                    raise RuntimeError("Neither local Ollama nor OpenAI API key available for SK")
                chat_service = OpenAIChatCompletion(
                    ai_model_id=config.model_name,
                    api_key=api_key,
                )
    else:
        logger.info(f"Creating Semantic Kernel with OpenAI model: {config.model_name}")
        chat_service = OpenAIChatCompletion(
            ai_model_id=config.model_name,
            api_key=config.api_key or _load_openai_api_key(),
        )

    kernel.add_service(chat_service)

    # Add core plugins
    kernel.add_plugin(TextPlugin(), plugin_name="text")
    kernel.add_plugin(TimePlugin(), plugin_name="time")

    # Add memory store if configured
    if config.enable_memory and config.memory_store:
        kernel.register_memory_store(config.memory_store)
        logger.info("Memory store registered with kernel")

    logger.info("Semantic Kernel initialized successfully")
    return kernel


def create_sk_kernel_with_chroma(
    chroma_persist_directory: str,
    config: Optional[SKConfig] = None
) -> Kernel:
    """
    Create SK kernel with ChromaDB memory integration.

    Args:
        chroma_persist_directory: Path to ChromaDB persistence directory
        config: SK configuration

    Returns:
        Kernel with ChromaDB memory store configured

    Example:
        >>> kernel = create_sk_kernel_with_chroma(
        ...     chroma_persist_directory="path/to/chroma/collections"
        ... )
        >>> # Now SK functions can query Chroma collections
    """
    try:
        from semantic_kernel.memory import ChromaMemoryStore

        # Create Chroma memory store
        memory_store = ChromaMemoryStore(
            persist_directory=chroma_persist_directory
        )

        # Update config with memory store
        if config is None:
            config = SKConfig()
        config.memory_store = memory_store

        logger.info(f"ChromaDB memory store created at: {chroma_persist_directory}")

        return create_sk_kernel(config)

    except ImportError:
        logger.warning("ChromaDB not available. Creating kernel without memory store.")
        return create_sk_kernel(config)


def get_default_sk_config(use_local: bool = True) -> SKConfig:
    """Get default SK configuration optimized for legal document drafting."""
    if use_local:
        return SKConfig(
            use_local=True,
            local_model="qwen2.5:14b",  # Qwen2.5 14B - better reasoning for legal briefs
            temperature=0.3,      # Lower for more consistent legal writing
            max_tokens=4000,      # Sufficient for section drafting
            enable_memory=True
        )
    else:
        return SKConfig(
            model_name="gpt-4o",  # Best for complex reasoning
            temperature=0.3,      # Lower for more consistent legal writing
            max_tokens=4000,      # Sufficient for section drafting
            enable_memory=True
        )


def get_exploration_sk_config(use_local: bool = True) -> SKConfig:
    """Get SK configuration optimized for creative exploration."""
    if use_local:
        return SKConfig(
            use_local=True,
            local_model="qwen2.5:14b",  # Qwen2.5 14B - better reasoning for legal briefs
            temperature=0.7,      # Higher for creative exploration
            max_tokens=2000,      # Shorter for brainstorming
            enable_memory=True
        )
    else:
        return SKConfig(
            model_name="gpt-4o",
            temperature=0.7,      # Higher for creative exploration
            max_tokens=2000,      # Shorter for brainstorming
            enable_memory=True
        )


# Convenience functions for common kernel creation patterns
def create_drafting_kernel() -> Kernel:
    """Create kernel optimized for document drafting (defaults to local LLM)."""
    return create_sk_kernel(get_default_sk_config(use_local=True))


def create_exploration_kernel() -> Kernel:
    """Create kernel optimized for creative exploration (defaults to local LLM)."""
    return create_sk_kernel(get_exploration_sk_config(use_local=True))


def create_kernel_with_existing_chroma() -> Kernel:
    """
    Create kernel with ChromaDB integration using existing collections.

    Looks for Chroma collections in common locations:
    - writer_agents/code/memory_store/
    - case_law_data/
    - temp_enriched_samples/
    """
    # Common Chroma locations in the project
    possible_chroma_paths = [
        "writer_agents/code/memory_store",
        "case_law_data",
        "temp_enriched_samples",
        "memory_store"
    ]

    for path in possible_chroma_paths:
        chroma_path = Path(path)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            logger.info(f"Found ChromaDB at: {chroma_path}")
            return create_sk_kernel_with_chroma(str(chroma_path))

    logger.warning("No ChromaDB collections found. Creating kernel without memory.")
    return create_sk_kernel()


def create_multi_model_sk_config(
    primary_model: str = "phi3:mini",
    secondary_model: str = "qwen2.5:14b",
    ollama_base_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 4000
) -> Dict[str, Kernel]:
    """
    Create SK kernels for multiple models (multi-model ensemble).

    Args:
        primary_model: Primary model name (e.g., "phi3:mini")
        secondary_model: Secondary model name (e.g., "qwen2.5:14b")
        ollama_base_url: Ollama server URL
        temperature: Temperature for generation
        max_tokens: Maximum tokens per model

    Returns:
        Dictionary with 'primary' and 'secondary' kernels
    """
    primary_config = SKConfig(
        use_local=True,
        local_model=primary_model,
        local_base_url=ollama_base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )

    secondary_config = SKConfig(
        use_local=True,
        local_model=secondary_model,
        local_base_url=ollama_base_url,
        temperature=temperature,
        max_tokens=max_tokens
    )

    primary_kernel = create_sk_kernel(primary_config)
    secondary_kernel = create_sk_kernel(secondary_config)

    logger.info(f"Created multi-model SK kernels: {primary_model} + {secondary_model}")

    return {
        "primary": primary_kernel,
        "secondary": secondary_kernel
    }


# Export commonly used functions
__all__ = [
    "SKConfig",
    "create_sk_kernel",
    "create_sk_kernel_with_chroma",
    "create_drafting_kernel",
    "create_exploration_kernel",
    "create_kernel_with_existing_chroma",
    "get_default_sk_config",
    "get_exploration_sk_config",
    "create_multi_model_sk_config"
]
