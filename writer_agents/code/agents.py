"""Wrappers for Microsoft AutoGen agents used in the writing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)


try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.openai import OpenAIChatCompletionClient
except ImportError: # pragma: no cover - optional dependency
    AssistantAgent = None # type: ignore
    OpenAIChatCompletionClient = None # type: ignore

# Import Ollama support
try:
    # Try relative import first (if in same package)
    try:
        from .ollama_client import OllamaConfig, OllamaChatClient, find_available_model, check_ollama_server
    except ImportError:
        # Try absolute import
        from code.ollama_client import OllamaConfig, OllamaChatClient, find_available_model, check_ollama_server
    OLLAMA_AVAILABLE = True
except ImportError as e:
    OllamaConfig = None  # type: ignore
    OllamaChatClient = None  # type: ignore
    find_available_model = None  # type: ignore
    check_ollama_server = None  # type: ignore
    OLLAMA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.debug(f"Ollama client not available: {e}")


class AutoGenUnavailableError(RuntimeError):
    """Raised when AutoGen is not installed but the pipeline is invoked."""


@dataclass(slots=True)
class ModelConfig:
    """Configuration for shared model clients."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 4096
    use_local: bool = True  # Default to local (Ollama); OpenAI is backup
    local_model: str = "qwen2.5:14b"  # Ollama model name (better reasoning for review)
    local_base_url: str = "http://localhost:11434"  # Ollama server URL

    # Model selection for different tasks
    review_model: Optional[str] = None  # If None, uses local_model or model
    review_use_local: Optional[bool] = None  # If None, uses use_local


class AgentFactory:
    """Creates and manages AutoGen agents with a shared model client."""

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        self._config = config or ModelConfig()
        self._client: Optional[Any] = None
        # Environment override: MATRIX_USE_LOCAL=true/false
        env_toggle = os.environ.get("MATRIX_USE_LOCAL")
        if env_toggle is not None:
            val = env_toggle.strip().lower()
            self._config.use_local = val in {"1", "true", "yes", "on"}
        self._use_local = self._config.use_local
        # Optional env for local model/base URL
        env_model = os.environ.get("MATRIX_LOCAL_MODEL")
        if env_model:
            self._config.local_model = env_model.strip()
        env_url = os.environ.get("MATRIX_LOCAL_URL")
        if env_url:
            self._config.local_base_url = env_url.strip()
        # Ensure review client attr exists
        self._review_client: Optional[Any] = None

    def _ensure_support(self) -> None:
        if AssistantAgent is None:
            raise AutoGenUnavailableError(
                "Microsoft AutoGen is required. Install 'autogen-agentchat' and 'autogen-ext'."
            )

        # Do not hard-fail here; fallback logic in client() will handle unavailability
        if not self._use_local and OpenAIChatCompletionClient is None:
            raise AutoGenUnavailableError(
                "OpenAI client required. Install 'autogen-ext' or enable local models."
            )

    def client(self):
        """Return a lazily constructed shared model client (OpenAI or Ollama)."""
        self._ensure_support()

        if self._client is None:
            if self._use_local:
                # Prefer Ollama; fallback to OpenAI if unavailable
                try:
                    if not OLLAMA_AVAILABLE:
                        raise RuntimeError("Ollama not installed")
                    ollama_config = OllamaConfig(
                        model=self._config.local_model,
                        base_url=self._config.local_base_url,
                        temperature=self._config.temperature,
                        max_tokens=self._config.max_tokens
                    )
                    logger.info(f"Attempting to create Ollama client with model={self._config.local_model}, base_url={self._config.local_base_url}")
                    self._client = OllamaChatClient(ollama_config)
                    logger.info(f"Successfully created Ollama client: {type(self._client)}")
                except Exception as e:
                    # Log the error before falling back
                    logger.warning(f"Ollama client creation failed: {e}. Falling back to OpenAI.")
                    import traceback
                    logger.debug(f"Ollama client creation traceback: {traceback.format_exc()}")
                    # Fallback to OpenAI if possible
                    if OpenAIChatCompletionClient is None:
                        raise RuntimeError(f"Local model unavailable and OpenAI client missing: {e}")
                    # Load API key for fallback
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        # Try loading from config file
                        try:
                            from pathlib import Path
                            config_path = Path(__file__).resolve().parents[2] / "config" / "OpenaiConfig.json"
                            if config_path.exists():
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                    api_key = config.get("openai_api_key")
                        except Exception:
                            pass
                    if not api_key:
                        raise RuntimeError(
                            f"Local model unavailable ({e}) and no OpenAI API key found. "
                            "Set OPENAI_API_KEY environment variable or ensure config/OpenaiConfig.json exists."
                        )
                    logger.warning(f"Using OpenAI fallback with model={self._config.model}")
                    self._client = OpenAIChatCompletionClient(
                        model=self._config.model,
                        api_key=api_key,
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.max_tokens,
                    )
            else:
                # Use OpenAI explicitly
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    # Try loading from config file
                    try:
                        from pathlib import Path
                        config_path = Path(__file__).resolve().parents[2] / "config" / "OpenaiConfig.json"
                        if config_path.exists():
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                api_key = config.get("openai_api_key")
                    except Exception:
                        pass
                if not api_key:
                    raise RuntimeError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable or ensure config/OpenaiConfig.json exists."
                    )
                self._client = OpenAIChatCompletionClient(
                    model=self._config.model,
                    api_key=api_key,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_tokens,
                )

        return self._client

    def review_client(self):
        """Return a lazily constructed model client for review tasks (defaults to Qwen2.5 for better reasoning)."""
        self._ensure_support()

        if self._review_client is None:
            # Use review model if specified, otherwise use default
            use_local_review = self._config.review_use_local if self._config.review_use_local is not None else self._config.use_local
            review_model = self._config.review_model or self._config.local_model

            if use_local_review:
                # Prefer Ollama; fallback to OpenAI
                try:
                    if not OLLAMA_AVAILABLE:
                        raise RuntimeError("Ollama not installed")
                    ollama_config = OllamaConfig(
                        model=review_model,
                        base_url=self._config.local_base_url,
                        temperature=self._config.temperature,
                        max_tokens=self._config.max_tokens
                    )
                    self._review_client = OllamaChatClient(ollama_config)
                except Exception as e:
                    if OpenAIChatCompletionClient is None:
                        raise RuntimeError(f"Local review model unavailable and OpenAI client missing: {e}")
                    self._review_client = OpenAIChatCompletionClient(
                        model=self._config.model,
                        temperature=self._config.temperature,
                        max_output_tokens=self._config.max_tokens,
                    )
            else:
                # Use OpenAI for review explicitly
                if OpenAIChatCompletionClient is None:
                    raise AutoGenUnavailableError("OpenAI client required for review. Install 'autogen-ext' or enable local models.")
                self._review_client = OpenAIChatCompletionClient(
                    model=self._config.model,
                    temperature=self._config.temperature,
                    max_output_tokens=self._config.max_tokens,
                )

        return self._review_client

    def create(self, name: str, system_message: str, use_review_model: bool = False) -> AssistantAgent:
        """Return a configured AutoGen assistant agent.

        Args:
            name: Agent name
            system_message: System message for the agent
            use_review_model: If True, use review model (Qwen2.5) instead of default model
        """
        self._ensure_support()
        client = self.review_client() if use_review_model else self.client()
        return AssistantAgent(name, system_message=system_message, model_client=client)

    async def close(self) -> None:
        """Close the underlying model client if it was created."""
        if self._client is not None:
            await self._client.close()
            self._client = None


class BaseAutoGenAgent:
    """Convenience wrapper to expose a uniform async interface."""

    def __init__(self, factory: AgentFactory, name: str, system_message: str) -> None:
        self._inner = factory.create(name=name, system_message=system_message)

    async def run(self, task: str) -> str:
        """Execute the agent with a given task prompt."""
        result = await self._inner.run(task=task)
        # Extract content from TaskResult if needed
        if hasattr(result, 'content'):
            return result.content
        elif hasattr(result, 'text'):
            return result.text
        elif isinstance(result, str):
            return result
        elif hasattr(result, 'messages') and result.messages:
            # Extract content from the last message
            last_message = result.messages[-1]
            if hasattr(last_message, 'content'):
                return last_message.content
            else:
                return str(last_message)
        else:
            return str(result)


class PlannerAgent(BaseAutoGenAgent):
    """Planner agent responsible for producing section plans."""


class WriterAgent(BaseAutoGenAgent):
    """Writer agent responsible for drafting prose."""


class EditorAgent(BaseAutoGenAgent):
    """Editor agent responsible for holistic review."""


class DoubleCheckerAgent(BaseAutoGenAgent):
    """Double checker agent that focuses on factual validation."""


class StylistAgent(BaseAutoGenAgent):
    """Optional stylist agent that suggests refinements before editing."""


__all__ = [
    "AgentFactory",
    "PlannerAgent",
    "WriterAgent",
    "EditorAgent",
    "DoubleCheckerAgent",
    "StylistAgent",
    "ModelConfig",
    "AutoGenUnavailableError",
]
