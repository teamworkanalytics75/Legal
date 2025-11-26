"""Ollama client adapter for AutoGen and Semantic Kernel integration."""

import asyncio
import logging
import requests
from typing import Optional, List, Dict, Any, AsyncIterator, Sequence, Union, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# AutoGen imports
try:
    from autogen_core.models import (
        CreateResult,
        LLMMessage,
        RequestUsage,
        FinishReasons,
        SystemMessage,
        UserMessage,
        AssistantMessage,
    )
    AUTOGEN_TYPES_AVAILABLE = True
except ImportError:
    AUTOGEN_TYPES_AVAILABLE = False
    CreateResult = None
    RequestUsage = None
    FinishReasons = None

OLLAMA_AVAILABLE = False
OLLAMA_ASYNC_AVAILABLE = False
try:
    import ollama
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
    OLLAMA_ASYNC_AVAILABLE = True
except ImportError:
    try:
        import ollama
        OLLAMA_AVAILABLE = True
        OLLAMA_ASYNC_AVAILABLE = False
    except ImportError:
        ollama = None
        AsyncClient = None


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: float = 120.0


def check_ollama_server(base_url: str = "http://localhost:11434") -> tuple[bool, List[Dict[str, Any]]]:
    """Check if Ollama server is running and return available models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return True, models
        return False, []
    except Exception as e:
        logger.debug(f"Ollama server check failed: {e}")
        return False, []


def find_available_model(
    preferred_model: str = "qwen2.5:14b",
    base_url: str = "http://localhost:11434"
) -> Optional[str]:
    """Find an available Ollama model, with fallback logic."""
    is_running, models = check_ollama_server(base_url)

    if not is_running:
        logger.warning("Ollama server is not running")
        return None

    if not models:
        logger.warning("No models available in Ollama")
        return None

    model_names = [m.get("name", "") for m in models]

    # Try preferred model first
    if preferred_model in model_names:
        return preferred_model

    # Try partial match
    for name in model_names:
        if preferred_model.split(":")[0] in name:
            logger.info(f"Using similar model: {name} (requested: {preferred_model})")
            return name

    # Try fallback models
    fallback_models = ["qwen", "phi3", "llama3", "mistral"]
    for fallback in fallback_models:
        for name in model_names:
            if fallback.lower() in name.lower():
                logger.info(f"Using fallback model: {name}")
                return name

    # Use first available
    if model_names:
        logger.warning(f"Using first available model: {model_names[0]} (requested: {preferred_model})")
        return model_names[0]

    return None


class OllamaChatClient:
    """Ollama client adapter for chat completions (AutoGen-compatible)."""

    def __init__(self, config: OllamaConfig):
        """Initialize Ollama chat client."""
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install with: pip install ollama")

        self.config = config
        self.model = find_available_model(config.model, config.base_url)

        if not self.model:
            raise RuntimeError(
                f"No Ollama model available. "
                f"Requested: {config.model}, "
                f"Server: {config.base_url}"
            )

        # Add model_info attribute required by AutoGen
        self.model_info = {
            "vision": False,  # Ollama models typically don't support vision
            "function_calling": False,  # Basic function calling support
            "max_tokens": config.max_tokens
        }

        # Create async client for true parallel requests
        if OLLAMA_ASYNC_AVAILABLE:
            self.async_client = AsyncClient(host=config.base_url)
            logger.info(f"Ollama async client initialized for parallel requests")
        else:
            self.async_client = None
            logger.warning("Ollama AsyncClient not available - requests will be queued, not parallel")

        logger.info(f"Ollama client initialized with model: {self.model}")

    async def create(
        self,
        messages: Sequence[Any],  # LLMMessage if AutoGen available, otherwise Any
        *,
        tools: Sequence[Any] = [],
        tool_choice: Any = "auto",
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[Any] = None,
    ) -> Union[Any, Dict[str, Any]]:  # CreateResult if AutoGen available, otherwise Any
        """Create a chat completion (AutoGen-compatible interface).

        Returns CreateResult if AutoGen types are available, otherwise returns dict.
        """
        # Convert AutoGen messages to prompt format
        prompt = self._messages_to_prompt_autogen(messages) if AUTOGEN_TYPES_AVAILABLE else self._messages_to_prompt(messages)

        # Get options
        temperature = extra_create_args.get("temperature", self.config.temperature)
        max_tokens = extra_create_args.get("max_tokens", extra_create_args.get("max_output_tokens", self.config.max_tokens))

        # Call Ollama - use async client for true parallel execution
        try:
            if self.async_client is not None:
                # Use async client for true parallel requests (no queuing)
                response = await self.async_client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    stream=False
                )
                content = response.get("response", "")
            else:
                # Fallback to sync API (will be queued, not parallel)
                response = await asyncio.to_thread(
                    ollama.generate,
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                    stream=False
                )
                content = response.get("response", "")

            # Calculate usage (approximate)
            prompt_tokens = len(prompt.split())
            completion_tokens = len(content.split())

            # Return CreateResult if AutoGen types available
            if AUTOGEN_TYPES_AVAILABLE and CreateResult is not None:
                usage = RequestUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # FinishReasons is a Literal type, so use string directly
                return CreateResult(
                    finish_reason="stop",  # FinishReasons is Literal["stop", "length", ...]
                    content=content,
                    usage=usage,
                    cached=False,
                    logprobs=None,
                    thought=None,
                )
            else:
                # Fallback to dict format
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": "stop"
                    }],
                    "model": self.model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def _messages_to_prompt_autogen(self, messages: Sequence[Any]) -> str:
        """Convert AutoGen LLMMessage objects to a single prompt string."""
        parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                parts.append(f"System: {msg.content}\n")
            elif isinstance(msg, UserMessage):
                # Handle both string and list content
                if isinstance(msg.content, str):
                    parts.append(f"User: {msg.content}\n")
                elif isinstance(msg.content, list):
                    # Handle multimodal content
                    text_parts = [str(item) for item in msg.content if isinstance(item, str)]
                    parts.append(f"User: {' '.join(text_parts)}\n")
                else:
                    parts.append(f"User: {str(msg.content)}\n")
            elif isinstance(msg, AssistantMessage):
                if isinstance(msg.content, str):
                    parts.append(f"Assistant: {msg.content}\n")
                else:
                    parts.append(f"Assistant: {str(msg.content)}\n")
            else:
                # Fallback for other message types
                if hasattr(msg, 'content'):
                    parts.append(f"User: {str(msg.content)}\n")
                else:
                    parts.append(f"User: {str(msg)}\n")

        return "\n".join(parts) + "\nAssistant:"

    def _messages_to_prompt(self, messages: List[Any]) -> str:
        """Convert chat messages to a single prompt string.

        Handles both dict messages and AutoGen Pydantic message objects.
        """
        parts = []
        for msg in messages:
            # Handle Pydantic message objects (AutoGen)
            if hasattr(msg, 'content'):
                content = msg.content
                # Determine role from message type
                msg_type = type(msg).__name__
                if 'System' in msg_type:
                    role = "system"
                elif 'User' in msg_type or 'Text' in msg_type:
                    role = "user"
                elif 'Assistant' in msg_type:
                    role = "assistant"
                else:
                    role = "user"  # Default
            # Handle dict messages (legacy)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # Fallback: convert to string
                role = "user"
                content = str(msg)

            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")

        return "\n".join(parts) + "\nAssistant:"

    async def close(self):
        """Close the client (no-op for Ollama)."""
        pass


class OllamaCompletion:
    """Ollama completion response (AutoGen-compatible)."""

    def __init__(self, content: str, model: str):
        self.content = content
        self.model = model
        self.choices = [type('obj', (object,), {"message": type('obj', (object,), {"content": content})()})()]


def create_ollama_client(config: OllamaConfig) -> OllamaChatClient:
    """Factory function to create Ollama chat client."""
    return OllamaChatClient(config)

