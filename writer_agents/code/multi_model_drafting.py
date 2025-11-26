"""Parallel draft generation from multiple models."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Awaitable
from dataclasses import dataclass

from semantic_kernel import Kernel
from sk_config import SKConfig, create_sk_kernel
from .sk_compat import get_chat_components

logger = logging.getLogger(__name__)


@dataclass
class DraftResult:
    """Result from a single model draft generation."""
    model_name: str
    content: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


class ParallelDraftGenerator:
    """Generates drafts from multiple models simultaneously."""

    def __init__(
        self,
        primary_model: str = "phi3:mini",
        secondary_model: str = "qwen2.5:14b",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        max_retries: int = 1,
    ):
        """
        Initialize parallel draft generator.

        Args:
            primary_model: Primary model name (e.g., "phi3:mini")
            secondary_model: Secondary model name (e.g., "qwen2.5:14b")
            ollama_base_url: Ollama server URL
            temperature: Temperature for generation
            max_tokens: Maximum tokens per model
        """
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.ollama_base_url = ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)

        # Create SK kernels for each model
        self.primary_kernel: Optional[Kernel] = None
        self.secondary_kernel: Optional[Kernel] = None
        self._initialize_kernels()

    def _initialize_kernels(self) -> None:
        """Initialize SK kernels for both models."""
        try:
            # Primary model kernel (Phi-3)
            primary_config = SKConfig(
                use_local=True,
                local_model=self.primary_model,
                local_base_url=self.ollama_base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.primary_kernel = create_sk_kernel(primary_config)
            logger.info(f"Initialized primary model kernel: {self.primary_model}")

            # Secondary model kernel (Qwen2.5)
            secondary_config = SKConfig(
                use_local=True,
                local_model=self.secondary_model,
                local_base_url=self.ollama_base_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            self.secondary_kernel = create_sk_kernel(secondary_config)
            logger.info(f"Initialized secondary model kernel: {self.secondary_model}")

        except Exception as e:
            logger.error(f"Failed to initialize kernels: {e}")
            self.primary_kernel = None
            self.secondary_kernel = None

    async def generate_parallel_drafts(
        self,
        prompt: str,
        context: Dict[str, Any],
        timeout: float = 300.0  # 5 minutes timeout per model
    ) -> Tuple[DraftResult, DraftResult]:
        """
        Generate drafts from both models in parallel.

        Args:
            prompt: Generation prompt
            context: Additional context for generation
            timeout: Timeout in seconds for each model

        Returns:
            Tuple of (primary_draft_result, secondary_draft_result)
        """
        logger.info("Starting parallel draft generation...")

        pending = {"primary": True, "secondary": True}
        model_names = {"primary": self.primary_model, "secondary": self.secondary_model}
        results: Dict[str, Optional[DraftResult]] = {"primary": None, "secondary": None}
        last_errors: Dict[str, BaseException] = {}

        attempt = 0
        max_attempts = max(1, self.max_retries + 1)

        while any(pending.values()) and attempt < max_attempts:
            attempt += 1
            logger.debug(f"Parallel generation attempt {attempt}/{max_attempts}")

            coroutines: List[Awaitable[DraftResult]] = []
            task_labels: List[str] = []

            if pending["primary"]:
                coroutines.append(
                    self._generate_draft(
                        kernel=self.primary_kernel,
                        model_name=self.primary_model,
                        prompt=prompt,
                        context=context,
                        timeout=timeout
                    )
                )
                task_labels.append("primary")

            if pending["secondary"]:
                coroutines.append(
                    self._generate_draft(
                        kernel=self.secondary_kernel,
                        model_name=self.secondary_model,
                        prompt=prompt,
                        context=context,
                        timeout=timeout
                    )
                )
                task_labels.append("secondary")

            gather_results = await asyncio.gather(*coroutines, return_exceptions=True)

            for label, result in zip(task_labels, gather_results):
                if isinstance(result, Exception):
                    last_errors[label] = result
                    if self._should_retry(result) and attempt < max_attempts:
                        logger.warning(
                            f"{model_names[label]} transient failure ({result}); retrying "
                            f"({attempt}/{max_attempts})"
                        )
                        continue

                    logger.error(f"{model_names[label]} failed: {result}")
                    results[label] = self._build_error_result(model_names[label], result)
                    pending[label] = False
                else:
                    results[label] = result
                    pending[label] = False

        # Populate any remaining slots with error results
        for label in ("primary", "secondary"):
            if results[label] is None:
                error = last_errors.get(label, RuntimeError("Draft generation failed"))
                results[label] = self._build_error_result(model_names[label], error)

        primary_result = results["primary"]
        secondary_result = results["secondary"]

        logger.info(
            "Parallel generation complete. Primary: %d chars, Secondary: %d chars",
            len(primary_result.content) if primary_result.content else 0,
            len(secondary_result.content) if secondary_result.content else 0,
        )
        return primary_result, secondary_result

    async def _generate_draft(
        self,
        kernel: Optional[Kernel],
        model_name: str,
        prompt: str,
        context: Dict[str, Any],
        timeout: float
    ) -> DraftResult:
        """
        Generate draft from a single model.

        Args:
            kernel: SK kernel for the model
            model_name: Name of the model
            prompt: Generation prompt
            context: Additional context
            timeout: Timeout in seconds

        Returns:
            DraftResult from the model
        """
        if kernel is None:
            return DraftResult(
                model_name=model_name,
                content="",
                metadata={"error": "Kernel not initialized"},
                error="Kernel not initialized"
            )

        try:
            # Build full prompt with context
            full_prompt = self._build_prompt(prompt, context)

            # Generate using SK kernel's chat service
            # Use asyncio.wait_for for timeout
            result = await asyncio.wait_for(
                self._invoke_kernel_generation(kernel, full_prompt),
                timeout=timeout
            )

            return DraftResult(
                model_name=model_name,
                content=result,
                metadata={
                    "model": model_name,
                    "prompt_length": len(full_prompt),
                    "output_length": len(result),
                    "success": True
                }
            )

        except asyncio.TimeoutError:
            logger.error(f"{model_name} generation timed out after {timeout}s")
            return DraftResult(
                model_name=model_name,
                content="",
                metadata={"timeout": timeout},
                error=f"Generation timed out after {timeout} seconds"
            )
        except Exception as e:
            logger.error(f"{model_name} generation failed: {e}")
            return DraftResult(
                model_name=model_name,
                content="",
                metadata={"error": str(e)},
                error=str(e)
            )

    async def _invoke_kernel_generation(self, kernel: Kernel, prompt: str) -> str:
        """
        Invoke SK kernel to generate text using chat service.

        Args:
            kernel: SK kernel instance
            prompt: Generation prompt

        Returns:
            Generated text
        """
        try:
            # Get chat service from kernel (same approach as WorkflowOrchestrator)
            chat_service = None
            try:
                # Try different methods to get chat service
                # Method 1: Try get_service without type_id (newer API)
                if hasattr(kernel, 'get_service'):
                    try:
                        # Try with service_id parameter
                        chat_service = kernel.get_service(service_id=list(kernel.services.keys())[0]) if hasattr(kernel, 'services') and kernel.services else None
                    except (TypeError, AttributeError, IndexError):
                        pass

                # Method 2: Try get_service with type_id (older API)
                if not chat_service:
                    try:
                        chat_service = kernel.get_service(type_id="chat_completion")
                    except (AttributeError, KeyError, ValueError, TypeError):
                        pass

                # Method 3: Get first service directly
                if not chat_service:
                    services = getattr(kernel, 'services', {})
                    if services:
                        if isinstance(services, dict):
                            chat_service = list(services.values())[0]
                        elif hasattr(services, '__iter__'):
                            chat_service = next(iter(services)) if services else None
            except Exception:
                pass

            if not chat_service:
                raise RuntimeError("No chat service available in kernel")

            # Use chat service to generate (same as _generate_motion_with_llm)
            ChatHistory, _, _ = get_chat_components()

            chat_history = ChatHistory()
            chat_history.add_user_message(prompt)

            response = await chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=kernel.get_prompt_execution_settings_from_service_id(
                    service_id=chat_service.service_id
                ) if hasattr(kernel, 'get_prompt_execution_settings_from_service_id') else None
            )

            # Extract content from response
            if isinstance(response, list) and len(response) > 0:
                message = response[0]
                if hasattr(message, 'content'):
                    return str(message.content)
                elif hasattr(message, 'text'):
                    return str(message.text)
                else:
                    return str(message)
            elif hasattr(response, 'content'):
                return str(response.content)
            elif hasattr(response, 'value'):
                return str(response.value)
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Failed to invoke kernel generation: {e}")
            raise

    @staticmethod
    def _should_retry(error: BaseException) -> bool:
        """Return True when the error looks transient."""
        transient_types = (asyncio.TimeoutError, ConnectionError, OSError)
        return isinstance(error, transient_types)

    def _build_error_result(self, model_name: str, error: BaseException) -> DraftResult:
        """Create a placeholder DraftResult for failed generations."""
        return DraftResult(
            model_name=model_name,
            content="",
            metadata={"error": str(error)},
            error=str(error)
        )

    def _build_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """Build full prompt with context."""
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items() if v])
        return f"{prompt}\n\nContext:\n{context_str}"


__all__ = ["ParallelDraftGenerator", "DraftResult"]
