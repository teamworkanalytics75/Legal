"""Base class for background agents."""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama-python not installed. Install with: pip install ollama-python")


class AgentPriority(Enum):
    """Agent priority levels."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class AgentConfig:
    """Configuration for a background agent."""
    name: str
    model: Optional[str] = None  # None for deterministic agents
    priority: AgentPriority = AgentPriority.MEDIUM
    interval_minutes: Optional[int] = None
    interval_hours: Optional[int] = None
    enabled: bool = True
    max_concurrent_tasks: int = 1
    temperature: float = 0.7
    max_tokens: int = 2000


class BackgroundAgent(ABC):
    """Base class for all background agents."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"agent.{config.name}")
        self._running = False
        self._task_count = 0
        self._error_count = 0
        self._last_checkpoint: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        """Agent name."""
        return self.config.name

    @property
    def is_running(self) -> bool:
        """Check if agent is currently running."""
        return self._running

    @abstractmethod
    async def process(self, task: Any) -> Any:
        """
        Process a task. Must be implemented by subclasses.

        Args:
            task: The task to process

        Returns:
            The result of processing
        """
        pass

    async def initialize(self) -> None:
        """
        Initialize the agent. Override if needed.
        """
        self.logger.info(f"Initializing agent: {self.name}")

    async def shutdown(self) -> None:
        """
        Shutdown the agent. Override if needed.
        """
        self.logger.info(f"Shutting down agent: {self.name}")

    async def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """
        Save checkpoint state.

        Args:
            state: State to checkpoint
        """
        self._last_checkpoint = state
        # Subclasses can override to persist to disk

    async def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint state.

        Returns:
            Saved state if available
        """
        return self._last_checkpoint

    async def llm_query(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Query the LLM using Ollama.

        Args:
            prompt: The prompt to send
            temperature: Temperature override
            max_tokens: Max tokens override
            model: Model override

        Returns:
            The LLM response
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Ollama not available. Install with: pip install ollama-python")

        if not self.config.model and not model:
            raise ValueError(f"Agent {self.name} has no model configured")

        model_name = model or self.config.model
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        try:
            response = await asyncio.to_thread(
                ollama.generate,
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": temp,
                    "num_predict": max_tok,
                }
            )

            return response['response']

        except Exception as e:
            self.logger.error(f"LLM query failed: {e}")
            raise

    async def run_task(self, task: Any) -> Any:
        """
        Run a task with error handling and logging.

        Args:
            task: The task to run

        Returns:
            Result of processing
        """
        self._running = True
        self._task_count += 1

        try:
            self.logger.info(f"Processing task {self._task_count}")
            result = await self.process(task)
            self.logger.info(f"Task {self._task_count} completed successfully")
            return result

        except Exception as e:
            self._error_count += 1
            self.logger.error(f"Task {self._task_count} failed: {e}", exc_info=True)
            raise

        finally:
            self._running = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics.

        Returns:
            Dictionary of stats
        """
        return {
            "name": self.name,
            "running": self._running,
            "total_tasks": self._task_count,
            "errors": self._error_count,
            "success_rate": (
                (self._task_count - self._error_count) / self._task_count
                if self._task_count > 0 else 0.0
            ),
        }

