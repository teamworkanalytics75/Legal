"""Worker pool for concurrent atomic agent execution.

Manages concurrent execution capacity with semaphore-based throttling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List, Optional


class WorkerPool:
    """Manages concurrent worker capacity for atomic agents.

    Uses semaphore to limit parallel execution and prevent resource exhaustion.
    Tracks active workers and provides backpressure when capacity is reached.
    """

    def __init__(self, max_workers: int = 6) -> None:
        """Initialize worker pool.

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active = 0
        self._lock = asyncio.Lock()

    @property
    def available(self) -> int:
        """Get number of available worker slots.

        Returns:
            Number of workers that can be started immediately
        """
        return self.max_workers - self.active

    @property
    def utilization(self) -> float:
        """Get pool utilization percentage.

        Returns:
            Utilization as fraction (0.0 to 1.0)
        """
        return self.active / self.max_workers if self.max_workers > 0 else 0.0

    async def run(self, job: Callable[[], Awaitable[Any]]) -> Any:
        """Run a job with worker pool management.

        Args:
            job: Async callable to execute

        Returns:
            Job result
        """
        async with self.semaphore:
            async with self._lock:
                self.active += 1

            try:
                return await job()
            finally:
                async with self._lock:
                    self.active -= 1

    async def run_batch(
        self,
        jobs: List[Callable[[], Awaitable[Any]]],
        return_exceptions: bool = False,
    ) -> List[Any]:
        """Run a batch of jobs with worker pool management.

        Args:
            jobs: List of async callables to execute
            return_exceptions: If True, exceptions are returned as results

        Returns:
            List of job results
        """
        tasks = [self.run(job) for job in jobs]
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    async def drain(self) -> None:
        """Wait for all active workers to complete."""
        while self.active > 0:
            await asyncio.sleep(0.1)

    def is_full(self) -> bool:
        """Check if pool is at capacity.

        Returns:
            True if no workers available
        """
        return self.available == 0

    def is_empty(self) -> bool:
        """Check if pool is idle.

        Returns:
            True if no active workers
        """
        return self.active == 0

    def __repr__(self) -> str:
        """String representation of pool state."""
        return (
            f"WorkerPool(active={self.active}/{self.max_workers}, "
            f"utilization={self.utilization:.1%})"
        )
