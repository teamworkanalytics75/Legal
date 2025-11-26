#!/usr/bin/env python3
"""
Performance benchmark for refinement loop.

Measures performance metrics for feature extraction, weak feature identification,
plugin edit request generation, and edit application.
"""

import asyncio
import time
import logging
from pathlib import Path
from typing import Dict, Any, List
import tracemalloc

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce logging for benchmarks
logger = logging.getLogger(__name__)

# Add project root to path
import sys
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from writer_agents.code.sk_plugins.FeaturePlugin.feature_orchestrator import RefinementLoop

# Sample draft text for benchmarking
SAMPLE_DRAFT = """
Motion for Seal and Pseudonym

This motion seeks to protect the privacy of the plaintiff who has been subjected to harassment.
The plaintiff's safety is at risk due to the nature of the allegations.
Retaliation is a concern in this case.
The privacy interests outweigh any public interest in disclosure.

The court should grant this motion because the privacy interests are significant.
The plaintiff has demonstrated a legitimate need for protection.
"""


async def benchmark_feature_extraction(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark feature extraction performance."""
    logger.info("Benchmarking feature extraction...")

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    times = []

    for i in range(iterations):
        start = time.perf_counter()
        await loop.analyze_draft(SAMPLE_DRAFT)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "operation": "feature_extraction",
        "iterations": iterations,
        "avg_time_seconds": avg_time,
        "min_time_seconds": min_time,
        "max_time_seconds": max_time,
        "total_time_seconds": sum(times)
    }


async def benchmark_weak_feature_identification(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark weak feature identification performance."""
    logger.info("Benchmarking weak feature identification...")

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    times = []

    for i in range(iterations):
        start = time.perf_counter()
        weak_features = await loop.analyze_draft(SAMPLE_DRAFT)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    return {
        "operation": "weak_feature_identification",
        "iterations": iterations,
        "avg_time_seconds": avg_time,
        "total_time_seconds": sum(times)
    }


async def benchmark_plugin_edit_requests(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark plugin edit request generation performance."""
    logger.info("Benchmarking plugin edit request generation...")

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    weak_features = {"test": {"current": 1.0, "target": 2.0, "gap": 1.0}}
    context = {"weak_features": weak_features}

    times = []

    for i in range(iterations):
        start = time.perf_counter()
        await loop.collect_edit_requests(SAMPLE_DRAFT, weak_features, context)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    return {
        "operation": "plugin_edit_request_generation",
        "iterations": iterations,
        "avg_time_seconds": avg_time,
        "total_time_seconds": sum(times)
    }


async def benchmark_edit_application(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark edit application performance."""
    logger.info("Benchmarking edit application...")

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    weak_features = {"test": {"current": 1.0, "target": 2.0, "gap": 1.0}}
    context = {"weak_features": weak_features}

    times = []

    for i in range(iterations):
        start = time.perf_counter()
        await loop.strengthen_draft(SAMPLE_DRAFT, weak_features, context)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    return {
        "operation": "edit_application",
        "iterations": iterations,
        "avg_time_seconds": avg_time,
        "total_time_seconds": sum(times)
    }


async def benchmark_refinement_iteration(iterations: int = 3) -> Dict[str, Any]:
    """Benchmark full refinement loop iteration."""
    logger.info("Benchmarking refinement loop iteration...")

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    context = {}

    start = time.perf_counter()
    try:
        result = await loop.run_feedback_loop(SAMPLE_DRAFT, max_iterations=iterations, context=context)
        elapsed = time.perf_counter() - start

        return {
            "operation": "refinement_loop_iteration",
            "iterations_completed": result.get("iterations_completed", 0),
            "total_time_seconds": elapsed,
            "avg_time_per_iteration": elapsed / max(result.get("iterations_completed", 1), 1)
        }
    except Exception as e:
        logger.warning(f"Feedback loop benchmark skipped: {e}")
        return {
            "operation": "refinement_loop_iteration",
            "error": str(e),
            "total_time_seconds": time.perf_counter() - start
        }


def benchmark_memory_usage() -> Dict[str, Any]:
    """Benchmark memory usage."""
    logger.info("Benchmarking memory usage...")

    tracemalloc.start()

    plugins = {}
    loop = RefinementLoop(plugins=plugins)

    # Perform operations
    import asyncio
    weak_features = asyncio.run(loop.analyze_draft(SAMPLE_DRAFT))
    asyncio.run(loop.strengthen_draft(SAMPLE_DRAFT, weak_features, {}))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "operation": "memory_usage",
        "current_memory_mb": current / 1024 / 1024,
        "peak_memory_mb": peak / 1024 / 1024
    }


async def run_all_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("Refinement Loop Performance Benchmarks")
    print("=" * 60)
    print()

    results = []

    # Feature extraction
    print("1. Feature Extraction...")
    result = await benchmark_feature_extraction(iterations=5)
    results.append(result)
    print(f"   Average: {result['avg_time_seconds']:.3f}s")
    print()

    # Weak feature identification
    print("2. Weak Feature Identification...")
    result = await benchmark_weak_feature_identification(iterations=5)
    results.append(result)
    print(f"   Average: {result['avg_time_seconds']:.3f}s")
    print()

    # Plugin edit requests
    print("3. Plugin Edit Request Generation...")
    result = await benchmark_plugin_edit_requests(iterations=5)
    results.append(result)
    print(f"   Average: {result['avg_time_seconds']:.3f}s")
    print()

    # Edit application
    print("4. Edit Application...")
    result = await benchmark_edit_application(iterations=5)
    results.append(result)
    print(f"   Average: {result['avg_time_seconds']:.3f}s")
    print()

    # Refinement iteration
    print("5. Refinement Loop Iteration...")
    result = await benchmark_refinement_iteration(iterations=2)
    results.append(result)
    if "error" not in result:
        print(f"   Total: {result['total_time_seconds']:.3f}s")
        print(f"   Per iteration: {result['avg_time_per_iteration']:.3f}s")
    else:
        print(f"   Skipped: {result.get('error', 'Unknown error')}")
    print()

    # Memory usage
    print("6. Memory Usage...")
    result = benchmark_memory_usage()
    results.append(result)
    print(f"   Current: {result['current_memory_mb']:.2f} MB")
    print(f"   Peak: {result['peak_memory_mb']:.2f} MB")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total_time = sum(r.get('total_time_seconds', 0) for r in results if 'total_time_seconds' in r)
    print(f"Total benchmark time: {total_time:.3f}s")
    print()

    # Performance targets
    print("Performance Targets:")
    print("  - Feature extraction: < 2.0s")
    print("  - Edit application: < 5.0s")
    print("  - Refinement iteration: < 30.0s")
    print()

    return results


if __name__ == "__main__":
    results = asyncio.run(run_all_benchmarks())

