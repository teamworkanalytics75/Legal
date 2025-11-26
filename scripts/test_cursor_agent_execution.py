#!/usr/bin/env python3
"""
Test script that simulates Cursor multi-agent execution.

This script mimics what Cursor does when you select multiple agents (2x, 3x, etc.)
and tests if they can execute in parallel properly.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


class Colors:
    """ANSI color codes."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


class MockAgent:
    """Mock agent that simulates Cursor agent execution."""

    def __init__(self, agent_id: int, task: str):
        self.agent_id = agent_id
        self.task = task
        self.status = "idle"
        self.result = None
        self.start_time = None
        self.end_time = None
        self.error = None

    async def execute(self) -> Dict[str, Any]:
        """Simulate agent execution."""
        self.status = "running"
        self.start_time = datetime.now()

        print_info(f"Agent {self.agent_id}: Starting task: {self.task[:50]}...")

        try:
            # Simulate work - check environment, access files, etc.
            await self._check_environment()
            await self._access_files()
            await self._simulate_work()

            self.status = "completed"
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            self.result = {
                "agent_id": self.agent_id,
                "task": self.task,
                "status": "completed",
                "duration": duration,
                "output": f"Task completed by agent {self.agent_id}"
            }

            print_success(f"Agent {self.agent_id}: Completed in {duration:.2f}s")
            return self.result

        except Exception as e:
            self.status = "failed"
            self.end_time = datetime.now()
            self.error = str(e)

            print_error(f"Agent {self.agent_id}: Failed - {e}")
            return {
                "agent_id": self.agent_id,
                "task": self.task,
                "status": "failed",
                "error": str(e)
            }

    async def _check_environment(self):
        """Check environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not set")

        workspace = os.getcwd()
        if not workspace:
            raise Exception("Workspace path not available")

        # Small delay to simulate work
        await asyncio.sleep(0.1)

    async def _access_files(self):
        """Test file system access."""
        workspace = Path(os.getcwd())

        # Try to read a file
        test_file = workspace / "README.md"
        if test_file.exists():
            try:
                test_file.read_text()
            except Exception as e:
                raise Exception(f"Cannot read files: {e}")

        # Try to write a test file
        test_write = workspace / f".agent_{self.agent_id}_test"
        try:
            test_write.write_text(f"Test from agent {self.agent_id}")
            test_write.unlink()
        except Exception as e:
            raise Exception(f"Cannot write files: {e}")

        # Small delay
        await asyncio.sleep(0.1)

    async def _simulate_work(self):
        """Simulate actual agent work."""
        # Simulate API call or processing
        await asyncio.sleep(0.5)

        # Test if we can import required libraries
        try:
            import requests
        except ImportError:
            raise Exception("requests library not available")

        # Test OpenAI client creation (if API key is set)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                import openai
                # Just create client, don't make actual call
                client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise Exception("openai library not available")
            except Exception as e:
                # Don't fail on API errors, just log
                print_info(f"Agent {self.agent_id}: OpenAI client creation warning: {e}")


async def run_parallel_agents(num_agents: int, task: str) -> List[Dict[str, Any]]:
    """Run multiple agents in parallel (simulating Cursor's multi-agent mode)."""
    print_header(f"Running {num_agents} Agents in Parallel")

    # Create agents
    agents = [MockAgent(i + 1, f"{task} (Agent {i + 1})") for i in range(num_agents)]

    print_info(f"Created {num_agents} agents")
    print_info(f"Task: {task}")
    print_info("Starting parallel execution...\n")

    start_time = time.time()

    # Run all agents in parallel (this is what Cursor does)
    results = await asyncio.gather(
        *[agent.execute() for agent in agents],
        return_exceptions=True
    )

    end_time = time.time()
    total_duration = end_time - start_time

    # Process results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
    failed = len(results) - successful

    print_header("Execution Results")
    print_info(f"Total duration: {total_duration:.2f}s")
    print_info(f"Successful agents: {successful}/{num_agents}")
    print_info(f"Failed agents: {failed}/{num_agents}")

    if successful == num_agents:
        print_success("All agents completed successfully!")
    elif successful > 0:
        print(f"{Colors.YELLOW}⚠️  Some agents failed{Colors.RESET}")
    else:
        print_error("All agents failed!")

    # Print individual results
    print("\nIndividual Agent Results:")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print_error(f"Agent {i + 1}: Exception - {result}")
        elif isinstance(result, dict):
            status = result.get("status", "unknown")
            if status == "completed":
                duration = result.get("duration", 0)
                print_success(f"Agent {i + 1}: Completed in {duration:.2f}s")
            else:
                error = result.get("error", "Unknown error")
                print_error(f"Agent {i + 1}: Failed - {error}")

    return results


async def test_sequential_execution(num_agents: int, task: str):
    """Test sequential execution for comparison."""
    print_header(f"Running {num_agents} Agents Sequentially (for comparison)")

    agents = [MockAgent(i + 1, f"{task} (Agent {i + 1})") for i in range(num_agents)]

    start_time = time.time()
    results = []

    for agent in agents:
        result = await agent.execute()
        results.append(result)

    end_time = time.time()
    total_duration = end_time - start_time

    print_info(f"Sequential execution duration: {total_duration:.2f}s")

    return results


async def main():
    """Main test function."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print("Cursor Multi-Agent Execution Test")
    print("="*70)
    print(f"{Colors.RESET}\n")

    # Check prerequisites
    print_header("Prerequisites Check")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print_error("OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    else:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print_success(f"OPENAI_API_KEY is set: {masked}")

    workspace = os.getcwd()
    print_success(f"Workspace: {workspace}")

    # Test with 2 agents (most common case)
    task = "Analyze code and provide suggestions"

    print("\n" + "="*70)
    print("Test 1: Parallel Execution (2 agents)")
    print("="*70)
    parallel_results = await run_parallel_agents(2, task)

    # Test with 3 agents
    print("\n" + "="*70)
    print("Test 2: Parallel Execution (3 agents)")
    print("="*70)
    parallel_results_3 = await run_parallel_agents(3, task)

    # Compare with sequential
    print("\n" + "="*70)
    print("Test 3: Sequential Execution (for comparison)")
    print("="*70)
    sequential_results = await test_sequential_execution(2, task)

    # Summary
    print_header("Test Summary")

    parallel_success = sum(1 for r in parallel_results if isinstance(r, dict) and r.get("status") == "completed")
    parallel_3_success = sum(1 for r in parallel_results_3 if isinstance(r, dict) and r.get("status") == "completed")

    if parallel_success == 2 and parallel_3_success == 3:
        print_success("✅ All parallel execution tests passed!")
        print_success("✅ Multi-agent execution is working correctly")
        print("\n💡 If Cursor's UI still shows stuck agents, the issue is likely:")
        print("   - Cursor UI bug (not execution environment)")
        print("   - Agent communication/coordination issue")
        print("   - Cursor cache needs clearing")
    else:
        print_error("❌ Some agents failed during execution")
        print("\n💡 Check the errors above and:")
        print("   1. Run: python scripts/diagnose_cursor_agents.py")
        print("   2. Verify environment variables are set")
        print("   3. Check file permissions")
        print("   4. Ensure workspace is in WSL home directory")

    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

