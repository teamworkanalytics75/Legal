#!/usr/bin/env python3
"""
Workaround script for Cursor multi-agent execution.

If Cursor's UI multi-agent feature is broken, this script allows you to
run multiple agents in parallel and get their results, which you can then
use in Cursor manually.

Usage:
    python scripts/cursor_multi_agent_workaround.py "your task here" --agents 2
"""

import os
import sys
import asyncio
import argparse
import json
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


class CursorAgent:
    """Agent that simulates Cursor's agent execution."""

    def __init__(self, agent_id: int, task: str, model: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.task = task
        self.model = model
        self.status = "idle"
        self.result = None
        self.start_time = None
        self.end_time = None
        self.error = None

    async def execute(self) -> Dict[str, Any]:
        """Execute the agent task."""
        self.status = "running"
        self.start_time = datetime.now()

        print_info(f"Agent {self.agent_id}: Starting...")

        try:
            # Check environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise Exception("OPENAI_API_KEY not set")

            # Import OpenAI
            try:
                import openai
            except ImportError:
                raise Exception("openai library not installed. Run: pip install openai")

            # Create client
            client = openai.OpenAI(api_key=api_key)

            # Execute task
            print_info(f"Agent {self.agent_id}: Processing task...")

            # Split task for different agents
            agent_specific_task = self._get_agent_specific_task()

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are Agent {self.agent_id} working on a multi-agent task. "
                                 f"Focus on your specific part of the task and provide clear, actionable results."
                    },
                    {
                        "role": "user",
                        "content": agent_specific_task
                    }
                ],
                temperature=0.7,
                max_tokens=2000
            )

            result_text = response.choices[0].message.content

            self.status = "completed"
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()

            self.result = {
                "agent_id": self.agent_id,
                "task": self.task,
                "agent_specific_task": agent_specific_task,
                "status": "completed",
                "duration": duration,
                "result": result_text,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
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

    def _get_agent_specific_task(self) -> str:
        """Get agent-specific version of the task."""
        if self.agent_id == 1:
            return f"""Task: {self.task}

You are Agent 1. Focus on:
- Understanding the requirements
- Breaking down the problem
- Identifying key components

Provide a structured analysis and approach."""

        elif self.agent_id == 2:
            return f"""Task: {self.task}

You are Agent 2. Focus on:
- Implementation details
- Code or solution approach
- Practical considerations

Provide concrete implementation suggestions."""

        elif self.agent_id == 3:
            return f"""Task: {self.task}

You are Agent 3. Focus on:
- Testing and validation
- Edge cases
- Quality assurance

Provide testing strategy and validation approach."""

        else:
            return f"""Task: {self.task}

You are Agent {self.agent_id}. Provide your perspective and analysis on this task."""


async def run_multi_agent_workaround(task: str, num_agents: int = 2, model: str = "gpt-4o-mini", output_file: str = None):
    """Run multiple agents in parallel as a workaround for Cursor's broken UI."""
    print_header("Cursor Multi-Agent Workaround")

    print_info(f"Task: {task}")
    print_info(f"Number of agents: {num_agents}")
    print_info(f"Model: {model}")
    print()

    # Create agents
    agents = [
        CursorAgent(i + 1, task, model)
        for i in range(num_agents)
    ]

    print_info("Starting parallel execution...\n")

    start_time = datetime.now()

    # Run all agents in parallel
    results = await asyncio.gather(
        *[agent.execute() for agent in agents],
        return_exceptions=True
    )

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Process results
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "completed"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "failed" or isinstance(r, Exception)]

    # Print results
    print_header("Results")

    print_info(f"Total duration: {total_duration:.2f}s")
    print_info(f"Successful: {len(successful)}/{num_agents}")
    print_info(f"Failed: {len(failed)}/{num_agents}")
    print()

    # Print individual results
    for result in results:
        if isinstance(result, Exception):
            print_error(f"Agent exception: {result}")
        elif isinstance(result, dict):
            agent_id = result.get("agent_id", "?")
            status = result.get("status", "unknown")

            if status == "completed":
                duration = result.get("duration", 0)
                result_text = result.get("result", "")
                print_success(f"\nAgent {agent_id} (Completed in {duration:.2f}s):")
                print("-" * 70)
                print(result_text[:500] + ("..." if len(result_text) > 500 else ""))
                print("-" * 70)
            else:
                error = result.get("error", "Unknown error")
                print_error(f"Agent {agent_id}: {error}")

    # Save results to file
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(os.getcwd()) / f"cursor_agent_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "task": task,
        "num_agents": num_agents,
        "model": model,
        "total_duration": total_duration,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }

    try:
        output_path.write_text(json.dumps(output_data, indent=2))
        print_success(f"\nResults saved to: {output_path}")
    except Exception as e:
        print_error(f"Could not save results: {e}")

    # Summary
    print_header("Summary")

    if len(successful) == num_agents:
        print_success("✅ All agents completed successfully!")
        print("\n💡 You can now:")
        print("   1. Review the results above")
        print("   2. Check the saved JSON file for full results")
        print("   3. Use the results in Cursor manually")
    else:
        print_error(f"❌ {len(failed)} agent(s) failed")
        print("\n💡 Check the errors above and:")
        print("   1. Verify OPENAI_API_KEY is set")
        print("   2. Check your internet connection")
        print("   3. Ensure openai library is installed: pip install openai")

    return output_data


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Workaround for Cursor multi-agent execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 2 agents on a task
  python scripts/cursor_multi_agent_workaround.py "Analyze this code and suggest improvements" --agents 2

  # Run 3 agents with custom model
  python scripts/cursor_multi_agent_workaround.py "Write a function to parse JSON" --agents 3 --model gpt-4o

  # Save results to specific file
  python scripts/cursor_multi_agent_workaround.py "Task here" --output results.json
        """
    )

    parser.add_argument(
        "task",
        help="The task for the agents to work on"
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=2,
        help="Number of agents to run in parallel (default: 2)"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--output",
        help="Output file path for results (default: auto-generated)"
    )

    args = parser.parse_args()

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print_error("OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Run workaround
    try:
        asyncio.run(run_multi_agent_workaround(
            task=args.task,
            num_agents=args.agents,
            model=args.model,
            output_file=args.output
        ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

