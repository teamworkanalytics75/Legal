#!/usr/bin/env python3
"""
Cursor Auto/Codex Agent Management CLI

Manage multiple Cursor Auto Agents or Codex agents through the terminal.

Usage:
    python3 scripts/manage_cursor_agents.py launch --agents 2 --prompt "your task"
    python3 scripts/manage_cursor_agents.py status
    python3 scripts/manage_cursor_agents.py list
    python3 scripts/manage_cursor_agents.py stop [agent_id]
"""

import argparse
import sys
import subprocess
import json
import time
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class CursorAgentManager:
    """Manager for Cursor Auto/Codex agents."""

    def __init__(self):
        """Initialize agent manager."""
        self.project_root = Path(__file__).parent.parent
        self.agents_dir = self.project_root / ".cursor" / "agents"
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.worktrees_config = self.project_root / ".cursor" / "worktrees.json"

    def check_cursor_cli(self) -> bool:
        """Check if Cursor CLI is installed."""
        try:
            result = subprocess.run(
                ["cursor", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def install_cursor_cli(self) -> bool:
        """Install Cursor CLI."""
        print("📦 Installing Cursor CLI...")
        try:
            # Try Linux/WSL installation
            result = subprocess.run(
                ["bash", "-c", "curl https://cursor.com/install -fsS | bash"],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                print("✅ Cursor CLI installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error installing Cursor CLI: {e}")
            print("\n💡 Manual installation:")
            print("   Visit: https://docs.cursor.com/en/cli")
            return False

    def launch_agents(
        self,
        num_agents: int,
        prompt: str,
        model: Optional[str] = None,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Launch multiple Cursor agents."""
        print(f"🚀 Launching {num_agents} Cursor agents...\n")

        # Check Cursor CLI
        if not self.check_cursor_cli():
            print("⚠️  Cursor CLI not found")
            if input("Install Cursor CLI? (y/n): ").lower() == 'y':
                if not self.install_cursor_cli():
                    return {"error": "Cursor CLI installation failed"}
            else:
                return {"error": "Cursor CLI required"}

        # Prepare tasks
        if tasks is None:
            tasks = [f"Task {i+1}: {prompt}" for i in range(num_agents)]
        elif len(tasks) < num_agents:
            # Extend tasks if not enough provided
            tasks.extend([f"Task {i+1}: {prompt}" for i in range(len(tasks), num_agents)])

        # Create coordination prompt
        coordination_prompt = self._create_coordination_prompt(tasks)

        # Launch agents
        agent_processes = []
        agent_info = []

        for i in range(num_agents):
            agent_id = i + 1
            agent_dir = self.agents_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)

            # Save agent info
            info = {
                "agent_id": agent_id,
                "task": tasks[i],
                "started_at": datetime.now().isoformat(),
                "status": "starting",
                "pid": None
            }
            agent_info.append(info)

            # Launch agent via Cursor CLI
            try:
                # Use cursor-agent command if available
                cmd = ["cursor-agent", "--task", tasks[i]]
                if model:
                    cmd.extend(["--model", model])

                process = subprocess.Popen(
                    cmd,
                    cwd=str(agent_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                info["pid"] = process.pid
                info["status"] = "running"
                agent_processes.append((agent_id, process))

                print(f"✅ Agent {agent_id} started (PID: {process.pid})")

            except FileNotFoundError:
                # Fallback: Use cursor command
                print(f"⚠️  cursor-agent not found, trying alternative method for Agent {agent_id}...")
                # Alternative: Open Cursor with specific workspace
                info["status"] = "manual"
                info["note"] = "Use Cursor UI to launch this agent"

        # Save agent info
        self._save_agent_info(agent_info)

        return {
            "status": "launched",
            "agents": agent_info,
            "coordination_prompt": coordination_prompt
        }

    def _create_coordination_prompt(self, tasks: List[str]) -> str:
        """Create coordination prompt for parallel agents."""
        prompt = """PARALLEL AGENT COORDINATION

First, read the .agent-id file in your working directory. If it doesn't exist yet,
wait a few seconds and try again (the setup script may still be running).

TASKS:
"""
        for i, task in enumerate(tasks, 1):
            prompt += f"{i}. {task}\n"

        prompt += """
Your .agent-id file contains a number (1-{num_agents}). Execute ONLY the task that matches your number.
Report your agent ID and task completion when done.
""".format(num_agents=len(tasks))

        return prompt

    def _save_agent_info(self, agents: List[Dict[str, Any]]):
        """Save agent information to file."""
        info_file = self.agents_dir / "agents.json"
        with open(info_file, 'w') as f:
            json.dump(agents, f, indent=2)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        info_file = self.agents_dir / "agents.json"
        if not info_file.exists():
            return {"agents": [], "count": 0}

        try:
            with open(info_file, 'r') as f:
                agents = json.load(f)

            # Check if processes are still running
            for agent in agents:
                if agent.get("pid"):
                    try:
                        os.kill(agent["pid"], 0)  # Check if process exists
                        agent["status"] = "running"
                    except OSError:
                        agent["status"] = "stopped"

            return {"agents": agents, "count": len(agents)}
        except Exception as e:
            return {"error": str(e), "agents": []}

    def list_agents(self, format_type: str = "table") -> None:
        """List all agents."""
        status = self.get_agent_status()
        agents = status.get("agents", [])

        if format_type == "json":
            print(json.dumps(status, indent=2))
            return

        if not agents:
            print("📋 No active agents")
            return

        print(f"\n📋 Active Agents ({len(agents)})\n")
        print(f"{'ID':<5} {'Status':<12} {'Started':<20} {'Task':<40}")
        print("=" * 80)

        for agent in agents:
            agent_id = agent.get("agent_id", "?")
            status = agent.get("status", "unknown")
            started = agent.get("started_at", "?")
            if len(started) > 19:
                started = started[:19]
            task = agent.get("task", "?")
            if len(task) > 38:
                task = task[:35] + "..."

            print(f"{agent_id:<5} {status:<12} {started:<20} {task:<40}")

        print()

    def stop_agent(self, agent_id: Optional[int] = None) -> bool:
        """Stop agent(s)."""
        status = self.get_agent_status()
        agents = status.get("agents", [])

        if not agents:
            print("⚠️  No active agents to stop")
            return False

        if agent_id:
            # Stop specific agent
            agent = next((a for a in agents if a.get("agent_id") == agent_id), None)
            if not agent:
                print(f"❌ Agent {agent_id} not found")
                return False

            pid = agent.get("pid")
            if pid:
                try:
                    os.kill(pid, 15)  # SIGTERM
                    print(f"✅ Stopped agent {agent_id} (PID: {pid})")
                    return True
                except OSError:
                    print(f"⚠️  Agent {agent_id} process not found")
                    return False
            else:
                print(f"⚠️  Agent {agent_id} has no process ID")
                return False
        else:
            # Stop all agents
            stopped = 0
            for agent in agents:
                pid = agent.get("pid")
                if pid:
                    try:
                        os.kill(pid, 15)
                        stopped += 1
                    except OSError:
                        pass

            print(f"✅ Stopped {stopped} agent(s)")
            return stopped > 0

    def show_coordination_prompt(self, num_agents: int) -> None:
        """Show coordination prompt for manual use."""
        tasks = [f"Task {i+1}" for i in range(num_agents)]
        prompt = self._create_coordination_prompt(tasks)
        print("\n📋 Coordination Prompt (Copy this for Cursor UI):\n")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        print("\n💡 Use this prompt when launching parallel agents in Cursor UI")
        print("   (Select 2x, 3x, 4x, etc. agents in Cursor chat)\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage multiple Cursor Auto/Codex agents through CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch 2 agents with a task
  python3 scripts/manage_cursor_agents.py launch --agents 2 --prompt "Refactor authentication module"

  # Launch 3 agents with specific tasks
  python3 scripts/manage_cursor_agents.py launch --agents 3 --tasks "Task 1" "Task 2" "Task 3"

  # List all agents
  python3 scripts/manage_cursor_agents.py list

  # Check agent status
  python3 scripts/manage_cursor_agents.py status

  # Stop specific agent
  python3 scripts/manage_cursor_agents.py stop --agent 1

  # Stop all agents
  python3 scripts/manage_cursor_agents.py stop

  # Show coordination prompt for manual use
  python3 scripts/manage_cursor_agents.py prompt --agents 4
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Launch command
    launch_parser = subparsers.add_parser('launch', help='Launch multiple agents')
    launch_parser.add_argument('--agents', '-n', type=int, required=True,
                              help='Number of agents to launch')
    launch_parser.add_argument('--prompt', '-p', type=str,
                              help='Task prompt for all agents')
    launch_parser.add_argument('--tasks', '-t', nargs='+',
                              help='Specific tasks for each agent')
    launch_parser.add_argument('--model', '-m', type=str,
                              help='Model to use (optional)')

    # List command
    list_parser = subparsers.add_parser('list', help='List all agents')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show agent status')
    status_parser.add_argument('--format', choices=['table', 'json'], default='table',
                             help='Output format')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop agent(s)')
    stop_parser.add_argument('--agent', '-a', type=int,
                            help='Agent ID to stop (stops all if not specified)')

    # Prompt command
    prompt_parser = subparsers.add_parser('prompt', help='Show coordination prompt')
    prompt_parser.add_argument('--agents', '-n', type=int, required=True,
                              help='Number of agents')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = CursorAgentManager()

    try:
        if args.command == 'launch':
            if not args.prompt and not args.tasks:
                print("❌ Error: --prompt or --tasks required")
                sys.exit(1)

            result = manager.launch_agents(
                num_agents=args.agents,
                prompt=args.prompt or "",
                model=args.model,
                tasks=args.tasks
            )

            if "error" in result:
                print(f"❌ Error: {result['error']}")
                sys.exit(1)

            print("\n✅ Agents launched successfully!")
            if "coordination_prompt" in result:
                print("\n📋 Coordination Prompt:")
                print(result["coordination_prompt"])

        elif args.command == 'list':
            manager.list_agents(format_type=args.format)

        elif args.command == 'status':
            status = manager.get_agent_status()
            if args.format == "json":
                print(json.dumps(status, indent=2))
            else:
                manager.list_agents()

        elif args.command == 'stop':
            manager.stop_agent(agent_id=args.agent)

        elif args.command == 'prompt':
            manager.show_coordination_prompt(num_agents=args.agents)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

