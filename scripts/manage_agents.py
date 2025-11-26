#!/usr/bin/env python3
"""
Agent Management CLI - Easy terminal control for multiple codex agents

Usage:
    python scripts/manage_agents.py list
    python scripts/manage_agents.py start <agent_name>
    python scripts/manage_agents.py stop <agent_name>
    python scripts/manage_agents.py status
    python scripts/manage_agents.py logs <agent_name>
    python scripts/manage_agents.py monitor [agent_name]
"""

import argparse
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import yaml
except ImportError:
    print("⚠️  Warning: PyYAML not installed. Install with: pip install pyyaml")
    yaml = None


class AgentManager:
    """CLI manager for background agents."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize agent manager."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "background_agents" / "config.yaml"
        self.config_path = config_path
        self.config = self._load_config()
        self.logs_dir = Path(__file__).parent.parent / "background_agents" / "logs"
        self.outputs_dir = Path(__file__).parent.parent / "background_agents" / "outputs"

    def _load_config(self) -> Dict[str, Any]:
        """Load agent configuration."""
        if not self.config_path.exists():
            return {}
        
        if yaml is None:
            print("⚠️  Cannot load config without PyYAML")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return {}

    def list_agents(self, format_type: str = "table") -> None:
        """List all configured agents."""
        agents = self.config.get('agents', {})
        
        if not agents:
            print("⚠️  No agents found in configuration")
            return

        if format_type == "json":
            print(json.dumps(agents, indent=2))
            return

        # Table format
        print("\n📋 Configured Agents\n")
        print(f"{'Agent Name':<25} {'Status':<10} {'Priority':<10} {'Model':<20}")
        print("=" * 75)

        for name, config in agents.items():
            status = "✅ Enabled" if config.get('enabled', False) else "❌ Disabled"
            priority = config.get('priority', 'N/A').upper()
            model = config.get('model', 'N/A') or 'None'
            if len(model) > 18:
                model = model[:15] + "..."
            
            print(f"{name:<25} {status:<10} {priority:<10} {model:<20}")

        print()

    def get_agent_status(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of agent(s)."""
        # This is a simplified version - in production, you'd check actual process status
        agents = self.config.get('agents', {})
        
        if agent_name:
            if agent_name not in agents:
                return {"error": f"Agent '{agent_name}' not found"}
            return {
                agent_name: {
                    "enabled": agents[agent_name].get('enabled', False),
                    "config": agents[agent_name]
                }
            }
        
        return {
            name: {
                "enabled": config.get('enabled', False),
                "priority": config.get('priority', 'N/A'),
                "model": config.get('model', 'N/A')
            }
            for name, config in agents.items()
        }

    def show_status(self, format_type: str = "table") -> None:
        """Display agent status."""
        status = self.get_agent_status()
        
        if format_type == "json":
            print(json.dumps(status, indent=2))
            return

        print("\n📊 Agent Status\n")
        print(f"{'Agent Name':<25} {'Enabled':<10} {'Priority':<10} {'Model':<20}")
        print("=" * 75)

        for name, info in status.items():
            enabled = "✅ Yes" if info.get('enabled') else "❌ No"
            priority = info.get('priority', 'N/A').upper()
            model = info.get('model', 'N/A') or 'None'
            if len(model) > 18:
                model = model[:15] + "..."
            
            print(f"{name:<25} {enabled:<10} {priority:<10} {model:<20}")

        print()

    def show_logs(self, agent_name: Optional[str] = None, follow: bool = False, tail: int = 50) -> None:
        """Show agent logs."""
        if not self.logs_dir.exists():
            print(f"⚠️  Logs directory not found: {self.logs_dir}")
            return

        if agent_name:
            # Look for agent-specific log file
            log_files = list(self.logs_dir.glob(f"*{agent_name}*.log"))
            if not log_files:
                log_files = list(self.logs_dir.glob("*.log"))
        else:
            log_files = list(self.logs_dir.glob("*.log"))

        if not log_files:
            print(f"⚠️  No log files found in {self.logs_dir}")
            return

        if follow:
            # Follow logs (like tail -f)
            print(f"📝 Following logs (Ctrl+C to stop)...\n")
            try:
                for log_file in sorted(log_files):
                    print(f"\n{'='*60}")
                    print(f"📄 {log_file.name}")
                    print(f"{'='*60}\n")
                    
                    # Use tail -f if available
                    try:
                        subprocess.run(["tail", "-f", str(log_file)], check=False)
                    except (FileNotFoundError, subprocess.SubprocessError):
                        # Fallback: read and watch file
                        self._watch_file(log_file)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopped following logs")
        else:
            # Show last N lines
            for log_file in sorted(log_files):
                print(f"\n{'='*60}")
                print(f"📄 {log_file.name}")
                print(f"{'='*60}\n")
                
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-tail:]:
                            print(line, end='')
                except Exception as e:
                    print(f"❌ Error reading log: {e}")

    def _watch_file(self, log_file: Path) -> None:
        """Watch a log file for changes (simple implementation)."""
        try:
            with open(log_file, 'r') as f:
                # Go to end of file
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        print(line, end='')
                    else:
                        time.sleep(0.1)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"❌ Error watching file: {e}")

    def monitor(self, agent_name: Optional[str] = None, interval: int = 5) -> None:
        """Monitor agent status in real-time."""
        print(f"📊 Agent Monitor (updates every {interval}s, Ctrl+C to stop)\n")
        
        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end='')
                print(f"📊 Agent Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                self.show_status()
                
                if agent_name:
                    print(f"\n📝 Recent logs for {agent_name}:\n")
                    self.show_logs(agent_name, follow=False, tail=10)
                
                print(f"\n⏳ Refreshing in {interval} seconds... (Ctrl+C to stop)")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped")

    def start_agent_system(self, dry_run: bool = False) -> None:
        """Start the background agent system."""
        script_path = Path(__file__).parent.parent / "background_agents" / "start_agents.py"
        
        if not script_path.exists():
            print(f"❌ Agent start script not found: {script_path}")
            return

        cmd = [sys.executable, str(script_path)]
        if dry_run:
            cmd.append("--dry-run")

        print(f"🚀 Starting agent system...")
        if dry_run:
            print("   (dry-run mode)")
        
        try:
            subprocess.run(cmd, check=False)
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage multiple codex agents through CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all agents
  python scripts/manage_agents.py list

  # Show agent status
  python scripts/manage_agents.py status

  # Show logs for specific agent
  python scripts/manage_agents.py logs document_monitor

  # Follow logs in real-time
  python scripts/manage_agents.py logs --follow

  # Monitor all agents
  python scripts/manage_agents.py monitor

  # Monitor specific agent
  python scripts/manage_agents.py monitor document_monitor

  # Start agent system
  python scripts/manage_agents.py start-system
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser('list', help='List all configured agents')
    list_parser.add_argument('--format', choices=['table', 'json'], default='table',
                           help='Output format (default: table)')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show agent status')
    status_parser.add_argument('--format', choices=['table', 'json'], default='table',
                             help='Output format (default: table)')

    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Show agent logs')
    logs_parser.add_argument('agent_name', nargs='?', help='Agent name (optional)')
    logs_parser.add_argument('--follow', '-f', action='store_true',
                           help='Follow logs in real-time')
    logs_parser.add_argument('--tail', '-n', type=int, default=50,
                           help='Number of lines to show (default: 50)')
    logs_parser.add_argument('--all', '-a', action='store_true',
                           help='Show all agent logs')

    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor agents in real-time')
    monitor_parser.add_argument('agent_name', nargs='?', help='Agent name to monitor (optional)')
    monitor_parser.add_argument('--interval', '-i', type=int, default=5,
                              help='Update interval in seconds (default: 5)')

    # Start system command
    start_parser = subparsers.add_parser('start-system', help='Start background agent system')
    start_parser.add_argument('--dry-run', action='store_true',
                            help='Run in dry-run mode')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    manager = AgentManager()

    try:
        if args.command == 'list':
            manager.list_agents(format_type=args.format)
        elif args.command == 'status':
            manager.show_status(format_type=args.format)
        elif args.command == 'logs':
            if args.all:
                manager.show_logs(agent_name=None, follow=args.follow, tail=args.tail)
            else:
                manager.show_logs(agent_name=args.agent_name, follow=args.follow, tail=args.tail)
        elif args.command == 'monitor':
            manager.monitor(agent_name=args.agent_name, interval=args.interval)
        elif args.command == 'start-system':
            manager.start_agent_system(dry_run=args.dry_run)
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

