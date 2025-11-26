"""Check the status of the background agent system."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from background_agents.core import TaskQueue


def main():
    """Display system status."""
    db_path = Path("background_agents/agents.db")

    if not db_path.exists():
        print("❌ System not running or not initialized")
        print(f"   Database not found at: {db_path}")
        return

    print("\n" + "="*60)
    print("📊 Background Agent System Status")
    print("="*60 + "\n")

    # Load task queue
    queue = TaskQueue(db_path)

    # Get stats
    stats = queue.get_stats()

    if not stats:
        print("ℹ️  No tasks in queue yet")
        return

    # Display stats by agent
    for agent_name, agent_stats in stats.items():
        print(f"\n🤖 {agent_name.upper()}")
        print("   " + "-"*50)

        total = sum(agent_stats.values())

        for status, count in agent_stats.items():
            percentage = (count / total * 100) if total > 0 else 0

            # Status icons
            icon = {
                'pending': '⏳',
                'in_progress': '🔄',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '🚫'
            }.get(status, '📋')

            print(f"   {icon} {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

        print(f"   📊 Total Tasks: {total}")

    print("\n" + "="*60)
    print("\n💡 Tip: Use 'python background_agents/view_insights.py' to see outputs")
    print()


if __name__ == "__main__":
    main()

