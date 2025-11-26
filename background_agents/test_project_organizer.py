"""Test script for the enhanced project organizer agent."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from background_agents.core import AgentSystem, AgentConfig, AgentPriority
from background_agents.agents import ProjectOrganizerAgent


async def test_project_organizer():
    """Test the enhanced project organizer agent."""
    print("=" * 60)
    print("Testing Enhanced Project Organizer Agent")
    print("=" * 60)

    # Create agent config
    agent_config = AgentConfig(
        name='project_organizer',
        model=None,  # No LLM needed
        priority=AgentPriority.HIGH,
        interval_hours=6,
        enabled=True
    )

    # Create agent
    agent = ProjectOrganizerAgent(agent_config)
    agent.default_root = Path("C:/Users/User/Desktop/TheMatrix")

    print(f"[OK] Project organizer agent created")
    print(f"[OK] Root directory: {agent.default_root}")

    # Test task data (simulating what the scheduler would create)
    task_data = {
        'root': str(agent.default_root),
        'mode': 'organize',
        'enable_file_naming_standardization': True,
        'enable_folder_organization': True,
        'enable_cleanup_empty_folders': True,
        'enable_duplicate_detection': True
    }

    print(f"[OK] Task data prepared: {task_data}")

    try:
        # Initialize agent
        await agent.initialize()
        print("[OK] Agent initialized")

        # Process the organization task
        print("\nStarting project organization...")
        result = await agent.process(task_data)

        print(f"[OK] Organization completed!")
        print(f"   Files renamed: {result.get('files_renamed', 0)}")
        print(f"   Files moved: {result.get('files_moved', 0)}")
        print(f"   Empty folders removed: {result.get('empty_folders_removed', 0)}")
        print(f"   Duplicates found: {result.get('duplicates_found', 0)}")
        print(f"   Report: {result.get('report_path', 'N/A')}")

        if result.get('actions_performed'):
            print("\nActions performed:")
            for action in result['actions_performed']:
                print(f"   [OK] {action}")

        if result.get('errors'):
            print("\nErrors encountered:")
            for error in result['errors']:
                print(f"   [ERROR] {error}")

        # Shutdown agent
        await agent.shutdown()
        print("[OK] Agent shutdown")

        print("\n" + "=" * 60)
        print("[SUCCESS] PROJECT ORGANIZER TEST PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(test_project_organizer())
    sys.exit(0 if success else 1)
