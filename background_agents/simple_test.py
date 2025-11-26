"""Simple test to verify the background agent system works."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from background_agents.core import AgentSystem, AgentConfig, AgentPriority, Task, TaskPriority
from background_agents.core.task_queue import TaskStatus
from background_agents.agents import DocumentMonitorAgent, SettlementOptimizerAgent


async def test_basic_system():
    """Test basic system startup and task processing."""
    print("=" * 60)
    print("Testing Background Agent System")
    print("=" * 60 + "\n")

    # Create a minimal config for testing
    config_path = Path("background_agents/config.yaml")

    if not config_path.exists():
        print(f"[FAIL] Config file not found: {config_path}")
        return False

    print("[OK] Config file found")

    # Create system
    try:
        system = AgentSystem(config_path)
        print("[OK] AgentSystem created")
    except Exception as e:
        print(f"[FAIL] Failed to create AgentSystem: {e}")
        return False

    # Register a simple agent (settlement optimizer - no LLM needed)
    try:
        config = AgentConfig(
            name='settlement_optimizer',
            model=None,
            priority=AgentPriority.LOW,
            enabled=True
        )
        agent = SettlementOptimizerAgent(config)
        system.register_agent(agent)
        print("[OK] Settlement optimizer agent registered")
    except Exception as e:
        print(f"[FAIL] Failed to register agent: {e}")
        return False

    # Create a test task manually
    try:
        task = Task.create(
            agent_name='settlement_optimizer',
            task_type='test',
            data={
                'case_name': 'test_case',
                'success_probability': 0.65,
                'damages_mean': 500000,
                'damages_std': 100000,
                'legal_costs': 50000,
                'risk_aversion': 0.5,
                'iterations': 1000  # Small for testing
            },
            priority=TaskPriority.HIGH
        )

        system.task_queue.add_task(task)
        print("[OK] Test task created and queued")
    except Exception as e:
        print(f"[FAIL] Failed to create task: {e}")
        return False

    # Initialize and process one task
    try:
        await system.initialize_agents()
        print("[OK] Agents initialized")

        # Get and process the task
        next_task = system.task_queue.get_next_task('settlement_optimizer')
        if next_task:
            print(f"[OK] Retrieved task: {next_task.id}")

            # Mark as in progress
            system.task_queue.update_task_status(next_task.id, TaskStatus.IN_PROGRESS)

            # Process it
            result = await agent.run_task(next_task.data)
            print("[OK] Task processed successfully")
            print(f"   Optimal settlement: ${result['optimal_settlement']['optimal_settlement']:,.0f}")

            # Mark as complete
            system.task_queue.update_task_status(
                next_task.id,
                TaskStatus.COMPLETED,
                result={'output': 'success'}
            )
            print("[OK] Task marked as completed")
        else:
            print("[WARN] No task retrieved from queue")
            return False

    except Exception as e:
        print(f"[FAIL] Failed during task processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await system.shutdown_agents()

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    result = asyncio.run(test_basic_system())
    sys.exit(0 if result else 1)

