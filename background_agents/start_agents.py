"""Main entry point for starting the background agent system."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from background_agents.core import AgentSystem, AgentConfig, AgentPriority
from background_agents.agents import (
    DocumentMonitorAgent,
    LegalResearchAgent,
    CitationNetworkAgent,
    PatternDetectionAgent,
    SettlementOptimizerAgent,
    ProjectOrganizerAgent,
)


def create_agents(config: dict) -> list:
    """Create agent instances from configuration."""
    agents = []

    agent_configs = config['agents']

    # Document Monitor Agent
    if agent_configs.get('document_monitor', {}).get('enabled', False):
        cfg = agent_configs['document_monitor']
        agent_config = AgentConfig(
            name='document_monitor',
            model=cfg.get('model'),
            priority=AgentPriority[cfg.get('priority', 'MEDIUM').upper()],
            interval_minutes=cfg.get('interval_minutes'),
            enabled=cfg.get('enabled', True),
            max_concurrent_tasks=cfg.get('max_concurrent_tasks', 1)
        )
        agents.append(DocumentMonitorAgent(agent_config))

    # Legal Research Agent
    if agent_configs.get('legal_research', {}).get('enabled', False):
        cfg = agent_configs['legal_research']
        agent_config = AgentConfig(
            name='legal_research',
            model=cfg.get('model'),
            priority=AgentPriority[cfg.get('priority', 'MEDIUM').upper()],
            interval_minutes=cfg.get('interval_minutes'),
            enabled=cfg.get('enabled', True),
            temperature=cfg.get('temperature', 0.7),
            max_tokens=cfg.get('max_tokens', 2000)
        )
        agents.append(LegalResearchAgent(agent_config))

    # Citation Network Agent
    if agent_configs.get('citation_network', {}).get('enabled', False):
        cfg = agent_configs['citation_network']
        agent_config = AgentConfig(
            name='citation_network',
            model=cfg.get('model'),
            priority=AgentPriority[cfg.get('priority', 'LOW').upper()],
            interval_hours=cfg.get('interval_hours'),
            enabled=cfg.get('enabled', True)
        )
        agents.append(CitationNetworkAgent(agent_config))

    # Pattern Detection Agent
    if agent_configs.get('pattern_detection', {}).get('enabled', False):
        cfg = agent_configs['pattern_detection']
        agent_config = AgentConfig(
            name='pattern_detection',
            model=cfg.get('model'),
            priority=AgentPriority[cfg.get('priority', 'LOW').upper()],
            interval_hours=cfg.get('interval_hours'),
            enabled=cfg.get('enabled', True)
        )
        agents.append(PatternDetectionAgent(agent_config))

    # Settlement Optimizer Agent
    if agent_configs.get('settlement_optimizer', {}).get('enabled', False):
        cfg = agent_configs['settlement_optimizer']
        agent_config = AgentConfig(
            name='settlement_optimizer',
            model=None,  # Deterministic
            priority=AgentPriority[cfg.get('priority', 'LOW').upper()],
            interval_hours=cfg.get('interval_hours'),
            enabled=cfg.get('enabled', True)
        )
        agents.append(SettlementOptimizerAgent(agent_config))

    # Project Organizer Agent
    if agent_configs.get('project_organizer', {}).get('enabled', False):
        cfg = agent_configs['project_organizer']
        agent_config = AgentConfig(
            name='project_organizer',
            model=None,
            priority=AgentPriority[cfg.get('priority', 'MEDIUM').upper()],
            interval_hours=cfg.get('interval_hours'),
            interval_minutes=cfg.get('interval_minutes'),
            enabled=cfg.get('enabled', True)
        )
        agent = ProjectOrganizerAgent(agent_config)
        root_dir = cfg.get('root_directory')
        if root_dir:
            agent.default_root = Path(root_dir).resolve()
        agents.append(agent)

    return agents


async def main():
    """Main entry point."""
    print("🚀 Starting Background Agent System...")

    # Get config path
    config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        print(f"❌ Error: Configuration file not found at {config_path}")
        print("Please create config.yaml before starting the system.")
        return

    # Create agent system
    system = AgentSystem(config_path)

    # Create and register agents
    agents = create_agents(system.config)

    if not agents:
        print("⚠️  Warning: No agents enabled in configuration")
        print("Edit config.yaml to enable agents")
        return

    print(f"\n📋 Registered Agents:")
    for agent in agents:
        system.register_agent(agent)
        print(f"  ✅ {agent.name} (Priority: {agent.config.priority.name})")

    print(f"\n🔧 System Configuration:")
    print(f"  Max RAM: {system.config['system']['max_ram_usage_gb']}GB")
    print(f"  Max CPU Cores: {system.config['system']['max_cpu_cores']}")
    print(f"  Log Level: {system.config['system']['log_level']}")

    print(f"\n📂 Output Directory: background_agents/outputs/")
    print(f"📊 Database: {system.config['database']['path']}")
    print(f"📝 Logs: {system.config['logging']['directory']}/")

    print("\n" + "="*60)
    print("System is running! Press Ctrl+C to stop.")
    print("="*60 + "\n")

    # Start system
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping system...")
        await system.stop()
        print("✅ System stopped cleanly")


if __name__ == "__main__":
    asyncio.run(main())
