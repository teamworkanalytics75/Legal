"""Test script to verify background agent system setup."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required packages can be imported."""
    print("🧪 Testing imports...")

    try:
        import yaml
        print("  ✅ yaml")
    except ImportError:
        print("  ❌ yaml - Install: pip install pyyaml")
        return False

    try:
        import schedule
        print("  ✅ schedule")
    except ImportError:
        print("  ❌ schedule - Install: pip install schedule")
        return False

    try:
        import watchdog
        print("  ✅ watchdog")
    except ImportError:
        print("  ❌ watchdog - Install: pip install watchdog")
        return False

    try:
        import networkx
        print("  ✅ networkx")
    except ImportError:
        print("  ⚠️  networkx - Optional but recommended: pip install networkx")

    try:
        import numpy
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy - Install: pip install numpy")
        return False

    return True


def test_ollama():
    """Test Ollama connectivity."""
    print("\n🤖 Testing Ollama...")

    try:
        import ollama
        print("  ✅ ollama-python installed")
    except ImportError:
        print("  ❌ ollama-python not installed")
        print("     Install: pip install ollama-python")
        return False

    try:
        # Try to list models
        models = ollama.list()
        print(f"  ✅ Ollama server running")

        if models.get('models'):
            print(f"  📦 Installed models:")
            for model in models['models']:
                print(f"     - {model['name']}")
        else:
            print("  ⚠️  No models installed yet")
            print("     Run: ollama pull llama3.2:7b")

        return True

    except Exception as e:
        print(f"  ❌ Ollama server not running: {e}")
        print("     Start Ollama or install from: https://ollama.com/download/windows")
        return False


def test_config():
    """Test configuration file."""
    print("\n⚙️  Testing configuration...")

    config_path = Path("background_agents/config.yaml")

    if not config_path.exists():
        print(f"  ❌ Config file not found: {config_path}")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        print("  ✅ Config file valid")

        # Check key sections
        if 'system' in config:
            print(f"  ✅ System config: {config['system']['max_ram_usage_gb']}GB RAM limit")

        if 'agents' in config:
            enabled = [name for name, cfg in config['agents'].items() if cfg.get('enabled')]
            print(f"  ✅ Enabled agents: {', '.join(enabled)}")

        return True

    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_directories():
    """Test that required directories exist or can be created."""
    print("\n📁 Testing directories...")

    dirs = [
        "background_agents/core",
        "background_agents/agents",
        "background_agents/outputs",
        "background_agents/logs",
    ]

    all_ok = True
    for dir_path in dirs:
        p = Path(dir_path)
        if p.exists():
            print(f"  ✅ {dir_path}")
        else:
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"  ✅ {dir_path} (created)")
            except Exception as e:
                print(f"  ❌ {dir_path}: {e}")
                all_ok = False

    return all_ok


def test_core_modules():
    """Test that core modules can be imported."""
    print("\n📦 Testing core modules...")

    try:
        from background_agents.core import BackgroundAgent, AgentConfig, AgentSystem, TaskQueue
        print("  ✅ Core modules import successfully")
        return True
    except Exception as e:
        print(f"  ❌ Core import error: {e}")
        return False


def test_agent_modules():
    """Test that agent modules can be imported."""
    print("\n🤖 Testing agent modules...")

    try:
        from background_agents.agents import (
            DocumentMonitorAgent,
            LegalResearchAgent,
            CitationNetworkAgent,
            PatternDetectionAgent,
            SettlementOptimizerAgent,
        )
        print("  ✅ All agent modules import successfully")
        return True
    except Exception as e:
        print(f"  ❌ Agent import error: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("🧪 Background Agent System - Setup Test")
    print("="*60 + "\n")

    results = {
        "Imports": test_imports(),
        "Ollama": test_ollama(),
        "Configuration": test_config(),
        "Directories": test_directories(),
        "Core Modules": test_core_modules(),
        "Agent Modules": test_agent_modules(),
    }

    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60 + "\n")

    for test_name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {test_name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! System is ready to start.")
        print("\nNext step: python background_agents/start_agents.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nSee QUICK_START.md for installation instructions.")
    print("="*60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

