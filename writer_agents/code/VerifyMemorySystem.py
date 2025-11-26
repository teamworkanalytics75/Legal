#!/usr/bin/env python3
"""Verification script for the complete memory system implementation."""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all memory system components can be imported."""
    print("🔍 Testing imports...")

    try:
        from memory_system import MemoryStore, AgentMemory
        print("✅ MemoryStore and AgentMemory imported")
    except ImportError as e:
        print(f"❌ Failed to import memory_system: {e}")
        return False

    try:
        from embeddings import EmbeddingService
        print("✅ EmbeddingService imported")
    except ImportError as e:
        print(f"❌ Failed to import embeddings: {e}")
        return False

    try:
        from atomic_agent import AtomicAgent
        print("✅ AtomicAgent imported")
    except ImportError as e:
        print(f"❌ Failed to import atomic_agent: {e}")
        return False

    try:
        from master_supervisor import MemoryConfig, SupervisorConfig
        print("✅ MemoryConfig and SupervisorConfig imported")
    except ImportError as e:
        print(f"❌ Failed to import config classes: {e}")
        return False

    return True

def test_memory_store():
    """Test MemoryStore functionality."""
    print("\n🔍 Testing MemoryStore...")

    try:
        from memory_system import MemoryStore, AgentMemory

        # Test local embeddings
        store_local = MemoryStore(use_local_embeddings=True)
        print("✅ MemoryStore with local embeddings created")

        # Test OpenAI embeddings (without API key)
        store_openai = MemoryStore(use_local_embeddings=False)
        print("✅ MemoryStore with OpenAI embeddings created")

        # Test memory creation
        memory = AgentMemory(
            agent_type="test_agent",
            input_summary="Test input",
            output_summary="Test output",
            key_insights=["Test insight"],
            timestamp="2025-01-01T00:00:00Z"
        )

        store_local.add(memory)
        print("✅ Memory added to store")

        # Test retrieval
        memories = store_local.get_memories("test_agent", "test query", k=1)
        print(f"✅ Retrieved {len(memories)} memories")

        return True

    except Exception as e:
        print(f"❌ MemoryStore test failed: {e}")
        return False

def test_embedding_service():
    """Test EmbeddingService functionality."""
    print("\n🔍 Testing EmbeddingService...")

    try:
        from embeddings import EmbeddingService

        # Test local embeddings
        local_service = EmbeddingService(use_local=True)
        embedding = local_service.embed_text("Test text")
        print(f"✅ Local embedding generated: {len(embedding)} dimensions")

        # Test OpenAI embeddings (will fail without API key, but that's expected)
        try:
            openai_service = EmbeddingService(use_local=False)
            print("✅ OpenAI EmbeddingService created (API key required for actual embedding)")
        except Exception as e:
            print(f"⚠️ OpenAI EmbeddingService creation failed (expected without API key): {e}")

        return True

    except Exception as e:
        print(f"❌ EmbeddingService test failed: {e}")
        return False

def test_config_integration():
    """Test configuration integration."""
    print("\n🔍 Testing configuration integration...")

    try:
        from master_supervisor import MemoryConfig, SupervisorConfig

        # Test MemoryConfig
        mem_config = MemoryConfig(
            enabled=True,
            use_local_embeddings=True,
            embedding_model="all-MiniLM-L6-v2",
            k_neighbors=5,
            max_memory_tokens=500
        )
        print("✅ MemoryConfig created")

        # Test SupervisorConfig with MemoryConfig
        supervisor_config = SupervisorConfig(memory_config=mem_config)
        print("✅ SupervisorConfig with MemoryConfig created")

        # Test environment variable support
        import os
        os.environ["WITCHWEB_USE_LOCAL_EMBEDDINGS"] = "false"
        mem_config_env = MemoryConfig()
        print(f"✅ Environment variable support: use_local_embeddings={mem_config_env.use_local_embeddings}")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_agent_templates():
    """Test agent context templates."""
    print("\n🔍 Testing agent context templates...")

    try:
        from agent_context_templates import get_agent_context

        # Test a few agent types
        test_agents = ["CitationFinderAgent", "OutlineBuilderAgent", "FactExtractorAgent"]

        for agent_type in test_agents:
            context = get_agent_context(agent_type)
            if context and "project_overview" in context:
                print(f"✅ Context for {agent_type}: {len(context['project_overview'])} chars")
            else:
                print(f"⚠️ No context found for {agent_type}")

        return True

    except Exception as e:
        print(f"❌ Agent templates test failed: {e}")
        return False

def test_scripts():
    """Test that all scripts exist and are properly structured."""
    print("\n🔍 Testing scripts...")

    script_dir = Path("scripts")
    required_scripts = [
        "refresh_agent_memories.py",
        "populate_initial_memories.py",
        "test_memory_writing.py",
        "test_memory_pilot.py"
    ]

    for script in required_scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"✅ {script} exists")
        else:
            print(f"❌ {script} missing")
            return False

    return True

def main():
    """Run all verification tests."""
    print("🚀 The Matrix Memory System Verification")
    print("=" * 50)

    tests = [
        test_imports,
        test_memory_store,
        test_embedding_service,
        test_config_integration,
        test_agent_templates,
        test_scripts
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Memory system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
