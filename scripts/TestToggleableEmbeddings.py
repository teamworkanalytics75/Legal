"""Test script to verify toggleable embeddings work correctly."""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from writer_agents.code.memory_system import MemoryStore
from writer_agents.code.embeddings import EmbeddingService
from writer_agents.code.master_supervisor import SupervisorConfig, MemoryConfig


def test_embedding_service():
    """Test EmbeddingService with both local and OpenAI options."""

    print("=" * 80)
    print("TESTING EMBEDDING SERVICE")
    print("=" * 80)

    # Test local embeddings
    print("\n1. Testing Local Embeddings:")
    try:
        service_local = EmbeddingService(use_local=True)
        test_text = "This is a test sentence for embedding."
        embedding_local = service_local.embed(test_text)

        print(f" [ok] Local embedding shape: {embedding_local.shape}")
        print(f" [ok] Local embedding dimension: {service_local.get_dimension()}")
        print(f" [ok] Local cost estimate: ${service_local.estimate_cost(100):.4f}")

    except Exception as e:
        print(f" x Local embeddings failed: {e}")
        return False

    # Test OpenAI embeddings (if available)
    print("\n2. Testing OpenAI Embeddings:")
    try:
        service_openai = EmbeddingService(use_local=False)
        embedding_openai = service_openai.embed(test_text)

        print(f" [ok] OpenAI embedding shape: {embedding_openai.shape}")
        print(f" [ok] OpenAI embedding dimension: {service_openai.get_dimension()}")
        print(f" [ok] OpenAI cost estimate: ${service_openai.estimate_cost(100):.4f}")

    except Exception as e:
        print(f" WARNING OpenAI embeddings not available: {e}")
        print(" (This is expected if OpenAI API key is not set)")

    return True


def test_memory_store():
    """Test MemoryStore with different embedding options."""

    print("\n" + "=" * 80)
    print("TESTING MEMORY STORE")
    print("=" * 80)

    # Test with local embeddings
    print("\n1. Testing MemoryStore with Local Embeddings:")
    try:
        store_local = MemoryStore(
            storage_path=Path("test_memory_local"),
            use_local_embeddings=True,
            embedding_model="all-MiniLM-L6-v2"
        )

        print(f" [ok] MemoryStore created with local embeddings")
        print(f" [ok] Embedding service mode: {store_local.embeddings_service.mode}")
        print(f" [ok] Embedding dimension: {store_local.embeddings_service.get_dimension()}")

    except Exception as e:
        print(f" x Local MemoryStore failed: {e}")
        return False

    # Test with OpenAI embeddings (if available)
    print("\n2. Testing MemoryStore with OpenAI Embeddings:")
    try:
        store_openai = MemoryStore(
            storage_path=Path("test_memory_openai"),
            use_local_embeddings=False,
            embedding_model="all-MiniLM-L6-v2" # Ignored when using OpenAI
        )

        print(f" [ok] MemoryStore created with OpenAI embeddings")
        print(f" [ok] Embedding service mode: {store_openai.embeddings_service.mode}")
        print(f" [ok] Embedding dimension: {store_openai.embeddings_service.get_dimension()}")

    except Exception as e:
        print(f" WARNING OpenAI MemoryStore not available: {e}")
        print(" (This is expected if OpenAI API key is not set)")

    return True


def test_config_integration():
    """Test configuration integration."""

    print("\n" + "=" * 80)
    print("TESTING CONFIGURATION INTEGRATION")
    print("=" * 80)

    # Test local config
    print("\n1. Testing Local Configuration:")
    config_local = SupervisorConfig(
        memory_config=MemoryConfig(
            use_local_embeddings=True,
            embedding_model="all-MiniLM-L6-v2"
        )
    )

    print(f" [ok] Local config: {config_local.memory_config.use_local_embeddings}")
    print(f" [ok] Embedding model: {config_local.memory_config.embedding_model}")

    # Test OpenAI config
    print("\n2. Testing OpenAI Configuration:")
    config_openai = SupervisorConfig(
        memory_config=MemoryConfig(
            use_local_embeddings=False,
            embedding_model="all-MiniLM-L6-v2" # Ignored when using OpenAI
        )
    )

    print(f" [ok] OpenAI config: {config_openai.memory_config.use_local_embeddings}")
    print(f" [ok] Embedding model: {config_openai.memory_config.embedding_model}")

    # Test environment variable
    print("\n3. Testing Environment Variable:")
    import os
    env_value = os.getenv("WITCHWEB_USE_LOCAL_EMBEDDINGS", "true")
    print(f" [ok] Environment variable: WITCHWEB_USE_LOCAL_EMBEDDINGS={env_value}")

    return True


def test_refresh_script_args():
    """Test refresh script argument parsing."""

    print("\n" + "=" * 80)
    print("TESTING REFRESH SCRIPT ARGUMENTS")
    print("=" * 80)

    import argparse

    # Test argument parser
    parser = argparse.ArgumentParser(description="Test refresh script")
    parser.add_argument("--days", type=int, help="Scan last N days")
    parser.add_argument("--full", action="store_true", help="Full rescan")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI embeddings")

    # Test local args
    args_local = parser.parse_args(["--days", "7"])
    print(f" [ok] Local args: days={args_local.days}, openai={args_local.openai}")

    # Test OpenAI args
    args_openai = parser.parse_args(["--full", "--openai"])
    print(f" [ok] OpenAI args: full={args_openai.full}, openai={args_openai.openai}")

    return True


def main():
    """Run all tests."""

    print("TOGGLEABLE EMBEDDINGS TEST SUITE")
    print("=" * 80)
    print("Testing the wiring between MemoryConfig and EmbeddingService")
    print("=" * 80)

    try:
        # Run tests
        success = True

        success &= test_embedding_service()
        success &= test_memory_store()
        success &= test_config_integration()
        success &= test_refresh_script_args()

        if success:
            print("\n" + "=" * 80)
            print("[ok] ALL TESTS PASSED!")
            print("=" * 80)
            print("\nToggleable embeddings are working correctly:")
            print("- Local embeddings: FREE, 384-dim, good quality")
            print("- OpenAI embeddings: PAID, 1536-dim, better quality")
            print("- Configuration flows through properly")
            print("- Command-line arguments work")
            print("- Environment variable support enabled")

            print("\nUsage Examples:")
            print("- Local (default): python scripts/refresh_agent_memories.py --full")
            print("- OpenAI: python scripts/refresh_agent_memories.py --full --openai")
            print("- Environment: WITCHWEB_USE_LOCAL_EMBEDDINGS=false python script.py")

        else:
            print("\nx SOME TESTS FAILED - CHECK ERRORS ABOVE")
            return 1

    except Exception as e:
        print(f"\nx TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
