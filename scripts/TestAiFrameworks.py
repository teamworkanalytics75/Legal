"""
Test script for CrewAI and LlamaIndex installations
Tests basic functionality without requiring API keys
"""

import sys
import os
from typing import Optional

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        # Fallback: disable emojis if encoding fails
        pass


def test_crewai_import() -> bool:
    """Test CrewAI import and basic functionality."""
    try:
        from crewai import Agent, Task, Crew
        print("✅ CrewAI imported successfully")

        # Test that classes are available (don't create instances without API keys)
        print(f"✅ Agent class available: {Agent.__name__}")
        print(f"✅ Task class available: {Task.__name__}")
        print(f"✅ Crew class available: {Crew.__name__}")

        return True
    except Exception as e:
        print(f"❌ CrewAI test failed: {e}")
        return False


def test_llamaindex_import() -> bool:
    """Test LlamaIndex import and basic functionality."""
    try:
        from llama_index.core import Document, Settings
        from llama_index.core.node_parser import SentenceSplitter
        print("✅ LlamaIndex imported successfully")

        # Test document creation (no API calls)
        doc = Document(text="This is a test document for validation.")
        print(f"✅ Created test document with {len(doc.text)} characters")

        # Test node parser
        parser = SentenceSplitter(chunk_size=100, chunk_overlap=10)
        print("✅ Created sentence splitter")

        return True
    except Exception as e:
        print(f"❌ LlamaIndex test failed: {e}")
        return False


def test_llamaindex_readers() -> bool:
    """Test LlamaIndex data readers."""
    try:
        from llama_index.readers.file import PDFReader
        print("✅ LlamaIndex PDFReader available")

        # Check other readers
        try:
            from llama_index.core import SimpleDirectoryReader
            print("✅ SimpleDirectoryReader available")
        except ImportError:
            print("⚠️  SimpleDirectoryReader not available")

        return True
    except Exception as e:
        print(f"❌ LlamaIndex readers test failed: {e}")
        return False


def test_chromadb_integration() -> bool:
    """Test ChromaDB integration with CrewAI."""
    try:
        import chromadb
        version = chromadb.__version__
        print(f"✅ ChromaDB v{version} available")
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False


def check_optional_dependencies() -> None:
    """Check for optional but useful dependencies."""
    print("\n📦 Checking optional dependencies:")

    optional_packages = {
        'openai': 'OpenAI API client',
        'anthropic': 'Anthropic API client',
        'sentence_transformers': 'Local embeddings',
        'tiktoken': 'Token counting',
        'pypdf': 'PDF processing',
        'beautifulsoup4': 'HTML parsing',
    }

    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ⚠️  {package}: {description} (not installed)")


def main() -> None:
    """Run all tests."""
    print("🔬 Testing AI Research Frameworks\n")
    print("=" * 60)

    results = {
        'CrewAI Import': test_crewai_import(),
        'LlamaIndex Import': test_llamaindex_import(),
        'LlamaIndex Readers': test_llamaindex_readers(),
        'ChromaDB Integration': test_chromadb_integration(),
    }

    print("\n" + "=" * 60)
    print("\n📊 Test Results Summary:")
    print("-" * 60)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")

    check_optional_dependencies()

    print("\n" + "=" * 60)

    if all(results.values()):
        print("\n🎉 All core tests passed! Frameworks are ready to use.")
        print("\n💡 Next steps:")
        print("  1. Set up your API keys (OpenAI, Anthropic, etc.)")
        print("  2. Review AI_RESEARCH_FRAMEWORKS_GUIDE.md")
        print("  3. Try the example code in the guide")
        print("  4. Integrate with your existing TheMatrix project")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

