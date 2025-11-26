"""
Quick Start: 1782 Case Research with CrewAI + LlamaIndex

This script provides a simple, ready-to-use research tool for your 1782 case database.
Just set your OPENAI_API_KEY and run!

Usage:
    python quick_start_1782_research.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def check_setup() -> tuple[bool, str]:
    """Check if environment is properly configured."""

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False, """
❌ OpenAI API key not found!

Please set your API key:

PowerShell:
    $env:OPENAI_API_KEY="your-key-here"

Or create a .env file in this directory:
    OPENAI_API_KEY=your-key-here

Get your API key from: https://platform.openai.com/api-keys
"""

    # Check for 1782 database
    db_path = Path("1782 Case PDF Database")
    if not db_path.exists():
        return False, f"""
⚠️  1782 Case PDF Database not found at: {db_path.absolute()}

Please update DB_PATH in this script to point to your PDF directory.
"""

    # Count PDFs
    pdf_count = len(list(db_path.glob("*.pdf")))
    if pdf_count == 0:
        return False, f"""
⚠️  No PDF files found in: {db_path.absolute()}

Please ensure your PDFs are in this directory.
"""

    return True, f"✅ Found {pdf_count} PDFs in database"


def create_legal_index(force_rebuild: bool = False):
    """Create or load the legal document index."""
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI

    storage_dir = Path("./storage/1782_legal_index")
    db_path = Path("1782 Case PDF Database")

    # Configure settings
    Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding()

    # Check if index exists
    if storage_dir.exists() and not force_rebuild:
        print(f"📂 Loading existing index from {storage_dir}")
        try:
            from llama_index.core import StorageContext, load_index_from_storage
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
            index = load_index_from_storage(storage_context)
            print("✅ Index loaded successfully")
            return index
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            print("   Building new index...")

    # Build new index
    print(f"📚 Loading documents from {db_path}")
    documents = SimpleDirectoryReader(str(db_path)).load_data()
    print(f"✅ Loaded {len(documents)} documents")

    print("🔨 Creating vector index (this may take a few minutes)...")
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )

    print(f"💾 Saving index to {storage_dir}")
    storage_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(storage_dir))

    print("✅ Index created and saved")
    return index


def simple_query_interface(index):
    """Simple command-line query interface."""
    print("\n" + "="*60)
    print("🔍 1782 Case Research - Query Interface")
    print("="*60)
    print("\nCommands:")
    print("  - Type your question to search the database")
    print("  - 'examples' to see example queries")
    print("  - 'quit' to exit")
    print("="*60 + "\n")

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="tree_summarize"
    )

    example_queries = [
        "What are the Intel factors for 1782 applications?",
        "What are common reasons courts deny 1782 applications?",
        "Are there cases involving international arbitration?",
        "What factors indicate discoverable interest?",
        "How do courts analyze foreign discoverability?",
    ]

    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == 'examples':
                print("\n📋 Example queries:")
                for i, example in enumerate(example_queries, 1):
                    print(f"  {i}. {example}")
                continue

            print("\n🔎 Searching database...")
            response = query_engine.query(user_input)

            print("\n📝 Answer:")
            print("-" * 60)
            print(response)
            print("-" * 60)

            # Show sources if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"\n📚 Based on {len(response.source_nodes)} source(s)")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def run_example_queries(index):
    """Run a few example queries to demonstrate functionality."""
    print("\n" + "="*60)
    print("📊 Running Example Queries")
    print("="*60)

    queries = [
        "What are the main factors courts consider in 1782 applications?",
        "Are there any cases discussing international arbitration?",
        "What are common reasons for denial?",
    ]

    query_engine = index.as_query_engine(similarity_top_k=3)

    for i, query in enumerate(queries, 1):
        print(f"\n{i}. ❓ {query}")
        print("-" * 60)

        try:
            response = query_engine.query(query)
            print(f"📝 {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main entry point."""
    print("\n🏛️  1782 Case Research Tool")
    print("   Powered by CrewAI + LlamaIndex\n")

    # Check setup
    is_ready, message = check_setup()
    print(message)

    if not is_ready:
        print("\n💡 Tip: Once configured, this tool will:")
        print("   • Index all your 1782 case PDFs")
        print("   • Enable semantic search across all cases")
        print("   • Answer questions about case law")
        print("   • Identify relevant precedents")
        sys.exit(1)

    print("\n🚀 Initializing research tool...")

    try:
        # Create or load index
        index = create_legal_index()

        # Ask user what they want to do
        print("\n" + "="*60)
        print("What would you like to do?")
        print("="*60)
        print("1. Run example queries (demo)")
        print("2. Interactive query interface")
        print("3. Both")
        print("="*60)

        choice = input("\nYour choice (1-3): ").strip()

        if choice == "1":
            run_example_queries(index)
        elif choice == "2":
            simple_query_interface(index)
        elif choice == "3":
            run_example_queries(index)
            input("\n⏸️  Press Enter to continue to interactive mode...")
            simple_query_interface(index)
        else:
            print("Invalid choice. Running interactive mode...")
            simple_query_interface(index)

        print("\n✅ Research session complete!")

    except ImportError as e:
        print(f"\n❌ Missing package: {e}")
        print("\nPlease install required packages:")
        print("  pip install llama-index")
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

