"""
Legal Research with Qwen2.5 14B - Your local model!
Completely free, private, and powerful for legal analysis
"""

import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def test_qwen_simple():
    """Simple test of Qwen2.5 14B model."""
    print("\n🚀 Testing Qwen2.5 14B Model")
    print("="*60 + "\n")

    try:
        from llama_index.llms.ollama import Ollama

        # Configure Qwen2.5 14B
        llm = Ollama(
            model="qwen2.5:14b",
            request_timeout=120.0,
            temperature=0.1  # Lower = more focused
        )

        # Test with legal question
        print("❓ Question: What are the four Intel factors in 1782 applications?")
        print("\n💭 Qwen2.5 thinking...\n")
        print("-" * 60)

        response = llm.complete(
            """Explain the four Intel factors that courts consider
            when deciding 28 U.S.C. § 1782 applications for discovery.
            Be specific and concise."""
        )

        print(response.text)
        print("-" * 60)
        print("\n✅ Qwen2.5 14B is working!\n")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def index_with_qwen():
    """Index your 1782 PDFs with Qwen2.5 and local embeddings."""
    print("\n" + "="*60)
    print("📚 Indexing 1782 PDFs with Qwen2.5 14B")
    print("="*60 + "\n")

    try:
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import StorageContext, load_index_from_storage

        # Check if index already exists
        storage_dir = Path("./storage/qwen_1782_index")

        if storage_dir.exists():
            print(f"📂 Loading existing index from {storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
            index = load_index_from_storage(storage_context)
            print("✅ Index loaded!\n")
            return index

        # Configure Qwen2.5 14B
        print("⚙️  Configuring Qwen2.5 14B...")
        Settings.llm = Ollama(
            model="qwen2.5:14b",
            request_timeout=120.0,
            temperature=0.1
        )

        # Configure local embeddings (FREE!)
        print("⚙️  Configuring local embeddings...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Load PDFs
        pdf_dir = Path("1782 Case PDF Database")
        if not pdf_dir.exists():
            print(f"⚠️  Directory not found: {pdf_dir}")
            print("   Please update the path to your PDF directory")
            return None

        print(f"📚 Loading PDFs from {pdf_dir}...")
        documents = SimpleDirectoryReader(str(pdf_dir)).load_data()
        print(f"✅ Loaded {len(documents)} documents")

        # Ask user how many to index
        print(f"\n💡 Found {len(documents)} documents")
        print("   Start with 5-10 for testing, then scale up")

        try:
            count = input(f"\nHow many to index? (press Enter for 5): ").strip()
            count = int(count) if count else 5
            count = min(count, len(documents))
        except ValueError:
            count = 5

        print(f"\n🔨 Creating index with {count} documents...")
        print("   (This will take a few minutes on first run)\n")

        index = VectorStoreIndex.from_documents(
            documents[:count],
            show_progress=True
        )

        # Save for next time
        print(f"\n💾 Saving index to {storage_dir}...")
        storage_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_dir))

        print("✅ Index created and saved!\n")
        return index

    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n💡 Install missing packages:")
        print("   pip install llama-index-embeddings-huggingface sentence-transformers")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_legal_database(index):
    """Interactive query interface."""
    if index is None:
        print("⚠️  No index available. Run indexing first.")
        return

    print("\n" + "="*60)
    print("🔍 Legal Research Query Interface")
    print("   Powered by Qwen2.5 14B (Local, Free, Private)")
    print("="*60)

    example_queries = [
        "What are the Intel factors for 1782 applications?",
        "What factors indicate discoverable interest?",
        "Common reasons courts deny 1782 requests?",
        "Cases involving international arbitration?",
        "How do courts analyze foreign discoverability?",
    ]

    print("\n📋 Example queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"   {i}. {q}")

    print("\n" + "-"*60)
    print("Commands: 'quit' to exit, 'examples' to see examples again")
    print("-"*60 + "\n")

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="tree_summarize"
    )

    while True:
        try:
            query = input("❓ Your question: ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if query.lower() == 'examples':
                print("\n📋 Example queries:")
                for i, q in enumerate(example_queries, 1):
                    print(f"   {i}. {q}")
                continue

            print("\n🔎 Searching with Qwen2.5 14B...\n")
            print("-" * 60)

            response = query_engine.query(query)
            print(response)

            print("-" * 60)
            print(f"✅ Query complete (0 API costs)\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main workflow."""
    print("\n🏛️  1782 Legal Research with Qwen2.5 14B")
    print("   Local Model - Zero Costs - Complete Privacy")
    print("\n" + "="*60)

    # Step 1: Test the model
    print("\n📍 Step 1: Testing Qwen2.5 14B")
    if not test_qwen_simple():
        print("\n⚠️  Model test failed. Please check Ollama is running.")
        return

    # Step 2: Create/load index
    print("\n📍 Step 2: Creating Vector Index")
    index = index_with_qwen()

    if index is None:
        print("\n⚠️  Indexing failed. Check errors above.")
        return

    # Step 3: Interactive queries
    print("\n📍 Step 3: Query Your Legal Database")
    query_legal_database(index)

    print("\n" + "="*60)
    print("✅ Session complete!")
    print("\n💰 Total API costs: $0.00")
    print("🔒 Privacy: 100% (all data stayed local)")
    print("📊 Quality: High (Qwen2.5 14B is a powerful model)")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

