"""
DEEP RESEARCH: Local LLMs + Internet Access
Uses YOUR actual setup:
- Qwen2.5 14B (local, free)
- 9 PDFs from 1782 Case PDF Database
- Internet search via DuckDuckGo (no API needed!)
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def test_ollama_connection():
    """Test connection to Ollama."""
    print("🔍 Checking Ollama connection...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama running with {len(models)} models:")
            for m in models:
                print(f"   • {m['name']}")
            return True
    except Exception as e:
        print(f"❌ Ollama not responding: {e}")
        print("\n💡 Start Ollama:")
        print("   1. Search 'Ollama' in Windows Start menu")
        print("   2. Click to start the app")
        return False


def search_internet(query: str) -> str:
    """Search the internet using DuckDuckGo (no API key needed!)."""
    try:
        from duckduckgo_search import DDGS

        print(f"🌐 Searching internet for: {query}")

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   {result['body']}")
            output.append(f"   Source: {result['href']}\n")

        return "\n".join(output)

    except ImportError:
        return "❌ duckduckgo-search not installed. Run: pip install duckduckgo-search"
    except Exception as e:
        return f"❌ Search error: {e}"


def analyze_with_qwen(prompt: str) -> str:
    """Analyze with Qwen2.5 14B local model."""
    try:
        from llama_index.llms.ollama import Ollama

        llm = Ollama(model="qwen2.5:14b", request_timeout=120.0, temperature=0.1)
        response = llm.complete(prompt)
        return response.text

    except Exception as e:
        return f"❌ Error: {e}"


def deep_research_query(question: str, search_internet_flag: bool = True):
    """
    Perform deep research combining internet + local LLM.

    Args:
        question: Your research question
        search_internet_flag: Whether to search the internet
    """
    print("\n" + "="*70)
    print(f"🔬 DEEP RESEARCH: {question}")
    print("="*70 + "\n")

    context = ""

    # Step 1: Search internet if requested
    if search_internet_flag:
        print("📍 Step 1: Searching the Internet")
        print("-"*70)
        search_results = search_internet(question)
        print(search_results)
        print("-"*70 + "\n")
        context += f"\n\nInternet Search Results:\n{search_results}\n"

    # Step 2: Analyze with local LLM
    print("📍 Step 2: Analyzing with Qwen2.5 14B (Local)")
    print("-"*70)

    analysis_prompt = f"""Based on the following information, provide a comprehensive answer to this question:

Question: {question}

{context}

Provide a detailed, well-structured analysis. Include:
1. Key findings
2. Important details
3. Any relevant context
4. Conclusions or recommendations

Be thorough but concise."""

    print("💭 Qwen2.5 14B analyzing...\n")
    analysis = analyze_with_qwen(analysis_prompt)
    print(analysis)
    print("-"*70 + "\n")

    # Summary
    print("✅ Research Complete!")
    print(f"   • Internet Search: {'✅ Used' if search_internet_flag else '⏭️ Skipped'}")
    print(f"   • Local Analysis: ✅ Qwen2.5 14B")
    print(f"   • Cost: $0.00")
    print(f"   • Privacy: {'Partial (web search)' if search_internet_flag else 'Complete (100%)'}")


def index_local_pdfs():
    """Index the 9 PDFs from 1782 Case PDF Database."""
    print("\n" + "="*70)
    print("📚 Indexing Local 1782 Case PDFs")
    print("="*70 + "\n")

    pdf_dir = Path("1782 Case PDF Database")

    if not pdf_dir.exists():
        print(f"❌ Directory not found: {pdf_dir}")
        return None

    # Count PDFs
    pdfs = list(pdf_dir.glob("*.pdf"))
    print(f"📂 Found {len(pdfs)} PDF files")

    if len(pdfs) == 0:
        print("⚠️  No PDFs found in directory")
        return None

    try:
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.core import StorageContext, load_index_from_storage

        # Check for existing index
        storage_dir = Path("./storage/qwen_hybrid_index")
        if storage_dir.exists():
            print(f"📂 Loading existing index from {storage_dir}")
            storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
            index = load_index_from_storage(storage_context)
            print("✅ Index loaded!\n")
            return index

        # Configure local models
        print("⚙️  Configuring Qwen2.5 14B...")
        Settings.llm = Ollama(model="qwen2.5:14b", request_timeout=120.0, temperature=0.1)

        print("⚙️  Configuring local embeddings (free)...")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Load PDFs
        print(f"📚 Loading {len(pdfs)} PDFs...")
        documents = SimpleDirectoryReader(str(pdf_dir)).load_data()
        print(f"✅ Loaded {len(documents)} document chunks")

        # Create index
        print(f"🔨 Creating vector index with Qwen2.5 14B...")
        index = VectorStoreIndex.from_documents(documents, show_progress=True)

        # Save
        print(f"\n💾 Saving index to {storage_dir}...")
        storage_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=str(storage_dir))

        print("✅ Index created and saved!\n")
        return index

    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n💡 Install: pip install llama-index-embeddings-huggingface sentence-transformers")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_local_docs(index, question: str):
    """Query the local PDF database."""
    if index is None:
        print("⚠️  No index available")
        return None

    print("\n" + "="*70)
    print(f"📄 Querying Local PDFs: {question}")
    print("="*70 + "\n")

    query_engine = index.as_query_engine(similarity_top_k=3)

    print("🔎 Searching local documents with Qwen2.5 14B...\n")
    print("-"*70)

    response = query_engine.query(question)
    print(response)
    print("-"*70)

    print("\n✅ Local query complete (0 API costs)")
    return response


def hybrid_research(question: str, index=None):
    """
    ULTIMATE HYBRID: Search internet + Query local docs + Synthesize
    """
    print("\n" + "="*70)
    print(f"🔥 HYBRID DEEP RESEARCH")
    print(f"Question: {question}")
    print("="*70 + "\n")

    # Step 1: Internet search
    print("📍 Step 1: Internet Research")
    internet_results = search_internet(question)
    print(f"✅ Found internet sources\n")

    # Step 2: Local docs search
    print("📍 Step 2: Local Documents")
    local_results = ""
    if index:
        query_engine = index.as_query_engine(similarity_top_k=3)
        local_response = query_engine.query(question)
        local_results = str(local_response)
        print(f"✅ Found relevant local content\n")
    else:
        print("⏭️  No local index available\n")

    # Step 3: Synthesize with Qwen2.5
    print("📍 Step 3: Synthesis with Qwen2.5 14B")
    print("-"*70)

    synthesis_prompt = f"""You are a legal research expert. Synthesize information from multiple sources to answer this question:

QUESTION: {question}

INTERNET SOURCES:
{internet_results}

LOCAL DOCUMENTS:
{local_results}

Provide a comprehensive, well-structured answer that:
1. Combines insights from both internet and local sources
2. Identifies key findings
3. Notes any contradictions or gaps
4. Provides actionable conclusions

Be thorough, accurate, and cite when information comes from internet vs local sources."""

    synthesis = analyze_with_qwen(synthesis_prompt)
    print(synthesis)
    print("-"*70 + "\n")

    print("✅ HYBRID RESEARCH COMPLETE!")
    print(f"   • Internet Sources: ✅")
    print(f"   • Local Documents: {'✅' if index else '⏭️'}")
    print(f"   • AI Analysis: ✅ Qwen2.5 14B")
    print(f"   • Total Cost: $0.00")


def interactive_mode():
    """Interactive research interface."""
    print("\n" + "="*70)
    print("🚀 DEEP RESEARCH - Interactive Mode")
    print("="*70)
    print("\nCommands:")
    print("  1. Type your question to research")
    print("  2. 'internet <question>' - Internet only")
    print("  3. 'local <question>' - Local docs only")
    print("  4. 'hybrid <question>' - Both sources")
    print("  5. 'quit' - Exit")
    print("="*70 + "\n")

    # Try to load local index
    pdf_dir = Path("1782 Case PDF Database")
    if pdf_dir.exists() and len(list(pdf_dir.glob("*.pdf"))) > 0:
        print("📚 Local PDFs available. Creating index...")
        index = index_local_pdfs()
    else:
        index = None
        print("⚠️  No local PDFs found. Internet research only.")

    print("\n" + "-"*70 + "\n")

    while True:
        try:
            user_input = input("🔬 Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Parse command
            if user_input.lower().startswith('internet '):
                question = user_input[9:].strip()
                deep_research_query(question, search_internet_flag=True)

            elif user_input.lower().startswith('local '):
                question = user_input[6:].strip()
                if index:
                    query_local_docs(index, question)
                else:
                    print("⚠️  No local index available")

            elif user_input.lower().startswith('hybrid '):
                question = user_input[7:].strip()
                hybrid_research(question, index)

            else:
                # Default to hybrid if index available, otherwise internet
                if index:
                    hybrid_research(user_input, index)
                else:
                    deep_research_query(user_input, search_internet_flag=True)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Main entry point."""
    print("\n🔬 DEEP RESEARCH: Local LLMs + Internet Access")
    print("   Your Setup: Qwen2.5 14B + 9 PDFs + Web Search")
    print("\n" + "="*70)

    # Test connection
    if not test_ollama_connection():
        return

    print("\n✅ System Ready!")
    print("\nWhat would you like to do?")
    print("  1. Interactive research mode (recommended)")
    print("  2. Quick internet search test")
    print("  3. Index local PDFs only")
    print("  4. Exit")

    choice = input("\nYour choice (1-4): ").strip()

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        question = input("\nWhat do you want to research? ")
        deep_research_query(question, search_internet_flag=True)
    elif choice == "3":
        index_local_pdfs()
    else:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

