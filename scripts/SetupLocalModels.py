"""
Setup script for local LLM models
Configures Phi-3, Qwen2, and other local models for use with CrewAI + LlamaIndex
"""

import subprocess
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


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60 + '\n')


def install_packages():
    """Install required packages for local models."""
    print_header("📦 Installing Required Packages")

    packages = [
        'llama-index-llms-ollama',
        'llama-index-embeddings-huggingface',
        'sentence-transformers',
        'torch',  # Required for local embeddings
        'transformers',  # Additional support
    ]

    for package in packages:
        print(f"📥 Installing {package}...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', package, '--quiet'],
                check=True,
                capture_output=True
            )
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to install {package}: {e}")

    print("\n✅ Package installation complete")


def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    print_header("🔍 Checking Ollama Installation")

    try:
        result = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✅ Ollama installed: {result.stdout.strip()}")
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ Ollama not found!")
        print("\n📥 Please install Ollama:")
        print("   Option 1: winget install Ollama.Ollama")
        print("   Option 2: Download from https://ollama.ai/download")
        print("\nAfter installing, run this script again.")
        return False


def list_available_models() -> list:
    """List currently available Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = [line.split()[0] for line in lines if line.strip()]
            return models
        return []
    except Exception:
        return []


def pull_recommended_models():
    """Pull recommended models for legal research."""
    print_header("📥 Downloading Recommended Models")

    # Check what's already available
    existing_models = list_available_models()
    print(f"📋 Currently installed: {', '.join(existing_models) if existing_models else 'None'}\n")

    recommended = {
        'phi3:mini': 'Phi-3 Mini (Best for legal - 128K context)',
        'llama3.2:latest': 'Llama 3.2 (Alternative option)',
        'qwen2.5:1.5b': 'Qwen 2.5 (Fast and efficient)',
    }

    print("Recommended models for legal research:")
    for i, (model, desc) in enumerate(recommended.items(), 1):
        status = "✅ Installed" if model in existing_models else "⬇️  Download"
        print(f"  {i}. {model} - {desc} [{status}]")

    print("\n" + "-"*60)
    choice = input("\nDownload Phi-3 Mini now? (recommended) [Y/n]: ").strip().lower()

    if choice in ['', 'y', 'yes']:
        print("\n📥 Pulling Phi-3 Mini (this may take a few minutes)...")
        try:
            subprocess.run(['ollama', 'pull', 'phi3:mini'], check=True)
            print("✅ Phi-3 Mini downloaded successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to download Phi-3")
            print("   You can download it later with: ollama pull phi3:mini")
    else:
        print("\n💡 Skipped. Download later with: ollama pull phi3:mini")


def test_local_llm():
    """Test local LLM connection."""
    print_header("🧪 Testing Local LLM")

    try:
        from llama_index.llms.ollama import Ollama

        print("🔌 Connecting to Ollama...")
        llm = Ollama(model="phi3:mini", request_timeout=30.0)

        print("💬 Sending test query...")
        response = llm.complete("Respond with exactly: 'Local LLM working!'")

        print(f"📝 Response: {response.text}")
        print("✅ Local LLM is working!\n")
        return True

    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is running: ollama serve")
        print("   2. Check if model is pulled: ollama list")
        print("   3. Try pulling: ollama pull phi3:mini")
        return False


def test_local_embeddings():
    """Test local embeddings."""
    print_header("🧪 Testing Local Embeddings")

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        print("📥 Loading BGE embedding model (first time may download)...")
        embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        print("🔢 Creating test embedding...")
        embedding = embed_model.get_text_embedding("This is a test sentence.")

        print(f"✅ Embeddings working! (Dimension: {len(embedding)})\n")
        return True

    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print("\n💡 Try reinstalling: pip install sentence-transformers")
        return False


def check_local_models():
    """Check for locally cached models."""
    print_header("📂 Checking Local Model Cache")

    models_dir = Path("models_cache")
    if not models_dir.exists():
        print("⚠️  models_cache directory not found")
        return

    found_models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            found_models.append(model_dir.name)

    if found_models:
        print("✅ Found local models:")
        for model in found_models:
            print(f"   • {model}")
        print("\n💡 These can be used with transformers/HuggingFace")
    else:
        print("⚠️  No local models found in models_cache/")


def create_config_file():
    """Create a configuration file for local models."""
    print_header("📝 Creating Configuration File")

    config = """# Local Models Configuration
# Generated by setup_local_models.py

[llm]
# Primary model for reasoning
provider = ollama
model = phi3:mini
temperature = 0.1
request_timeout = 120

[embeddings]
# Local embedding model
provider = huggingface
model = BAAI/bge-small-en-v1.5

[ollama]
# Ollama server settings
base_url = http://localhost:11434
timeout = 120

[storage]
# Where to store indices
index_dir = ./storage/local_indices
cache_dir = ./cache

[performance]
# Performance tuning
chunk_size = 1024
chunk_overlap = 100
similarity_top_k = 5
"""

    config_path = Path("local_models_config.ini")
    config_path.write_text(config)
    print(f"✅ Configuration saved to: {config_path}")
    print("   You can edit this file to customize settings")


def create_example_script():
    """Create an example script using local models."""
    print_header("📝 Creating Example Script")

    example = '''"""
Example: Using Local Models for Legal Research
NO API COSTS - Completely free and private!
"""

from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main():
    print("🚀 Local Legal Research Example\\n")

    # Configure local models
    print("⚙️  Configuring local models...")
    Settings.llm = Ollama(
        model="phi3:mini",
        request_timeout=120.0,
        temperature=0.1
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Load a test document
    print("📚 Loading documents...")
    # Replace with your actual document directory
    documents = SimpleDirectoryReader('1782 Case PDF Database').load_data()
    print(f"✅ Loaded {len(documents)} documents")

    # Create index
    print("🔨 Creating vector index...")
    index = VectorStoreIndex.from_documents(documents[:5])  # Start with 5 docs

    # Query the index
    print("🔍 Querying with local models...\\n")
    query_engine = index.as_query_engine(similarity_top_k=3)

    question = "What are the Intel factors in 1782 applications?"
    print(f"❓ Question: {question}\\n")

    response = query_engine.query(question)
    print(f"📝 Answer:\\n{response}\\n")

    print("✅ Local research complete - no API costs!")

if __name__ == "__main__":
    main()
'''

    example_path = Path("local_legal_research_example.py")
    example_path.write_text(example)
    print(f"✅ Example saved to: {example_path}")
    print("   Run it with: python local_legal_research_example.py")


def print_summary(llm_ok: bool, embed_ok: bool):
    """Print setup summary."""
    print_header("📊 Setup Summary")

    print("Component Status:")
    print(f"  • Ollama: {'✅ Working' if llm_ok else '❌ Not working'}")
    print(f"  • Local LLM: {'✅ Working' if llm_ok else '❌ Not working'}")
    print(f"  • Embeddings: {'✅ Working' if embed_ok else '❌ Not working'}")

    if llm_ok and embed_ok:
        print("\n🎉 All systems ready for local research!")
        print("\n📚 Next Steps:")
        print("   1. Review: LOCAL_MODELS_SETUP_GUIDE.md")
        print("   2. Run: python local_legal_research_example.py")
        print("   3. Start researching with NO API costs!")
        print("\n💰 Cost savings: 100% (vs OpenAI API)")
        print("🔒 Privacy: 100% (all data stays local)")
    else:
        print("\n⚠️  Some components need attention")
        print("\n💡 Troubleshooting:")
        if not llm_ok:
            print("   • Install Ollama: https://ollama.ai/download")
            print("   • Pull model: ollama pull phi3:mini")
        if not embed_ok:
            print("   • Install: pip install sentence-transformers torch")


def main():
    """Main setup process."""
    print("\n🤖 Local LLM Models Setup for Legal Research")
    print("   Using: Phi-3, Qwen, Llama (your downloaded models)")

    # Step 1: Install packages
    install_packages()

    # Step 2: Check Ollama
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("\n⏸️  Setup paused. Install Ollama and run again.")
        return

    # Step 3: Check/download models
    pull_recommended_models()

    # Step 4: Check local cache
    check_local_models()

    # Step 5: Test LLM
    llm_ok = test_local_llm()

    # Step 6: Test embeddings
    embed_ok = test_local_embeddings()

    # Step 7: Create config
    create_config_file()

    # Step 8: Create example
    create_example_script()

    # Step 9: Summary
    print_summary(llm_ok, embed_ok)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

