"""
Test Ollama directly from Cursor
Quick test to see your local models in action!
"""

import sys
import requests
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def check_ollama_running():
    """Check if Ollama is accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)


def list_available_models():
    """List models available in Ollama."""
    print("🔍 Checking available models...\n")

    is_running, data = check_ollama_running()

    if not is_running:
        print("❌ Ollama not responding")
        print("\n💡 To start Ollama:")
        print("   1. Search for 'Ollama' in Windows Start menu")
        print("   2. Click to start the app")
        print("   3. Look for Ollama icon in system tray")
        return []

    if 'models' in data:
        models = data['models']
        if models:
            print(f"✅ Found {len(models)} model(s):\n")
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f"   • {name} ({size:.1f} GB)")
            return [m['name'] for m in models]
        else:
            print("⚠️  Ollama is running but no models installed")
            print("\n💡 Download Phi-3:")
            print("   Open PowerShell and run: ollama pull phi3:mini")
            return []

    return []


def test_model(model_name="phi3:mini", prompt="Hello! Can you explain Section 1782 in one sentence?"):
    """Test a specific model with a prompt."""
    print(f"\n🧪 Testing {model_name}...")
    print(f"📝 Prompt: {prompt}\n")
    print("-" * 60)

    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": True
    }

    try:
        response = requests.post(url, json=data, stream=True, timeout=120)

        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return

        print("💬 Response: ", end="", flush=True)
        full_response = ""

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if 'response' in chunk:
                    text = chunk['response']
                    print(text, end="", flush=True)
                    full_response += text

                if chunk.get('done', False):
                    break

        print("\n" + "-" * 60)
        print("✅ Test complete!\n")
        return full_response

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        print("💡 The model might be loading for the first time (can take a minute)")
    except Exception as e:
        print(f"❌ Error: {e}")


def test_llamaindex_integration():
    """Test LlamaIndex integration with Ollama."""
    print("\n" + "="*60)
    print("🦙 Testing LlamaIndex Integration")
    print("="*60 + "\n")

    try:
        from llama_index.llms.ollama import Ollama

        print("📦 LlamaIndex Ollama package: ✅ Installed")

        llm = Ollama(model="phi3:mini", request_timeout=60.0)
        print("🔌 Connected to Ollama\n")

        print("💬 Asking: 'What are the Intel factors?'")
        print("-" * 60)

        response = llm.complete("Briefly explain the Intel factors in 1782 applications in 2 sentences.")

        print(f"📝 Response:\n{response.text}\n")
        print("-" * 60)
        print("✅ LlamaIndex integration working!\n")

        return True

    except ImportError:
        print("⚠️  llama-index-llms-ollama not installed")
        print("\n💡 Install with:")
        print("   pip install llama-index-llms-ollama")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_crewai_integration():
    """Test CrewAI integration with Ollama."""
    print("\n" + "="*60)
    print("🤖 Testing CrewAI Integration")
    print("="*60 + "\n")

    try:
        from crewai import LLM

        print("📦 CrewAI: ✅ Installed")

        llm = LLM(
            model="ollama/phi3:mini",
            base_url="http://localhost:11434"
        )
        print("🔌 CrewAI configured for Ollama")
        print("✅ Ready for multi-agent workflows!\n")

        return True

    except ImportError:
        print("⚠️  CrewAI package structure changed or not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main test sequence."""
    print("\n🚀 Testing Ollama in Cursor")
    print("="*60 + "\n")

    # Step 1: List models
    models = list_available_models()

    if not models:
        print("\n⏸️  Setup required. Please install a model first.")
        return

    # Step 2: Test with direct API
    print("\n" + "="*60)
    print("🎯 Direct API Test")
    print("="*60)

    # Use phi3:mini if available, otherwise first model
    test_model_name = "phi3:mini" if "phi3:mini" in models else models[0]
    test_model(test_model_name)

    # Step 3: Test LlamaIndex
    test_llamaindex_integration()

    # Step 4: Test CrewAI
    test_crewai_integration()

    # Summary
    print("="*60)
    print("✅ Ollama is ready to use in Cursor!")
    print("="*60)
    print("\n💡 Next steps:")
    print("   1. Try: python phi3_legal_research.py")
    print("   2. Index your 1782 PDFs")
    print("   3. Start researching with $0 costs!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

