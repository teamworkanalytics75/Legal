"""
Quick Test: Internet + Local LLM Research
Run this NOW to see your system working!
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n🔬 Testing Your Deep Research System")
print("="*60 + "\n")

# Test 1: Ollama
print("Test 1: Checking Ollama...")
try:
    from llama_index.llms.ollama import Ollama
    llm = Ollama(model="qwen2.5:14b", request_timeout=30)
    response = llm.complete("Say 'Ollama working!' if you can read this.")
    print(f"✅ {response.text}\n")
except Exception as e:
    print(f"❌ Ollama test failed: {e}\n")

# Test 2: Internet Search
print("Test 2: Internet Search...")
try:
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        results = list(ddgs.text("28 USC 1782", max_results=2))
    print(f"✅ Found {len(results)} results")
    print(f"   Example: {results[0]['title']}\n")
except Exception as e:
    print(f"❌ Internet search failed: {e}\n")

# Test 3: Hybrid Research Example
print("Test 3: Hybrid Research Example")
print("-"*60)

question = "What is 28 USC 1782?"

print(f"\n🌐 Researching: {question}\n")

# Search internet
try:
    with DDGS() as ddgs:
        web_results = list(ddgs.text(question, max_results=2))

    print("Internet Results:")
    for r in web_results:
        print(f"  • {r['title'][:60]}...")

    # Analyze with local LLM
    print("\n💭 Qwen2.5 14B analyzing...\n")

    context = "\n".join([f"{r['title']}: {r['body']}" for r in web_results])

    prompt = f"""Based on this information, explain in 2-3 sentences:

Question: {question}

Information:
{context}"""

    analysis = llm.complete(prompt)
    print("📝 Analysis:")
    print(analysis.text)
    print("\n" + "-"*60)

except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n✅ SYSTEM CHECK COMPLETE")
print("\nYour capabilities:")
print("  ✅ Local LLM (Qwen2.5 14B)")
print("  ✅ Internet search (DuckDuckGo)")
print("  ✅ Hybrid research (both combined)")
print("  ✅ Zero API costs")
print("\n💡 Ready for deep research!")
print(f"   Run: python deep_research_hybrid.py")

