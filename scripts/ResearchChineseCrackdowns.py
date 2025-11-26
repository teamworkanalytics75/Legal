"""
Deep Research: Margaret Lewis on Chinese Crackdowns
LOCAL LLM ONLY (Qwen2.5 14B) - No OpenAI API
"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ddgs import DDGS
from llama_index.llms.ollama import Ollama

print("\n" + "="*80)
print("🔬 DEEP RESEARCH: Margaret Lewis on Chinese Crackdowns")
print("   Using: Local LLM ONLY (Qwen2.5 14B)")
print("="*80 + "\n")

# Research question
question = """According to Margaret Lewis, what are the defining temporal patterns of Chinese crackdowns—e.g., short-term intensity, legal exceptionalism, selective enforcement—and how do they differ from ongoing bureaucratic control?"""

print(f"📍 Research Question:\n{question}\n")
print("-"*80 + "\n")

# Step 1: Internet Search
print("🌐 Step 1: Searching Internet for Margaret Lewis + Chinese crackdowns\n")

search_queries = [
    "Margaret Lewis Chinese crackdowns temporal patterns",
    "Margaret Lewis China legal exceptionalism selective enforcement",
    "Margaret Lewis China bureaucratic control crackdowns"
]

all_results = []
for query in search_queries:
    print(f"   Searching: {query}")
    try:
        results = DDGS().text(query, max_results=5)
        if results:
            all_results.extend(results)
            print(f"   ✅ Found {len(results)} sources")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

print(f"\n📚 Total sources found: {len(all_results)}\n")

if all_results:
    print("Top sources:\n")
    # Remove duplicates by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        if r['href'] not in seen_urls:
            seen_urls.add(r['href'])
            unique_results.append(r)

    for i, r in enumerate(unique_results[:10], 1):
        print(f"{i}. {r['title']}")
        print(f"   {r['body'][:150]}...")
        print(f"   Source: {r['href']}\n")

    # Build context
    context = "\n\n".join([
        f"Source {i}: {r['title']}\n{r['body']}"
        for i, r in enumerate(unique_results[:10], 1)
    ])
else:
    print("⚠️  No internet sources found. Using LLM knowledge only.\n")
    context = ""

# Step 2: Deep Analysis with LOCAL LLM
print("-"*80)
print("\n💭 Step 2: Deep Analysis with Qwen2.5 14B (LOCAL - No API costs)\n")
print("   Model: qwen2.5:14b (9GB local model)")
print("   Temperature: 0.1 (focused)")
print("   Timeout: 180 seconds\n")

try:
    # Configure LOCAL LLM only
    llm = Ollama(
        model="qwen2.5:14b",
        request_timeout=180,
        temperature=0.1
    )

    if context:
        prompt = f"""You are a legal and political science expert analyzing Chinese governance patterns.

Research Question:
{question}

Internet Sources Retrieved:
{context}

Task:
1. Synthesize information from the sources about Margaret Lewis's analysis
2. Identify the temporal patterns of Chinese crackdowns she describes
3. Explain concepts like:
   - Short-term intensity
   - Legal exceptionalism
   - Selective enforcement
4. Contrast these crackdown patterns with ongoing bureaucratic control
5. Provide specific examples if mentioned in the sources
6. Note any scholarly citations or publications

Provide a comprehensive, well-structured academic analysis."""
    else:
        prompt = f"""You are a legal and political science expert.

Research Question:
{question}

Based on your knowledge of Chinese legal systems and governance:
1. Explain temporal patterns of Chinese crackdowns
2. Discuss concepts like short-term intensity, legal exceptionalism, selective enforcement
3. Contrast crackdown patterns with routine bureaucratic control
4. Note: We don't have specific sources from Margaret Lewis, so explain these concepts generally

Provide a thorough academic analysis."""

    print("🤖 Qwen2.5 14B analyzing...\n")
    print("-"*80 + "\n")

    response = llm.complete(prompt)

    print("📝 ANALYSIS:\n")
    print(response.text)
    print("\n" + "-"*80)

except Exception as e:
    print(f"❌ LLM Error: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*80)
print("✅ DEEP RESEARCH COMPLETE")
print("="*80)
print("\nSystem Details:")
print(f"   • Internet Sources: {len(unique_results) if all_results else 0}")
print(f"   • Analysis Model: Qwen2.5 14B (LOCAL)")
print(f"   • OpenAI API Used: ❌ NO")
print(f"   • Cost: $0.00")
print(f"   • Privacy: Partial (web search) + 100% local AI")
print("\n" + "="*80)

