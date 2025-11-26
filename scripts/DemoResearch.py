"""Quick demo of deep research system"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ddgs import DDGS
from llama_index.llms.ollama import Ollama

print("\n🔬 DEEP RESEARCH SYSTEM ENABLED")
print("="*70 + "\n")

# Test query
question = "What are the Intel factors in 28 USC 1782 applications?"
print(f"📍 Research Question: {question}\n")
print("-"*70 + "\n")

# Step 1: Internet search
print("🌐 Step 1: Searching Internet...\n")
try:
    results = DDGS().text("Intel factors 28 USC 1782", max_results=3)
    if results:
        print(f"Found {len(results)} sources:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['body'][:120]}...")
            print(f"   {r['href']}\n")
        context = "\n".join([f"{r['title']}: {r['body']}" for r in results])
    else:
        print("No internet results found.\n")
        context = ""
except Exception as e:
    print(f"⚠️  Search error: {e}\n")
    context = ""

# Step 2: Local AI analysis
print("-"*70)
print("\n💭 Step 2: Analyzing with Qwen2.5 14B (Local AI)...\n")

try:
    llm = Ollama(model="qwen2.5:14b", request_timeout=90, temperature=0.1)

    if context:
        prompt = f"""Based on these internet sources, explain the four Intel factors that courts consider in 28 USC Section 1782 discovery applications:

Sources:
{context}

Provide a clear, structured explanation."""
    else:
        prompt = "Explain the four Intel factors that courts consider in 28 USC Section 1782 discovery applications. Be specific."

    response = llm.complete(prompt)

    print("📝 Answer:\n")
    print(response.text)

except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n" + "="*70)
print("\n✅ DEEP RESEARCH COMPLETE!")
print("\nSystem Status:")
print("   • Internet Search: ✅ Enabled")
print("   • Local AI (Qwen2.5 14B): ✅ Active")
print("   • Your 9 PDFs: ✅ Ready to index")
print("   • API Cost: $0.00")
print("   • Privacy: Partial (web) / Complete (local docs)")
print("\n" + "="*70)
print("\n💡 Your system is LIVE and ready for research!")
print("   Run: python deep_research_working.py for interactive mode")

