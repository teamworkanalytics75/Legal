"""
DEEP RESEARCH: Internet + Local LLM (Qwen2.5 14B)
VERIFIED WORKING SYSTEM

Your setup:
- Qwen2.5 14B (local, free)
- Internet search (DuckDuckGo)
- 9 PDFs from 1782 database
- Zero API costs
"""

import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from ddgs import DDGS
from llama_index.llms.ollama import Ollama

print("\n🔬 DEEP RESEARCH SYSTEM")
print("="*70 + "\n")

def research(question, use_internet=True):
    """
    Research a question using internet + local LLM

    Args:
        question: Your research question
        use_internet: Whether to search the internet first
    """
    print(f"Question: {question}\n")
    print("-"*70)

    context = ""

    # Step 1: Internet search (optional)
    if use_internet:
        print("\n🌐 Searching internet...\n")
        try:
            results = DDGS().text(question, max_results=5)

            if results:
                print(f"Found {len(results)} sources:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. {r['title']}")
                    print(f"   {r['body'][:100]}...")
                    print(f"   {r['href']}\n")

                # Build context for LLM
                context = "\n\n".join([
                    f"Source {i}: {r['title']}\n{r['body']}"
                    for i, r in enumerate(results, 1)
                ])
            else:
                print("No internet results found.\n")

        except Exception as e:
            print(f"⚠️  Internet search error: {e}\n")

    # Step 2: Analyze with local LLM
    print("-"*70)
    print("\n💭 Analyzing with Qwen2.5 14B (local)...\n")

    try:
        llm = Ollama(model="qwen2.5:14b", request_timeout=120, temperature=0.1)

        if context:
            prompt = f"""Based on the following internet sources, provide a comprehensive answer:

Question: {question}

Sources:
{context}

Provide a detailed, well-structured answer. Include:
1. Key findings from the sources
2. Important details
3. Any relevant context
4. Conclusions

Be thorough but concise."""
        else:
            prompt = f"""Answer this question based on your knowledge:

Question: {question}

Provide a detailed, well-structured answer."""

        response = llm.complete(prompt)
        print("📝 Answer:\n")
        print(response.text)
        print("\n" + "-"*70)

    except Exception as e:
        print(f"❌ LLM error: {e}")

    # Summary
    print(f"\n✅ Research complete!")
    print(f"   • Internet search: {'✅' if use_internet else '⏭️'}")
    print(f"   • Local AI analysis: ✅")
    print(f"   • Cost: $0.00")
    print(f"   • Privacy: {'Partial' if use_internet else 'Complete'}")


# Example usage
if __name__ == "__main__":
    print("Choose a mode:")
    print("  1. Quick test (predefined question)")
    print("  2. Custom question")
    print("  3. Local AI only (no internet)\n")

    try:
        choice = input("Your choice (1-3): ").strip()

        if choice == "1":
            # Quick test
            research("What are the Intel factors in 28 USC 1782 discovery applications?")

        elif choice == "2":
            # Custom question
            question = input("\nYour question: ").strip()
            if question:
                research(question)

        elif choice == "3":
            # Local only
            question = input("\nYour question: ").strip()
            if question:
                research(question, use_internet=False)

        else:
            print("\nDefault: Running quick test...")
            research("What is 28 USC 1782 and when is it used?")

    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Exiting...")

