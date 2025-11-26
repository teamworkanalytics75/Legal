"""Quick CrewAI Research - Command Line"""
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get question from command line or use default
if len(sys.argv) > 1:
    question = " ".join(sys.argv[1:])
else:
    question = "According to Margaret Lewis, what are the defining temporal patterns of Chinese crackdowns—e.g., short-term intensity, legal exceptionalism, selective enforcement—and how do they differ from ongoing bureaucratic control?"

print(f"\n🔬 CrewAI Research Starting...")
print(f"📋 Question: {question}\n")

# Import and run
from crewai_deep_research import conduct_research

try:
    conduct_research(question)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

