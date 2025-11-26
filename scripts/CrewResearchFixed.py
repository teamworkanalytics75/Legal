"""
CrewAI Deep Research - Fixed Version
Multi-agent collaboration with local LLMs -> Markdown + PDF
"""

import os
from pathlib import Path
from datetime import datetime

# Import CrewAI components
try:
    from crewai import Agent, Task, Crew
    from crewai.llm import LLM
except ImportError:
    print("Installing CrewAI...")
    import subprocess
    subprocess.run(["pip", "install", "--user", "crewai", "crewai-tools"])
    from crewai import Agent, Task, Crew
    from crewai.llm import LLM

from ddgs import DDGS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER


def web_search(query: str) -> str:
    """Search internet with DuckDuckGo"""
    try:
        results = DDGS().text(query, max_results=8)
        if not results:
            return f"No results for: {query}"

        output = [f"Search: {query}\n"]
        for i, r in enumerate(results, 1):
            output.append(f"{i}. {r['title']}")
            output.append(f"   {r['body'][:200]}...")
            output.append(f"   {r['href']}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def create_pdf_report(content: str, filepath: str, title: str):
    """Generate PDF from content"""
    try:
        doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        story = []
        styles = getSampleStyleSheet()

        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=20, alignment=TA_CENTER, spaceAfter=30)
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2*inch))

        # Metadata
        story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%B %d, %Y')}</i>", styles['Normal']))
        story.append(Paragraph("<i>System: CrewAI + Qwen2.5 14B (Local)</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # Content
        for line in content.split('\n'):
            if line.strip():
                story.append(Paragraph(line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'), styles['BodyText']))
                story.append(Spacer(1, 0.1*inch))

        doc.build(story)
        return True
    except Exception as e:
        print(f"PDF error: {e}")
        return False


def research(question: str):
    """Main research function with CrewAI agents"""

    print("\n" + "="*70)
    print("CrewAI Multi-Agent Research")
    print("="*70)
    print(f"\nQuestion: {question}\n")
    print("-"*70 + "\n")

    # Configure local LLM
    try:
        local_llm = LLM(model="ollama/qwen2.5:14b", base_url="http://localhost:11434")
    except Exception as e:
        print(f"LLM config error: {e}")
        print("Trying alternative configuration...")
        local_llm = "ollama/qwen2.5:14b"

    # Create agents
    print("[1/3] Creating agents...")

    researcher = Agent(
        role='Internet Researcher',
        goal='Search and gather comprehensive information',
        backstory='Expert researcher skilled at finding quality sources',
        llm=local_llm,
        verbose=False
    )

    analyst = Agent(
        role='Academic Analyst',
        goal='Analyze findings and synthesize insights',
        backstory='PhD-level analyst expert at connecting concepts',
        llm=local_llm,
        verbose=False
    )

    writer = Agent(
        role='Report Writer',
        goal='Create professional research reports',
        backstory='Experienced academic writer',
        llm=local_llm,
        verbose=False
    )

    print("    Agents created\n")

    # Create tasks
    print("[2/3] Defining tasks...")

    task1 = Task(
        description=f"""Search internet for: {question}

Gather information from multiple sources. Find academic papers, expert opinions, credible sources.
Summarize key findings with source citations.""",
        agent=researcher,
        expected_output="Research summary with sources"
    )

    task2 = Task(
        description=f"""Analyze research on: {question}

Synthesize findings, identify patterns, draw conclusions.
Provide critical analysis and insights.""",
        agent=analyst,
        expected_output="Analytical synthesis",
        context=[task1]
    )

    task3 = Task(
        description=f"""Write comprehensive report on: {question}

Structure:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusion
5. Sources

Professional academic style, 1500-2000 words.""",
        agent=writer,
        expected_output="Complete professional report",
        context=[task1, task2]
    )

    print("    Tasks defined\n")

    # Execute
    print("[3/3] Executing research (this may take 5-10 minutes)...\n")
    print("-"*70 + "\n")

    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[task1, task2, task3],
        verbose=True
    )

    result = crew.kickoff()

    # Save outputs
    print("\n" + "="*70)
    print("Saving reports...")
    print("="*70 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in question[:40] if c.isalnum() or c in ' ').replace(' ', '_')

    Path("reports").mkdir(exist_ok=True)

    # Markdown
    md_file = f"reports/{timestamp}_{safe_name}.md"
    md_content = f"""# Research Report

**Question:** {question}

**Date:** {datetime.now().strftime("%B %d, %Y %I:%M %p")}

**System:** CrewAI Multi-Agent (Local LLMs)

---

{result}

---

**Generated by:**
- 3 AI Agents (Researcher, Analyst, Writer)
- Model: Qwen2.5 14B (Local)
- Cost: $0.00
"""

    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"[OK] Markdown: {md_file}")

    # PDF
    pdf_file = f"reports/{timestamp}_{safe_name}.pdf"
    if create_pdf_report(str(result), pdf_file, question):
        print(f"[OK] PDF: {pdf_file}")

    # Summary
    print("\n" + "="*70)
    print("RESEARCH COMPLETE!")
    print("="*70)
    print(f"\nFiles:")
    print(f"   - {md_file}")
    print(f"   - {pdf_file}")
    print(f"\nCost: $0.00")
    print(f"Agents: 3 (all local)")
    print(f"Privacy: 100% local\n")

    return result


if __name__ == "__main__":
    import sys

    # Get question
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = "According to Margaret Lewis, what are the defining temporal patterns of Chinese crackdowns—e.g., short-term intensity, legal exceptionalism, selective enforcement—and how do they differ from ongoing bureaucratic control?"

    try:
        research(q)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

