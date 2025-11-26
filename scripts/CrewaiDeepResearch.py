"""
CrewAI Deep Research System
Multi-agent collaboration with local LLMs
Outputs: Markdown + PDF reports for humans
"""

import sys
import os
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from crewai import Agent, Task, Crew, LLM
from ddgs import DDGS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import markdown2


# Configure local LLM
LOCAL_LLM = LLM(
    model="ollama/qwen2.5:14b",
    base_url="http://localhost:11434"
)

print("\n" + "="*80)
print("🤖 CrewAI Deep Research System")
print("   Multi-Agent Collaboration with Local LLMs")
print("="*80 + "\n")


# ============================================================================
# TOOLS: Web Search
# ============================================================================

def web_search_tool(query: str) -> str:
    """
    Search the internet using DuckDuckGo.

    Args:
        query: Search query string

    Returns:
        Formatted search results as string
    """
    try:
        results = DDGS().text(query, max_results=10)

        if not results:
            return f"No results found for: {query}"

        output = [f"Search results for: {query}\n"]
        output.append("="*60 + "\n")

        for i, result in enumerate(results, 1):
            output.append(f"\n{i}. {result['title']}")
            output.append(f"   {result['body']}")
            output.append(f"   Source: {result['href']}\n")

        return "\n".join(output)

    except Exception as e:
        return f"Search error: {str(e)}"


# ============================================================================
# AGENTS: Specialized Research Team
# ============================================================================

# Agent 1: Internet Researcher
researcher = Agent(
    role='Senior Internet Researcher',
    goal='Search the internet thoroughly and gather comprehensive information on the research topic',
    backstory="""You are an expert internet researcher with years of experience
    finding high-quality sources. You know how to craft effective search queries,
    evaluate source credibility, and gather comprehensive information. You excel
    at finding academic papers, expert opinions, and reliable data.""",
    llm=LOCAL_LLM,
    verbose=True,
    allow_delegation=False
)

# Agent 2: Academic Analyst
analyst = Agent(
    role='Academic Research Analyst',
    goal='Analyze research findings and synthesize insights into coherent analysis',
    backstory="""You are a PhD-level academic analyst with expertise across
    multiple disciplines. You excel at identifying patterns, connecting concepts,
    and drawing meaningful conclusions from diverse sources. You can critically
    evaluate arguments and synthesize complex information into clear insights.""",
    llm=LOCAL_LLM,
    verbose=True,
    allow_delegation=False
)

# Agent 3: Report Writer
writer = Agent(
    role='Professional Research Writer',
    goal='Transform research and analysis into well-structured, professional reports',
    backstory="""You are an experienced academic and professional writer. You
    specialize in creating clear, comprehensive research reports that are both
    scholarly rigorous and accessible to educated readers. You know how to
    structure arguments, cite sources properly, and present complex ideas clearly.""",
    llm=LOCAL_LLM,
    verbose=True,
    allow_delegation=False
)


# ============================================================================
# RESEARCH WORKFLOW
# ============================================================================

def conduct_research(research_question: str, output_dir: str = "reports"):
    """
    Conduct multi-agent research using CrewAI.

    Args:
        research_question: The question to research
        output_dir: Directory to save reports
    """

    print(f"\n📋 Research Question:")
    print(f"   {research_question}\n")
    print("-"*80 + "\n")

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in research_question[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_filename = safe_filename.replace(' ', '_')
    base_filename = f"{timestamp}_{safe_filename}"

    # ========================================================================
    # TASK 1: Internet Research
    # ========================================================================

    research_task = Task(
        description=f"""Research the following question thoroughly using internet sources:

Question: {research_question}

Instructions:
1. Conduct multiple searches with different query variations
2. Gather information from at least 10 diverse sources
3. Focus on academic sources, expert opinions, and credible publications
4. Note key facts, arguments, and perspectives
5. Identify any controversies or debates

Use the web_search_tool to search. Try multiple search queries to get comprehensive coverage.

Provide a detailed summary of your findings with source citations.""",
        agent=researcher,
        expected_output="Comprehensive research summary with 10+ sources and key findings"
    )

    # ========================================================================
    # TASK 2: Analysis
    # ========================================================================

    analysis_task = Task(
        description=f"""Analyze the research findings on this question:

Question: {research_question}

Using the research gathered, provide a deep academic analysis that:

1. Identifies main themes and patterns
2. Synthesizes information across sources
3. Evaluates the quality and credibility of arguments
4. Identifies gaps or contradictions
5. Draws meaningful conclusions
6. Provides critical insights

Your analysis should be thorough, well-reasoned, and academically rigorous.""",
        agent=analyst,
        expected_output="Comprehensive analytical synthesis with critical insights",
        context=[research_task]
    )

    # ========================================================================
    # TASK 3: Report Writing
    # ========================================================================

    writing_task = Task(
        description=f"""Create a professional research report on:

Question: {research_question}

Using the research and analysis provided, write a comprehensive report that includes:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
2. INTRODUCTION
   - Background and context
   - Research question and significance

3. KEY FINDINGS
   - Main discoveries organized by theme
   - Supporting evidence from sources

4. DETAILED ANALYSIS
   - In-depth examination of findings
   - Critical insights and implications
   - Synthesis across sources

5. DISCUSSION
   - Patterns and themes
   - Controversies or debates
   - Gaps in knowledge

6. CONCLUSION
   - Summary of main points
   - Implications
   - Recommendations for further research

7. SOURCES
   - List all sources consulted

Format: Professional academic style, clear structure, proper citations.
Length: Comprehensive (aim for 2000-3000 words).
Tone: Scholarly but accessible.""",
        agent=writer,
        expected_output="Complete professional research report in markdown format",
        context=[research_task, analysis_task]
    )

    # ========================================================================
    # EXECUTE CREW
    # ========================================================================

    print("🚀 Starting Multi-Agent Research...\n")
    print("   Agent 1: Researcher - Gathering information")
    print("   Agent 2: Analyst - Analyzing findings")
    print("   Agent 3: Writer - Creating report\n")
    print("-"*80 + "\n")

    # Create crew
    research_crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=True
    )

    # Execute
    result = research_crew.kickoff()

    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================

    print("\n" + "="*80)
    print("💾 Saving Outputs...")
    print("="*80 + "\n")

    # Save Markdown
    md_path = Path(output_dir) / f"{base_filename}.md"

    md_content = f"""# Research Report

**Question:** {research_question}

**Date:** {datetime.now().strftime("%B %d, %Y at %I:%M %p")}

**Generated by:** CrewAI Multi-Agent System (Local LLMs)

**Cost:** $0.00

---

{result}

---

**Metadata:**
- Research System: CrewAI v1.0.0
- LLM Model: Qwen2.5 14B (Local)
- Agents: Researcher, Analyst, Writer
- Processing: 100% Local (No external APIs)
"""

    md_path.write_text(md_content, encoding='utf-8')
    print(f"✅ Markdown saved: {md_path}")

    # Save PDF
    pdf_path = Path(output_dir) / f"{base_filename}.pdf"
    create_pdf(md_content, str(pdf_path), research_question)
    print(f"✅ PDF saved: {pdf_path}")

    # Also save plain text version of result
    txt_path = Path(output_dir) / f"{base_filename}_raw.txt"
    txt_path.write_text(str(result), encoding='utf-8')
    print(f"✅ Raw text saved: {txt_path}")

    print("\n" + "="*80)
    print("✅ RESEARCH COMPLETE")
    print("="*80)
    print(f"\n📄 Your reports:")
    print(f"   • Markdown: {md_path}")
    print(f"   • PDF: {pdf_path}")
    print(f"   • Raw: {txt_path}")
    print(f"\n💰 Total Cost: $0.00 (Local LLMs)")
    print(f"🔒 Privacy: 100% Local Processing")
    print(f"🤖 Agents: 3 specialized agents collaborated")

    return result, md_path, pdf_path


# ============================================================================
# PDF GENERATION
# ============================================================================

def create_pdf(content: str, output_path: str, title: str):
    """
    Create a professional PDF from markdown content.

    Args:
        content: Markdown content
        output_path: Where to save PDF
        title: Report title
    """
    try:
        # Create PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Container for PDF elements
        story = []

        # Styles
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor='#1a1a1a',
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor='#2c3e50',
            spaceAfter=12,
            spaceBefore=12
        )

        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
            leading=14
        )

        # Add title
        story.append(Paragraph(f"<b>{title}</b>", title_style))
        story.append(Spacer(1, 0.2*inch))

        # Add timestamp
        timestamp = datetime.now().strftime("%B %d, %Y")
        story.append(Paragraph(f"<i>Generated: {timestamp}</i>", styles['Normal']))
        story.append(Paragraph("<i>System: CrewAI Multi-Agent Research (Local LLMs)</i>", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # Process content
        lines = content.split('\n')

        for line in lines:
            line = line.strip()

            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue

            # Headers
            if line.startswith('# '):
                text = line[2:].strip()
                story.append(Paragraph(f"<b>{text}</b>", title_style))
            elif line.startswith('## '):
                text = line[3:].strip()
                story.append(Paragraph(f"<b>{text}</b>", heading_style))
            elif line.startswith('### '):
                text = line[4:].strip()
                story.append(Paragraph(f"<b>{text}</b>", styles['Heading3']))
            elif line.startswith('---'):
                story.append(Spacer(1, 0.2*inch))
            else:
                # Regular text
                # Escape HTML
                line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                # Bold
                line = line.replace('**', '<b>').replace('**', '</b>')
                # Italic
                line = line.replace('*', '<i>').replace('*', '</i>')

                story.append(Paragraph(line, body_style))

        # Build PDF
        doc.build(story)

    except Exception as e:
        print(f"⚠️  PDF generation error: {e}")
        print("   Markdown report is still available")


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def main():
    """Interactive research interface."""

    print("\n📚 What would you like to research?")
    print("\nExamples:")
    print("  • Academic: 'What are the key theories of social contract in political philosophy?'")
    print("  • Legal: 'What are recent developments in AI regulation in the EU?'")
    print("  • Technical: 'What are the main approaches to quantum error correction?'")
    print("  • Current: 'What are the latest findings on climate change tipping points?'\n")

    try:
        question = input("Your research question: ").strip()

        if not question:
            print("\n⚠️  No question provided. Using example...")
            question = "What are the Intel factors in 28 USC 1782 discovery applications?"

        print(f"\n✅ Researching: {question}\n")

        # Run research
        conduct_research(question)

    except (EOFError, KeyboardInterrupt):
        print("\n\n👋 Exiting...")


if __name__ == "__main__":
    main()

