"""
CrewAI 49-Agent Research System - Based on The Matrix Architecture
Replaces OpenAI 4o-mini with local LLMs for maximum length reports
"""

import os
import sys
import io
import json
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import urllib.request
import requests

# Import CrewAI components
try:
    from crewai import Agent
except ImportError:
    print("Installing CrewAI...")
    subprocess.run(["pip", "install", "--user", "crewai", "crewai-tools"], check=False)
    from crewai import Agent

from ddgs import DDGS
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from crewai.llms.base_llm import BaseLLM


# Ensure UTF-8 console output (prevents CrewAI event bus encoding errors on Windows)
if os.name == "nt":
    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class TaskSpec:
    name: str
    agent_index: int
    description: str
    min_words: int
    iterations: int
    context_names: List[str] = field(default_factory=list)
    category: str = ""
    focus_variants: Optional[List[str]] = None


@dataclass
class IterationLog:
    iteration: int
    prompt_focus: str
    attempt: int
    start_time: float
    end_time: float
    word_count: int
    success: bool


@dataclass
class TaskLog:
    task_name: str
    agent_role: str
    category: str
    iterations: List[IterationLog] = field(default_factory=list)
    total_word_count: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0


class LocalOllamaLLM(BaseLLM):
    """Minimal CrewAI-compatible LLM that talks to Ollama's HTTP API directly."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 900,
        temperature: float = 0.7,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model, provider="ollama", temperature=temperature, base_url=base_url, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.options = options.copy() if options else {}
        self.options.setdefault("temperature", temperature)
        self.options.setdefault("num_ctx", 4096)
        self.options.setdefault("num_predict", 2048)
        self.session = requests.Session()

    def _format_prompt(self, messages: str | List[Dict[str, Any]]) -> str:
        if isinstance(messages, str):
            return messages.strip()

        parts: List[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def call(  # type: ignore[override]
        self,
        messages: str | List[Dict[str, Any]],
        tools: List[dict] | None = None,
        callbacks: List[Any] | None = None,
        available_functions: Dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
    ) -> str:
        prompt = self._format_prompt(messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": dict(self.options),
        }

        if self.stop:
            payload["options"]["stop"] = self.stop

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        data = response.json()
        text = data.get("response", "")
        if not text:
            raise RuntimeError(f"Ollama returned empty response: {data}")
        return text.strip()

    def supports_stop_words(self) -> bool:
        return True


def safe_word_count(text: str) -> int:
    """Return a conservative word count for generated content."""
    if not text:
        return 0
    return len([w for w in text.replace("\n", " ").split(" ") if w.strip()])


def verify_ollama(model_tag: str, base_url: str = "http://localhost:11434") -> bool:
    """Ensure Ollama is reachable and the requested model is available."""
    candidate_commands = [["ollama", "list"]]
    if os.name == "nt":
        candidate_commands.append(
            [os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe"), "list"]
        )

    for cmd in candidate_commands:
        if not cmd[0]:
            continue
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        except FileNotFoundError:
            continue
        except Exception:
            continue
        if result.returncode == 0 and model_tag.split(":")[0] in result.stdout:
            if model_tag in result.stdout:
                break
    else:
        print(f"[WARN] Ollama CLI or model '{model_tag}' not detected. Please run 'ollama pull {model_tag}'.")
        return False

    # Ping Ollama HTTP API
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        available = {tag.get("name") for tag in data.get("models", [])}
        if model_tag not in available:
            print(f"[WARN] Ollama server reachable but model '{model_tag}' not loaded. Run 'ollama pull {model_tag}'.")
            return False
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Unable to contact Ollama at {base_url}: {exc}")
        return False

    return True


def gather_context(context_names: List[str], outputs: Dict[str, Dict[str, Any]]) -> str:
    """Aggregate context from previously completed tasks."""
    if not context_names:
        return "No prior task context supplied."

    context_sections = []
    for name in context_names:
        payload = outputs.get(name)
        if not payload:
            continue
        context_sections.append(
            f"### Context from {name}\n"
            f"(Words: {payload.get('word_count', 0)}, Iterations: {payload.get('iterations_completed', 0)})\n"
            f"{payload.get('content', '')}\n"
        )

    if not context_sections:
        return "Context references provided, but no content was available yet."

    return "\n".join(context_sections)


DEFAULT_FOCUS_VARIANTS = [
    "Deliver a broad thematic overview covering the most important dimensions.",
    "Drill into granular evidence, case studies, statistics, and named sources.",
    "Provide critical evaluation, identify gaps, and propose future research questions.",
    "Highlight comparative or temporal patterns that illuminate shifts over time.",
    "Translate findings into practical policy or governance recommendations.",
]


def build_iteration_prompt(
    agent: Agent,
    spec: TaskSpec,
    iteration_index: int,
    context_text: str,
    prior_feedback: Optional[str] = None,
) -> str:
    """Build a rich prompt that incorporates agent persona, iteration focus, and context."""
    focus_list = spec.focus_variants or DEFAULT_FOCUS_VARIANTS
    focus_instruction = focus_list[iteration_index % len(focus_list)]

    feedback_block = ""
    if prior_feedback:
        feedback_block = f"\nPrevious attempt feedback:\n{prior_feedback}\n"

    return f"""
You are {agent.role}.
Your overarching goal: {agent.goal}
Background: {agent.backstory}

Task Name: {spec.name}
Task Category: {spec.category}
Primary Research Question: {spec.description.splitlines()[0]}

Iteration #{iteration_index + 1} Focus:
{focus_instruction}

Context provided:
{context_text}

Instructions:
- Produce a detailed, well-structured markdown section.
- Minimum length: {spec.min_words} words (no exceptions).
- Use inline citations where appropriate.
- Maintain professional, academic tone.
- Explicitly cover new angles compared to previous iterations.
{feedback_block}
Begin when ready.
""".strip()


def run_task_spec(
    agent: Agent,
    spec: TaskSpec,
    outputs: Dict[str, Dict[str, Any]],
    max_attempts: int = 3,
) -> TaskLog:
    """Execute a task spec with iterative prompting and word-count enforcement."""
    task_log = TaskLog(
        task_name=spec.name,
        agent_role=agent.role,
        category=spec.category,
        started_at=time.time(),
    )

    context_text = gather_context(spec.context_names, outputs)
    iteration_outputs: List[str] = []
    focus_sequence = spec.focus_variants or DEFAULT_FOCUS_VARIANTS

    for idx in range(spec.iterations):
        prior_feedback = None
        iteration_success = False
        attempt = 0
        while attempt < max_attempts and not iteration_success:
            attempt += 1
            start_time = time.time()
            prompt = build_iteration_prompt(
                agent=agent,
                spec=spec,
                iteration_index=idx,
                context_text=context_text,
                prior_feedback=prior_feedback,
            )

            try:
                response = agent.llm.call(prompt)
            except Exception as exc:  # noqa: BLE001
                response = f"Generation error: {exc}"

            end_time = time.time()
            word_count = safe_word_count(response)
            iteration_log = IterationLog(
                iteration=idx + 1,
                prompt_focus=focus_sequence[idx % len(focus_sequence)],
                attempt=attempt,
                start_time=start_time,
                end_time=end_time,
                word_count=word_count,
                success=word_count >= spec.min_words,
            )
            task_log.iterations.append(iteration_log)

            if word_count >= spec.min_words:
                iteration_outputs.append(response.strip())
                iteration_success = True
            else:
                prior_feedback = (
                    f"The previous attempt contained only {word_count} words. Expand significantly and ensure the "
                    f"output surpasses the {spec.min_words} word requirement. Provide deeper details, additional "
                    f"evidence, and broader coverage."
                )

        if not iteration_success:
            iteration_outputs.append(
                f"WARNING: Unable to reach {spec.min_words} words after {max_attempts} attempts. "
                f"Best effort output:\n{response}"
            )

    task_log.completed_at = time.time()
    combined_output = "\n\n".join(iteration_outputs)
    total_words = safe_word_count(combined_output)
    task_log.total_word_count = total_words

    outputs[spec.name] = {
        "content": combined_output,
        "word_count": total_words,
        "iterations_completed": spec.iterations,
        "agent_role": agent.role,
        "category": spec.category,
        "context_used": spec.context_names,
    }

    return task_log


def write_execution_log(logs: List[TaskLog], filepath: Path) -> None:
    """Serialize execution metrics to JSON for auditing."""
    serializable = []
    for entry in logs:
        serializable.append(
            {
                "task_name": entry.task_name,
                "agent_role": entry.agent_role,
                "category": entry.category,
                "started_at": entry.started_at,
                "completed_at": entry.completed_at,
                "total_word_count": entry.total_word_count,
                "iterations": [
                    {
                        "iteration": log.iteration,
                        "prompt_focus": log.prompt_focus,
                        "attempt": log.attempt,
                        "start_time": log.start_time,
                        "end_time": log.end_time,
                        "word_count": log.word_count,
                        "success": log.success,
                        "duration_seconds": log.end_time - log.start_time,
                    }
                    for log in entry.iterations
                ],
            }
        )

    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def web_search(query: str) -> str:
    """Ultra-comprehensive internet search for 49-agent system"""
    try:
        # Multiple search variations for maximum coverage
        search_queries = [
            query,
            f"{query} academic research papers",
            f"{query} expert analysis scholarly",
            f"{query} policy implications legal",
            f"{query} historical context",
            f"{query} Margaret Lewis research",
            f"{query} Chinese crackdowns analysis",
            f"{query} temporal patterns enforcement",
            f"{query} legal exceptionalism China",
            f"{query} selective enforcement tactics",
            f"{query} bureaucratic control systems",
            f"{query} authoritarian governance",
            f"{query} human rights violations",
            f"{query} international law",
            f"{query} comparative politics"
        ]

        all_results = []
        for search_query in search_queries:
            results = DDGS().text(search_query, max_results=20)
            if results:
                all_results.extend(results)

        # Remove duplicates and limit to top 50
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r['href'] not in seen_urls:
                seen_urls.add(r['href'])
                unique_results.append(r)
                if len(unique_results) >= 50:
                    break

        if not unique_results:
            return f"No results for: {query}"

        output = [f"ULTRA-COMPREHENSIVE SEARCH RESULTS for: {query}\n"]
        output.append(f"Total Sources Found: {len(unique_results)}\n")
        output.append("="*100 + "\n")

        for i, r in enumerate(unique_results, 1):
            output.append(f"SOURCE {i}: {r['title']}")
            output.append(f"URL: {r['href']}")
            output.append(f"CONTENT PREVIEW: {r['body'][:600]}...")
            output.append(f"RELEVANCE SCORE: {i/len(unique_results)*100:.1f}%")
            output.append(f"CATEGORY: {'Academic' if any(word in r['title'].lower() for word in ['journal', 'academic', 'research', 'study']) else 'General'}")
            output.append("-"*80 + "\n")

        return "\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def create_pdf_report(content: str, filepath: str, title: str):
    """Generate comprehensive PDF from content"""
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
        story.append(Paragraph("<i>System: CrewAI 49-Agent Matrix Architecture (Local LLMs, Iterative)</i>", styles['Normal']))
        story.append(Paragraph("<i>Agents: 49 Specialized AI Researchers</i>", styles['Normal']))
        story.append(Paragraph("<i>Target Word Count: 25,000+ words</i>", styles['Normal']))
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


def create_49_agents(local_llm) -> List[Agent]:
    """Create all 49 agents based on The Matrix architecture"""

    agents = []

    # === CITATION PROCESSING (5 agents - all deterministic) ===

    # 1. CitationFinderAgent
    agents.append(Agent(
        role='Citation Finder Specialist',
        goal='Find raw citation strings using regex patterns',
        backstory='Expert in legal citation patterns with 15+ years experience. Specializes in identifying case citations, statutes, codes, and federal rules using sophisticated regex patterns.',
        llm=local_llm,
        verbose=False
    ))

    # 2. CitationNormalizerAgent
    agents.append(Agent(
        role='Citation Normalizer Specialist',
        goal='Normalize citations to Bluebook format',
        backstory='Legal citation expert specializing in Bluebook formatting. Expert in reporter abbreviations, case name formatting, and statute standardization.',
        llm=local_llm,
        verbose=False
    ))

    # 3. CitationVerifierAgent
    agents.append(Agent(
        role='Citation Verifier Specialist',
        goal='Verify citations against case law database',
        backstory='Database specialist with expertise in legal citation verification. Expert in SQL queries and case law database management.',
        llm=local_llm,
        verbose=False
    ))

    # 4. CitationLocatorAgent
    agents.append(Agent(
        role='Citation Locator Specialist',
        goal='Map citations to file paths and URLs',
        backstory='Document management expert specializing in citation mapping. Expert in database lookups and file system organization.',
        llm=local_llm,
        verbose=False
    ))

    # 5. CitationInserterAgent
    agents.append(Agent(
        role='Citation Inserter Specialist',
        goal='Insert formatted citations into text',
        backstory='Text processing expert specializing in citation insertion. Expert in string manipulation and position-aware insertion.',
        llm=local_llm,
        verbose=False
    ))

    # === RESEARCH (6 agents - mix) ===

    # 6. FactExtractorAgent
    agents.append(Agent(
        role='Fact Extractor Specialist',
        goal='Extract discrete facts from documents',
        backstory='Research analyst with 20+ years experience in fact extraction. Expert in structured data extraction and confidence scoring.',
        llm=local_llm,
        verbose=False
    ))

    # 7. PrecedentFinderAgent
    agents.append(Agent(
        role='Precedent Finder Specialist',
        goal='Find relevant precedent cases',
        backstory='Legal research expert specializing in precedent identification. Expert in case matching and relevance scoring.',
        llm=local_llm,
        verbose=False
    ))

    # 8. PrecedentRankerAgent
    agents.append(Agent(
        role='Precedent Ranker Specialist',
        goal='Rank precedent cases by relevance',
        backstory='Legal analyst expert in case ranking and prioritization. Expert in relevance criteria and ranking algorithms.',
        llm=local_llm,
        verbose=False
    ))

    # 9. PrecedentSummarizerAgent
    agents.append(Agent(
        role='Precedent Summarizer Specialist',
        goal='Summarize precedent cases',
        backstory='Legal writing expert specializing in case summarization. Expert in concise legal writing and case analysis.',
        llm=local_llm,
        verbose=False
    ))

    # 10. StatuteLocatorAgent
    agents.append(Agent(
        role='Statute Locator Specialist',
        goal='Find relevant statutes and regulations',
        backstory='Legal database expert specializing in statute identification. Expert in regulatory research and statutory analysis.',
        llm=local_llm,
        verbose=False
    ))

    # 11. ExhibitFetcherAgent
    agents.append(Agent(
        role='Exhibit Fetcher Specialist',
        goal='Fetch exhibits and supporting documents',
        backstory='Document management expert specializing in exhibit retrieval. Expert in file system queries and document metadata.',
        llm=local_llm,
        verbose=False
    ))

    # === DRAFTING (4 agents - all LLM) ===

    # 12. OutlineBuilderAgent
    agents.append(Agent(
        role='Outline Builder Specialist',
        goal='Create document structure and outline',
        backstory='Document architect with 25+ years experience in legal document structure. Expert in hierarchical organization and logical flow.',
        llm=local_llm,
        verbose=False
    ))

    # 13. SectionWriterAgent
    agents.append(Agent(
        role='Section Writer Specialist',
        goal='Write individual document sections',
        backstory='Senior legal writer with 20+ years experience in section writing. Expert in comprehensive legal analysis and argumentation.',
        llm=local_llm,
        verbose=False
    ))

    # 14. ParagraphWriterAgent
    agents.append(Agent(
        role='Paragraph Writer Specialist',
        goal='Write individual paragraphs',
        backstory='Legal writing expert specializing in paragraph construction. Expert in clear argumentation and logical flow.',
        llm=local_llm,
        verbose=False
    ))

    # 15. TransitionAgent
    agents.append(Agent(
        role='Transition Specialist',
        goal='Connect sections with transitions',
        backstory='Writing flow expert specializing in section transitions. Expert in maintaining logical coherence and narrative flow.',
        llm=local_llm,
        verbose=False
    ))

    # === QA/REVIEW (7 agents - mix) ===

    # 16. GrammarFixerAgent
    agents.append(Agent(
        role='Grammar Fixer Specialist',
        goal='Fix grammar and syntax errors',
        backstory='Grammar expert with 15+ years experience in legal writing. Expert in grammar correction and syntax improvement.',
        llm=local_llm,
        verbose=False
    ))

    # 17. StyleCheckerAgent
    agents.append(Agent(
        role='Style Checker Specialist',
        goal='Enforce style guide compliance',
        backstory='Style guide expert specializing in legal writing standards. Expert in Bluebook compliance and legal writing conventions.',
        llm=local_llm,
        verbose=False
    ))

    # 18. LogicCheckerAgent
    agents.append(Agent(
        role='Logic Checker Specialist',
        goal='Verify logical consistency',
        backstory='Logic expert with 20+ years experience in legal reasoning. Expert in logical analysis and consistency verification.',
        llm=local_llm,
        verbose=False
    ))

    # 19. ConsistencyCheckerAgent
    agents.append(Agent(
        role='Consistency Checker Specialist',
        goal='Check term consistency throughout document',
        backstory='Consistency expert specializing in document coherence. Expert in terminology management and consistency verification.',
        llm=local_llm,
        verbose=False
    ))

    # 20. RedactionAgent
    agents.append(Agent(
        role='Redaction Specialist',
        goal='Apply PII redaction',
        backstory='Privacy expert specializing in PII redaction. Expert in data protection and privacy compliance.',
        llm=local_llm,
        verbose=False
    ))

    # 21. ComplianceAgent
    agents.append(Agent(
        role='Compliance Specialist',
        goal='Ensure format compliance',
        backstory='Compliance expert specializing in document formatting. Expert in legal document standards and compliance requirements.',
        llm=local_llm,
        verbose=False
    ))

    # 22. ExpertQAAgent
    agents.append(Agent(
        role='Expert QA Specialist',
        goal='Expert-level quality assurance',
        backstory='Senior legal expert with 30+ years experience in quality assurance. Expert in comprehensive document review and quality control.',
        llm=local_llm,
        verbose=False
    ))

    # === OUTPUT (3 agents - all deterministic) ===

    # 23. MarkdownExporterAgent
    agents.append(Agent(
        role='Markdown Exporter Specialist',
        goal='Export document to Markdown format',
        backstory='Document conversion expert specializing in Markdown formatting. Expert in template-based conversion and formatting.',
        llm=local_llm,
        verbose=False
    ))

    # 24. DocxExporterAgent
    agents.append(Agent(
        role='DOCX Exporter Specialist',
        goal='Export document to DOCX format',
        backstory='Document conversion expert specializing in DOCX formatting. Expert in Microsoft Word document generation.',
        llm=local_llm,
        verbose=False
    ))

    # 25. MetadataTaggerAgent
    agents.append(Agent(
        role='Metadata Tagger Specialist',
        goal='Add metadata tags to document',
        backstory='Metadata expert specializing in document tagging. Expert in document classification and metadata management.',
        llm=local_llm,
        verbose=False
    ))

    # === ADDITIONAL RESEARCH AGENTS (24 agents for maximum coverage) ===

    # 26-35. Specialized Research Agents
    research_specializations = [
        'Academic Literature Specialist',
        'Policy Analysis Specialist',
        'Historical Context Specialist',
        'Legal Framework Specialist',
        'International Law Specialist',
        'Comparative Analysis Specialist',
        'Statistical Analysis Specialist',
        'Case Study Specialist',
        'Expert Opinion Specialist',
        'Media Analysis Specialist'
    ]

    for specialization in research_specializations:
        agents.append(Agent(
            role=specialization,
            goal=f'Provide specialized research in {specialization.lower()}',
            backstory=f'Expert researcher with 20+ years experience in {specialization.lower()}. Specializes in comprehensive analysis and detailed research.',
            llm=local_llm,
            verbose=False
        ))

    # 36-45. Analysis and Synthesis Agents
    analysis_specializations = [
        'Critical Analysis Specialist',
        'Synthesis Specialist',
        'Pattern Recognition Specialist',
        'Causal Analysis Specialist',
        'Temporal Analysis Specialist',
        'Comparative Analysis Specialist',
        'Cross-Reference Specialist',
        'Evidence Evaluation Specialist',
        'Argumentation Specialist',
        'Conclusion Synthesis Specialist'
    ]

    for specialization in analysis_specializations:
        agents.append(Agent(
            role=specialization,
            goal=f'Provide specialized analysis in {specialization.lower()}',
            backstory=f'Expert analyst with 20+ years experience in {specialization.lower()}. Specializes in comprehensive analysis and detailed evaluation.',
            llm=local_llm,
            verbose=False
        ))

    # 46-49. Final Review and Quality Agents
    quality_specializations = [
        'Final Review Specialist',
        'Quality Assurance Specialist',
        'Comprehensive Editor Specialist',
        'Master Synthesis Specialist'
    ]

    for specialization in quality_specializations:
        agents.append(Agent(
            role=specialization,
            goal=f'Provide final {specialization.lower()}',
            backstory=f'Expert reviewer with 25+ years experience in {specialization.lower()}. Specializes in comprehensive review and quality control.',
            llm=local_llm,
            verbose=False
        ))

    return agents


def _legacy_create_49_tasks(question: str, agents: List[Agent]):
    """Legacy CrewAI Task construction (kept for reference)."""

    tasks = []

    # === CITATION PROCESSING TASKS (5 tasks) ===

    # Task 1: Citation Finding
    tasks.append(Task(
        description=f"""CITATION FINDING: {question}

Find all citation strings in research materials using regex patterns:
- Case citations: "123 U.S. 456"
- Case names: "Smith v. Jones"
- Statutes: "42 U.S.C. Section 1983"
- Codes: "Cal. Civ. Code Section 1234"
- Federal rules: "Fed. R. Civ. P. 12(b)(6)"
- Years: "(2023)"

MINIMUM OUTPUT: 500 words with comprehensive citation list""",
        agent=agents[0],
        expected_output="Comprehensive citation list with 500+ words"
    ))

    # Task 2: Citation Normalization
    tasks.append(Task(
        description=f"""CITATION NORMALIZATION: {question}

Normalize all citations to Bluebook format:
- Use Bluebook Table 1 reporter abbreviations
- Format case names with italic hints
- Standardize statute formatting
- Ensure consistent citation style

MINIMUM OUTPUT: 400 words with normalized citations""",
        agent=agents[1],
        expected_output="Normalized citations with 400+ words",
        context=[tasks[0]]
    ))

    # Task 3: Citation Verification
    tasks.append(Task(
        description=f"""CITATION VERIFICATION: {question}

Verify citations against case law database:
- Check citation accuracy
- Verify case existence
- Validate statutory references
- Confirm regulatory citations

MINIMUM OUTPUT: 300 words with verification results""",
        agent=agents[2],
        expected_output="Citation verification with 300+ words",
        context=[tasks[0], tasks[1]]
    ))

    # Task 4: Citation Location
    tasks.append(Task(
        description=f"""CITATION LOCATION: {question}

Map citations to file paths and URLs:
- Locate source documents
- Map to database entries
- Provide access links
- Document metadata

MINIMUM OUTPUT: 300 words with location mapping""",
        agent=agents[3],
        expected_output="Citation location mapping with 300+ words",
        context=[tasks[0], tasks[1], tasks[2]]
    ))

    # Task 5: Citation Insertion
    tasks.append(Task(
        description=f"""CITATION INSERTION: {question}

Insert formatted citations into text:
- Position-aware insertion
- Maintain text flow
- Ensure proper formatting
- Verify placement accuracy

MINIMUM OUTPUT: 300 words with inserted citations""",
        agent=agents[4],
        expected_output="Text with inserted citations (300+ words)",
        context=[tasks[0], tasks[1], tasks[2], tasks[3]]
    ))

    # === RESEARCH TASKS (6 tasks) ===

    # Task 6: Fact Extraction
    tasks.append(Task(
        description=f"""FACT EXTRACTION: {question}

Extract discrete facts from all research materials:
- Identify key facts and evidence
- Categorize by relevance
- Assign confidence scores
- Structure for analysis

MINIMUM OUTPUT: 800 words with structured facts""",
        agent=agents[5],
        expected_output="Structured facts with 800+ words"
    ))

    # Task 7: Precedent Finding
    tasks.append(Task(
        description=f"""PRECEDENT FINDING: {question}

Find relevant precedent cases:
- Identify applicable precedents
- Match to research question
- Score relevance
- Categorize by type

MINIMUM OUTPUT: 700 words with precedent analysis""",
        agent=agents[6],
        expected_output="Precedent analysis with 700+ words",
        context=[tasks[5]]
    ))

    # Task 8: Precedent Ranking
    tasks.append(Task(
        description=f"""PRECEDENT RANKING: {question}

Rank precedent cases by relevance:
- Apply ranking criteria
- Explain rankings
- Identify key precedents
- Prioritize by importance

MINIMUM OUTPUT: 600 words with ranked precedents""",
        agent=agents[7],
        expected_output="Ranked precedents with 600+ words",
        context=[tasks[5], tasks[6]]
    ))

    # Task 9: Precedent Summarization
    tasks.append(Task(
        description=f"""PRECEDENT SUMMARIZATION: {question}

Summarize precedent cases:
- Provide concise summaries
- Highlight key points
- Extract relevant holdings
- Maintain accuracy

MINIMUM OUTPUT: 800 words with case summaries""",
        agent=agents[8],
        expected_output="Case summaries with 800+ words",
        context=[tasks[5], tasks[6], tasks[7]]
    ))

    # Task 10: Statute Location
    tasks.append(Task(
        description=f"""STATUTE LOCATION: {question}

Find relevant statutes and regulations:
- Identify applicable laws
- Locate regulatory provisions
- Find relevant sections
- Document sources

MINIMUM OUTPUT: 600 words with statutory analysis""",
        agent=agents[9],
        expected_output="Statutory analysis with 600+ words",
        context=[tasks[5]]
    ))

    # Task 11: Exhibit Fetching
    tasks.append(Task(
        description=f"""EXHIBIT FETCHING: {question}

Fetch exhibits and supporting documents:
- Locate supporting materials
- Retrieve relevant documents
- Organize by relevance
- Document metadata

MINIMUM OUTPUT: 500 words with exhibit analysis""",
        agent=agents[10],
        expected_output="Exhibit analysis with 500+ words",
        context=[tasks[5]]
    ))

    # === DRAFTING TASKS (4 tasks) ===

    # Task 12: Outline Building
    tasks.append(Task(
        description=f"""OUTLINE BUILDING: {question}

Create comprehensive document structure:
- Develop hierarchical outline
- Organize by themes
- Ensure logical flow
- Plan comprehensive coverage

MINIMUM OUTPUT: 1000 words with detailed outline""",
        agent=agents[11],
        expected_output="Detailed outline with 1000+ words",
        context=[tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10]]
    ))

    # Task 13: Section Writing
    tasks.append(Task(
        description=f"""SECTION WRITING: {question}

Write comprehensive document sections:
- Develop detailed analysis
- Provide thorough coverage
- Maintain academic rigor
- Ensure comprehensive treatment

MINIMUM OUTPUT: 2000 words per major section""",
        agent=agents[12],
        expected_output="Comprehensive sections with 2000+ words",
        context=[tasks[11]]
    ))

    # Task 14: Paragraph Writing
    tasks.append(Task(
        description=f"""PARAGRAPH WRITING: {question}

Write detailed paragraphs:
- Develop clear arguments
- Provide supporting evidence
- Maintain logical flow
- Ensure comprehensive coverage

MINIMUM OUTPUT: 1500 words with detailed paragraphs""",
        agent=agents[13],
        expected_output="Detailed paragraphs with 1500+ words",
        context=[tasks[11], tasks[12]]
    ))

    # Task 15: Transition Writing
    tasks.append(Task(
        description=f"""TRANSITION WRITING: {question}

Create smooth section transitions:
- Connect sections logically
- Maintain narrative flow
- Ensure coherence
- Provide bridging content

MINIMUM OUTPUT: 800 words with transitions""",
        agent=agents[14],
        expected_output="Smooth transitions with 800+ words",
        context=[tasks[11], tasks[12], tasks[13]]
    ))

    # === QA/REVIEW TASKS (7 tasks) ===

    # Task 16: Grammar Fixing
    tasks.append(Task(
        description=f"""GRAMMAR FIXING: {question}

Fix grammar and syntax errors:
- Correct grammatical errors
- Improve sentence structure
- Enhance readability
- Maintain academic tone

MINIMUM OUTPUT: 1000 words with corrections""",
        agent=agents[15],
        expected_output="Grammar corrections with 1000+ words",
        context=[tasks[11], tasks[12], tasks[13], tasks[14]]
    ))

    # Task 17: Style Checking
    tasks.append(Task(
        description=f"""STYLE CHECKING: {question}

Enforce style guide compliance:
- Apply Bluebook standards
- Ensure consistency
- Check formatting
- Verify conventions

MINIMUM OUTPUT: 800 words with style corrections""",
        agent=agents[16],
        expected_output="Style corrections with 800+ words",
        context=[tasks[15]]
    ))

    # Task 18: Logic Checking
    tasks.append(Task(
        description=f"""LOGIC CHECKING: {question}

Verify logical consistency:
- Check argument logic
- Verify reasoning
- Ensure coherence
- Identify fallacies

MINIMUM OUTPUT: 900 words with logic analysis""",
        agent=agents[17],
        expected_output="Logic analysis with 900+ words",
        context=[tasks[15], tasks[16]]
    ))

    # Task 19: Consistency Checking
    tasks.append(Task(
        description=f"""CONSISTENCY CHECKING: {question}

Check term consistency:
- Verify terminology
- Ensure consistency
- Check definitions
- Maintain coherence

MINIMUM OUTPUT: 700 words with consistency analysis""",
        agent=agents[18],
        expected_output="Consistency analysis with 700+ words",
        context=[tasks[15], tasks[16], tasks[17]]
    ))

    # Task 20: Redaction
    tasks.append(Task(
        description=f"""REDACTION: {question}

Apply PII redaction:
- Identify PII
- Apply redaction
- Maintain readability
- Ensure compliance

MINIMUM OUTPUT: 600 words with redaction analysis""",
        agent=agents[19],
        expected_output="Redaction analysis with 600+ words",
        context=[tasks[15], tasks[16], tasks[17], tasks[18]]
    ))

    # Task 21: Compliance Checking
    tasks.append(Task(
        description=f"""COMPLIANCE CHECKING: {question}

Ensure format compliance:
- Check document format
- Verify standards
- Ensure compliance
- Document requirements

MINIMUM OUTPUT: 600 words with compliance analysis""",
        agent=agents[20],
        expected_output="Compliance analysis with 600+ words",
        context=[tasks[15], tasks[16], tasks[17], tasks[18], tasks[19]]
    ))

    # Task 22: Expert QA
    tasks.append(Task(
        description=f"""EXPERT QA: {question}

Provide expert-level quality assurance:
- Comprehensive review
- Quality assessment
- Expert evaluation
- Final recommendations

MINIMUM OUTPUT: 1200 words with expert analysis""",
        agent=agents[21],
        expected_output="Expert analysis with 1200+ words",
        context=[tasks[15], tasks[16], tasks[17], tasks[18], tasks[19], tasks[20]]
    ))

    # === OUTPUT TASKS (3 tasks) ===

    # Task 23: Markdown Export
    tasks.append(Task(
        description=f"""MARKDOWN EXPORT: {question}

Export to Markdown format:
- Format for Markdown
- Ensure readability
- Maintain structure
- Optimize presentation

MINIMUM OUTPUT: 500 words with formatting""",
        agent=agents[22],
        expected_output="Markdown formatting with 500+ words",
        context=[tasks[21]]
    ))

    # Task 24: DOCX Export
    tasks.append(Task(
        description=f"""DOCX EXPORT: {question}

Export to DOCX format:
- Format for Word
- Ensure compatibility
- Maintain formatting
- Optimize layout

MINIMUM OUTPUT: 500 words with DOCX formatting""",
        agent=agents[23],
        expected_output="DOCX formatting with 500+ words",
        context=[tasks[21]]
    ))

    # Task 25: Metadata Tagging
    tasks.append(Task(
        description=f"""METADATA TAGGING: {question}

Add metadata tags:
- Classify document
- Add metadata
- Ensure completeness
- Document properties

MINIMUM OUTPUT: 400 words with metadata""",
        agent=agents[24],
        expected_output="Metadata with 400+ words",
        context=[tasks[21]]
    ))

    # === ADDITIONAL RESEARCH TASKS (24 tasks) ===

    # Tasks 26-35: Specialized Research
    research_topics = [
        'Academic Literature Review',
        'Policy Analysis',
        'Historical Context',
        'Legal Framework Analysis',
        'International Law Review',
        'Comparative Analysis',
        'Statistical Analysis',
        'Case Study Analysis',
        'Expert Opinion Analysis',
        'Media Analysis'
    ]

    for i, topic in enumerate(research_topics):
        tasks.append(Task(
            description=f"""{topic.upper()}: {question}

Provide comprehensive {topic.lower()}:
- Conduct thorough research
- Analyze findings
- Provide detailed insights
- Ensure comprehensive coverage

MINIMUM OUTPUT: 1500 words with detailed analysis""",
            agent=agents[25 + i],
            expected_output=f"{topic} with 1500+ words",
            context=[tasks[5], tasks[6], tasks[7], tasks[8], tasks[9], tasks[10]]
        ))

    # Tasks 36-45: Analysis and Synthesis
    analysis_topics = [
        'Critical Analysis',
        'Synthesis',
        'Pattern Recognition',
        'Causal Analysis',
        'Temporal Analysis',
        'Comparative Analysis',
        'Cross-Reference Analysis',
        'Evidence Evaluation',
        'Argumentation Analysis',
        'Conclusion Synthesis'
    ]

    for i, topic in enumerate(analysis_topics):
        tasks.append(Task(
            description=f"""{topic.upper()}: {question}

Provide comprehensive {topic.lower()}:
- Analyze all findings
- Synthesize insights
- Identify patterns
- Draw conclusions

MINIMUM OUTPUT: 1200 words with detailed analysis""",
            agent=agents[35 + i],
            expected_output=f"{topic} with 1200+ words",
            context=tasks[25:35]  # All research tasks
        ))

    # Tasks 46-49: Final Review and Quality
    quality_topics = [
        'Final Review',
        'Quality Assurance',
        'Comprehensive Editing',
        'Master Synthesis'
    ]

    for i, topic in enumerate(quality_topics):
        tasks.append(Task(
            description=f"""{topic.upper()}: {question}

Provide comprehensive {topic.lower()}:
- Review all content
- Ensure quality
- Finalize document
- Complete synthesis

MINIMUM OUTPUT: 2000 words with comprehensive review""",
            agent=agents[45 + i],
            expected_output=f"{topic} with 2000+ words",
            context=tasks[25:]  # All previous tasks
        ))

    return tasks


def create_task_specs(question: str) -> List[TaskSpec]:
    """Return structured task specifications for the enhanced 49-agent flow."""

    specs: List[TaskSpec] = []

    def add_spec(
        name: str,
        agent_index: int,
        category: str,
        min_words: int,
        iterations: int,
        instructions: str,
        context: Optional[List[str]] = None,
        focus: Optional[List[str]] = None,
    ) -> None:
        description = f"{name.upper()}: {question}\n\n{instructions.strip()}"
        specs.append(
            TaskSpec(
                name=name,
                agent_index=agent_index,
                description=description,
                min_words=min_words,
                iterations=iterations,
                category=category,
                context_names=context or [],
                focus_variants=focus,
            )
        )

    # Citation Processing
    add_spec(
        "Citation Finding",
        0,
        "Citation Processing",
        600,
        2,
        """Objectives:
- Sweep every research artifact for potential citations (cases, statutes, regulations, comparatives, grey literature).
- Capture raw strings with surrounding context, categorize by type, and flag ambiguous references.""",
    )
    add_spec(
        "Citation Normalization",
        1,
        "Citation Processing",
        600,
        2,
        """Tasks:
- Convert raw citations to Bluebook-compliant format with reporter abbreviations, italic hints, and section symbols.
- Produce a crosswalk mapping raw and normalized citations, noting assumptions or missing data.""",
        context=["Citation Finding"],
    )
    add_spec(
        "Citation Verification",
        2,
        "Citation Processing",
        500,
        2,
        """Tasks:
- Verify each normalized citation against authoritative databases.
- Record confidence levels, URLs/IDs, and remediation steps for unresolved items.""",
        context=["Citation Finding", "Citation Normalization"],
    )
    add_spec(
        "Citation Location",
        3,
        "Citation Processing",
        500,
        2,
        """Tasks:
- Map verified citations to downloadable resources with metadata (jurisdiction, publisher, publication year).
- Prioritize official or archival sources and note access restrictions.""",
        context=["Citation Verification"],
    )
    add_spec(
        "Citation Insertion",
        4,
        "Citation Processing",
        600,
        2,
        """Tasks:
- Recommend precise placement of citations within draft sections.
- Provide inline markers or footnote-ready copy and highlight gaps needing additional sourcing.""",
        context=["Citation Location"],
    )

    # Core Research
    add_spec(
        "Fact Extraction",
        5,
        "Core Research",
        1200,
        3,
        """Goals:
- Extract discrete facts, statistics, quotations, and anecdotes illuminating temporal crackdowns and bureaucratic controls.
- Attribute each fact, assess reliability, and group findings by analytical pillars (short-term intensity, legal exceptionalism, selective enforcement, bureaucratic governance).""",
        context=["Citation Finding"],
    )
    add_spec(
        "Precedent Finding",
        6,
        "Core Research",
        1000,
        3,
        """Goals:
- Identify domestic and comparative precedents relevant to Chinese crackdowns and administrative governance.
- Summarize holdings, factual contexts, and relevance to Margaret Lewis's framing.""",
        context=["Fact Extraction"],
    )
    add_spec(
        "Precedent Ranking",
        7,
        "Core Research",
        900,
        3,
        """Goals:
- Develop ranking criteria (relevance, authority, factual alignment, recency).
- Score and justify prioritization while noting doctrinal tensions or gaps.""",
        context=["Precedent Finding"],
    )
    add_spec(
        "Precedent Summaries",
        8,
        "Core Research",
        1200,
        3,
        """Goals:
- Produce executive summaries for top-ranked precedents, including holdings, reasoning, and implications for crackdown patterns vs bureaucratic control.
- Compare cross-jurisdictional insights and temporal evolutions.""",
        context=["Precedent Ranking"],
    )
    add_spec(
        "Statutory Mapping",
        9,
        "Core Research",
        1000,
        3,
        """Goals:
- Catalogue statutes, regulations, and policy directives shaping campaign-style crackdowns and routine governance.
- Explain procedural deviations during campaigns and contrast them with ordinary bureaucratic mechanisms.""",
        context=["Fact Extraction"],
    )
    add_spec(
        "Exhibit Compilation",
        10,
        "Core Research",
        900,
        3,
        """Goals:
- Curate supporting exhibits (reports, datasets, interviews, media investigations) with annotations on credibility and thematic relevance.
- Group materials by analytical pillar for rapid downstream use.""",
        context=["Fact Extraction"],
    )

    # Drafting & Argumentation
    add_spec(
        "Outline Architect",
        11,
        "Drafting",
        1500,
        3,
        """Deliverables:
- Build a detailed outline with at least twelve major sections, nested subsections, and estimated word counts.
- Ensure logical flow from historical backdrop to comparative synthesis and policy implications.""",
        context=["Fact Extraction", "Precedent Summaries", "Statutory Mapping", "Exhibit Compilation"],
    )
    add_spec(
        "Primary Drafting",
        12,
        "Drafting",
        2500,
        3,
        """Deliverables:
- Produce dense narrative prose aligned with the outline, integrating citations and comparative insights.
- Blend empirical findings with doctrinal analysis to progress toward the 25,000+ word target.""",
        context=["Outline Architect", "Fact Extraction", "Precedent Summaries", "Statutory Mapping"],
    )
    add_spec(
        "Argument Development",
        13,
        "Drafting",
        2000,
        3,
        """Deliverables:
- Strengthen argumentative scaffolding linking Margaret Lewis's temporal patterns to evidence.
- Address doctrinal, policy, and historical implications while maintaining academic rigor.""",
        context=["Primary Drafting"],
    )
    add_spec(
        "Counterargument Response",
        14,
        "Drafting",
        1800,
        3,
        """Deliverables:
- Surface plausible critiques and alternative explanations.
- Provide evidence-backed rebuttals and note residual uncertainties requiring future monitoring.""",
        context=["Argument Development"],
    )

    # Quality Assurance
    qa_specs = [
        ("Fact Checking", 15, 1200, ["Primary Drafting", "Argument Development", "Counterargument Response"]),
        ("Legal Accuracy Review", 16, 1100, ["Citation Normalization", "Primary Drafting", "Argument Development"]),
        ("Consistency Review", 17, 900, ["Primary Drafting", "Argument Development"]),
        ("Tone and Style Review", 18, 900, ["Primary Drafting"]),
        ("Structural Review", 19, 1000, ["Outline Architect", "Primary Drafting"]),
        ("Citation QA", 20, 900, ["Citation Insertion", "Primary Drafting"]),
        ("Completeness Review", 21, 1100, ["Primary Drafting", "Fact Extraction", "Argument Development"]),
    ]

    for name, agent_idx, min_words, context in qa_specs:
        add_spec(
            name,
            agent_idx,
            "Quality Assurance",
            min_words,
            2,
            f"""Review Focus:
- Conduct a comprehensive {name.lower()} sweep.
- Document issues, severity, and remediation steps.
- Confirm resolution status or escalate remaining dependencies.""",
            context=context,
        )

    # Output Preparation
    output_specs = [
        ("Markdown Formatting", 22, 1200),
        ("Executive Summary", 23, 1500),
        ("DOCX Preparation", 24, 1000),
    ]

    for name, agent_idx, min_words in output_specs:
        add_spec(
            name,
            agent_idx,
            "Output Preparation",
            min_words,
            2,
            f"""Output Requirements:
- Deliver polished {name.lower()} content ready for production.
- Maintain consistent styling, navigability, and reference integrity.
- Flag outstanding editorial decisions.""",
            context=["Primary Drafting", "Argument Development", "Counterargument Response"],
        )

    add_spec(
        "PDF Formatting Guide",
        24,
        "Output Preparation",
        900,
        2,
        """Output Requirements:
- Provide detailed instructions for generating a publication-ready PDF (pagination, headers/footers, typography).
- Address tables, figures, appendices, accessibility, and metadata population.""",
        context=["Markdown Formatting"],
    )
    add_spec(
        "Metadata Tagging",
        24,
        "Output Preparation",
        800,
        2,
        """Output Requirements:
- Propose metadata schema, keywords, abstracts, rights statements, and identifiers.
- Suggest automation or scripting paths for knowledge-base integration.""",
        context=["PDF Formatting Guide"],
    )

    # Specialized Research
    specialized_topics = [
        ("Academic Literature Review", 25),
        ("Policy Analysis", 26),
        ("Historical Context", 27),
        ("Legal Framework Analysis", 28),
        ("International Law Review", 29),
        ("Comparative Crackdown Study", 30),
        ("Statistical Trend Analysis", 31),
        ("Case Study Deep Dive", 32),
        ("Expert Opinion Synthesis", 33),
        ("Media Narrative Analysis", 34),
    ]

    for name, agent_idx in specialized_topics:
        add_spec(
            name,
            agent_idx,
            "Specialized Research",
            1800,
            3,
            f"""Research Tasks:
- Conduct an exhaustive {name.lower()} drawing from at least twelve authoritative sources.
- Compare methodologies, geographic perspectives, and temporal findings.
- Identify controversies, blind spots, and future research needs.""",
            context=["Fact Extraction", "Precedent Summaries", "Statutory Mapping"],
        )

    # Analysis & Synthesis
    synthesis_topics = [
        ("Critical Analysis", 35),
        ("Integrated Synthesis", 36),
        ("Pattern Recognition", 37),
        ("Causal Analysis", 38),
        ("Temporal Sequencing", 39),
        ("Comparative Synthesis", 40),
        ("Cross-Reference Matrix", 41),
        ("Evidence Evaluation", 42),
        ("Argumentation Audit", 43),
        ("Conclusion Builder", 44),
    ]

    context_for_synthesis = [name for name, _ in specialized_topics]

    for name, agent_idx in synthesis_topics:
        add_spec(
            name,
            agent_idx,
            "Analysis & Synthesis",
            1600,
            3,
            f"""Analytical Tasks:
- Integrate outputs from specialized research modules.
- Surface convergences, divergences, and causal pathways.
- Translate insights into actionable conclusions for final synthesis.""",
            context=context_for_synthesis,
        )

    # Precision Team
    precision_specs = [
        ("Final Review", 45),
        ("Quality Assurance Lead", 46),
        ("Comprehensive Editing", 47),
        ("Master Synthesis", 48),
    ]

    for name, agent_idx in precision_specs:
        add_spec(
            name,
            agent_idx,
            "Precision Team",
            2200,
            3,
            """Mandate:
- Audit the entire corpus for coherence, completeness, and rigor.
- Resolve outstanding issues from earlier reviewers.
- Ensure the final manuscript exceeds 25,000 words and meets academic standards.
- Provide strategic recommendations and monitoring priorities.""",
            context=[spec.name for spec in specs],
            focus=[
                "Holistic audit and risk identification.",
                "Integrative revisions and quality elevation.",
                "Executive-ready synthesis and forward-looking guidance.",
            ],
        )

    return specs


def research_49_agents(question: str, max_tasks: Optional[int] = None):
    """Run the 49-agent research workflow with iterative prompting and verification."""

    start_time = time.time()
    phase_times: Dict[str, float] = {}
    task_logs: List[TaskLog] = []

    print("\n" + "=" * 100)
    print("CrewAI 49-Agent Matrix Research System (Iterative Local Execution)")
    print("=" * 100)
    print(f"\nQuestion: {question}\n")
    print(f"Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 100 + "\n")

    if not verify_ollama("mistral:latest"):
        raise RuntimeError("Ollama verification failed. Ensure `ollama serve` is running and mistral:latest is installed.")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    phase_start = time.time()
    try:
        local_llm = LocalOllamaLLM(
            model="mistral:latest",
            base_url="http://localhost:11434",
            timeout=900,
            temperature=0.7,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Unable to initialize local LLM via Ollama: {exc}") from exc
    phase_times["llm_initialization"] = time.time() - phase_start

    # Build agents
    phase_start = time.time()
    agents = create_49_agents(local_llm)
    phase_times["agent_creation"] = time.time() - phase_start
    print(f"[OK] {len(agents)} specialized agents configured ({phase_times['agent_creation']:.2f}s)")

    # Build task specifications
    phase_start = time.time()
    specs = create_task_specs(question)
    phase_times["spec_definition"] = time.time() - phase_start
    print(f"[OK] {len(specs)} task specifications prepared ({phase_times['spec_definition']:.2f}s)\n")

    # Execute tasks iteratively
    outputs: Dict[str, Dict[str, Any]] = {}
    category_totals: Dict[str, int] = {}
    category_order: List[str] = []
    execution_start = time.time()

    preview_path = reports_dir / "preview_in_progress.md"

    for idx, spec in enumerate(specs, 1):
        if max_tasks is not None and idx > max_tasks:
            print(f"[INFO] Max task limit reached ({max_tasks}); halting execution loop early.")
            break
        if spec.category not in category_order:
            category_order.append(spec.category)
        agent = agents[spec.agent_index]
        print(f"[{idx:02}/{len(specs):02}] {spec.category} → {spec.name} | Target: {spec.iterations} iteration(s) × {spec.min_words} words")
        task_log = run_task_spec(agent, spec, outputs)
        task_logs.append(task_log)
        category_totals.setdefault(spec.category, 0)
        category_totals[spec.category] += task_log.total_word_count
        print(
            f"    Completed {task_log.total_word_count} words across "
            f"{spec.iterations} iteration(s) "
            f"(last attempt {task_log.iterations[-1].word_count} words)"
        )

        # Refresh live preview for human monitoring
        try:
            preview_sections = [
                "# Live Preview",
                f"**Question:** {question}",
                f"**Updated:** {datetime.now():%B %d, %Y %I:%M:%S %p}",
                "",
            ]
            for category in category_order:
                preview_sections.append(f"## {category}")
                for spec_candidate in filter(lambda s: s.category == category, specs):
                    payload = outputs.get(spec_candidate.name)
                    if not payload:
                        continue
                    preview_sections.append(
                        f"### {spec_candidate.name}\n\n"
                        f"*Words:* {payload['word_count']:,} "
                        f"(iterations: {payload.get('iterations_completed', 0)})\n\n"
                        f"{payload['content']}\n"
                    )
            preview_path.write_text("\n".join(preview_sections), encoding="utf-8")
        except Exception as preview_exc:  # noqa: BLE001
            print(f"[WARN] Unable to refresh live preview: {preview_exc}")

    phase_times["execution"] = time.time() - execution_start

    total_words = sum(payload["word_count"] for payload in outputs.values())
    print(f"\n[INFO] Aggregate word count after primary pass: {total_words:,} words")

    # Top-up if total word count is below 25,000
    supplemental_content = ""
    if total_words < 25_000:
        shortfall = 25_000 - total_words
        print(f"[WARN] Total word count below target by {shortfall:,} words. Generating supplemental synthesis.")
        master_spec = next(spec for spec in specs if spec.name == "Master Synthesis")
        master_agent = agents[master_spec.agent_index]
        supplement_prompt = f"""
You are {master_agent.role}. The compiled research currently totals {total_words} words.
Generate an additional {shortfall + 800} words of fresh analysis that:
- Deepens the comparative and temporal synthesis.
- Avoids repeating phrasing from prior outputs.
- Provides forward-looking insights, monitoring priorities, and policy implications.
- Includes inline references to earlier task names where relevant.
"""
        supplemental_content = master_agent.llm.call(supplement_prompt.strip())
        supplemental_words = safe_word_count(supplemental_content)
        total_words += supplemental_words
        outputs["Supplemental Expansion"] = {
            "content": supplemental_content,
            "word_count": supplemental_words,
            "iterations_completed": 1,
            "agent_role": master_agent.role,
            "category": "Precision Team",
            "context_used": [spec.name for spec in specs],
        }
        category_totals.setdefault("Precision Team", 0)
        category_totals["Precision Team"] += supplemental_words
        if "Precision Team" not in category_order:
            category_order.append("Precision Team")
        print(f"[OK] Supplemental expansion added {supplemental_words:,} words. New total: {total_words:,} words.")

    # Assemble markdown report
    phase_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c for c in question[:40] if c.isalnum() or c in " ").replace(" ", "_")

    header_lines = [
        "# 49-Agent Matrix Research Report",
        f"**Research Question:** {question}",
        f"**Generated:** {datetime.now():%B %d, %Y %I:%M %p}",
        "**System:** CrewAI 49-Agent Iterative Matrix (Local Ollama - mistral:latest)",
        f"**Total Word Count:** {total_words:,}",
        "**Execution:** 49 agents x multi-iteration loops",
        "**Verification:** Local Ollama connectivity, per-iteration word counts, JSON execution log",
        "",
    ]

    body_sections: List[str] = []
    for category in category_order:
        body_sections.append(f"## {category}")
        for spec in filter(lambda s: s.category == category, specs):
            payload = outputs.get(spec.name)
            if not payload:
                continue
            body_sections.append(f"### {spec.name}\n\n*Word count:* {payload['word_count']:,}\n\n{payload['content']}\n")
        if category == "Precision Team" and supplemental_content:
            payload = outputs.get("Supplemental Expansion")
            if payload:
                body_sections.append(
                    f"### Supplemental Expansion\n\n*Word count:* {payload['word_count']:,}\n\n{payload['content']}\n"
                )

    final_report = "\n".join(header_lines + body_sections)

    # Save markdown and PDF
    md_file = reports_dir / f"{timestamp}_{safe_name}_49_AGENTS_ITERATIVE.md"
    md_file.write_text(final_report, encoding="utf-8")
    print(f"[OK] Markdown report saved: {md_file}")

    pdf_file = reports_dir / f"{timestamp}_{safe_name}_49_AGENTS_ITERATIVE.pdf"
    if create_pdf_report(final_report, str(pdf_file), question):
        print(f"[OK] PDF report saved: {pdf_file}")
    else:
        print("[WARN] PDF generation failed; markdown remains available.")

    # Persist execution log
    log_path = reports_dir / "results" / f"{timestamp}_49_agents_execution.json"
    write_execution_log(task_logs, log_path)
    print(f"[OK] Execution log saved: {log_path}")

    phase_times["report_saving"] = time.time() - phase_start

    total_duration = time.time() - start_time
    end_time = datetime.now()

    print("\n" + "=" * 100)
    print("49-AGENT MATRIX RESEARCH COMPLETE")
    print("=" * 100)
    print(f"\nTIMING ANALYSIS:")
    print(f"   Start Time: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End Time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print("\nPHASE BREAKDOWN:")
    for key in ["llm_initialization", "agent_creation", "spec_definition", "execution", "report_saving"]:
        if key in phase_times:
            print(f"   {key.replace('_', ' ').title()}: {phase_times[key]:.2f}s ({phase_times[key]/total_duration*100:.1f}%)")
    print("\nCATEGORY WORD COUNTS:")
    for category in category_order:
        print(f"   {category}: {category_totals.get(category, 0):,} words")
    print(f"\nPERFORMANCE METRICS:")
    executed_count = len(task_logs) or 1
    print(f"   Total Words: {total_words:,}")
    print(f"   Average Words per Task: {total_words // executed_count:,}")
    print(f"   Average Task Duration: {phase_times['execution']/executed_count:.2f}s")
    print(f"   Output Characters per Second: {len(final_report)/total_duration:.0f}")
    print(f"   Execution Log: {log_path}")
    print(f"\nDeliverables:")
    print(f"   - Markdown: {md_file}")
    print(f"   - PDF:      {pdf_file}")

    return final_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the 49-agent CCP crackdown research workflow.")
    parser.add_argument(
        "question",
        nargs="*",
        help=(
            "Optional research prompt. If omitted, a default question exploring CCP narrative conflicts with "
            "U.S. Ivy League statements is used."
        ),
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on the number of task specifications to execute (useful for smoke-testing).",
    )
    args = parser.parse_args()

    question = (
        " ".join(args.question)
        if args.question
        else (
            "Investigate whether crackdowns like the ESUWIKI campaign or comparable CCP political campaigns could "
            "plausibly be triggered by major U.S. Ivy League universities issuing public statements that contradict "
            "official Chinese media narratives. Analyze historical precedents, policy patterns, and legal frameworks "
            "that would support arguments about narrative conflicts triggering political signaling or crackdowns."
        )
    )

    try:
        research_49_agents(question, max_tasks=args.max_tasks)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()



