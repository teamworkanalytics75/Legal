"""
CrewAI deep research workflow using a local GGUF model driven by CTransformers.
Builds a multi-agent crew, executes tasks, and writes a markdown report.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Sequence, Tuple

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from ddgs import DDGS
from langchain_community.llms import CTransformers


class LocalCTransformersLLM(BaseLLM):
    """CrewAI-compatible wrapper around LangChain's CTransformers."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        *,
        temperature: float = 0.2,
        max_new_tokens: int = 600,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> None:
        super().__init__(model=model_path, temperature=temperature, provider="local")
        self._client = CTransformers(
            model=model_path,
            model_type=model_type,
            config={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            },
        )

    def call(  # type: ignore[override]
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
    ):
        if isinstance(messages, str):
            prompt = messages
        else:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                prompt_parts.append(f"{role.capitalize()}: {content}")
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)

        response = self._client.invoke(prompt)
        if isinstance(response, str):
            return self._apply_stop_words(response)
        return response

    def supports_stop_words(self) -> bool:
        return True


class DuckDuckGoTool(BaseTool):
    """Simple DuckDuckGo text search exposed as a CrewAI tool."""

    name: str = "web_search"
    description: str = "Search DuckDuckGo and return recent sources with snippets."

    def _run(self, query: str) -> str:
        try:
            results = DDGS().text(query, max_results=12)
        except Exception as exc:  # noqa: BLE001
            return f"Search error: {exc}"

        if not results:
            return f"No results for: {query}"

        lines = [f"Search results for: {query}", "=" * 72]
        for idx, item in enumerate(results, 1):
            lines.append(f"{idx}. {item['title']}")
            lines.append(f"   {item['body']}")
            lines.append(f"   Source: {item['href']}")
        return "\n".join(lines)

    async def _arun(self, query: str) -> str:  # pragma: no cover - async shim
        return self._run(query)


def build_agents(local_llm: BaseLLM) -> Tuple[Agent, ...]:
    """Create the specialized CrewAI agents."""
    return (
        Agent(
            role="Lead Internet Researcher",
            goal="Collect comprehensive open-source intelligence relevant to the brief.",
            backstory="Veteran researcher adept at multi-angle source discovery.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
        Agent(
            role="Academic Analyst",
            goal="Surface scholarly consensus, debates, and theoretical framing.",
            backstory="Interdisciplinary academic who synthesizes legal and political research.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
        Agent(
            role="Policy and Legal Specialist",
            goal="Explain enforcement mechanics, regulatory context, and campaign logistics.",
            backstory="Policy practitioner translating statutes into real-world implementation.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
        Agent(
            role="Critical Methodologist",
            goal="Stress-test claims, highlight evidence gaps, and flag bias risks.",
            backstory="Methodology expert focused on reliability and epistemic humility.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
        Agent(
            role="Synthesis Architect",
            goal="Integrate all findings into a coherent perspective aligned to the question.",
            backstory="Systems thinker who builds clear narratives from diverse inputs.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
        Agent(
            role="Lead Writer",
            goal="Deliver a polished markdown report with citations and actionable takeaways.",
            backstory="Experienced research communicator writing for legal-policy audiences.",
            llm=local_llm,
            allow_delegation=False,
            verbose=False,
        ),
    )


def build_tasks(question: str, agents: Sequence[Agent], tool: BaseTool) -> Sequence[Task]:
    """Define the multi-stage research plan."""
    researcher_task = Task(
        description=f"""Collect current reporting, think tank briefs, and investigative journalism on:
{question}

Instructions:
- Issue multiple targeted web searches (campaign names, crackdowns, enforcement campaigns).
- Capture timelines, responsible agencies, and representative incidents.
- Present findings as bullet lists with inline source URLs.""",
        expected_output="Structured OSINT memo with numbered sources.",
        agent=agents[0],
        tools=[tool],
    )

    academic_task = Task(
        description="""Survey scholarly literature with emphasis on Margaret K. Lewis and comparable China-law experts.
- Extract definitions of "crackdown", temporal markers, and legal exceptionalism.
- Summarize consensus positions, disagreements, and methodological caveats.
Produce a 3-5 paragraph synthesis with inline citations.""",
        expected_output="Concise academic literature review.",
        agent=agents[1],
        tools=[tool],
    )

    policy_task = Task(
        description="""Explain how bureaucratic control operates outside crackdowns.
- Compare routine enforcement methods with campaign-style crackdowns.
- Detail how selective enforcement and legal workarounds are operationalized.
Highlight implications for human-rights and compliance monitoring.""",
        expected_output="Policy analysis contrasting routine governance vs campaign crackdowns.",
        agent=agents[2],
        tools=[tool],
    )

    critic_task = Task(
        description="""Assess evidence reliability.
- Identify thin or contested areas in the source base.
- Flag potential bias (Western analysis vs PRC narratives).
- Recommend follow-up research leads or data needs.""",
        expected_output="Critical appraisal summarizing uncertainties and future research ideas.",
        agent=agents[3],
    )

    synthesis_task = Task(
        description="""Integrate prior outputs into a coherent summary.
- Distill the temporal patterns of Chinese crackdowns.
- Contrast them with ongoing bureaucratic control pathways.
- Provide bullet-point takeaways accessible to executive readers.""",
        expected_output="Synthesis brief linking all upstream findings.",
        agent=agents[4],
        context=[researcher_task, academic_task, policy_task, critic_task],
    )

    writer_task = Task(
        description="""Produce a polished markdown report (~1,800 words).
Sections:
1. Executive Summary
2. Temporal Pattern Overview
3. Crackdowns vs Bureaucratic Control
4. Legal Mechanisms and Exceptionalism
5. Evidence Quality & Gaps
6. Implications & Next Questions
Include inline citations and finish with a bibliography of consulted sources.""",
        expected_output="Complete markdown report with citations and recommendations.",
        agent=agents[5],
        context=[
            researcher_task,
            academic_task,
            policy_task,
            critic_task,
            synthesis_task,
        ],
    )

    return (
        researcher_task,
        academic_task,
        policy_task,
        critic_task,
        synthesis_task,
        writer_task,
    )


def run_research(question: str) -> Path:
    """Execute the local CrewAI workflow and save the markdown output."""
    candidate_models = [
        ("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "llama", "TinyLlama 1.1B Chat Q4_K_M"),
        ("phi-2.Q4_K_M.gguf", "phi", "Phi-2 Q4_K_M"),
        ("Phi-3-mini-4k-instruct-Q4_K_M.gguf", "phi", "Phi-3 Mini 4K Instruct Q4_K_M"),
    ]

    model_path: Path | None = None
    model_type: str | None = None
    model_label: str | None = None
    for filename, llm_type, label in candidate_models:
        candidate = Path("models") / filename
        if candidate.exists():
            model_path = candidate
            model_type = llm_type
            model_label = label
            break

    if model_path is None or model_type is None or model_label is None:
        raise FileNotFoundError("No supported GGUF model found in ./models.")

    timing = {"start": time.time()}
    llm = LocalCTransformersLLM(str(model_path), model_type)
    agents = build_agents(llm)
    search_tool = DuckDuckGoTool()
    tasks = build_tasks(question, agents, search_tool)

    crew = Crew(
        agents=list(agents),
        tasks=list(tasks),
        verbose=True,
    )

    timing["crew_start"] = time.time()
    report_markdown = crew.kickoff()
    timing["crew_end"] = time.time()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = "".join(c if c.isalnum() else "_" for c in question.lower())[:60]
    output_path = reports_dir / f"{timestamp}_{slug}_LOCAL_CREW.md"

    meta = [
        "# Deep Research Report",
        f"**Question:** {question}",
        f"**Generated:** {datetime.now():%B %d, %Y %I:%M %p}",
        f"**LLM:** {model_label} (GGUF via CTransformers, local CPU inference)",
        "**Agents:** Researcher, Academic Analyst, Policy Specialist, Critical Methodologist, "
        "Synthesis Architect, Lead Writer",
        f"**Runtime:** {timing['crew_end'] - timing['start']:.1f}s "
        f"(Crew execution {timing['crew_end'] - timing['crew_start']:.1f}s)",
        "",
        report_markdown,
    ]
    output_path.write_text("\n".join(meta), encoding="utf-8")
    return output_path


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Run CrewAI research with a local GGUF model.")
    parser.add_argument("question", nargs="*", help="Research question to explore.")
    args = parser.parse_args()

    question = (
        "According to Margaret Lewis, what are the defining temporal patterns of Chinese crackdowns "
        "and how do they differ from ongoing bureaucratic control?"
        if not args.question
        else " ".join(args.question)
    )

    try:
        output = run_research(question)
    except Exception as exc:  # noqa: BLE001
        print(f"Research run failed: {exc}")
    else:
        print(f"\nReport saved to {output}")


if __name__ == "__main__":
    main()

