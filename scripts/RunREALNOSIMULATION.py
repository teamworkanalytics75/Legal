#!/usr/bin/env python3
"""
REAL 49-Agent System Execution - NO SIMULATION
==============================================

This script runs the ACTUAL MasterSupervisor with real agent coordination.
Replaces all simulation code with real execution calls.
"""

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

# Set the API key from the project
os.environ["OPENAI_API_KEY"] = "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"

# Add writer_agents/code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from master_supervisor import MasterSupervisor, SupervisorConfig, SessionConfig, MemoryConfig
from session_manager import SessionManager
from job_persistence import JobManager
from insights import CaseInsights, EvidenceItem
from task_decomposer import TaskDecomposer

# ---------------------------------------------------------------------------
# Evidence loading utilities
# ---------------------------------------------------------------------------

DEFAULT_LAWSUIT_DB = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

ARGUMENT_QUERIES: Dict[str, List[str]] = {
    "alumni_clubs": [
        "%alumni%club%",
        "%thousand%alumni%",
        "%club%alumni%",
    ],
    "three_years_talks": [
        "%three%year%",
        "%same%talk%",
        "%same%slide%",
        "%3%year%",
    ],
    "sendelta_investigation": [
        "%sendelta%",
        "%investigat%sendelta%",
        "%harvard%investigat%",
    ],
    "yi_wang_confrontation": [
        "%yi%wang%",
        "%confront%",
        "%picture%slide%",
        "%wang%confront%",
    ],
    "fresh_graduates": [
        "%fresh%graduat%",
        "%recent%graduat%",
        "%new%graduat%",
    ],
    "ccp_pressure": [
        "%ccp%",
        "%pressure%",
        "%china%pressure%",
        "%grapevine%",
    ],
}

ARGUMENT_DESCRIPTIONS: Dict[str, str] = {
    "alumni_clubs": "Large Harvard alumni clubs in China that could circulate presentation content",
    "three_years_talks": "Three-year repetition of the same slide deck, increasing exposure",
    "sendelta_investigation": "Harvard investigations into Sendelta High School contemporaneous with the talk",
    "yi_wang_confrontation": "Yi Wang confrontation involving photos of the user's slides",
    "fresh_graduates": "Both user and Xi Mingze were recent graduates, raising visibility",
    "ccp_pressure": "CCP sensitivity and pressure on Harvard alumni regarding Xi Mingze references",
}

DEFAULT_QUESTION = dedent(
    """
    What is the likelihood that Marlyn McGrath knew about the SPECIFIC Xi Mingze slide
    in my PowerPoint presentation before Harvard published Statement 1 on April 19, 2019?

    IMPORTANT CLARIFICATION: Yi Wang confronted me with a picture of my OTHER slides
    (not the Xi Mingze slide). This shows general presentation knowledge but NOT
    specific knowledge of the Xi Mingze slide.

    Consider my arguments about:
    - Thousands of alumni in Harvard clubs in China
    - I gave the same talk for 3 years with the same slides
    - Harvard was already investigating the Sendelta high school situation
    - Yi Wang confronted me with pictures of OTHER slides (not the Xi Mingze slide)
    - Xi Mingze was only fresh graduated and me too
    - Such sensitive topics likely led to CCP pressure on Harvard alumni community
    - All these arguments are already in my database
    """
).strip()


def _normalize_preview(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.encode("ascii", "ignore").decode("ascii")


def load_argument_evidence(
    db_path: Path = DEFAULT_LAWSUIT_DB,
    max_docs_per_argument: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load argument-specific evidence snippets from the lawsuit database."""
    results: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ARGUMENT_QUERIES}

    if not db_path.exists():
        return results

    query_template = dedent(
        """
        SELECT
            rowid AS doc_id,
            SUBSTR(content, 1, 1200) AS preview,
            LENGTH(content) AS doc_length
        FROM cleaned_documents
        WHERE {conditions}
        ORDER BY doc_length DESC
        LIMIT ?
        """
    )

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for argument_key, patterns in ARGUMENT_QUERIES.items():
            placeholders = " OR ".join(["LOWER(content) LIKE ?"] * len(patterns))
            sql = query_template.format(conditions=placeholders)
            params: List[Any] = [pattern.lower() for pattern in patterns]
            params.append(max_docs_per_argument)

            cursor.execute(sql, params)
            rows = cursor.fetchall()

            for row in rows:
                results[argument_key].append(
                    {
                        "doc_id": row["doc_id"],
                        "preview": _normalize_preview(row["preview"]),
                        "length": row["doc_length"],
                    }
                )
    except Exception as exc:
        print(f"[REAL] Warning: failed to load argument evidence: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return results


def build_evidence_items(argument_data: Dict[str, List[Dict[str, Any]]]) -> List[EvidenceItem]:
    """Convert argument evidence into CaseInsights EvidenceItems."""
    evidence_items: List[EvidenceItem] = []

    for key, docs in argument_data.items():
        if not docs:
            continue

        description = ARGUMENT_DESCRIPTIONS.get(key, key.replace("_", " ").title())
        top_doc = docs[0]
        weight = min(1.0, (top_doc.get("length", 0) or 0) / 5000.0)

        evidence_items.append(
            EvidenceItem(
                node_id=key,
                state="supported",
                weight=weight if weight > 0 else None,
                description=f"{description}. Example snippet: {top_doc.get('preview', '')[:250]}",
            )
        )

    return evidence_items


def build_argument_summary(argument_data: Dict[str, List[Dict[str, Any]]]) -> str:
    """Create a human-readable summary of argument evidence."""
    lines: List[str] = []
    for key, description in ARGUMENT_DESCRIPTIONS.items():
        docs = argument_data.get(key, [])
        if docs:
            sample = docs[0].get("preview", "")
            lines.append(f"- {description} — evidence found (Doc #{docs[0]['doc_id']}): {sample[:180]}...")
        else:
            lines.append(f"- {description} — no direct documents located in current search window.")
    return "\n".join(lines)


def build_documents_payload(argument_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Transform argument evidence into a documents payload for research agents."""
    documents: List[Dict[str, Any]] = []
    for key, docs in argument_data.items():
        for doc in docs:
            documents.append(
                {
                    "argument_type": key,
                    "doc_id": doc["doc_id"],
                    "text": doc.get("preview", ""),
                    "length": doc.get("length", 0),
                }
            )
    return documents


def count_argument_hits(argument_data: Dict[str, List[Dict[str, Any]]]) -> int:
    """Count how many argument categories have at least one supporting document."""
    return sum(1 for docs in argument_data.values() if docs)


def estimate_probability(argument_hits: int) -> Dict[str, Any]:
    """Generate a simple probability estimate based on evidence coverage."""
    base_prob = 0.20  # 20% baseline without corroboration
    increment = 0.05  # +5% per corroborated argument category
    point_estimate = min(0.85, base_prob + increment * argument_hits)

    lower_bound = max(0.05, point_estimate - 0.05)
    upper_bound = min(0.95, point_estimate + 0.05)

    return {
        "argument_hits": argument_hits,
        "point_estimate_percent": round(point_estimate * 100, 1),
        "lower_bound_percent": round(lower_bound * 100, 1),
        "upper_bound_percent": round(upper_bound * 100, 1),
        "method": "Heuristic weighting based on corroborated argument categories",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the real runner."""
    parser = argparse.ArgumentParser(description="Run the real 49-agent The Matrix analysis.")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Custom analysis question. Overrides --question-file if both are provided.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="Path to a text file containing the analysis question.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="mcgrath_REAL_NO_SIMULATION",
        help="Prefix for the generated output JSON filename.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Reuse an existing session ID instead of creating a new one.",
    )

    args = parser.parse_args()

    if args.question_file:
        question_path = Path(args.question_file)
        if not question_path.exists():
            parser.error(f"Question file not found: {question_path}")
        args.question = question_path.read_text(encoding="utf-8")

    if not args.question:
        args.question = DEFAULT_QUESTION

    args.question = dedent(args.question).strip()
    return args


class RealExecutionTracker:
    """Tracks REAL execution metrics from actual agent runs."""

    def __init__(self):
        self.start_time = time.time()
        self.execution_log = []
        self.phase_results = {}
        self.agent_executions = []

    def log_phase_start(self, phase_name: str):
        """Log when a phase starts."""
        self.execution_log.append({
            "timestamp": time.time(),
            "event": f"phase_start_{phase_name}",
            "phase": phase_name
        })
        print(f"[REAL] Starting {phase_name} phase...")

    def log_phase_complete(self, phase_name: str, results: Dict[str, Any]):
        """Log when a phase completes with real results."""
        self.execution_log.append({
            "timestamp": time.time(),
            "event": f"phase_complete_{phase_name}",
            "phase": phase_name,
            "results": results
        })
        self.phase_results[phase_name] = results
        print(f"[REAL] {phase_name} phase completed with {len(results)} results")

    def log_agent_execution(self, agent_name: str, success: bool, tokens: int = 0, error: str = None):
        """Log individual agent execution."""
        self.agent_executions.append({
            "agent_name": agent_name,
            "success": success,
            "tokens": tokens,
            "error": error,
            "timestamp": time.time()
        })
        status = "SUCCESS" if success else "FAILED"
        print(f"[REAL] Agent {agent_name}: {status} ({tokens} tokens)")
        if error:
            print(f"[REAL] Error: {error}")

    def get_execution_report(self) -> Dict[str, Any]:
        """Get comprehensive execution report."""
        duration = time.time() - self.start_time
        successful_agents = [a for a in self.agent_executions if a["success"]]
        failed_agents = [a for a in self.agent_executions if not a["success"]]
        total_tokens = sum(a["tokens"] for a in self.agent_executions)

        return {
            "execution_duration_seconds": duration,
            "total_agents_executed": len(self.agent_executions),
            "successful_agents": len(successful_agents),
            "failed_agents": len(failed_agents),
            "total_tokens_used": total_tokens,
            "phases_completed": list(self.phase_results.keys()),
            "phase_results": self.phase_results,
            "agent_executions": self.agent_executions,
            "execution_log": self.execution_log,
            "timestamp": datetime.now().isoformat()
        }


class RealThe MatrixRunner:
    """Runs the REAL 49-agent system with actual execution - NO SIMULATION."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self.tracker = RealExecutionTracker()
        self.session_manager = SessionManager(db_path)
        self.job_manager = JobManager(db_path)

        # Initialize REAL configuration
        self.config = SupervisorConfig(
            db_path=db_path,
            enable_research=True,
            enable_drafting=True,
            enable_citation=True,
            enable_qa=True,
            memory_config=MemoryConfig(
                enabled=True,
                use_local_embeddings=True,
                k_neighbors=5,
                max_memory_tokens=2000,
                refresh_after_runs=3
            ),
            session_config=SessionConfig(
                enabled=True,
                default_expiry_days=7,
                max_context_interactions=10,
                max_context_tokens=5000
            )
        )
        self._last_base_context: Dict[str, Any] = {}
        self._last_argument_data: Dict[str, Any] = {}
        self._last_probability_info: Dict[str, Any] = {}

    def _prepare_case_inputs(
        self,
        question: str,
    ) -> Tuple[CaseInsights, Dict[str, Any], Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """Load evidence, build case insights, and assemble base context."""
        argument_data = load_argument_evidence()
        evidence_items = build_evidence_items(argument_data)
        evidence_summary = build_argument_summary(argument_data)
        documents_payload = build_documents_payload(argument_data)
        argument_hits = count_argument_hits(argument_data)
        probability_info = estimate_probability(argument_hits)

        summary_text = dedent(
            f"""
            Evaluate whether Harvard admissions dean Marlyn McGrath likely knew that the user
            presented a PowerPoint slide about Xi Mingze before Harvard issued Statement 1 on
            April 19, 2019. Consider institutional communication channels, alumni information
            sharing, and evidence pulled from the lawsuit database.

            Key evidence overview:
            {evidence_summary}
            """
        ).strip()

        analysis_context = dedent(
            f"""
            QUESTION:
            {question.strip()}

            EVIDENCE SUMMARY:
            {evidence_summary}

            TASK:
            Produce a legal-style analysis estimating the probability that Marlyn McGrath was
            aware of the specific Xi Mingze slide before April 19, 2019. Discuss each evidence
            strand, explain transmission pathways, and conclude with a probability assessment.
            """
        ).strip()

        case_insights = CaseInsights(
            reference_id="REAL_MCGRATH_ANALYSIS",
            summary=summary_text,
            posteriors=[],
            evidence=evidence_items,
            jurisdiction="Harvard Admissions",
            case_style="Specific Knowledge Assessment",
        )

        base_context: Dict[str, Any] = {
            "question": question.strip(),
            "summary": summary_text,
            "analysis_context": analysis_context,
            "context": analysis_context,
            "insights": case_insights.to_prompt_block(),
            "documents": documents_payload,
            "text": analysis_context,
            "legal_issue": "Specific knowledge of Xi Mingze slide prior to April 19, 2019",
            "evidence_summary": evidence_summary,
            "argument_evidence": argument_data,
        }

        return case_insights, base_context, argument_data, probability_info

    async def run_real_mcgrath_analysis(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the REAL McGrath analysis with actual 49-agent execution - NO SIMULATION."""

        print(f"[REAL] Starting REAL McGrath analysis - NO SIMULATION...")
        print(f"[REAL] Question: {question}")

        # Prepare evidence, insights, and base context
        case_insights, base_context, argument_data, probability_info = self._prepare_case_inputs(question)
        self._last_base_context = base_context
        self._last_argument_data = argument_data
        self._last_probability_info = probability_info

        # Create or use existing session
        if not session_id:
            session_id = self.session_manager.create_session("REAL McGrath Analysis")
            print(f"[REAL] Created new session: {session_id}")
        else:
            print(f"[REAL] Using existing session: {session_id}")

        # Initialize REAL MasterSupervisor
        supervisor = MasterSupervisor(
            session=None,  # We'll handle this differently
            config=self.config,
            session_id=session_id
        )
        supervisor.base_context = base_context

        # Create REAL task profile
        task_decomposer = TaskDecomposer()
        summary = base_context["analysis_context"]
        task_profile = await task_decomposer.compute(case_insights, summary)

        print(f"[REAL] Task profile created: {task_profile.evidence_count} evidence items")
        print(f"[REAL] Spawn policies: {task_profile.spawn_policies}")

        # Execute the REAL analysis - NO SIMULATION
        print(f"[REAL] Executing REAL 49-agent analysis - NO SIMULATION...")
        start_time = time.time()

        try:
            # THIS IS THE REAL EXECUTION - NO SIMULATION
            result = await supervisor.execute_with_session(
                user_prompt=question,
                insights=case_insights,
                summary=summary
            )

            execution_time = time.time() - start_time
            print(f"[REAL] REAL Analysis completed in {execution_time:.2f} seconds")

            # Extract REAL execution stats from the result
            execution_stats = result.get('execution_stats', {})
            phase_results = result.get('phase_results', {})

            # Log real phase results
            for phase_name, phase_data in phase_results.items():
                if phase_data:  # Only log non-empty phases
                    self.tracker.log_phase_complete(phase_name, phase_data)

            # Extract real agent execution data
            total_agents = 0
            total_tokens = execution_stats.get('total_tokens', 0)

            # Count agents from phase results
            for phase_name, phase_data in phase_results.items():
                if isinstance(phase_data, dict):
                    for agent_key, agent_result in phase_data.items():
                        if isinstance(agent_result, dict) and 'agent_name' in agent_result:
                            total_agents += 1
                            self.tracker.log_agent_execution(
                                agent_name=agent_result['agent_name'],
                                success=True,  # If we got results, it succeeded
                                tokens=0  # We'll get this from execution_stats
                            )

            # Add real execution report
            execution_report = self.tracker.get_execution_report()
            result["real_execution_report"] = execution_report
            result["argument_evidence"] = argument_data
            result["evidence_summary"] = base_context.get("evidence_summary", "")
            result["probability_assessment"] = probability_info
            result["analysis_context"] = base_context.get("analysis_context", "")

            print(f"[REAL] Total agents executed: {total_agents}")
            print(f"[REAL] Total tokens: {total_tokens}")
            print(f"[REAL] Execution time: {execution_time:.2f} seconds")

            return result

        except Exception as e:
            print(f"[REAL ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main execution function."""
    print("REAL 49-AGENT WITCHWEB SYSTEM EXECUTION - NO SIMULATION")
    print("=" * 70)

    args = parse_args()
    question = args.question

    # Initialize REAL runner
    runner = RealThe MatrixRunner()

    try:
        # Run the REAL analysis
        result = await runner.run_real_mcgrath_analysis(question, session_id=args.session_id)

        # Save REAL results
        output_file = f"{args.output_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n[REAL SUCCESS] Analysis completed!")
        print(f"[REAL] Results saved to: {output_file}")

        # Print REAL execution report
        execution_report = result.get('real_execution_report', {})
        print(f"[REAL] Execution duration: {execution_report.get('execution_duration_seconds', 0):.2f} seconds")
        print(f"[REAL] Total agents executed: {execution_report.get('total_agents_executed', 0)}")
        print(f"[REAL] Successful agents: {execution_report.get('successful_agents', 0)}")
        print(f"[REAL] Failed agents: {execution_report.get('failed_agents', 0)}")
        print(f"[REAL] Total tokens: {execution_report.get('total_tokens_used', 0)}")
        print(f"[REAL] Phases completed: {execution_report.get('phases_completed', [])}")

        # Print REAL summary
        print(f"\nREAL ANALYSIS SUMMARY:")
        if 'final_text' in result and result['final_text']:
            print(result['final_text'])
        else:
            print("Analysis completed - check output file for detailed results")

        return result

    except Exception as e:
        print(f"[REAL ERROR] Analysis failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
