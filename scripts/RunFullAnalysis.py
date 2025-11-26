#!/usr/bin/env python3
"""
REAL 49-Agent System Execution
=============================

This script runs the ACTUAL MasterSupervisor with real agent coordination.
No simulations - this executes the real 49-agent system.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Set the API key from the project
os.environ["OPENAI_API_KEY"] = "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"

# Add writer_agents/code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from master_supervisor import MasterSupervisor, SupervisorConfig, SessionConfig, MemoryConfig
from session_manager import SessionManager
from job_persistence import JobManager
from insights import CaseInsights
from task_decomposer import TaskDecomposer


class RealCostTracker:
    """Tracks REAL API costs from actual agent execution."""

    def __init__(self):
        self.start_time = time.time()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.agent_calls = 0
        self.phase_executions = 0

    def add_execution(self, tokens: int, cost: float):
        """Add real execution data."""
        self.total_tokens += tokens
        self.total_cost += cost
        self.agent_calls += 1

    def add_phase(self):
        """Track phase execution."""
        self.phase_executions += 1

    def get_report(self) -> Dict[str, Any]:
        """Get real cost report."""
        duration = time.time() - self.start_time
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "agent_calls": self.agent_calls,
            "phase_executions": self.phase_executions,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }


class RealThe MatrixRunner:
    """Runs the REAL 49-agent system with actual execution."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self.cost_tracker = RealCostTracker()
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

    async def run_real_mcgrath_analysis(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the REAL McGrath analysis with actual 49-agent execution."""

        print(f"[REAL] Starting REAL McGrath analysis...")
        print(f"[REAL] Question: {question}")

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

        # Create REAL case insights
        case_insights = CaseInsights(
            reference_id="REAL_MCGRATH_ANALYSIS_001",
            summary=f"REAL Analysis of McGrath's knowledge of Xi Mingze slide before April 19, 2019 - Yi Wang confrontation was about OTHER slides",
            posteriors=[],
            evidence=[],
            jurisdiction="Federal",
            case_style="REAL McGrath Knowledge Assessment"
        )

        # Create REAL task profile
        task_decomposer = TaskDecomposer()
        summary = f"REAL Analysis: Whether Marlyn McGrath knew about the SPECIFIC Xi Mingze slide in the user's PowerPoint presentation before Harvard's Statement 1 on April 19, 2019. CRITICAL: Yi Wang confronted user with pictures of OTHER slides (not the Xi Mingze slide), which shows general presentation knowledge but NOT specific slide knowledge."
        task_profile = await task_decomposer.compute(case_insights, summary)

        print(f"[REAL] Task profile created: {task_profile.evidence_count} evidence items")
        print(f"[REAL] Spawn policies: {task_profile.spawn_policies}")

        # Execute the REAL analysis
        print(f"[REAL] Executing REAL 49-agent analysis...")
        start_time = time.time()

        try:
            # This is the REAL execution - no simulation
            result = await supervisor.execute_with_session(
                user_prompt=question,
                insights=case_insights,
                summary=summary
            )

            execution_time = time.time() - start_time
            print(f"[REAL] REAL Analysis completed in {execution_time:.2f} seconds")

            # Add real execution stats
            execution_stats = result.get('execution_stats', {})
            self.cost_tracker.add_execution(
                tokens=execution_stats.get('total_tokens', 0),
                cost=execution_stats.get('total_cost', 0.0)
            )

            # Generate REAL cost report
            cost_report = self.cost_tracker.get_report()
            result["real_cost_report"] = cost_report

            print(f"[REAL] Total cost: ${cost_report['total_cost']:.4f}")
            print(f"[REAL] Total tokens: {cost_report['total_tokens']:,}")
            print(f"[REAL] Agent calls: {cost_report['agent_calls']}")
            print(f"[REAL] Phase executions: {cost_report['phase_executions']}")

            return result

        except Exception as e:
            print(f"[REAL ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise


async def main():
    """Main execution function."""
    print("REAL 49-AGENT WITCHWEB SYSTEM EXECUTION")
    print("=" * 60)

    # The REAL McGrath question with proper clarification
    real_mcgrath_question = """
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

    # Initialize REAL runner
    runner = RealThe MatrixRunner()

    try:
        # Run the REAL analysis
        result = await runner.run_real_mcgrath_analysis(real_mcgrath_question)

        # Save REAL results
        output_file = f"mcgrath_REAL_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[REAL SUCCESS] Analysis completed!")
        print(f"[REAL] Results saved to: {output_file}")
        print(f"[REAL] Total cost: ${result['real_cost_report']['total_cost']:.4f}")
        print(f"[REAL] Total tokens: {result['real_cost_report']['total_tokens']:,}")
        print(f"[REAL] Agent calls: {result['real_cost_report']['agent_calls']}")
        print(f"[REAL] Phase executions: {result['real_cost_report']['phase_executions']}")

        # Print REAL summary
        print(f"\nREAL ANALYSIS SUMMARY:")
        if 'summary' in result:
            print(result['summary'])
        else:
            print("Analysis completed - check output file for detailed results")

        return result

    except Exception as e:
        print(f"[REAL ERROR] Analysis failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())
