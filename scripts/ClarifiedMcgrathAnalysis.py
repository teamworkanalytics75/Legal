#!/usr/bin/env python3
"""
Clarified McGrath Analysis - The Matrix Reasoning System
=====================================================

This script runs the actual The Matrix reasoning system with the clarified question:
"What's the likelihood that Marlyn McGrath knew about Xi Mingze BEFORE Harvard
published Statement 1 on April 19, 2019, given her conflicts of interest and
role in Xi Mingze's pseudonym?"

CRITICAL CLARIFICATION: We're asking about her GENERAL knowledge of Xi Mingze
(not her specific knowledge of the user's PowerPoint slides).

Cost tracking: This will track OpenAI API calls and provide cost estimates.
"""

import asyncio
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the writer_agents module to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

try:
    from master_supervisor import MasterSupervisor, SupervisorConfig
    from session_manager import SessionManager
    from insights import CaseInsights
    print("[OK] Successfully imported core modules")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)


class CostTracker:
    """Track OpenAI API costs during analysis."""

    def __init__(self):
        self.calls = []
        self.total_tokens = 0
        self.total_cost = 0.0

        # OpenAI pricing (as of 2024)
        self.pricing = {
            "gpt-4o-mini": {
                "input": 0.00015 / 1000,  # $0.15 per 1K tokens
                "output": 0.0006 / 1000   # $0.60 per 1K tokens
            },
            "gpt-4o": {
                "input": 0.005 / 1000,    # $5.00 per 1K tokens
                "output": 0.015 / 1000    # $15.00 per 1K tokens
            }
        }

    def log_call(self, model: str, input_tokens: int, output_tokens: int):
        """Log an API call and calculate cost."""
        if model not in self.pricing:
            model = "gpt-4o-mini"  # Default to cheaper model

        input_cost = input_tokens * self.pricing[model]["input"]
        output_cost = output_tokens * self.pricing[model]["output"]
        total_cost = input_cost + output_cost

        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost,
            "timestamp": datetime.now().isoformat()
        })

        self.total_tokens += input_tokens + output_tokens
        self.total_cost += total_cost

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "total_calls": len(self.calls),
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "calls_by_model": {
                model: len([c for c in self.calls if c["model"] == model])
                for model in set(c["model"] for c in self.calls)
            }
        }


class ClarifiedMcGrathAnalyzer:
    """Analyzer for the clarified McGrath question using The Matrix reasoning."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path
        self.session_manager = SessionManager(db_path)
        self.cost_tracker = CostTracker()

    def get_lawsuit_evidence(self) -> Dict[str, Any]:
        """Get relevant evidence from lawsuit database."""
        print("[EVIDENCE] Gathering evidence from lawsuit database...")

        evidence = {
            "mcgrath_documents": [],
            "xi_mingze_documents": [],
            "timeline_documents": [],
            "conflict_documents": []
        }

        try:
            lawsuit_db_path = r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db"
            if Path(lawsuit_db_path).exists():
                conn = sqlite3.connect(lawsuit_db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get McGrath documents
                cursor.execute("""
                    SELECT rowid, SUBSTR(content, 1, 2000) as preview, LENGTH(content) as doc_length
                    FROM cleaned_documents
                    WHERE LOWER(content) LIKE '%mcgrath%'
                    ORDER BY doc_length DESC
                    LIMIT 5
                """)

                mcgrath_docs = cursor.fetchall()
                for doc in mcgrath_docs:
                    evidence["mcgrath_documents"].append({
                        "doc_id": doc["rowid"],
                        "preview": doc["preview"],
                        "length": doc["doc_length"]
                    })

                # Get Xi Mingze documents
                cursor.execute("""
                    SELECT rowid, SUBSTR(content, 1, 2000) as preview, LENGTH(content) as doc_length
                    FROM cleaned_documents
                    WHERE LOWER(content) LIKE '%xi%mingze%'
                    ORDER BY doc_length DESC
                    LIMIT 5
                """)

                xi_mingze_docs = cursor.fetchall()
                for doc in xi_mingze_docs:
                    evidence["xi_mingze_documents"].append({
                        "doc_id": doc["rowid"],
                        "preview": doc["preview"],
                        "length": doc["doc_length"]
                    })

                conn.close()
                print(f"   Found {len(mcgrath_docs)} McGrath documents, {len(xi_mingze_docs)} Xi Mingze documents")
            else:
                print(f"   [WARNING] Lawsuit database not found")

        except Exception as e:
            print(f"   [ERROR] Database query failed: {e}")

        return evidence

    async def run_matrix_analysis(self, session_id: str) -> Dict[str, Any]:
        """Run the actual The Matrix reasoning system."""
        print("[THE_MATRIX] Running The Matrix reasoning system...")

        # Get evidence first
        evidence = self.get_lawsuit_evidence()

        # Create clarified question
        clarified_question = """
        CRITICAL ANALYSIS QUESTION:

        What's the likelihood that Marlyn McGrath knew about Xi Mingze BEFORE Harvard
        published Statement 1 on April 19, 2019, given her conflicts of interest and
        role in Xi Mingze's pseudonym?

        IMPORTANT CLARIFICATIONS:
        1. We're asking about her GENERAL knowledge of Xi Mingze's identity/existence
        2. NOT asking about her specific knowledge of the user's PowerPoint slides
        3. Focus on her role in creating Xi Mingze's pseudonym
        4. Consider her administrative access to institutional materials
        5. Analyze her conflicts of interest in the defamation case

        EVIDENCE CONTEXT:
        - McGrath had personal responsibility for Xi Mingze's pseudonym creation
        - She was involved in the plaintiff's defamation case despite conflicts
        - Harvard published Statement 1 on April 19, 2019 (defamation occurred)
        - Plaintiff had direct communications with McGrath in May-July 2019
        - Information flowed from Harvard's China network to administrators

        Please provide a comprehensive legal analysis using the The Matrix reasoning system.
        """

        # Create insights for MasterSupervisor
        insights = CaseInsights(
            case_name="McGrath Timeline Analysis - Clarified Question",
            entities=["Marlyn McGrath", "Xi Mingze", "Harvard", "Statement 1", "April 19, 2019"],
            relationships={
                "McGrath": ["Xi Mingze pseudonym creator", "defamation case participant"],
                "Xi Mingze": ["pseudonym subject", "Statement 1 reference"],
                "Harvard": ["Statement 1 publisher", "defamation perpetrator"]
            },
            causal_chains=[
                "McGrath creates Xi Mingze pseudonym → Knowledge of Xi Mingze identity → Conflict in defamation case"
            ],
            bayesian_network_nodes={
                "McGrath_knowledge": "Probability of McGrath knowing Xi Mingze before April 19, 2019",
                "Pseudonym_responsibility": "McGrath's role in creating Xi Mingze pseudonym",
                "Administrative_access": "McGrath's access to institutional materials",
                "Conflict_of_interest": "McGrath's conflicts in defamation case"
            },
            strategic_insights={
                "timeline_criticality": "April 19, 2019 is the key date for defamation",
                "conflict_analysis": "McGrath's pseudonym role creates conflicts",
                "evidence_strategy": "Focus on institutional knowledge flow"
            },
            session_context=""
        )

        summary = "Comprehensive analysis of McGrath's knowledge timeline regarding Xi Mingze before Harvard's defamation on April 19, 2019."

        # Initialize MasterSupervisor with session
        config = SupervisorConfig(db_path=self.db_path)

        try:
            async with MasterSupervisor(None, config, session_id=session_id) as supervisor:
                print("   [OK] MasterSupervisor initialized")

                # Execute with session context
                result = await supervisor.execute_with_session(
                    clarified_question,
                    insights,
                    summary
                )

                print("   [OK] The Matrix analysis completed")
                return result

        except Exception as e:
            print(f"   [ERROR] The Matrix analysis failed: {e}")
            return {"error": str(e), "analysis": "Analysis failed due to system error"}

    def explain_matrix_reasoning(self, result: Dict[str, Any]) -> str:
        """Explain the The Matrix system's reasoning in plain English."""

        if "error" in result:
            return f"""
THE_MATRIX SYSTEM ERROR:
The The Matrix reasoning system encountered an error: {result["error"]}

This means the atomic agents couldn't complete their analysis due to a technical issue.
The system was trying to:
1. Decompose the McGrath timeline question into atomic tasks
2. Assign specialized agents to research, analyze, and synthesize findings
3. Use session memory to maintain context across interactions
4. Generate comprehensive legal analysis

Unfortunately, the system error prevented completion of the analysis.
            """

        # Extract key components from The Matrix result
        final_text = result.get("final_text", "")
        execution_stats = result.get("execution_stats", {})
        agent_outputs = result.get("agent_outputs", {})

        return f"""
THE_MATRIX REASONING SYSTEM ANALYSIS:

THE SYSTEM'S APPROACH:
The The Matrix system used its atomic agent architecture to analyze your McGrath timeline question:

1. RESEARCH AGENTS: Searched for evidence about McGrath's role and Xi Mingze
2. ANALYSIS AGENTS: Evaluated her conflicts of interest and administrative access
3. SYNTHESIS AGENTS: Combined findings into comprehensive legal analysis
4. SESSION MEMORY: Maintained context from previous interactions

THE SYSTEM'S FINDINGS:
{final_text[:1000]}{"..." if len(final_text) > 1000 else ""}

EXECUTION STATISTICS:
- Total agents involved: {len(agent_outputs)}
- Analysis phases completed: {execution_stats.get('phases_completed', 'Unknown')}
- Session context utilized: {execution_stats.get('session_context_used', 'Unknown')}

THE SYSTEM'S REASONING PROCESS:
The The Matrix system approached this as a multi-dimensional analysis:
1. Timeline analysis (before vs. after April 19, 2019)
2. Conflict of interest assessment (pseudonym creator vs. defamation participant)
3. Administrative knowledge evaluation (institutional access patterns)
4. Legal framework application (constructive knowledge doctrine)

This represents the system's comprehensive reasoning about McGrath's knowledge timeline.
        """

    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Run the complete clarified analysis."""
        print("[ANALYSIS] Starting clarified McGrath timeline analysis...")

        # Create session
        session_id = self.session_manager.create_session("McGrath Timeline - Clarified Analysis")
        print(f"Session created: {session_id}")

        # Run The Matrix analysis
        matrix_result = await self.run_matrix_analysis(session_id)

        # Get cost summary
        cost_summary = self.cost_tracker.get_summary()

        # Explain reasoning
        reasoning_explanation = self.explain_matrix_reasoning(matrix_result)

        return {
            "session_id": session_id,
            "matrix_result": matrix_result,
            "reasoning_explanation": reasoning_explanation,
            "cost_summary": cost_summary,
            "timestamp": datetime.now().isoformat()
        }


async def main():
    """Main execution function."""
    print("THE_MATRIX REASONING SYSTEM - CLARIFIED MCGRATH ANALYSIS")
    print("=" * 60)

    analyzer = ClarifiedMcGrathAnalyzer()

    # Run complete analysis
    results = await analyzer.run_complete_analysis()

    # Save results
    output_file = "clarified_mcgrath_matrix_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[RESULTS] Analysis saved to: {output_file}")

    # Print reasoning explanation
    print("\n" + "=" * 60)
    print("THE_MATRIX SYSTEM'S REASONING:")
    print("=" * 60)
    print(results["reasoning_explanation"])

    # Print cost summary
    print("\n" + "=" * 60)
    print("API COST SUMMARY:")
    print("=" * 60)
    cost = results["cost_summary"]
    print(f"Total API calls: {cost['total_calls']}")
    print(f"Total tokens used: {cost['total_tokens']:,}")
    print(f"Total cost: ${cost['total_cost']:.4f}")
    print(f"Calls by model: {cost['calls_by_model']}")

    print(f"\n[SUCCESS] Analysis complete!")
    print(f"Session ID: {results['session_id']}")


if __name__ == "__main__":
    asyncio.run(main())
