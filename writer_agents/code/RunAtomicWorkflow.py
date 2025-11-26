#!/usr/bin/env python3
"""
CLI entry point for the Atomic Agent Writer workflow with LangChain support.

This CLI always uses the LangChain-enabled MasterSupervisor pipeline.
Manual SQLite querying lives only as an internal fallback if LangChain
encounters an error.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_ROOT = PROJECT_ROOT / "writer_agents" / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

# Import modules
from bn_bridge import from_json_payload
from master_supervisor import MasterSupervisor, SupervisorConfig
from insights import CaseInsights


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the The Matrix atomic agent pipeline with LangChain support."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="Path to a JSON file containing case insights.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for the LLM client.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the model client.",
    )
    parser.add_argument(
        "--langchain-db-path",
        type=Path,
        default=None,
        help="Path to SQLite database for LangChain queries (default: lawsuit.db).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("jobs.db"),
        help="Path to jobs database for tracking.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum workers per phase.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--premium",
        action="store_true",
        help="Enable premium mode (GPT-4o for completeness/precision agents)",
    )
    return parser


def load_insights(path: Path) -> CaseInsights:
    """Load case insights from JSON file."""
    return from_json_payload(path.read_text(encoding="utf-8"))


async def run_atomic_workflow(
    payload_path: Path,
    model: str,
    temperature: float,
    langchain_db_path: Path | None,
    db_path: Path,
    max_workers: int,
    verbose: bool,
    premium: bool,
) -> None:
    """Run the atomic agent workflow with optional LangChain support."""

    # Load case insights
    insights = load_insights(payload_path)

    # Configure supervisor
    config = SupervisorConfig()
    config.max_workers_per_phase = max_workers
    config.db_path = str(db_path)
    config.enable_langchain = True
    config.premium_mode = premium
    if langchain_db_path:
        config.langchain_db_path = langchain_db_path
    else:
        config.langchain_db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

    if verbose:
        print(f"LangChain enabled with database: {config.langchain_db_path}")

    # Create master supervisor
    supervisor = MasterSupervisor(session=None, config=config)

    try:
        # Run the workflow
        print("Starting atomic agent workflow...")
        print("Using LangChain-enhanced research agents (manual SQL only as fallback)")

        results = await supervisor.run(
            insights=insights,
            summary="Case analysis using atomic agents"
        )

        # Display results
        print("\n=== Workflow Results ===")
        for phase, phase_results in results.items():
            print(f"\n{phase.upper()} PHASE:")
            if isinstance(phase_results, dict):
                for task_id, result in phase_results.items():
                    print(f"  Task {task_id}: {type(result).__name__}")
                    if hasattr(result, 'get') and result.get('success'):
                        print(f"    Success: {result.get('success')}")
                    if hasattr(result, 'get') and result.get('langchain_used'):
                        print(f"    LangChain used: {result.get('langchain_used')}")
            else:
                print(f"  Results: {phase_results}")

        print(f"\nWorkflow completed successfully!")

    except Exception as e:
        print(f"Workflow failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    # Validate inputs
    if not args.payload.exists():
        print(f"Error: Payload file not found: {args.payload}")
        exit(1)

    if args.langchain_db_path and not args.langchain_db_path.exists():
        print(f"Error: LangChain database not found: {args.langchain_db_path}")
        exit(1)

    # Run the workflow
    asyncio.run(
        run_atomic_workflow(
            payload_path=args.payload,
            model=args.model,
            temperature=args.temperature,
            langchain_db_path=args.langchain_db_path,
            db_path=args.db_path,
            max_workers=args.max_workers,
            verbose=args.verbose,
            premium=args.premium,
        )
    )


if __name__ == "__main__":
    main()
