#!/usr/bin/env python3
"""
Query Judge Allison D. Burroughs' Rulings on Motions to Seal, Pseudonym, and 1782 Applications

Uses LangChain SQL toolkit to query the MA federal motions database and analyze patterns.
"""

import argparse
import os
import sys
from pathlib import Path
import sqlite3
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ModuleNotFoundError:
    LANGCHAIN_AVAILABLE = False

DEFAULT_JUDGE = "Allison D. Burroughs"


def resolve_database_path(script_root: Path, user_selection: Optional[str]) -> Path:
    """
    Resolve which SQLite database to use.

    Priority:
      1. Explicit --db path (relative to repo or absolute)
      2. case_law_data/ma_federal_motions.db
      3. case_law_data/MaFederalMotions.db
    """
    candidate_paths: list[Path] = []

    if user_selection:
        user_path = Path(user_selection)
        if not user_path.is_absolute():
            user_path = script_root / user_path
        candidate_paths.append(user_path)

    candidate_paths.extend([
        script_root / "case_law_data" / "ma_federal_motions.db",
        script_root / "case_law_data" / "MaFederalMotions.db",
    ])

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to locate a motions database. Tried:\n"
        + "\n".join(str(p) for p in candidate_paths)
    )

def check_database_schema(db_path: Path):
    """Check what tables and columns exist in the database."""
    print("=" * 80)
    print("DATABASE SCHEMA EXPLORATION")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        print(f"  - {table[0]}")

    # Check if a judge column exists
    cursor.execute("PRAGMA table_info(cases);")
    columns = cursor.fetchall()
    print(f"\nColumns in 'cases' table:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")

    # Sample query to see if judge names are in the data
    cursor.execute("SELECT * FROM cases LIMIT 1;")
    sample = cursor.fetchone()
    if sample:
        col_names = [description[0] for description in cursor.description]
        print(f"\nSample case columns: {col_names}")

    conn.close()

def search_for_judge_basic(db_path: Path, judge_name: str, limit: int = 20):
    """Basic search for judge mentions."""
    print("\n" + "=" * 80)
    print(f"BASIC SEARCH FOR JUDGE {judge_name.upper()}")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Search in cleaned_text for mentions of Burroughs
    search_query = """
        SELECT
            cluster_id,
            case_name,
            court,
            docket_number,
            date_filed,
            SUBSTR(cleaned_text, 1, 500) as preview
        FROM cases
        WHERE LOWER(cleaned_text) LIKE ?
        OR LOWER(case_name) LIKE ?
        LIMIT ?;
    """

    like_pattern = f"%{judge_name.lower()}%"
    cursor.execute(search_query, (like_pattern, like_pattern, limit))
    results = cursor.fetchall()

    print(f"\nFound {len(results)} potential cases mentioning {judge_name}")

    for i, row in enumerate(results, 1):
        print(f"\n--- Case {i} ---")
        print(f"ID: {row[0]}")
        print(f"Name: {row[1]}")
        print(f"Court: {row[2]}")
        print(f"Docket: {row[3]}")
        print(f"Date: {row[4]}")
        print(f"Preview: {row[5][:200]}...")

    conn.close()
    return results

def create_langchain_agent(db_path: Path):
    """Create a LangChain SQL agent for natural language queries."""

    if not LANGCHAIN_AVAILABLE:
        raise RuntimeError("LangChain dependencies are not installed in this environment.")

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable required")

    print("\n" + "=" * 80)
    print("INITIALIZING LANGCHAIN SQL AGENT")
    print("=" * 80)

    uri = f"sqlite:///{db_path.as_posix()}"
    database = SQLDatabase.from_uri(uri)

    # Get table names for context
    tables = database.get_usable_table_names()
    print(f"\nAvailable tables: {tables}")

    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=4000
    )

    # Create toolkit
    toolkit = SQLDatabaseToolkit(db=database, llm=llm)

    # Create agent
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        top_k=20,
        max_iterations=5,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    return agent, database

def query_judge_rulings(agent, questions: list):
    """Query the agent with specific questions about Judge Burroughs."""
    print("\n" + "=" * 80)
    print("LANGCHAIN QUERIES ABOUT JUDGE BURROUGHS")
    print("=" * 80)

    results = {}

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(questions)}")
        print(f"{'='*80}")
        print(f"\nQUESTION: {question}\n")

        try:
            result = agent.invoke({"input": question})
            answer = result.get("output", "")

            print(f"\nANSWER:")
            print(f"{answer}\n")

            results[question] = {
                "answer": answer,
                "success": True,
                "intermediate_steps": result.get("intermediate_steps", [])
            }

        except Exception as e:
            print(f"\nERROR: {e}\n")
            results[question] = {
                "answer": None,
                "success": False,
                "error": str(e)
            }

    return results

def analyze_patterns(db_path: Path, judge_name: str):
    """Basic pattern analysis directly from database."""
    print("\n" + "=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get cases mentioning Burroughs
    cursor.execute("""
        SELECT
            cluster_id,
            case_name,
            cleaned_text
        FROM cases
        WHERE LOWER(cleaned_text) LIKE ?;
    """, (f"%{judge_name.lower()}%",))

    burroughs_cases = cursor.fetchall()
    print(f"\nFound {len(burroughs_cases)} cases mentioning Judge {judge_name}")

    # Analyze keywords for motion outcomes
    outcomes = {"granted": [], "denied": [], "partial": []}

    for case_id, case_name, text in burroughs_cases:
        text_lower = text.lower()

        granted_indicators = [
            "motion is granted",
            "motion granted",
            "granted in part",
            "may proceed under pseudonym",
            "may proceed anonymously",
            "may file under seal",
            "permission to file under seal",
            "good cause to seal",
            "impoundment allowed",
            "protective order granted",
        ]

        denied_indicators = [
            "motion is denied",
            "motion denied",
            "denied without prejudice",
            "denied with prejudice",
            "public interest in access",
            "strong presumption of public access",
            "transparency concerns",
            "fails to meet sealing standard",
        ]

        granted_count = sum(1 for ind in granted_indicators if ind in text_lower)
        denied_count = sum(1 for ind in denied_indicators if ind in text_lower)

        if granted_count > denied_count:
            outcome = "granted"
        elif denied_count > granted_count:
            outcome = "denied"
        else:
            outcome = "partial"

        outcomes[outcome].append({
            "case_id": case_id,
            "case_name": case_name
        })

    print(f"\nOutcome Distribution:")
    for outcome, cases in outcomes.items():
        print(f"  {outcome.capitalize()}: {len(cases)}")

    # Print details
    for outcome in ["granted", "denied", "partial"]:
        if outcomes[outcome]:
            print(f"\n{outcome.upper()} MOTIONS:")
            for case in outcomes[outcome][:5]:  # Show first 5
                print(f"  - {case['case_name']} (ID: {case['case_id']})")

    conn.close()

    return outcomes


def summarize_judge_mentions(db_path: Path, judge_name: str) -> None:
    """Provide quick stats about the judge's footprint inside the DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM cases;")
    total_cases = cursor.fetchone()[0]

    like_pattern = f"%{judge_name.lower()}%"
    cursor.execute(
        "SELECT COUNT(*) FROM cases WHERE LOWER(cleaned_text) LIKE ?;",
        (like_pattern,),
    )
    judge_cases = cursor.fetchone()[0]

    cursor.execute(
        """
        SELECT court, COUNT(*) as tally
        FROM cases
        WHERE LOWER(cleaned_text) LIKE ?
        GROUP BY court
        ORDER BY tally DESC
        LIMIT 5;
        """,
        (like_pattern,),
    )
    top_courts = cursor.fetchall()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM cases
        WHERE LOWER(cleaned_text) LIKE ?
          AND LOWER(cleaned_text) LIKE '%1782%';
        """,
        (like_pattern,),
    )
    section_1782_cases = cursor.fetchone()[0]

    conn.close()

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total cases in DB: {total_cases:,}")
    print(f"Cases mentioning Judge {judge_name}: {judge_cases:,}")
    print(f"Section 1782 + {judge_name}: {section_1782_cases:,}")

    if top_courts:
        print("\nTop courts for those mentions:")
        for court, tally in top_courts:
            print(f"  - {court}: {tally}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Judge-specific sealing / pseudonym rulings stored in the motions DB."
    )
    parser.add_argument(
        "--judge",
        default=DEFAULT_JUDGE,
        help=f"Judge name to search for (default: {DEFAULT_JUDGE})",
    )
    parser.add_argument(
        "--db",
        help="Path to the motions SQLite database (relative paths are resolved from the repo root).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of sample cases to display in the basic search section.",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip the expensive schema dump step.",
    )
    parser.add_argument(
        "--skip-langchain",
        action="store_true",
        help="Disable LangChain SQL agent queries even if OPENAI_API_KEY is set.",
    )
    return parser.parse_args()


def main():
    """Main execution function."""

    args = parse_args()
    judge_name = args.judge.strip() or DEFAULT_JUDGE

    script_root = Path(__file__).resolve().parent
    try:
        db_path = resolve_database_path(script_root, args.db)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    print("=" * 80)
    print(f"JUDGE {judge_name.upper()} RULINGS ANALYSIS")
    print("=" * 80)
    print(f"\nDatabase: {db_path}")
    print("Focus: Motions to seal, pseudonym practice, and Section 1782 references\n")

    if not args.skip_schema:
        check_database_schema(db_path)
    else:
        print("Skipping schema exploration per --skip-schema flag.")

    summarize_judge_mentions(db_path, judge_name)
    basic_results = search_for_judge_basic(db_path, judge_name, limit=args.limit)
    patterns = analyze_patterns(db_path, judge_name)

    should_run_langchain = (
        not args.skip_langchain
        and "OPENAI_API_KEY" in os.environ
        and LANGCHAIN_AVAILABLE
    )

    if should_run_langchain:
        try:
            agent, database = create_langchain_agent(db_path)
            questions = [
                f"Find all cases involving Judge {judge_name} and provide details about any motions to seal or pseudonym filings ruled on.",
                f"For cases involving Judge {judge_name}, what were the outcomes of motions to seal and pseudonym? Were they granted or denied?",
                f"Analyze patterns in Judge {judge_name}'s rulings on motions to seal and pseudonym. What factors led to grants vs denials?",
                f"Find any cases involving Judge {judge_name} and 1782 applications. What were the outcomes?",
                f"What are the most common reasons Judge {judge_name} grants or denies motions to seal in the available dataset?",
            ]
            results = query_judge_rulings(agent, questions)

            print("\n" + "=" * 80)
            print("SUMMARY")
            print("=" * 80)
            successes = sum(1 for r in results.values() if r.get("success"))
            print(f"\nSuccessfully completed {successes}/{len(results)} LangChain queries")

        except Exception as e:
            print(f"\nLangChain query failed: {e}")
            print("Continuing with deterministic analysis...")
    else:
        if not LANGCHAIN_AVAILABLE:
            reason = "LangChain dependencies not installed"
        elif args.skip_langchain:
            reason = "flagged via --skip-langchain"
        else:
            reason = "OPENAI_API_KEY not set"
        print(f"\nSkipping LangChain SQL agent ({reason}).")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
