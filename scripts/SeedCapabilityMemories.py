#!/usr/bin/env python3
"""
Seed LangChain capability-based memories for The Matrix agents.

Reads capability-based seed queries (universal skills + job primitives + domain anchors)
and executes them via LangChainSQLAgent to build agent memories.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Ensure repository modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = PROJECT_ROOT / "writer_agents" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

try:
    from agents import ModelConfig
    from langchain_integration import LangChainSQLAgent
    from memory_system import AgentMemory, MemoryStore
except ImportError as exc:  # pragma: no cover - script environment check
    raise SystemExit(f"Failed to import The Matrix modules: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LangChain capability-based seed queries and persist results into agent memory."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "config" / "langchain_capability_seeds.json",
        help="Path to the capability seed queries file (default: config/langchain_capability_seeds.json).",
    )
    parser.add_argument(
        "--section",
        type=str,
        choices=["universal_skills", "agents", "all"],
        default="all",
        help="Which section to seed: universal_skills, agents, or all (default: all).",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of agent names to seed (default: all in config).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of queries to process.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing LangChain or writing memory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output from LangChainSQLAgent.",
    )
    parser.add_argument(
        "--handle-parsing-errors",
        action="store_true",
        default=True,
        help="Enable parsing error handling in LangChain agent.",
    )
    return parser.parse_args()


def load_capability_seeds(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Capability seeds file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shorten(text: str, width: int = 300) -> str:
    """Collapse whitespace and trim long responses."""
    if not text:
        return ""
    single_line = " ".join(text.split())
    return textwrap.shorten(single_line, width=width, placeholder="...")


def iter_universal_skills(
    universal_skills: List[Dict[str, Any]],
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """Iterate over universal skills queries."""
    count = 0
    for skill in universal_skills:
        yield {
            "agent": "UniversalAgent",
            "question": skill["nl"],
            "codename": skill["id"],
            "context": f"Universal skill: {skill['category']}",
            "priority": skill.get("priority", 1)
        }
        count += 1
        if limit is not None and count >= limit:
            break


def iter_agent_queries(
    agents: Dict[str, Any],
    allowed_agents: Optional[List[str]],
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    """Iterate over agent-specific queries (primitives + anchors)."""
    count = 0
    for agent_name, agent_data in agents.items():
        if allowed_agents and agent_name not in allowed_agents:
            continue

        agent_info = agent_data.get("agent_info", {})

        # Job primitives
        primitives = agent_data.get("job_primitives", [])
        for primitive in primitives:
            yield {
                "agent": agent_name,
                "question": primitive["nl"],
                "codename": primitive["name"],
                "context": f"Job primitive: {primitive.get('category', 'generic')}",
                "priority": 1
            }
            count += 1
            if limit is not None and count >= limit:
                return

        # Domain anchors
        anchors = agent_data.get("domain_anchors", [])
        for anchor in anchors:
            yield {
                "agent": agent_name,
                "question": anchor["nl"],
                "codename": f"anchor_{len(anchors)}",
                "context": f"Domain anchor (limit: {anchor.get('limit', 'none')})",
                "priority": 2
            }
            count += 1
            if limit is not None and count >= limit:
                return


def build_memory_summary(
    agent: str,
    codename: Optional[str],
    question: str,
    executed_sql: Optional[str],
    answer: str,
) -> str:
    codename_part = f" ({codename})" if codename else ""
    sql_part = f"SQL: {shorten(executed_sql, 160)}" if executed_sql else "SQL: n/a"
    answer_part = shorten(answer, 220) if answer else "No answer returned."
    return (
        f"LangChain capability seed for {agent}{codename_part}: "
        f"question='{shorten(question, 160)}'; {sql_part}; answer={answer_part}"
    )


def run_capability_seeding(
    seeds: Dict[str, Any],
    section: str,
    agents_filter: Optional[List[str]],
    limit: Optional[int],
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run capability-based seeding process."""

    # Database and memory paths (hardcoded for now)
    database_path = Path("C:/Users/Owner/Desktop/LawsuitSQL/lawsuit.db")
    memory_dir = Path("memory_store")
    model_name = "gpt-4o-mini"

    if not dry_run and not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    if dry_run:
        print("[dry-run] No LangChain queries will be executed.")
    else:
        print(f"[info] Using database: {database_path}")
        print(f"[info] Memory directory: {memory_dir}")

    langchain_agent: Optional[LangChainSQLAgent] = None
    if not dry_run:
        model_config = ModelConfig(model=model_name, temperature=0.0, max_tokens=2000)
        langchain_agent = LangChainSQLAgent(
            db_path=database_path,
            model_config=model_config,
            verbose=verbose,
            memory_path=memory_dir / "langchain_meta_memory.sqlite",
        )

    memory_store: Optional[MemoryStore] = None
    if not dry_run:
        memory_store = MemoryStore(storage_path=memory_dir)

    successes = 0
    failures = 0

    # Process universal skills
    if section in ["universal_skills", "all"]:
        universal_skills = seeds.get("universal_skills", [])
        print(f"\n[seeding] Universal skills ({len(universal_skills)} queries)")

        for record in iter_universal_skills(universal_skills, limit):
            agent = record.get("agent")
            question = record.get("question")
            codename = record.get("codename")
            context = record.get("context")

            print(f"\n[seed] Agent={agent} Codename={codename} Question={shorten(question, 120)}")

            if dry_run:
                print(f"        context={context}")
                continue

            assert langchain_agent is not None
            assert memory_store is not None

            result = langchain_agent.query_evidence(question=question, context=context)

            if not result.get("success"):
                failures += 1
                print(f"  [fail] LangChain error: {result.get('error')}")
                continue

            answer = result.get("answer", "")
            executed_sql = result.get("executed_sql")
            cost_estimate = result.get("cost_estimate")

            summary = build_memory_summary(agent, codename, question, executed_sql, answer)
            print(f"  [ok] Stored memory summary: {shorten(summary, 140)}")
            if cost_estimate is not None:
                print(f"       estimated cost: ${cost_estimate:.6f}")

            memory = AgentMemory(
                agent_type=agent,
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context={
                    "codename": codename,
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "executed_sql": executed_sql,
                    "cost_estimate": cost_estimate,
                    "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "capability_seed_script",
                },
                embedding=None,
                source="capability_seed",
                timestamp=datetime.utcnow(),
            )

            memory_store.add(memory)
            successes += 1

    # Process agent-specific queries
    if section in ["agents", "all"]:
        agents = seeds.get("agents", {})
        print(f"\n[seeding] Agent-specific queries ({len(agents)} agents)")

        for record in iter_agent_queries(agents, agents_filter, limit):
            agent = record.get("agent")
            question = record.get("question")
            codename = record.get("codename")
            context = record.get("context")

            print(f"\n[seed] Agent={agent} Codename={codename} Question={shorten(question, 120)}")

            if dry_run:
                print(f"        context={context}")
                continue

            assert langchain_agent is not None
            assert memory_store is not None

            result = langchain_agent.query_evidence(question=question, context=context)

            if not result.get("success"):
                failures += 1
                print(f"  [fail] LangChain error: {result.get('error')}")
                continue

            answer = result.get("answer", "")
            executed_sql = result.get("executed_sql")
            cost_estimate = result.get("cost_estimate")

            summary = build_memory_summary(agent, codename, question, executed_sql, answer)
            print(f"  [ok] Stored memory summary: {shorten(summary, 140)}")
            if cost_estimate is not None:
                print(f"       estimated cost: ${cost_estimate:.6f}")

            memory = AgentMemory(
                agent_type=agent,
                memory_id=str(uuid.uuid4()),
                summary=summary,
                context={
                    "codename": codename,
                    "question": question,
                    "context": context,
                    "answer": answer,
                    "executed_sql": executed_sql,
                    "cost_estimate": cost_estimate,
                    "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                    "source": "capability_seed_script",
                },
                embedding=None,
                source="capability_seed",
                timestamp=datetime.utcnow(),
            )

            memory_store.add(memory)
            successes += 1

    if dry_run:
        print("\n[dry-run] Preview complete. No data persisted.")
        return

    assert memory_store is not None
    memory_store.save()
    print(
        f"\n[done] Capability seeding complete. Memories saved to {memory_dir}. "
        f"Successes={successes} Failures={failures}"
    )


def main() -> None:
    args = parse_args()
    seeds = load_capability_seeds(args.queries)
    try:
        run_capability_seeding(
            seeds=seeds,
            section=args.section,
            agents_filter=args.agents,
            limit=args.limit,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as exc:
        raise SystemExit(f"Capability seeding failed: {exc}") from exc


if __name__ == "__main__":
    main()
