#!/usr/bin/env python3
"""
Seed LangChain query memories for The Matrix agents.

Reads a configuration file with natural-language questions per agent,
executes each question via LangChainSQLAgent against the lawsuit database,
and stores the successful outcomes in the shared MemoryStore so GPT-4o-mini
agents start with warm context.
"""

from __future__ import annotations

import argparse
import json
import os
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


def load_api_key() -> str:
    """Load OpenAI API key from config file or environment."""
    # Try config file first
    config_file = PROJECT_ROOT / "config" / "openai_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('openai_api_key')
                if api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                    return api_key
        except Exception as e:
            print(f"Warning: Could not load API key from config: {e}")

    # Fall back to environment variable
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit("No OpenAI API key found. Set OPENAI_API_KEY environment variable or create config/openai_config.json")

    return api_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LangChain seed queries and persist results into agent memory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "langchain_seed_queries.json",
        help="Path to the seed query configuration file (default: config/langchain_seed_queries.json).",
    )
    parser.add_argument(
        "--agents", "--target-agents",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of agent names to seed (default: all in config).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of records to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Process at most this many seed records in this run (alias for --limit).",
    )
    parser.add_argument(
        "--max-memories",
        type=int,
        default=2,
        help="Maximum number of langchain_seed memories per agent before skipping (default: 2).",
    )
    parser.add_argument(
        "--force-reseed",
        action="store_true",
        help="Bypass memory-count checks and always add a new seed memory.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace existing langchain_seed memories instead of appending.",
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
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Seed config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shorten(text: str, width: int = 300) -> str:
    """Collapse whitespace and trim long responses."""
    if not text:
        return ""
    single_line = " ".join(text.split())
    return textwrap.shorten(single_line, width=width, placeholder="...")


def iter_records(
    records: Iterable[Dict[str, Any]],
    allowed_agents: Optional[List[str]],
    limit: Optional[int],
) -> Iterable[Dict[str, Any]]:
    count = 0
    for record in records:
        if allowed_agents and record.get("agent") not in allowed_agents:
            continue
        yield record
        count += 1
        if limit is not None and count >= limit:
            break


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
        f"LangChain seed for {agent}{codename_part}: "
        f"question='{shorten(question, 160)}'; {sql_part}; answer={answer_part}"
    )


def run_seed_process(
    config: Dict[str, Any],
    agents_filter: Optional[List[str]],
    limit: Optional[int],
    max_memories: int,
    force_reseed: bool,
    replace_existing: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    database_path = Path(config.get("database_path", "lawsuit.db"))
    memory_dir = Path(config.get("memory_dir", "memory_store"))
    model_name = config.get("model", "gpt-4o-mini")
    records = config.get("records", [])

    if not records:
        raise ValueError("Seed configuration contains no records.")

    if not dry_run and not database_path.exists():
        raise FileNotFoundError(f"Database not found: {database_path}")

    if dry_run:
        print("[dry-run] No LangChain queries will be executed.")
    else:
        print(f"[info] Using database: {database_path}")
        print(f"[info] Memory directory: {memory_dir}")

    langchain_agent: Optional[LangChainSQLAgent] = None
    if not dry_run:
        model_config = ModelConfig(model=model_name, temperature=0.0, max_tokens=4096)
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

    for record in iter_records(records, agents_filter, limit):
        agent = record.get("agent")
        question = record.get("question")
        if not agent or not question:
            print(f"[warn] Skipping malformed record: {record}")
            continue

        codename = record.get("codename")
        context = record.get("context")

        seed_memory_count = 0
        existing_memories = []
        if not dry_run and memory_store:
            existing_memories = memory_store.memories.get(agent, [])
            seed_memory_count = sum(
                1 for m in existing_memories if getattr(m, "source", "") == "langchain_seed"
            )
            if (
                not force_reseed
                and max_memories >= 0
                and seed_memory_count >= max_memories
            ):
                print(
                    f"\n[skip] Agent={agent} Codename={codename} "
                    f"Already has {seed_memory_count} langchain_seed memories "
                    f"(threshold={max_memories})"
                )
                continue

        print(f"\n[seed] Agent={agent} Codename={codename} Question={shorten(question, 120)}")

        if dry_run:
            print(f"        context={context}")
            continue

        assert langchain_agent is not None  # for type checkers
        assert memory_store is not None

        result = langchain_agent.query_evidence(question=question, context=context)

        if not result.get("success"):
            failures += 1
            print(f"  [fail] LangChain error: {result.get('error')}")
            continue

        answer = result.get("answer", "")
        executed_sql = result.get("executed_sql")
        sql_history = result.get("sql_history")
        meta_context = result.get("meta_context")
        cost_estimate = result.get("cost_estimate")

        summary = build_memory_summary(agent, codename, question, executed_sql, answer)
        print(f"  [ok] Stored memory summary: {shorten(summary, 140)}")
        if cost_estimate is not None:
            print(f"       estimated cost: ${cost_estimate:.6f}")
        if executed_sql:
            print(f"       executed sql: {shorten(executed_sql, 140)}")

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
                "sql_history": sql_history,
                "meta_context": meta_context,
                "cost_estimate": cost_estimate,
                "seed_timestamp": datetime.utcnow().isoformat() + "Z",
                "source": "langchain_seed_script",
            },
            embedding=None,
            source="langchain_seed",
            timestamp=datetime.utcnow(),
        )

        if replace_existing:
            filtered_memories = [
                m for m in existing_memories if getattr(m, "source", "") != "langchain_seed"
            ]
            if len(filtered_memories) != len(existing_memories):
                memory_store.memories[agent] = filtered_memories

        memory_store.add(memory)
        successes += 1

    if dry_run:
        print("\n[dry-run] Preview complete. No data persisted.")
        return

    assert memory_store is not None
    memory_store.save()
    per_agent_counts = {
        agent: len(memories)
        for agent, memories in sorted(memory_store.memories.items())
    }
    print("\n[stats] Memory counts per agent after seeding:")
    for agent, count in per_agent_counts.items():
        print(f"  - {agent}: {count}")
    print(
        f"\n[done] Seeding complete. Memories saved to {memory_dir}. "
        f"Successes={successes} Failures={failures}"
    )


def main() -> None:
    args = parse_args()

    # Load API key from config file
    try:
        api_key = load_api_key()
        print(f"[info] API key loaded successfully")
    except SystemExit as e:
        print(f"[error] {e}")
        return

    config = load_config(args.config)
    try:
        run_seed_process(
            config=config,
            agents_filter=args.agents,
            limit=args.batch_size or args.limit,
            max_memories=max(0, args.max_memories if args.max_memories is not None else 2),
            force_reseed=args.force_reseed,
            replace_existing=args.replace_existing,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as exc:
        raise SystemExit(f"Seeding failed: {exc}") from exc


if __name__ == "__main__":
    main()
