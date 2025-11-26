#!/usr/bin/env python3
"""
Generalized Agent Memory Population Script
==========================================

Loads a The Matrix analysis JSON (any question/run), derives per-agent learnings
from the recorded phase outputs, and stores them in the episodic memory store
so agents can leverage those insights on future tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Ensure we can import the The Matrix modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "writer_agents" / "code"))

from memory_system import AgentMemory, MemoryStore  # type: ignore


################################################################################
# Helper utilities
################################################################################

TEXT_FIELDS = (
    "section_text",
    "output_text",
    "corrected_text",
    "text",
    "summary",
)

LIST_FIELDS_TO_COUNT = {
    "facts": "facts",
    "violations": "violations",
    "issues": "issues",
    "sections": "sections",
    "located_citations": "located citations",
    "verified_citations": "verified citations",
}

IGNORE_KEYS = {
    "agent_name",
    "duty",
    "cached",
    "phase",
    "task_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate The Matrix agent memories from an analysis JSON artifact."
    )
    parser.add_argument(
        "--analysis-file",
        type=str,
        help="Path to analysis JSON. If omitted, the most recent *_analysis_*.json is used.",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default="memory_store",
        help="Directory for the vector memory store (default: memory_store).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated memories without saving them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of memories to save (useful for testing).",
    )
    return parser.parse_args()


def find_latest_analysis_file() -> Optional[Path]:
    candidates = sorted(
        PROJECT_ROOT.glob("*_analysis_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_analysis(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def shorten(text: str, width: int = 200) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder="…")


def summarize_agent_output(result: Dict[str, Any]) -> str:
    """Create a human-readable snippet summarizing an agent's contribution."""
    # Prefer textual fields
    for field in TEXT_FIELDS:
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            return f"{field}: {shorten(value, 220)}"

    # Then look for known lists
    for field, label in LIST_FIELDS_TO_COUNT.items():
        value = result.get(field)
        if isinstance(value, list) and value:
            prefix = f"{label}: {len(value)} item"
            prefix += "s" if len(value) != 1 else ""
            details = ""
            first_entry = value[0]
            if isinstance(first_entry, dict):
                detail_parts = [
                    f"{k}={shorten(str(v), 60)}"
                    for k, v in list(first_entry.items())[:3]
                ]
                if detail_parts:
                    details = f" (e.g. {', '.join(detail_parts)})"
            elif isinstance(first_entry, str):
                details = f" (e.g. {shorten(first_entry, 60)})"
            return prefix + details

    # Fallback: summarize key/value pairs
    kv_pairs: List[str] = []
    for key, value in result.items():
        if key in IGNORE_KEYS:
            continue
        if isinstance(value, (str, int, float)) and str(value).strip():
            kv_pairs.append(f"{key}={shorten(str(value), 80)}")
        elif isinstance(value, list) and value:
            kv_pairs.append(f"{key} (len={len(value)})")
        elif isinstance(value, dict) and value:
            kv_pairs.append(f"{key} (keys={list(value.keys())[:3]})")
        if len(kv_pairs) >= 3:
            break
    return ", ".join(kv_pairs) if kv_pairs else "Produced internal output (details omitted)."


def extract_agent_contributions(
    analysis_data: Dict[str, Any]
) -> Dict[str, List[Dict[str, str]]]:
    """Collect per-agent contribution summaries from phase results."""
    contributions: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    phase_results = analysis_data.get("phase_results", {})

    if not isinstance(phase_results, dict):
        return contributions

    for phase, tasks in phase_results.items():
        if not isinstance(tasks, dict):
            continue
        for task_id, result in tasks.items():
            if not isinstance(result, dict):
                continue
            agent_type = result.get("agent_name")
            if not agent_type:
                agent_type = task_id.split("_")[0]
            summary = summarize_agent_output(result)
            contributions[agent_type].append(
                {
                    "phase": phase,
                    "task_id": task_id,
                    "summary": summary,
                }
            )
    return contributions


def build_memory_summary(
    agent_type: str,
    entries: Iterable[Dict[str, str]],
    question: str,
    probability: Optional[Dict[str, Any]],
    analysis_type: Optional[str],
) -> str:
    lines = []
    lines.append(f"Question: {question.strip()}")
    if analysis_type:
        lines.append(f"Analysis type: {analysis_type}")
    if probability:
        point = probability.get("point_estimate_percent")
        lower = probability.get("lower_bound_percent")
        upper = probability.get("upper_bound_percent")
        lines.append(
            f"Run probability assessment: {point}% (range {lower}%–{upper}%)"
        )
    lines.append(f"Agent: {agent_type}")
    lines.append("Key contributions:")

    any_entries = False
    for entry in entries:
        any_entries = True
        phase = entry.get("phase", "unknown")
        task_id = entry.get("task_id", "")
        summary = entry.get("summary", "")
        lines.append(f"- [{phase}] {task_id}: {summary}")

    if not any_entries:
        lines.append("- (No explicit output recorded; agent likely idle.)")

    return "\n".join(lines)


def build_agent_memories(
    analysis_data: Dict[str, Any],
    analysis_file: Path,
) -> List[AgentMemory]:
    question = analysis_data.get("question") or analysis_data.get("analysis_context") or "Unknown question"
    analysis_type = analysis_data.get("analysis_type")
    probability = analysis_data.get("probability_assessment")
    timestamp_str = analysis_data.get("timestamp")
    timestamp = None
    if isinstance(timestamp_str, str):
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except ValueError:
            pass
    if timestamp is None:
        timestamp = datetime.now()

    contributions = extract_agent_contributions(analysis_data)
    memories: List[AgentMemory] = []

    for agent_type, entries in contributions.items():
        summary = build_memory_summary(
            agent_type,
            entries,
            question=question,
            probability=probability,
            analysis_type=analysis_type,
        )
        memory = AgentMemory(
            agent_type=agent_type,
            memory_id=str(uuid.uuid4()),
            summary=summary,
            context={
                "question": question,
                "analysis_file": str(analysis_file),
                "analysis_type": analysis_type,
                "probability_assessment": probability,
                "entries": entries,
            },
            source="analysis_artifact",
            timestamp=timestamp,
        )
        memories.append(memory)

    # Add an aggregate memory for MasterSupervisor if we have a compiled final text
    final_text = analysis_data.get("final_text")
    compiled_sections = analysis_data.get("compiled_sections")
    if final_text or compiled_sections:
        aggregate_summary_parts = [
            f"Question: {question.strip()}",
            f"Analysis type: {analysis_type or 'unknown'}",
        ]
        if probability:
            aggregate_summary_parts.append(
                f"Probability estimate: {probability.get('point_estimate_percent')}% "
                f"(range {probability.get('lower_bound_percent')}%–{probability.get('upper_bound_percent')}%)"
            )
        if final_text:
            aggregate_summary_parts.append(
                f"Final narrative: {shorten(final_text, 260)}"
            )
        elif compiled_sections:
            aggregate_summary_parts.append(
                f"Compiled sections: {len(compiled_sections)} sections recorded."
            )
        master_memory = AgentMemory(
            agent_type="MasterSupervisor",
            memory_id=str(uuid.uuid4()),
            summary="\n".join(aggregate_summary_parts),
            context={
                "question": question,
                "analysis_file": str(analysis_file),
                "probability_assessment": probability,
                "final_text": final_text,
                "compiled_sections": compiled_sections,
            },
            source="analysis_artifact",
            timestamp=timestamp,
        )
        memories.append(master_memory)

    return memories


################################################################################
# Main routine
################################################################################


def main() -> Optional[Dict[str, Any]]:
    args = parse_args()

    if args.analysis_file:
        analysis_path = Path(args.analysis_file)
        if not analysis_path.exists():
            print(f"[ERROR] Analysis file not found: {analysis_path}")
            return None
    else:
        latest = find_latest_analysis_file()
        if not latest:
            print("[ERROR] Could not find any *_analysis_*.json artifacts.")
            return None
        analysis_path = latest

    print("AGENT MEMORY POPULATION SCRIPT")
    print("=" * 40)
    print(f"[INFO] Using analysis file: {analysis_path}")

    try:
        analysis_data = load_analysis(analysis_path)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Failed to parse analysis JSON: {exc}")
        return None

    memories = build_agent_memories(analysis_data, analysis_path)
    if args.limit is not None:
        memories = memories[: args.limit]

    if not memories:
        print("[WARN] No agent outputs found to turn into memories.")
        return None

    print(f"[INFO] Prepared {len(memories)} memories "
          f"for {len({m.agent_type for m in memories})} agent types.")

    if args.dry_run:
        print("[DRY RUN] Previewing first few memories:")
        for mem in memories[:5]:
            print("-" * 60)
            print(f"Agent: {mem.agent_type}")
            print(mem.summary)
        return {
            "memories_prepared": len(memories),
            "saved": 0,
            "dry_run": True,
        }

    memory_store = MemoryStore(
        storage_path=Path(args.memory_dir),
        use_local_embeddings=True,
        embedding_model="all-MiniLM-L6-v2",
    )

    saved = 0
    for mem in memories:
        memory_store.add(mem)
        saved += 1
    memory_store.save()

    print(f"[SUCCESS] Saved {saved} memories to {args.memory_dir}")

    report = {
        "analysis_file": str(analysis_path),
        "memories_saved": saved,
        "agent_types": sorted({m.agent_type for m in memories}),
        "timestamp": datetime.now().isoformat(),
    }

    report_path = (
        analysis_path.parent
        / f"memory_population_report_{analysis_path.stem}.json"
    )
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Report written to: {report_path}")
    return report


if __name__ == "__main__":
    main()
