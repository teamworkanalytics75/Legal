#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


PLAN_PATH = Path("plans/MASTER_PLAN.json")
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TaskEval:
    task_id: str
    title: str
    current_status: str
    inferred: str  # done | maybe_done | pending | unknown
    reason: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def file_exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def file_contains(path: str, needles: List[str]) -> bool:
    p = REPO_ROOT / path
    if not p.exists():
        return False
    text = read_text(p)
    return all(n in text for n in needles)


def any_file_contains(paths: List[str], needles: List[str]) -> bool:
    for p in paths:
        if file_contains(p, needles):
            return True
    return False


def infer_status(task: Dict) -> TaskEval:
    tid = task.get("id", "")
    title = task.get("title", "")
    status = task.get("status", "pending")

    # Explicitly done/dropped
    if status in ("done", "dropped"):
        return TaskEval(tid, title, status, "done", "status explicitly marked done/dropped")

    # Heuristics per task id
    if tid == "SEARCH-001":
        # Adopt semantic/hybrid CourtListener search by default
        searcher = "case_law_data/scripts/citation_searchers/courtlistener_searcher.py"
        if file_contains(searcher, ["use_semantic=True"]) or file_contains(searcher, ["DEFAULT_USE_SEMANTIC", "= True"]):
            return TaskEval(tid, title, status, "done", "semantic default detected in courtlistener_searcher.py")
        return TaskEval(tid, title, status, "pending", "semantic default not detected")

    if tid == "MODEL-001":
        # Retrain unified outline model with TF‑IDF + Legal‑BERT
        has_model = file_exists("case_law_data/models/catboost_outline_unified.cbm")
        if has_model:
            return TaskEval(tid, title, status, "maybe_done", "model artifact present: catboost_outline_unified.cbm")
        return TaskEval(tid, title, status, "pending", "model artifact not found")

    if tid == "MODEL-002":
        # Calibration in codebase (search whitelisted code dirs)
        code_dirs = [
            REPO_ROOT / "Agents_1782_ML_Dataset",
            REPO_ROOT / "writer_agents",
            REPO_ROOT / "case_law_data",
        ]
        py_files = [str(p.relative_to(REPO_ROOT)) for d in code_dirs if d.exists() for p in d.rglob("*.py")]
        for rel in py_files:
            if file_contains(rel, ["CalibratedClassifierCV"]):
                return TaskEval(tid, title, status, "maybe_done", f"found calibration in {rel}")
        return TaskEval(tid, title, status, "pending", "no calibration usage detected")

    if tid == "MODEL-003":
        # Optuna usage in codebase (search whitelisted code dirs)
        code_dirs = [
            REPO_ROOT / "Agents_1782_ML_Dataset",
            REPO_ROOT / "writer_agents",
            REPO_ROOT / "case_law_data",
        ]
        py_files = [str(p.relative_to(REPO_ROOT)) for d in code_dirs if d.exists() for p in d.rglob("*.py")]
        for rel in py_files:
            if file_contains(rel, ["import optuna"]) or file_contains(rel, ["from optuna"]):
                return TaskEval(tid, title, status, "maybe_done", f"found optuna import in {rel}")
        return TaskEval(tid, title, status, "pending", "no optuna usage detected")

    if tid == "WORKFLOW-001":
        # Pre‑flight checker presence
        targets = [
            "writer_agents/scripts/generate_optimized_motion.py",
            "writer_agents/code/WorkflowStrategyExecutor.py",
            "writer_agents/code/Conductor.py",
        ]
        if any_file_contains(targets, ["preflight"]) or any_file_contains(targets, ["pre_flight"]):
            return TaskEval(tid, title, status, "maybe_done", "preflight check reference found")
        return TaskEval(tid, title, status, "pending", "no preflight check reference found")

    if tid == "MLOPS-001":
        if file_exists("reports/analysis_outputs/model_artifacts_inventory.csv") and file_exists("scripts/refresh_model_inventory.py"):
            return TaskEval(tid, title, status, "done", "inventory + refresh script present")
        return TaskEval(tid, title, status, "pending", "inventory or refresh script missing")

    if tid == "DOCS-001":
        if file_exists("docs/for_agents/ML_Lawsuit_Analysis_Docs_Overview.md") and file_exists("plans/MASTER_PLAN.md"):
            return TaskEval(tid, title, status, "maybe_done", "docs index + master plan present")
        return TaskEval(tid, title, status, "pending", "docs index or master plan missing")

    if tid == "DATA-001":
        return TaskEval(tid, title, status, "unknown", "manual review needed (labels via opinions)")

    # Default
    return TaskEval(tid, title, status, "pending", "no heuristic available")


def backup_plan(path: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f"_backup_{ts}" + path.suffix)
    shutil.copy2(path, backup)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audit_master_plan",
        description="Check MASTER_PLAN.json against repo state and optionally remove completed tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Do This Next:\n  audit_master_plan --dry-run\n  audit_master_plan --apply",
    )
    parser.add_argument("--apply", action="store_true", help="Persist removal of done tasks")
    parser.add_argument("--delete-maybe", action="store_true", help="Also remove tasks inferred as maybe_done")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying the plan")
    args = parser.parse_args()

    if not PLAN_PATH.exists():
        print(f"Plan not found: {PLAN_PATH}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    evals: List[TaskEval] = [infer_status(t) for t in tasks]

    # Report
    print("Task Status Summary:")
    for ev in evals:
        print(f"- {ev.task_id}: {ev.title} — status={ev.current_status} inferred={ev.inferred} ({ev.reason})")

    # Decide removals
    to_keep = []
    to_remove_ids = []
    for t, ev in zip(tasks, evals):
        remove = ev.inferred == "done" or (args.delete_maybe and ev.inferred == "maybe_done")
        if remove:
            to_remove_ids.append(ev.task_id)
        else:
            to_keep.append(t)

    print("")
    if to_remove_ids:
        print("Will remove tasks:")
        for tid in to_remove_ids:
            print(f"  - {tid}")
    else:
        print("No tasks to remove based on current heuristics.")

    if args.apply and to_remove_ids and not args.dry_run:
        backup = backup_plan(PLAN_PATH)
        data["tasks"] = to_keep
        PLAN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Updated plan written. Backup saved to: {backup}")
    else:
        print("Dry run or nothing to apply; plan not modified.")


if __name__ == "__main__":
    main()
