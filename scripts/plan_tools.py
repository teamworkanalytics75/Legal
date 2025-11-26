#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

PLAN_JSON = Path("plans/MASTER_PLAN.json")


def load_plan(path: Path = PLAN_JSON) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "tasks" not in data:
        data["tasks"] = []
    return data


def save_plan(data: Dict[str, Any], path: Path = PLAN_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_list(_: argparse.Namespace) -> None:
    plan = load_plan()
    tasks = plan.get("tasks", [])
    for t in tasks:
        print(f"{t.get('id','')}: {t.get('title','')} — {t.get('status','')} — {t.get('priority','')} — area: {t.get('area','')}")


def cmd_update(args: argparse.Namespace) -> None:
    plan = load_plan()
    for t in plan.get("tasks", []):
        if t.get("id") == args.id:
            if args.status:
                t["status"] = args.status
            if args.priority:
                t["priority"] = args.priority
            if args.notes:
                t["notes"] = args.notes
            save_plan(plan)
            print(f"Updated {args.id}")
            return
    raise SystemExit(f"Task id not found: {args.id}")


def cmd_add(args: argparse.Namespace) -> None:
    plan = load_plan()
    new_task = {
        "id": args.id,
        "title": args.title,
        "status": args.status or "pending",
        "priority": args.priority or "P2",
        "area": args.area or "docs",
        "tags": args.tags or [],
        "files": args.files or [],
        "commands": args.commands or [],
        "notes": args.notes or ""
    }
    plan.setdefault("tasks", []).append(new_task)
    save_plan(plan)
    print(f"Added {args.id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="plan_tools",
        description="Manage the MASTER_PLAN.json (list, add, update)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Do This Next:\n" """plan_tools list""",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s_list = sub.add_parser("list", help="List tasks")
    s_list.set_defaults(func=cmd_list)

    s_update = sub.add_parser("update", help="Update a task's status/priority/notes")
    s_update.add_argument("id", help="Task id (e.g., MODEL-001)")
    s_update.add_argument("--status", choices=["pending", "in_progress", "blocked", "done", "dropped"])
    s_update.add_argument("--priority", choices=["P0", "P1", "P2"])
    s_update.add_argument("--notes")
    s_update.set_defaults(func=cmd_update)

    s_add = sub.add_parser("add", help="Add a new task")
    s_add.add_argument("id", help="New task id")
    s_add.add_argument("title", help="Short title")
    s_add.add_argument("--status", default="pending")
    s_add.add_argument("--priority", default="P2")
    s_add.add_argument("--area", default="docs")
    s_add.add_argument("--tags", nargs="*")
    s_add.add_argument("--files", nargs="*")
    s_add.add_argument("--commands", nargs="*")
    s_add.add_argument("--notes")
    s_add.set_defaults(func=cmd_add)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

