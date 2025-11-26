#!/usr/bin/env python3
"""Build or refresh the Cursor/Codex chat embedding index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.cursor_chats import (
    CursorChatLibrary,
    DEFAULT_MODEL_NAME,
    CHAT_LOG_PATH,
)


def load_chat_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        raise FileNotFoundError(f"Chat log not found at {log_path}")
    with log_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("chats", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index captured Cursor/Codex chats.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformer model to use (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of chats when rebuilding the index.",
    )
    parser.add_argument(
        "--agent-type",
        action="append",
        dest="agent_types",
        help="Filter chats by agent type (e.g., cursor, codex). May repeat.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append chats from the log instead of rebuilding the index.",
    )
    parser.add_argument(
        "--log-path",
        default=str(CHAT_LOG_PATH),
        help="Path to the chat log JSON (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    library = CursorChatLibrary(model_name=args.model, auto_load=args.append)

    if args.append:
        try:
            chats = load_chat_log(log_path)
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Failed to parse chat log: {exc}")
            return

        if args.agent_types:
            chats = [chat for chat in chats if chat.get("agent_type") in args.agent_types]
        appended = library.append_chats(chats)
        print(f"Appended {appended} chats to the embedding index.")
    else:
        count = library.rebuild_from_log(
            limit=args.limit,
            agent_filter=args.agent_types,
        )
        print(f"Rebuilt chat embedding index with {count} chats.")


if __name__ == "__main__":
    main()
