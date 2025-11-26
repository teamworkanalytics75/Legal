#!/usr/bin/env python3
"""
Cursor Auto-Capture System
==========================

Automatically captures and stores Cursor chat conversations for learning.

Features:
- Monitors Cursor database for new chats
- Automatically imports to EpisodicMemoryBank
- Runs as background service or scheduled task
- Tracks conversation patterns for system improvement

Usage:
    python cursor_auto_capture.py --monitor          # Run continuously
    python cursor_auto_capture.py --sync             # One-time sync

    # To setup scheduled tasks, use:
    python scripts/schedule_cursor_capture.py --install
"""

import sqlite3
import json
import os
import sys
import argparse
import uuid
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import hashlib

# Import memory bank
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))
try:
    from EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryEntry
    MEMORY_BANK_AVAILABLE = True
except ImportError:
    MEMORY_BANK_AVAILABLE = False
    EpisodicMemoryBank = None
    EpisodicMemoryEntry = None


# Configuration
CURSOR_POSSIBLE_PATHS = [
    os.path.expanduser("~/AppData/Roaming/Cursor"),
    os.path.expanduser("~/AppData/Local/Cursor"),
    "C:/Users/User/AppData/Roaming/Cursor",
    "C:/Users/User/AppData/Local/Cursor",
    "C:/Users/Owner/AppData/Roaming/Cursor",
    "C:/Users/Owner/AppData/Local/Cursor"
]

MEMORY_STORE_PATH = Path("memory_store")
CAPTURE_LOG_PATH = Path("data/cursor_capture_log.json")
STATE_FILE = Path("data/cursor_capture_state.json")


@dataclass
class ChatMessage:
    """Individual chat message"""
    message_id: str
    timestamp: datetime
    role: str  # user or assistant
    content: str
    conversation_id: str


@dataclass
class CapturedConversation:
    """Captured conversation from Cursor"""
    conversation_id: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    messages: List[ChatMessage]
    project_context: str
    message_count: int
    hash: str  # For duplicate detection


class CursorAutoCapture:
    """Automatic Cursor chat capture system"""

    def __init__(
        self,
        cursor_path: Optional[str] = None,
        memory_store_path: Path = MEMORY_STORE_PATH
    ):
        self.cursor_path = cursor_path or self._find_cursor_data()
        self.memory_store_path = memory_store_path
        self.memory_bank = None
        self.processed_conversations: Set[str] = self._load_state()

        if MEMORY_BANK_AVAILABLE:
            self.memory_bank = EpisodicMemoryBank(storage_path=memory_store_path)
            print("[OK] Connected to EpisodicMemoryBank")
        else:
            print("[WARNING] EpisodicMemoryBank not available, chats will only be saved to JSON")

    def _find_cursor_data(self) -> Optional[str]:
        """Find Cursor's data directory"""
        for path in CURSOR_POSSIBLE_PATHS:
            if os.path.exists(path):
                print(f"[OK] Found Cursor data at: {path}")
                return path

        print("[ERROR] Could not find Cursor data directory")
        return None

    def _load_state(self) -> Set[str]:
        """Load processed conversation hashes"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_conversations', []))
            except Exception as e:
                print(f"Warning: Could not load state: {e}")
        return set()

    def _save_state(self):
        """Save processed conversation hashes"""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'processed_conversations': list(self.processed_conversations),
                'last_update': datetime.now().isoformat()
            }, f, indent=2)

    def _find_chat_databases(self) -> List[Path]:
        """Find all SQLite databases in Cursor directory"""
        if not self.cursor_path:
            return []

        db_files = []
        cursor_base = Path(self.cursor_path)

        try:
            # Look specifically in workspaceStorage for state.vscdb files
            workspace_storage = cursor_base / "User" / "workspaceStorage"
            if workspace_storage.exists():
                vscdb_files = list(workspace_storage.glob("*/state.vscdb"))
                print(f"[INFO] Found {len(vscdb_files)} workspace state databases")
                db_files.extend(vscdb_files)

            # Also search for any other .db/.sqlite files
            for root, dirs, files in os.walk(self.cursor_path):
                for file in files:
                    if file.endswith(('.db', '.sqlite', '.sqlite3')):
                        full_path = Path(root) / file
                        # Skip if already added
                        if full_path in db_files:
                            continue
                        # Check if file is accessible (not locked)
                        try:
                            with open(full_path, 'rb'):
                                pass
                            db_files.append(full_path)
                        except (PermissionError, IOError):
                            continue
        except Exception as e:
            print(f"[WARNING] Error searching databases: {e}")

        return db_files

    def _extract_conversations(self, db_path: Path, hours_back: int = 24) -> List[CapturedConversation]:
        """Extract conversations from a database"""
        conversations = []

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            # Look for chat-related tables
            chat_tables = [t for t in tables if any(
                keyword in t.lower() for keyword in ['chat', 'message', 'conversation', 'history']
            )]

            if not chat_tables:
                conn.close()
                return conversations

            # Try to extract from chat tables
            for table in chat_tables:
                try:
                    # Get schema
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = {row[1]: row[0] for row in cursor.fetchall()}

                    # Determine column mappings
                    timestamp_col = None
                    role_col = None
                    content_col = None
                    conv_id_col = None

                    for col in columns:
                        col_lower = col.lower()
                        if timestamp_col is None and any(kw in col_lower for kw in ['time', 'date', 'created']):
                            timestamp_col = col
                        if role_col is None and any(kw in col_lower for kw in ['role', 'type', 'sender']):
                            role_col = col
                        if content_col is None and any(kw in col_lower for kw in ['content', 'message', 'text', 'body']):
                            content_col = col
                        if conv_id_col is None and any(kw in col_lower for kw in ['conversation', 'session', 'thread', 'chat_id']):
                            conv_id_col = col

                    if not all([timestamp_col, role_col, content_col]):
                        continue

                    # Extract messages from last N hours
                    cutoff_time = datetime.now() - timedelta(hours=hours_back)

                    query = f"""
                    SELECT * FROM {table}
                    WHERE datetime({timestamp_col}) > '{cutoff_time.isoformat()}'
                    ORDER BY {timestamp_col}
                    """

                    cursor.execute(query)
                    rows = cursor.fetchall()

                    # Group by conversation
                    conv_groups: Dict[str, List] = {}
                    for row in rows:
                        conv_id = str(row[conv_id_col] if conv_id_col else "default")
                        if conv_id not in conv_groups:
                            conv_groups[conv_id] = []

                        try:
                            msg_timestamp = datetime.fromisoformat(row[timestamp_col])
                        except:
                            msg_timestamp = datetime.now()

                        conv_groups[conv_id].append({
                            'role': str(row[role_col]).lower(),
                            'content': str(row[content_col]),
                            'timestamp': msg_timestamp
                        })

                    # Create conversation objects
                    for conv_id, messages in conv_groups.items():
                        if len(messages) < 2:  # Need at least one exchange
                            continue

                        chat_msgs = []
                        for i, msg in enumerate(messages):
                            chat_msgs.append(ChatMessage(
                                message_id=f"{conv_id}_{i}",
                                timestamp=msg['timestamp'],
                                role=msg['role'],
                                content=msg['content'],
                                conversation_id=conv_id
                            ))

                        start_time = chat_msgs[0].timestamp
                        end_time = chat_msgs[-1].timestamp

                        # Create hash for duplicate detection
                        content_hash = hashlib.md5(
                            ''.join([msg.content for msg in chat_msgs]).encode()
                        ).hexdigest()

                        conversation = CapturedConversation(
                            conversation_id=conv_id,
                            session_id=f"session_{start_time.strftime('%Y%m%d_%H%M%S')}",
                            start_time=start_time,
                            end_time=end_time,
                            messages=chat_msgs,
                            project_context=self._extract_project_context(chat_msgs),
                            message_count=len(chat_msgs),
                            hash=content_hash
                        )

                        conversations.append(conversation)

                except Exception as e:
                    print(f"Error processing table {table}: {e}")
                    continue

            conn.close()

        except Exception as e:
            print(f"Error reading database {db_path}: {e}")

        return conversations

    def _extract_project_context(self, messages: List[ChatMessage]) -> str:
        """Extract project context from conversation"""
        # Look for file paths, imports, or project references in first messages
        context_cues = []
        for msg in messages[:5]:
            content = msg.content.lower()
            if any(indicator in content for indicator in ['file:', 'import ', 'from ', 'def ']):
                context_cues.append("Code-focused")
            if any(framework in content for framework in ['django', 'flask', 'react', 'vue', 'angular']):
                context_cues.append(framework.capitalize())

        if context_cues:
            return ', '.join(context_cues[:3])
        return "General Development"

    def _save_conversation(self, conversation: CapturedConversation):
        """Save conversation to memory bank and JSON"""
        # Save to JSON log
        CAPTURE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if CAPTURE_LOG_PATH.exists():
            with open(CAPTURE_LOG_PATH, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {'conversations': []}

        log_data['conversations'].append({
            'conversation_id': conversation.conversation_id,
            'session_id': conversation.session_id,
            'start_time': conversation.start_time.isoformat(),
            'end_time': conversation.end_time.isoformat() if conversation.end_time else None,
            'message_count': conversation.message_count,
            'project_context': conversation.project_context,
            'messages': [
                {
                    'role': msg.role,
                    'content': msg.content[:500],  # Truncate for storage
                    'timestamp': msg.timestamp.isoformat()
                }
                for msg in conversation.messages
            ],
            'captured_at': datetime.now().isoformat()
        })

        with open(CAPTURE_LOG_PATH, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Save to EpisodicMemoryBank
        if self.memory_bank and MEMORY_BANK_AVAILABLE:
            try:
                # Create summary from conversation
                user_messages = [m.content for m in conversation.messages if 'user' in m.role]
                if user_messages:
                    summary = f"Cursor chat: {user_messages[0][:200]}"
                else:
                    summary = f"Cursor conversation from {conversation.start_time.strftime('%Y-%m-%d %H:%M')}"

                memory = EpisodicMemoryEntry(
                    agent_type="CursorChatUser",
                    memory_id=str(uuid.uuid4()),
                    summary=summary,
                    context={
                        "conversation_id": conversation.conversation_id,
                        "session_id": conversation.session_id,
                        "message_count": conversation.message_count,
                        "project_context": conversation.project_context,
                        "messages": [
                            {"role": m.role, "content": m.content[:300]}
                            for m in conversation.messages[:10]  # Limit for storage
                        ],
                        "duration_minutes": (
                            (conversation.end_time - conversation.start_time).total_seconds() / 60
                            if conversation.end_time else 0
                        )
                    },
                    source="cursor_auto_capture",
                    timestamp=conversation.start_time,
                    memory_type="conversation"
                )

                self.memory_bank.add(memory)
                print(f"[OK] Saved to memory bank: {summary[:50]}...")
            except Exception as e:
                print(f"Error saving to memory bank: {e}")

    def sync(self, hours_back: int = 24, save_state: bool = True):
        """Sync conversations from Cursor databases"""
        if not self.cursor_path:
            print("[ERROR] Cursor path not found, cannot sync")
            return 0

        print(f"[SEARCH] Searching for conversations from last {hours_back} hours...")

        db_files = self._find_chat_databases()
        print(f"[INFO] Found {len(db_files)} database files")

        total_captured = 0

        for db_path in db_files:
            print(f"[SCAN] Scanning {db_path.name}...")
            conversations = self._extract_conversations(db_path, hours_back)

            for conversation in conversations:
                # Check if already processed
                if conversation.hash in self.processed_conversations:
                    continue

                # Save conversation
                self._save_conversation(conversation)
                self.processed_conversations.add(conversation.hash)
                total_captured += 1

                print(f"  [OK] Captured: {len(conversation.messages)} messages, "
                      f"Context: {conversation.project_context}")

        if save_state:
            self._save_state()

        print(f"\n[SUCCESS] Sync complete: Captured {total_captured} new conversations")
        return total_captured

    def monitor(self, interval_seconds: int = 300, max_runs: int = None):
        """Continuously monitor for new conversations"""
        print(f"[MONITOR] Starting continuous monitor (checking every {interval_seconds}s)...")
        print("Press Ctrl+C to stop")

        run_count = 0

        try:
            while True:
                if max_runs and run_count >= max_runs:
                    break

                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking for new conversations...")
                self.sync(hours_back=2, save_state=True)  # Only check last 2 hours

                run_count += 1
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Monitor stopped by user")


def main():
    parser = argparse.ArgumentParser(description='Automatic Cursor chat capture')
    parser.add_argument('--monitor', action='store_true',
                       help='Run continuous monitor')
    parser.add_argument('--sync', action='store_true',
                       help='One-time sync of conversations')
    parser.add_argument('--hours-back', type=int, default=24,
                       help='Hours of history to sync (default: 24)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Monitor interval in seconds (default: 300)')
    parser.add_argument('--cursor-path', type=str,
                       help='Path to Cursor data directory')
    parser.add_argument('--memory-path', type=str, default='memory_store',
                       help='Path to memory store (default: memory_store)')
    parser.add_argument('--test-run', action='store_true',
                       help='Run once and exit (for testing)')

    args = parser.parse_args()

    capture = CursorAutoCapture(
        cursor_path=args.cursor_path,
        memory_store_path=Path(args.memory_path)
    )

    if args.sync or args.test_run:
        capture.sync(hours_back=args.hours_back)

    elif args.monitor:
        # Run one sync first, then monitor
        capture.sync(hours_back=args.hours_back)
        capture.monitor(interval_seconds=args.interval)

    else:
        # Default: just sync
        capture.sync(hours_back=args.hours_back)


if __name__ == "__main__":
    main()

