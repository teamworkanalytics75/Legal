#!/usr/bin/env python3
"""
Working Cursor/Codex Chat Capture
==================================
Extracts chats from Cursor databases and saves to memory bank.
"""

import sqlite3
import json
import os
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "writer_agents" / "code"))
try:
    from EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryEntry
    MEMORY_BANK_AVAILABLE = True
except ImportError:
    MEMORY_BANK_AVAILABLE = False
    EpisodicMemoryBank = None
    EpisodicMemoryEntry = None

try:
    from knowledge_base.cursor_chats import CursorChatLibrary
    CHAT_LIBRARY_AVAILABLE = True
except ImportError:
    CursorChatLibrary = None  # type: ignore[assignment]
    CHAT_LIBRARY_AVAILABLE = False
WORKSPACE_STORAGE = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
CAPTURE_LOG = Path("data/cursor_chats.json")
STATE_FILE = Path("data/cursor_capture_state.json")


class CursorChatCapture:
    """Extract and save Cursor/Codex chats"""

    def __init__(self):
        self.processed = self._load_state()
        self.memory_bank = None

        if MEMORY_BANK_AVAILABLE:
            self.memory_bank = EpisodicMemoryBank(storage_path=Path("memory_store"))
            print("[OK] Connected to memory bank")

    def _load_state(self) -> set:
        """Load processed conversation hashes"""
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    return set(json.load(f).get('processed', []))
            except:
                pass
        return set()

    def _save_state(self):
        """Save processed hashes"""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump({'processed': list(self.processed)}, f)

    def extract_from_database(self, db_path: Path) -> List[Dict[str, Any]]:
        """Extract all chats from a database"""
        chats = []

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get composer metadata
            try:
                cursor.execute("SELECT value FROM ItemTable WHERE key='composer.composerData'")
                result = cursor.fetchone()
                composer_data = json.loads(result[0]) if result else {}
                all_composers = composer_data.get('allComposers', [])
            except:
                all_composers = []

            # Get prompts (actual chat content)
            try:
                cursor.execute("SELECT value FROM ItemTable WHERE key='aiService.prompts'")
                result = cursor.fetchone()
                prompts = json.loads(result[0]) if result else []
            except:
                prompts = []

            # Build composer map
            composer_map = {}
            for comp in all_composers:
                if isinstance(comp, dict):
                    comp_id = comp.get('composerId')
                    name = comp.get('name', 'Unnamed')
                    created = comp.get('createdAt', 0)
                    composer_map[comp_id] = {
                        'name': name,
                        'created': datetime.fromtimestamp(created / 1000) if created else datetime.now()
                    }

            # Extract chats from prompts
            for i, prompt in enumerate(prompts):
                if not isinstance(prompt, dict) or 'text' not in prompt:
                    continue

                text = prompt['text']
                if not text or len(text.strip()) < 10:
                    continue

                # Detect Codex vs Cursor
                agent_type = "codex" if 'codex' in text.lower() or 'Read this chat with codex' in text else "cursor"

                # Create hash
                content_hash = hashlib.md5(text.encode()).hexdigest()

                # Skip if already processed
                if content_hash in self.processed:
                    continue

                # Create chat entry
                chat = {
                    'id': content_hash,
                    'agent_type': agent_type,
                    'content': text,
                    'title': self._extract_title(text, agent_type),
                    'timestamp': datetime.now().isoformat(),
                    'length': len(text)
                }

                chats.append(chat)
                self.processed.add(content_hash)

            conn.close()

        except Exception as e:
            print(f"[ERROR] Error extracting from {db_path.name}: {e}")

        return chats

    def _extract_title(self, text: str, agent_type: str) -> str:
        """Extract title from chat content"""
        # Use first 60 chars as title
        title = text[:60].strip().replace('\n', ' ').replace('\r', ' ')
        return f"{agent_type.upper()}: {title}..."

    def save_chats(self, chats: List[Dict[str, Any]]):
        """Save chats to JSON and memory bank"""
        if not chats:
            print("[INFO] No new chats to save")
            return

        # Save to JSON
        CAPTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        if CAPTURE_LOG.exists():
            with open(CAPTURE_LOG, 'r') as f:
                existing = json.load(f)
        else:
            existing = {'chats': []}

        existing['chats'].extend(chats)
        existing['last_update'] = datetime.now().isoformat()
        existing['total_count'] = len(existing['chats'])

        with open(CAPTURE_LOG, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"[OK] Saved {len(chats)} chats to {CAPTURE_LOG}")

        # Save to memory bank
        if self.memory_bank and MEMORY_BANK_AVAILABLE:
            for chat in chats:
                memory = EpisodicMemoryEntry(
                    agent_type=f"ChatUser{chat['agent_type'].capitalize()}",
                    memory_id=str(uuid.uuid4()),
                    summary=chat['title'],
                    context={
                        'agent_type': chat['agent_type'],
                        'content': chat['content'][:500],  # First 500 chars
                        'length': chat['length'],
                        'timestamp': chat['timestamp']
                    },
                    source="cursor_auto_capture",
                    timestamp=datetime.now(),
                    memory_type="conversation"
                )
                self.memory_bank.add(memory)

            print(f"[OK] Saved {len(chats)} chats to memory bank")

        # Update vector index for downstream retrieval
        if CHAT_LIBRARY_AVAILABLE and chats:
            try:
                library = CursorChatLibrary(auto_load=True)
                appended = library.append_chats(chats)
                if appended:
                    print(f"[OK] Updated chat embeddings with {appended} entries")
            except Exception as exc:
                print(f"[WARN] Could not update chat embeddings: {exc}")
        elif chats:
            print("[INFO] Chat embedding library unavailable; skipping vector index update")

    def sync_all(self):
        """Sync all workspace databases"""
        if not WORKSPACE_STORAGE.exists():
            print(f"[ERROR] Workspace storage not found: {WORKSPACE_STORAGE}")
            return 0

        print(f"[SEARCH] Finding databases...")
        db_files = list(WORKSPACE_STORAGE.glob("*/state.vscdb"))
        print(f"[INFO] Found {len(db_files)} databases")

        all_chats = []

        for db_path in db_files:
            print(f"[SCAN] Scanning {db_path.parent.name}...")
            chats = self.extract_from_database(db_path)
            all_chats.extend(chats)
            if chats:
                print(f"  -> Found {len(chats)} chats")

        # Save all chats
        self.save_chats(all_chats)

        # Save state
        self._save_state()

        print(f"\n[SUCCESS] Captured {len(all_chats)} chats total")
        return len(all_chats)


def main():
    capture = CursorChatCapture()
    count = capture.sync_all()
    print(f"\nDone! Captured {count} chats.")


if __name__ == "__main__":
    main()
