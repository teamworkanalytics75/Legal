import sqlite3
import json
import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))

print("Checking ALL workspace databases...\n")

for ws_folder in workspace_dir.iterdir():
    if ws_folder.is_dir() and ws_folder.name != 'images':
        db_path = ws_folder / "state.vscdb"
        if db_path.exists():
            print(f"\n=== {ws_folder.name} ===")
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()

                # Check ItemTable
                cursor.execute("SELECT COUNT(*) FROM ItemTable")
                count = cursor.fetchone()[0]
                print(f"  ItemTable rows: {count}")

                # Get a sample of keys
                cursor.execute("SELECT key FROM ItemTable LIMIT 5")
                keys = [row[0] for row in cursor.fetchall()]
                print(f"  Sample keys: {keys}")

                # Check if cursorDiskKV exists
                try:
                    cursor.execute("SELECT COUNT(*) FROM cursorDiskKV")
                    kv_count = cursor.fetchone()[0]
                    if kv_count > 0:
                        print(f"  ⚠️ cursorDiskKV has {kv_count} entries!")
                        cursor.execute("SELECT key FROM cursorDiskKV")
                        kv_keys = [row[0] for row in cursor.fetchall()]
                        print(f"     Keys: {kv_keys}")
                except:
                    pass

                conn.close()
            except Exception as e:
                print(f"  Error: {e}")

