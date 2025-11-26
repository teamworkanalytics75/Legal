import os
from pathlib import Path

workspace_dir = Path(os.path.expanduser("~/AppData/Roaming/Cursor/User/workspaceStorage"))
print(f"Searching: {workspace_dir}\n")

# List all directories
for workspace_folder in sorted(workspace_dir.iterdir()):
    if workspace_folder.is_dir():
        print(f"\n=== {workspace_folder.name} ===")

        # List all files in this workspace
        files = list(workspace_folder.rglob("*"))
        for f in files[:10]:  # First 10 files per workspace
            if f.is_file():
                print(f"  {f.name} ({f.stat().st_size} bytes)")

