#!/usr/bin/env python3
"""
Verify and fix path issues for Cursor multi-agent execution.
"""

import os
import sys
from pathlib import Path

def check_workspace_path():
    """Check if workspace is in optimal location."""
    workspace = Path(os.getcwd())
    workspace_str = str(workspace)

    print("🔍 Checking workspace path...")
    print(f"   Current workspace: {workspace_str}")

    # Check if in WSL home
    if workspace_str.startswith("/home/"):
        print("✅ Workspace is in WSL home directory (optimal)")
        return True
    elif workspace_str.startswith("/mnt/"):
        print("⚠️  Workspace is on Windows mount (/mnt/)")
        print("   This may cause performance issues and path problems for agents")
        print("   Consider moving to: /home/serteamwork/projects/TheMatrix")
        return False
    else:
        print("⚠️  Workspace path format is unusual")
        return False

def check_permissions():
    """Check file permissions."""
    workspace = Path(os.getcwd())

    print("\n🔍 Checking file permissions...")

    # Check read permission
    if os.access(workspace, os.R_OK):
        print("✅ Workspace is readable")
    else:
        print("❌ Workspace is NOT readable")
        return False

    # Check write permission
    if os.access(workspace, os.W_OK):
        print("✅ Workspace is writable")
    else:
        print("❌ Workspace is NOT writable")
        return False

    # Test file creation
    test_file = workspace / ".cursor_path_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Can create and delete files")
    except Exception as e:
        print(f"❌ Cannot create files: {e}")
        return False

    return True

def check_python_path():
    """Check Python path configuration."""
    print("\n🔍 Checking Python path...")

    venv_path = Path(os.getcwd()) / ".venv" / "bin" / "python"
    if venv_path.exists():
        print(f"✅ Virtual environment found: {venv_path}")
        return True
    else:
        print(f"⚠️  Virtual environment not found at: {venv_path}")
        return False

def check_symlinks():
    """Check if symlinks are needed."""
    workspace = Path(os.getcwd())

    print("\n🔍 Checking for symlink needs...")

    # If workspace is on /mnt/, suggest symlink
    if str(workspace).startswith("/mnt/"):
        wsl_home = Path.home() / "projects" / "TheMatrix"
        if not wsl_home.exists():
            print(f"💡 Consider creating symlink:")
            print(f"   ln -s {workspace} {wsl_home}")
            print(f"   Then open workspace from: {wsl_home}")
        else:
            print(f"✅ Symlink target exists: {wsl_home}")

    return True

def main():
    """Run all path checks."""
    print("="*70)
    print("Path Verification for Cursor Multi-Agent Execution")
    print("="*70)

    all_ok = True

    all_ok &= check_workspace_path()
    all_ok &= check_permissions()
    all_ok &= check_python_path()
    check_symlinks()

    print("\n" + "="*70)
    if all_ok:
        print("✅ All path checks passed!")
    else:
        print("⚠️  Some path issues detected. See recommendations above.")
    print("="*70)

if __name__ == "__main__":
    main()

