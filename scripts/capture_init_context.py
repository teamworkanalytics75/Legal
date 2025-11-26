#!/usr/bin/env python3
"""
Capture Init Context

Main script to capture Codex /init context, detect module/focus area,
and generate structured documentation.

Usage:
    python scripts/capture_init_context.py --focus "CatBoost→SHAP integration audit"
    python scripts/capture_init_context.py --auto
    python scripts/capture_init_context.py --focus "description" --problem "Issue" --solution "Fix"
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from init_context_utils import (
    analyze_directory_structure,
    detect_module_from_path,
    extract_focus_area,
    find_key_files,
    format_file_list,
    get_repo_root,
    FocusArea,
    ModuleInfo,
)
from init_doc_manager import InitContextEntry, InitDocManager


def get_recent_git_changes(repo_root: Path, days: int = 1) -> List[str]:
    """Get recent git changes for context."""
    try:
        result = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--oneline", "--max-count=10"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def generate_architecture_summary(
    module_info: Optional[ModuleInfo],
    directory: Path,
    key_files: List[Path],
    repo_root: Path,
) -> str:
    """Generate a brief architecture summary."""
    lines = []
    
    if module_info:
        lines.append(f"Module: {module_info.name}")
        if module_info.has_code:
            lines.append("- Contains code directory")
        if module_info.has_docs:
            lines.append("- Contains documentation")
        if module_info.has_scripts:
            lines.append("- Contains scripts")
        lines.append("")
    
    # Analyze directory structure
    structure = analyze_directory_structure(directory)
    if structure.get("subdirectories"):
        lines.append("Key subdirectories:")
        for subdir in structure["subdirectories"][:5]:
            lines.append(f"- {subdir}")
        lines.append("")
    
    # Key files summary
    if key_files:
        lines.append("Key files identified:")
        for file_path in key_files[:5]:
            try:
                rel_path = file_path.relative_to(repo_root)
                lines.append(f"- {rel_path}")
            except ValueError:
                lines.append(f"- {file_path.name}")
    
    return "\n".join(lines) if lines else "No architecture information available."


def capture_init_context(
    focus_description: Optional[str] = None,
    module_override: Optional[str] = None,
    auto_detect: bool = False,
    problem: Optional[str] = None,
    solution: Optional[str] = None,
    files_changed: Optional[str] = None,
    validation: Optional[str] = None,
    update_only: bool = False,
) -> None:
    """
    Main function to capture init context.
    
    Args:
        focus_description: Explicit focus description from /init
        module_override: Override auto-detected module
        auto_detect: Auto-detect everything from current state
        problem: Problem encountered (for problem tracking)
        solution: Solution applied (for problem tracking)
        files_changed: Files changed (for problem tracking)
        validation: How it was validated (for problem tracking)
        update_only: Only update existing docs, don't create new
    """
    # Get repository root
    try:
        repo_root = get_repo_root()
    except Exception as e:
        print(f"Warning: Could not determine repo root: {e}", file=sys.stderr)
        repo_root = Path.cwd()
    
    current_dir = Path.cwd()
    
    print(f"Repository root: {repo_root}")
    print(f"Current directory: {current_dir}")
    
    # Detect module
    module_info: Optional[ModuleInfo] = None
    if module_override:
        module_path = repo_root / module_override
        if module_path.exists():
            module_info = ModuleInfo(
                name=module_override,
                path=module_path,
                has_docs=(module_path / "docs").exists(),
                has_code=(module_path / "code").exists(),
                has_scripts=(module_path / "scripts").exists(),
            )
    else:
        module_info = detect_module_from_path(current_dir, repo_root)
    
    if module_info:
        print(f"Detected module: {module_info.name}")
    else:
        print("No module detected (working at root level)")
    
    # Extract focus area
    if auto_detect and not focus_description:
        # Try to infer from recent git commits or directory
        recent_commits = get_recent_git_changes(repo_root, days=1)
        if recent_commits:
            focus_description = recent_commits[0]  # Use most recent commit message
        elif module_info:
            focus_description = f"Working on {module_info.name} module"
        else:
            focus_description = "General project work"
    
    if not focus_description:
        # Non-interactive: default to "General" instead of blocking input
        # Check if stdin is a TTY (interactive terminal)
        if sys.stdin.isatty():
            focus_description = input("Enter focus description (or press Enter for 'General'): ").strip()
        else:
            # Non-interactive: use default
            focus_description = "General"
        if not focus_description:
            focus_description = "General"
    
    focus_area = extract_focus_area(focus_description)
    print(f"Focus area: {focus_area.normalized_name}")
    print(f"Keywords: {', '.join(focus_area.keywords[:5])}")
    
    # Find key files (with timeout protection for large repos)
    search_dir = module_info.path if module_info else current_dir
    try:
        key_files = find_key_files(search_dir, max_files=20)
        print(f"Found {len(key_files)} key files")
    except Exception as e:
        logger.warning(f"Error finding key files: {e}, continuing with empty list")
        key_files = []
    
    # Generate architecture summary
    architecture = generate_architecture_summary(
        module_info,
        search_dir,
        key_files,
        repo_root,
    )
    
    # Get recent changes for context (non-blocking)
    try:
        recent_changes = get_recent_git_changes(repo_root, days=1)
    except Exception as e:
        logger.debug(f"Could not get recent git changes: {e}")
        recent_changes = []
    
    # Create init context entry
    key_files_str = [str(f.relative_to(repo_root)) for f in key_files]
    
    entry = InitContextEntry(
        date=datetime.now().isoformat(timespec="seconds"),
        focus=focus_description,
        directory=str(current_dir.relative_to(repo_root)),
        key_files=key_files_str,
        architecture_summary=architecture,
        problems_solved=[],
    )
    
    # Add problem if provided
    if problem:
        entry.problems_solved.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": problem[:60] + "..." if len(problem) > 60 else problem,
            "problem": problem,
            "solution": solution or "",
            "files_changed": files_changed or "",
            "validation": validation or "",
        })
    
    # Initialize documentation manager
    try:
        doc_manager = InitDocManager(repo_root)
    except Exception as e:
        print(f"Error initializing documentation manager: {e}", file=sys.stderr)
        print("Skipping documentation generation")
        return
    
    # Get documentation paths
    try:
        doc_paths = doc_manager.get_documentation_paths(
            module_name=module_info.name if module_info else None,
            focus_area=focus_area.path_safe_name if focus_area.path_safe_name != "general" else None,
        )
    except Exception as e:
        print(f"Error getting documentation paths: {e}", file=sys.stderr)
        print("Skipping documentation generation")
        return
    
    # Read existing entries (non-blocking)
    existing_module_entries = []
    existing_focus_entries = []
    
    if doc_paths.module_doc:
        try:
            existing_module_entries = doc_manager.read_existing_context(doc_paths.module_doc)
            print(f"Found {len(existing_module_entries)} existing module entries")
        except Exception as e:
            logger.debug(f"Could not read existing module entries: {e}")
    
    if doc_paths.focus_doc:
        try:
            existing_focus_entries = doc_manager.read_existing_context(doc_paths.focus_doc)
            print(f"Found {len(existing_focus_entries)} existing focus entries")
        except Exception as e:
            logger.debug(f"Could not read existing focus entries: {e}")
    
    # Write module documentation
    if doc_paths.module_doc and (not update_only or existing_module_entries):
        doc_manager.write_context_doc(
            doc_paths.module_doc,
            entry,
            existing_module_entries,
        )
        print(f"✓ Updated module documentation: {doc_paths.module_doc.relative_to(repo_root)}")
    
    # Write focus area documentation
    if doc_paths.focus_doc and (not update_only or existing_focus_entries):
        doc_manager.write_context_doc(
            doc_paths.focus_doc,
            entry,
            existing_focus_entries,
        )
        print(f"✓ Updated focus documentation: {doc_paths.focus_doc.relative_to(repo_root)}")
    
    # Handle problems documentation
    if problem:
        new_problems = entry.problems_solved
        
        # Module problems
        if doc_paths.module_problems:
            existing_module_problems = doc_manager.read_existing_problems(doc_paths.module_problems)
            doc_manager.write_problems_doc(
                doc_paths.module_problems,
                new_problems,
                existing_module_problems,
            )
            print(f"✓ Updated module problems: {doc_paths.module_problems.relative_to(repo_root)}")
        
        # Focus problems
        if doc_paths.focus_problems:
            existing_focus_problems = doc_manager.read_existing_problems(doc_paths.focus_problems)
            doc_manager.write_problems_doc(
                doc_paths.focus_problems,
                new_problems,
                existing_focus_problems,
            )
            print(f"✓ Updated focus problems: {doc_paths.focus_problems.relative_to(repo_root)}")
    
    # Update master index
    doc_manager.update_index(
        module_name=module_info.name if module_info else None,
        focus_area=focus_area.path_safe_name if focus_area.path_safe_name != "general" else None,
        entry=entry,
    )
    print(f"✓ Updated master index: {doc_paths.index.relative_to(repo_root)}")
    
    print("\n✓ Init context captured successfully!")
    print(f"\nDocumentation locations:")
    if doc_paths.module_doc:
        print(f"  - Module: {doc_paths.module_doc.relative_to(repo_root)}")
    if doc_paths.focus_doc:
        print(f"  - Focus: {doc_paths.focus_doc.relative_to(repo_root)}")
    print(f"  - Index: {doc_paths.index.relative_to(repo_root)}")
    
    # Link to activity digest
    activity_digest_path = repo_root / "reports" / "analysis_outputs" / "activity_digest.md"
    if activity_digest_path.exists():
        print(f"\n💡 Tip: Init context is cross-referenced in activity digest: {activity_digest_path.relative_to(repo_root)}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Capture Codex /init context and generate documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture with explicit focus
  python scripts/capture_init_context.py --focus "CatBoost→SHAP integration audit"
  
  # Auto-detect from current state
  python scripts/capture_init_context.py --auto
  
  # Add a problem/solution entry
  python scripts/capture_init_context.py --focus "Bug fix" \\
      --problem "Feature X not working" \\
      --solution "Fixed initialization order" \\
      --files-changed "src/feature.py" \\
      --validation "Ran tests, verified manually"
  
  # Update existing docs only
  python scripts/capture_init_context.py --focus "Update" --update-only
        """,
    )
    
    parser.add_argument(
        "--focus",
        type=str,
        help="Explicit focus description from /init command",
    )
    
    parser.add_argument(
        "--module",
        type=str,
        help="Override auto-detected module name",
    )
    
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect focus from current state (git commits, directory)",
    )
    
    parser.add_argument(
        "--problem",
        type=str,
        help="Problem encountered (for problem tracking)",
    )
    
    parser.add_argument(
        "--solution",
        type=str,
        help="Solution applied (for problem tracking)",
    )
    
    parser.add_argument(
        "--files-changed",
        type=str,
        help="Files changed (for problem tracking)",
    )
    
    parser.add_argument(
        "--validation",
        type=str,
        help="How it was validated (for problem tracking)",
    )
    
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only update existing documentation (don't create new files)",
    )
    
    args = parser.parse_args()
    
    try:
        capture_init_context(
            focus_description=args.focus,
            module_override=args.module,
            auto_detect=args.auto,
            problem=args.problem,
            solution=args.solution,
            files_changed=args.files_changed,
            validation=args.validation,
            update_only=args.update_only,
        )
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

