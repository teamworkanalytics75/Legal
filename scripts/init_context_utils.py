#!/usr/bin/env python3
"""
Init Context Utilities

Helper functions for module detection, focus area extraction, and file analysis
for the automatic /init context documentation system.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple


# Known module directories based on project structure
KNOWN_MODULES = {
    "background_agents",
    "writer_agents",
    "autogen_integration",
    "case_law_data",
    "case_law",
    "nlp_analysis",
    "factuality_filter",
    "document_ingestion",
    "bayesian_network",
    "voice_system",
    "matrix_ui",
    "revit_agent",
    "rules_registry",
    "Agents_1782_ML_Dataset",
    "section_1782_mining",
    "financial_system",
    "vida_datahub",
}


@dataclass
class ModuleInfo:
    """Information about a detected module."""
    name: str
    path: Path
    has_docs: bool
    has_code: bool
    has_scripts: bool


@dataclass
class FocusArea:
    """Extracted focus area information."""
    keywords: List[str]
    normalized_name: str
    path_safe_name: str


def detect_module_from_path(work_dir: Path, repo_root: Path) -> Optional[ModuleInfo]:
    """
    Detect which module the current working directory belongs to.
    
    Args:
        work_dir: Current working directory
        repo_root: Repository root directory
        
    Returns:
        ModuleInfo if a module is detected, None otherwise
    """
    # Normalize paths
    work_dir = work_dir.resolve()
    repo_root = repo_root.resolve()
    
    # Check if we're in a known module directory
    try:
        rel_path = work_dir.relative_to(repo_root)
    except ValueError:
        # Not within repo root
        return None
    
    # Check each path component
    parts = rel_path.parts
    for i, part in enumerate(parts):
        if part in KNOWN_MODULES:
            module_path = repo_root / part
            return ModuleInfo(
                name=part,
                path=module_path,
                has_docs=(module_path / "docs").exists(),
                has_code=(module_path / "code").exists(),
                has_scripts=(module_path / "scripts").exists(),
            )
    
    # Check if we're in a subdirectory of a module
    for module_name in KNOWN_MODULES:
        module_path = repo_root / module_name
        try:
            work_dir.relative_to(module_path)
            return ModuleInfo(
                name=module_name,
                path=module_path,
                has_docs=(module_path / "docs").exists(),
                has_code=(module_path / "code").exists(),
                has_scripts=(module_path / "scripts").exists(),
            )
        except ValueError:
            continue
    
    return None


def extract_focus_area(focus_description: str) -> FocusArea:
    """
    Extract focus area keywords and generate normalized names from focus description.
    
    Args:
        focus_description: The focus description from /init command
        
    Returns:
        FocusArea with keywords, normalized name, and path-safe name
    """
    # Clean up the description
    desc = focus_description.strip()
    
    # Extract keywords (common patterns: "X→Y", "X integration", "X audit", etc.)
    keywords = []
    
    # Split on common separators
    parts = re.split(r'[→\-\s,]+', desc)
    for part in parts:
        part = part.strip()
        if part and len(part) > 2:  # Ignore very short words
            keywords.append(part.lower())
    
    # Also look for common patterns
    patterns = [
        r'(\w+)\s*→\s*(\w+)',  # "X→Y" or "X -> Y"
        r'(\w+)\s+integration',
        r'(\w+)\s+audit',
        r'(\w+)\s+pipeline',
        r'(\w+)\s+system',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, desc, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                keywords.extend([m.lower() for m in match if len(m) > 2])
            elif isinstance(match, str) and len(match) > 2:
                keywords.append(match.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    # Generate normalized name (for display)
    normalized = "_".join(unique_keywords[:5])  # Limit to 5 keywords
    if not normalized:
        normalized = "general"
    
    # Generate path-safe name (lowercase, alphanumeric + underscores)
    path_safe = re.sub(r'[^a-z0-9_]+', '_', normalized.lower())
    path_safe = re.sub(r'_+', '_', path_safe).strip('_')
    if not path_safe:
        path_safe = "general"
    
    return FocusArea(
        keywords=unique_keywords,
        normalized_name=normalized,
        path_safe_name=path_safe,
    )


def find_key_files(directory: Path, max_files: int = 20) -> List[Path]:
    """
    Find key files in a directory (Python files, configs, READMEs).
    
    Args:
        directory: Directory to search
        max_files: Maximum number of files to return
        
    Returns:
        List of key file paths
    """
    key_files = []
    
    if not directory.exists():
        return key_files
    
    # Priority file patterns
    priority_patterns = [
        "*.py",
        "*.md",
        "*.json",
        "*.yaml",
        "*.yml",
        "*.toml",
        "*.txt",
        "README*",
        "requirements*.txt",
    ]
    
    # Directories to ignore
    ignore_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".pytest_cache",
        "mlruns",
        "dist",
        "build",
    }
    
    found = set()
    
    # First, look for high-priority files in root
    for pattern in priority_patterns:
        for path in directory.glob(pattern):
            if path.is_file() and path not in found:
                key_files.append(path)
                found.add(path)
                if len(key_files) >= max_files:
                    return key_files
    
    # Then search subdirectories (but not too deep) - optimized to avoid rglob on large repos
    # Use iterative depth-limited search instead of rglob
    try:
        # First level subdirectories only (depth 1)
        for item in directory.iterdir():
            if len(key_files) >= max_files:
                break
            if item.is_dir():
                # Skip ignored directories early
                if item.name in ignore_dirs:
                    continue
                # Search one level deep only
                try:
                    for subitem in item.iterdir():
                        if len(key_files) >= max_files:
                            break
                        if subitem.is_file() and subitem.suffix in {".py", ".md", ".json", ".yaml", ".yml"}:
                            if subitem not in found:
                                key_files.append(subitem)
                                found.add(subitem)
                except (PermissionError, OSError):
                    # Skip directories we can't read
                    continue
            elif item.is_file() and item.suffix in {".py", ".md", ".json", ".yaml", ".yml"}:
                if item not in found:
                    key_files.append(item)
                    found.add(item)
    except (PermissionError, OSError):
        # If we can't read the directory, just return what we have
        pass
    
    return key_files[:max_files]


def get_repo_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the repository root by looking for .git directory or other markers.
    
    Args:
        start_path: Starting path (defaults to current working directory)
        
    Returns:
        Path to repository root
    """
    if start_path is None:
        start_path = Path.cwd()
    
    current = Path(start_path).resolve()
    
    # Look for .git directory or other markers
    markers = [".git", "plans", "scripts", "docs"]
    
    # Also check if we're in a known module - if so, go up one level
    for module_name in KNOWN_MODULES:
        if module_name in current.parts:
            # Find the index of the module in the path
            try:
                module_index = current.parts.index(module_name)
                # Go up to the parent of the module
                potential_root = Path(*current.parts[:module_index])
                # Verify it has markers
                for marker in markers:
                    if (potential_root / marker).exists():
                        return potential_root
            except ValueError:
                continue
    
    while current != current.parent:
        # Check for markers
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
    
    # Fallback: return the starting path's parent
    return Path(start_path).resolve().parent


def analyze_directory_structure(directory: Path) -> dict:
    """
    Analyze directory structure to understand module organization.
    
    Args:
        directory: Directory to analyze
        
    Returns:
        Dictionary with structure information
    """
    if not directory.exists():
        return {}
    
    structure = {
        "has_code": False,
        "has_docs": False,
        "has_scripts": False,
        "has_tests": False,
        "has_config": False,
        "subdirectories": [],
    }
    
    # Check for common subdirectories
    common_dirs = ["code", "docs", "scripts", "tests", "config", "configs"]
    
    for item in directory.iterdir():
        if item.is_dir():
            dir_name = item.name.lower()
            structure["subdirectories"].append(item.name)
            
            if dir_name in ["code", "src"]:
                structure["has_code"] = True
            elif dir_name == "docs":
                structure["has_docs"] = True
            elif dir_name == "scripts":
                structure["has_scripts"] = True
            elif dir_name in ["tests", "test"]:
                structure["has_tests"] = True
            elif dir_name in ["config", "configs"]:
                structure["has_config"] = True
    
    return structure


def format_file_list(files: List[Path], repo_root: Path, max_display: int = 10) -> str:
    """
    Format a list of files for display in documentation.
    
    Args:
        files: List of file paths
        repo_root: Repository root for relative paths
        max_display: Maximum number of files to display
        
    Returns:
        Formatted string with file list
    """
    if not files:
        return "- (no key files found)"
    
    lines = []
    for i, file_path in enumerate(files[:max_display]):
        try:
            rel_path = file_path.relative_to(repo_root)
            lines.append(f"- `{rel_path}`")
        except ValueError:
            lines.append(f"- `{file_path}`")
    
    if len(files) > max_display:
        lines.append(f"- ... ({len(files) - max_display} more files)")
    
    return "\n".join(lines)

