#!/usr/bin/env python3
"""
Init Documentation Manager

Manages hierarchical documentation structure, updates existing docs, merges entries,
and maintains cross-references and indexes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class InitContextEntry:
    """Represents a single init context entry."""
    date: str
    focus: str
    directory: str
    key_files: List[str]
    architecture_summary: str
    problems_solved: List[Dict[str, str]]  # List of {problem, solution, files_changed, validation}


@dataclass
class DocumentationPaths:
    """Paths for all documentation files related to an init context."""
    module_doc: Optional[Path]  # {module}/docs/INIT_CONTEXT.md
    focus_doc: Optional[Path]  # docs/init_context/{focus}/CONTEXT.md
    module_problems: Optional[Path]  # {module}/docs/PROBLEMS_SOLVED.md
    focus_problems: Optional[Path]  # docs/init_context/{focus}/PROBLEMS.md
    index: Path  # docs/init_context/INDEX.md


class InitDocManager:
    """Manages init context documentation files."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.init_context_dir = self.repo_root / "docs" / "init_context"
        self.init_context_dir.mkdir(parents=True, exist_ok=True)
    
    def get_documentation_paths(
        self,
        module_name: Optional[str],
        focus_area: Optional[str],
    ) -> DocumentationPaths:
        """
        Determine all documentation paths for a given module and focus area.
        
        Args:
            module_name: Name of the module (if any)
            focus_area: Path-safe focus area name (if any)
            
        Returns:
            DocumentationPaths with all relevant paths
        """
        module_doc = None
        module_problems = None
        
        if module_name:
            module_path = self.repo_root / module_name
            docs_dir = module_path / "docs"
            if docs_dir.exists() or module_path.exists():
                docs_dir.mkdir(parents=True, exist_ok=True)
                module_doc = docs_dir / "INIT_CONTEXT.md"
                module_problems = docs_dir / "PROBLEMS_SOLVED.md"
        
        focus_doc = None
        focus_problems = None
        
        if focus_area:
            focus_dir = self.init_context_dir / focus_area
            focus_dir.mkdir(parents=True, exist_ok=True)
            focus_doc = focus_dir / "CONTEXT.md"
            focus_problems = focus_dir / "PROBLEMS.md"
        
        index = self.init_context_dir / "INDEX.md"
        
        return DocumentationPaths(
            module_doc=module_doc,
            focus_doc=focus_doc,
            module_problems=module_problems,
            focus_problems=focus_problems,
            index=index,
        )
    
    def read_existing_context(self, doc_path: Path) -> List[InitContextEntry]:
        """
        Parse existing INIT_CONTEXT.md file and extract entries.
        
        Args:
            doc_path: Path to the context documentation file
            
        Returns:
            List of InitContextEntry objects
        """
        if not doc_path.exists():
            return []
        
        entries = []
        content = doc_path.read_text(encoding="utf-8")
        
        # Parse sections - look for "## Context Snapshot" sections
        # Each section represents an entry
        sections = re.split(r'^##\s+Context\s+Snapshot', content, flags=re.MULTILINE)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first (header)
            entry = self._parse_context_section(section, content)
            if entry:
                entries.append(entry)
        
        # If no sections found, try to parse as single entry
        if not entries and "## Focus" in content:
            entry = self._parse_entire_file(content)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_context_section(self, section: str, full_content: str) -> Optional[InitContextEntry]:
        """Parse a single context snapshot section."""
        lines = section.splitlines()
        
        # Extract date
        date_match = re.search(r'Date:\s*(.+)', section)
        date = date_match.group(1).strip() if date_match else datetime.now().isoformat()
        
        # Extract directory
        dir_match = re.search(r'Directory:\s*(.+)', section)
        directory = dir_match.group(1).strip() if dir_match else ""
        
        # Extract key files
        key_files = []
        in_files = False
        for line in lines:
            if "Key Files:" in line:
                in_files = True
                continue
            if in_files and line.strip().startswith("-"):
                # Extract file path from markdown link or code block
                file_match = re.search(r'`([^`]+)`', line)
                if file_match:
                    key_files.append(file_match.group(1))
            elif in_files and line.strip() and not line.startswith(" "):
                in_files = False
        
        # Extract architecture summary
        arch_match = re.search(r'Architecture:\s*(.+?)(?:\n\n|\n##|$)', section, re.DOTALL)
        architecture = arch_match.group(1).strip() if arch_match else ""
        
        # Extract focus from previous section or full content
        focus_match = re.search(r'##\s+Focus\s*\n(.+?)(?:\n\n|\n##)', full_content, re.DOTALL)
        focus = focus_match.group(1).strip() if focus_match else ""
        
        return InitContextEntry(
            date=date,
            focus=focus,
            directory=directory,
            key_files=key_files,
            architecture_summary=architecture,
            problems_solved=[],
        )
    
    def _parse_entire_file(self, content: str) -> Optional[InitContextEntry]:
        """Parse entire file as a single entry."""
        focus_match = re.search(r'##\s+Focus\s*\n(.+?)(?:\n\n|\n##)', content, re.DOTALL)
        focus = focus_match.group(1).strip() if focus_match else ""
        
        date_match = re.search(r'Date:\s*(.+)', content)
        date = date_match.group(1).strip() if date_match else datetime.now().isoformat()
        
        dir_match = re.search(r'Directory:\s*(.+)', content)
        directory = dir_match.group(1).strip() if dir_match else ""
        
        return InitContextEntry(
            date=date,
            focus=focus,
            directory=directory,
            key_files=[],
            architecture_summary="",
            problems_solved=[],
        )
    
    def read_existing_problems(self, problems_path: Path) -> List[Dict[str, str]]:
        """
        Parse existing PROBLEMS_SOLVED.md file.
        
        Args:
            problems_path: Path to problems documentation file
            
        Returns:
            List of problem dictionaries
        """
        if not problems_path.exists():
            return []
        
        problems = []
        content = problems_path.read_text(encoding="utf-8")
        
        # Look for problem sections
        problem_sections = re.split(r'^###\s+(\d{4}-\d{2}-\d{2})\s*-\s*(.+?)$', content, flags=re.MULTILINE)
        
        for i in range(1, len(problem_sections), 3):
            if i + 2 < len(problem_sections):
                date = problem_sections[i]
                title = problem_sections[i + 1]
                section_content = problem_sections[i + 2]
                
                problem = self._parse_problem_section(date, title, section_content)
                if problem:
                    problems.append(problem)
        
        return problems
    
    def _parse_problem_section(self, date: str, title: str, content: str) -> Optional[Dict[str, str]]:
        """Parse a single problem section."""
        problem_match = re.search(r'Problem:\s*(.+?)(?:\n-|$)', content, re.DOTALL)
        solution_match = re.search(r'Solution:\s*(.+?)(?:\n-|$)', content, re.DOTALL)
        files_match = re.search(r'Files Changed:\s*(.+?)(?:\n-|$)', content, re.DOTALL)
        validation_match = re.search(r'Validation:\s*(.+?)(?:\n-|$)', content, re.DOTALL)
        
        return {
            "date": date,
            "title": title.strip(),
            "problem": problem_match.group(1).strip() if problem_match else "",
            "solution": solution_match.group(1).strip() if solution_match else "",
            "files_changed": files_match.group(1).strip() if files_match else "",
            "validation": validation_match.group(1).strip() if validation_match else "",
        }
    
    def write_context_doc(
        self,
        doc_path: Path,
        entry: InitContextEntry,
        existing_entries: List[InitContextEntry],
    ) -> None:
        """
        Write or update context documentation file.
        
        Args:
            doc_path: Path to write to
            entry: New entry to add
            existing_entries: Existing entries to preserve
        """
        # Determine title from path
        if "init_context" in str(doc_path.parent):
            # Focus area doc
            title = doc_path.parent.name.replace("_", " ").title()
        else:
            # Module doc
            title = doc_path.parent.parent.name.replace("_", " ").title()
        
        lines = [f"# {title} - Init Context", ""]
        
        # Add focus section
        lines.append("## Focus")
        lines.append(entry.focus or "(no focus specified)")
        lines.append("")
        
        # Add all entries (newest first)
        all_entries = [entry] + existing_entries
        all_entries.sort(key=lambda e: e.date, reverse=True)
        
        for i, e in enumerate(all_entries):
            if i > 0:
                lines.append("---")
                lines.append("")
            
            lines.append("## Context Snapshot")
            lines.append(f"- Date: {e.date}")
            lines.append(f"- Directory: {e.directory}")
            lines.append("")
            
            if e.key_files:
                lines.append("### Key Files")
                for file_path in e.key_files[:15]:  # Limit display
                    lines.append(f"- `{file_path}`")
                if len(e.key_files) > 15:
                    lines.append(f"- ... ({len(e.key_files) - 15} more files)")
                lines.append("")
            
            if e.architecture_summary:
                lines.append("### Architecture")
                lines.append(e.architecture_summary)
                lines.append("")
        
        # Add problems section if any
        all_problems = []
        for e in all_entries:
            all_problems.extend(e.problems_solved)
        
        if all_problems:
            lines.append("## Problems Encountered & Solved")
            for problem in all_problems:
                lines.append(f"### {problem.get('date', 'Unknown')} - {problem.get('title', 'Problem')}")
                if problem.get('problem'):
                    lines.append(f"- Problem: {problem['problem']}")
                if problem.get('solution'):
                    lines.append(f"- Solution: {problem['solution']}")
                if problem.get('files_changed'):
                    lines.append(f"- Files Changed: {problem['files_changed']}")
                if problem.get('validation'):
                    lines.append(f"- Validation: {problem['validation']}")
                lines.append("")
        
        # Add history section
        lines.append("## History")
        lines.append(f"- Total init sessions: {len(all_entries)}")
        for e in all_entries:
            lines.append(f"- {e.date}: {e.focus[:60]}...")
        lines.append("")
        
        doc_path.write_text("\n".join(lines), encoding="utf-8")
    
    def write_problems_doc(
        self,
        problems_path: Path,
        new_problems: List[Dict[str, str]],
        existing_problems: List[Dict[str, str]],
    ) -> None:
        """
        Write or update problems documentation file.
        
        Args:
            problems_path: Path to write to
            new_problems: New problems to add
            existing_problems: Existing problems to preserve
        """
        # Determine title
        if "init_context" in str(problems_path.parent):
            title = f"{problems_path.parent.name.replace('_', ' ').title()} - Problems Solved"
        else:
            title = f"{problems_path.parent.parent.name.replace('_', ' ').title()} - Problems Solved"
        
        lines = [f"# {title}", ""]
        lines.append("This document tracks problems encountered and solutions applied during development.")
        lines.append("")
        
        # Combine and sort by date (newest first)
        all_problems = new_problems + existing_problems
        all_problems.sort(key=lambda p: p.get("date", ""), reverse=True)
        
        for problem in all_problems:
            date = problem.get("date", datetime.now().strftime("%Y-%m-%d"))
            title_text = problem.get("title", "Problem")
            
            lines.append(f"### {date} - {title_text}")
            if problem.get("problem"):
                lines.append(f"- Problem: {problem['problem']}")
            if problem.get("solution"):
                lines.append(f"- Solution: {problem['solution']}")
            if problem.get("files_changed"):
                lines.append(f"- Files Changed: {problem['files_changed']}")
            if problem.get("validation"):
                lines.append(f"- Validation: {problem['validation']}")
            lines.append("")
        
        problems_path.write_text("\n".join(lines), encoding="utf-8")
    
    def update_index(
        self,
        module_name: Optional[str],
        focus_area: Optional[str],
        entry: InitContextEntry,
    ) -> None:
        """
        Update the master index file.
        
        Args:
            module_name: Module name (if any)
            focus_area: Focus area name (if any)
            entry: The new entry
        """
        index_path = self.init_context_dir / "INDEX.md"
        
        # Read existing index or create new
        if index_path.exists():
            content = index_path.read_text(encoding="utf-8")
        else:
            content = self._create_index_template()
        
        # Parse existing entries
        entries = self._parse_index_entries(content)
        
        # Add new entry
        new_entry = {
            "date": entry.date,
            "module": module_name or "root",
            "focus": focus_area or "general",
            "focus_description": entry.focus,
            "directory": entry.directory,
        }
        entries.append(new_entry)
        
        # Sort by date (newest first)
        entries.sort(key=lambda e: e.get("date", ""), reverse=True)
        
        # Rebuild index
        lines = ["# Init Context Documentation Index", ""]
        lines.append("Central index of all init context documentation across the project.")
        lines.append("")
        lines.append(f"- Last updated: {datetime.now().isoformat()}")
        lines.append(f"- Total sessions: {len(entries)}")
        lines.append("")
        
        # Group by module
        by_module: Dict[str, List[Dict]] = {}
        for entry in entries:
            module = entry.get("module", "root")
            if module not in by_module:
                by_module[module] = []
            by_module[module].append(entry)
        
        lines.append("## By Module")
        for module in sorted(by_module.keys()):
            lines.append(f"### {module}")
            for entry in by_module[module][:10]:  # Limit per module
                focus = entry.get("focus", "general")
                desc = entry.get("focus_description", "")[:50]
                date = entry.get("date", "")[:10]
                lines.append(f"- [{date}] [{focus}](docs/init_context/{focus}/CONTEXT.md): {desc}")
            if len(by_module[module]) > 10:
                lines.append(f"- ... ({len(by_module[module]) - 10} more)")
            lines.append("")
        
        # Recent sessions
        lines.append("## Recent Sessions")
        for entry in entries[:20]:
            module = entry.get("module", "root")
            focus = entry.get("focus", "general")
            desc = entry.get("focus_description", "")[:60]
            date = entry.get("date", "")[:10]
            lines.append(f"- [{date}] **{module}** / {focus}: {desc}")
        lines.append("")
        
        index_path.write_text("\n".join(lines), encoding="utf-8")
    
    def _create_index_template(self) -> str:
        """Create a template for the index file."""
        return """# Init Context Documentation Index

Central index of all init context documentation across the project.

## By Module

## Recent Sessions

"""
    
    def _parse_index_entries(self, content: str) -> List[Dict]:
        """Parse existing index entries."""
        entries = []
        # Simple parsing - look for list items with dates
        for line in content.splitlines():
            match = re.match(r'-\s*\[(\d{4}-\d{2}-\d{2}[^\]]*)\]\s*(.+)', line)
            if match:
                date = match.group(1)
                rest = match.group(2)
                # Try to extract module and focus
                module_match = re.search(r'\*\*([^*]+)\*\*', rest)
                focus_match = re.search(r'/(\w+)', rest)
                desc_match = re.search(r':\s*(.+)', rest)
                
                entries.append({
                    "date": date,
                    "module": module_match.group(1) if module_match else "root",
                    "focus": focus_match.group(1) if focus_match else "general",
                    "focus_description": desc_match.group(1) if desc_match else rest,
                })
        return entries

