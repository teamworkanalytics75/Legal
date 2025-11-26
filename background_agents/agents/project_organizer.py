"""Project organizer agent - scans and organizes project files."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.agent import BackgroundAgent, AgentConfig


@dataclass
class ScanTask:
    """Task definition for project organization scans."""

    root: Path
    mode: str = "full_scan"  # full_scan, quick_scan, or organize
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    enable_file_naming_standardization: bool = True
    enable_folder_organization: bool = True
    enable_cleanup_empty_folders: bool = True
    enable_duplicate_detection: bool = True

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], default_root: Path) -> "ScanTask":
        return cls(
            root=Path(payload.get("root", default_root)),
            mode=payload.get("mode", "organize"),  # Default to organize mode
            include_patterns=payload.get("include_patterns"),
            exclude_patterns=payload.get("exclude_patterns"),
            enable_file_naming_standardization=payload.get("enable_file_naming_standardization", True),
            enable_folder_organization=payload.get("enable_folder_organization", True),
            enable_cleanup_empty_folders=payload.get("enable_cleanup_empty_folders", True),
            enable_duplicate_detection=payload.get("enable_duplicate_detection", True),
        )


class ProjectOrganizerAgent(BackgroundAgent):
    """
    Scans the project directory, identifies clutter, and performs organization actions.

    Features:
      1. File naming standardization (snake_case, remove spaces)
      2. Folder organization (move loose files to appropriate folders)
      3. Empty folder cleanup
      4. Duplicate file detection
      5. Project structure optimization
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.default_root = Path(".").resolve()
        self.output_dir = Path("background_agents/outputs/project_organizer")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # File type mappings for organization
        self.file_type_mappings = {
            '.py': 'scripts',
            '.js': 'scripts',
            '.ts': 'scripts',
            '.ps1': 'scripts',
            '.bat': 'scripts',
            '.ahk': 'scripts',
            '.md': 'docs',
            '.txt': 'docs',
            '.json': 'data',
            '.csv': 'data',
            '.db': 'databases',
            '.sqlite': 'databases',
            '.pdf': 'documents',
            '.docx': 'documents',
            '.zip': 'archives',
            '.tar': 'archives',
            '.gz': 'archives',
            '.png': 'assets',
            '.jpg': 'assets',
            '.jpeg': 'assets',
            '.gif': 'assets',
            '.svg': 'assets',
        }

        # Directories to exclude from organization
        self.exclude_dirs = {
            '__pycache__', 'node_modules', '.git', '.venv', 'venv',
            'env', '.env', 'dist', 'build', '.pytest_cache', 'logs',
            'background_agents', 'outputs', 'temp_enrich_samples', 'temp_enriched_samples'
        }

    async def process(self, task: Any) -> Dict[str, Any]:
        """
        Entry point for scheduled tasks.

        Args:
            task: payload dict created by scheduler
        """
        scan_task = ScanTask.from_payload(task or {}, self.default_root)
        self.logger.info(
            "Starting project organization: root=%s mode=%s", scan_task.root, scan_task.mode
        )

        if scan_task.mode == "organize":
            # Perform actual organization tasks
            results = await self._perform_organization(scan_task)
        else:
            # Legacy inventory mode
            inventory = await self._build_inventory(scan_task)
            report_path = await self._write_inventory_report(inventory, scan_task)
            results = {
            "generated_at": datetime.utcnow().isoformat(),
            "root": str(scan_task.root),
            "mode": scan_task.mode,
            "file_count": len(inventory["files"]),
            "dir_count": len(inventory["directories"]),
            "total_size_bytes": inventory["totals"]["size_bytes"],
            "report_path": str(report_path),
        }

        self.logger.info(
            "Project organization complete: %s", results
        )
        return results

    async def _perform_organization(self, scan_task: ScanTask) -> Dict[str, Any]:
        """Perform actual organization tasks."""
        results = {
            "generated_at": datetime.utcnow().isoformat(),
            "root": str(scan_task.root),
            "mode": "organize",
            "actions_performed": [],
            "files_renamed": 0,
            "files_moved": 0,
            "empty_folders_removed": 0,
            "duplicates_found": 0,
            "errors": []
        }

        try:
            # 1. Standardize file names
            if scan_task.enable_file_naming_standardization:
                renamed_count = await self._standardize_file_names(scan_task.root)
                results["files_renamed"] = renamed_count
                results["actions_performed"].append(f"Renamed {renamed_count} files to standard format")

            # 2. Organize files into appropriate folders
            if scan_task.enable_folder_organization:
                moved_count = await self._organize_files_into_folders(scan_task.root)
                results["files_moved"] = moved_count
                results["actions_performed"].append(f"Moved {moved_count} files to appropriate folders")

            # 3. Clean up empty folders
            if scan_task.enable_cleanup_empty_folders:
                removed_count = await self._cleanup_empty_folders(scan_task.root)
                results["empty_folders_removed"] = removed_count
                results["actions_performed"].append(f"Removed {removed_count} empty folders")

            # 4. Detect duplicates
            if scan_task.enable_duplicate_detection:
                duplicates = await self._detect_duplicates(scan_task.root)
                results["duplicates_found"] = len(duplicates)
                results["actions_performed"].append(f"Found {len(duplicates)} duplicate files")

            # Write organization report
            report_path = await self._write_organization_report(results, scan_task)
            results["report_path"] = str(report_path)

        except Exception as e:
            error_msg = f"Organization error: {str(e)}"
            results["errors"].append(error_msg)
            self.logger.error(error_msg)

        return results

    async def _standardize_file_names(self, root: Path) -> int:
        """Standardize file names to snake_case format."""
        renamed_count = 0

        for file_path in root.rglob("*"):
            if file_path.is_file() and not self._should_exclude_path(file_path):
                old_name = file_path.name
                new_name = self._standardize_filename(old_name)

                if new_name != old_name:
                    new_path = file_path.parent / new_name
                    try:
                        # Avoid conflicts
                        if not new_path.exists():
                            file_path.rename(new_path)
                            renamed_count += 1
                            self.logger.info(f"Renamed: {old_name} -> {new_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to rename {old_name}: {e}")

        return renamed_count

    def _standardize_filename(self, filename: str) -> str:
        """Convert filename to proper capitalization standard."""
        # Remove extension
        name, ext = Path(filename).stem, Path(filename).suffix

        # Fix capitalization inconsistencies while preserving proper patterns
        name = self._fix_capitalization(name)

        # Replace non-alphanumeric with underscores
        name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
        name = name.strip('_')  # Remove leading/trailing underscores

        # Handle empty names
        if not name:
            name = "unnamed"

        return f"{name}{ext}"

    def _fix_capitalization(self, name: str) -> str:
        """Fix capitalization inconsistencies in filename."""
        if not name:
            return name

        # Common patterns to preserve proper capitalization
        proper_patterns = {
            # Legal terms
            'case': 'Case',
            'court': 'Court',
            'judge': 'Judge',
            'plaintiff': 'Plaintiff',
            'defendant': 'Defendant',
            'petition': 'Petition',
            'motion': 'Motion',
            'brief': 'Brief',
            'analysis': 'Analysis',
            'report': 'Report',
            'summary': 'Summary',
            'research': 'Research',
            'legal': 'Legal',
            'federal': 'Federal',
            'district': 'District',
            'circuit': 'Circuit',
            'supreme': 'Supreme',
            'appeals': 'Appeals',

            # Technical terms
            'api': 'API',
            'json': 'JSON',
            'xml': 'XML',
            'html': 'HTML',
            'css': 'CSS',
            'js': 'JS',
            'ts': 'TS',
            'py': 'PY',
            'sql': 'SQL',
            'db': 'DB',
            'pdf': 'PDF',
            'docx': 'DOCX',
            'txt': 'TXT',
            'md': 'MD',
            'csv': 'CSV',
            'xlsx': 'XLSX',

            # Common words
            'test': 'Test',
            'demo': 'Demo',
            'example': 'Example',
            'sample': 'Sample',
            'template': 'Template',
            'config': 'Config',
            'setup': 'Setup',
            'install': 'Install',
            'update': 'Update',
            'version': 'Version',
            'release': 'Release',
            'beta': 'Beta',
            'alpha': 'Alpha',
            'stable': 'Stable',
            'dev': 'Dev',
            'prod': 'Prod',
            'staging': 'Staging',
            'production': 'Production',
            'development': 'Development',
        }

        # Split by common separators and fix each part
        parts = re.split(r'[_\-\s\.]+', name)
        fixed_parts = []

        for part in parts:
            if not part:
                continue

            part_lower = part.lower()

            # Check if it matches a known pattern
            if part_lower in proper_patterns:
                fixed_parts.append(proper_patterns[part_lower])
            elif part.isupper() and len(part) > 1:
                # Preserve acronyms (all caps, 2+ chars)
                fixed_parts.append(part)
            elif part.islower() and len(part) > 1:
                # Convert to Title Case for regular words
                fixed_parts.append(part.title())
            else:
                # Mixed case - try to fix inconsistencies
                if part[0].islower() and any(c.isupper() for c in part[1:]):
                    # Fix camelCase inconsistencies
                    fixed_parts.append(part.title())
                else:
                    # Keep as is if it looks intentional
                    fixed_parts.append(part)

        return ''.join(fixed_parts)

    async def _organize_files_into_folders(self, root: Path) -> int:
        """Move loose files into appropriate folders based on file type."""
        moved_count = 0

        # Only organize files in the root directory (not in subdirectories)
        for file_path in root.iterdir():
            if file_path.is_file() and not self._should_exclude_path(file_path):
                target_folder = self._get_target_folder(file_path)
                if target_folder:
                    target_path = root / target_folder
                    target_path.mkdir(exist_ok=True)

                    new_path = target_path / file_path.name
                    try:
                        if not new_path.exists():
                            shutil.move(str(file_path), str(new_path))
                            moved_count += 1
                            self.logger.info(f"Moved: {file_path.name} -> {target_folder}/")
                    except Exception as e:
                        self.logger.warning(f"Failed to move {file_path.name}: {e}")

        return moved_count

    def _get_target_folder(self, file_path: Path) -> Optional[str]:
        """Determine target folder for a file based on its extension."""
        ext = file_path.suffix.lower()
        return self.file_type_mappings.get(ext)

    async def _cleanup_empty_folders(self, root: Path) -> int:
        """Remove empty folders (except important ones)."""
        removed_count = 0

        # Get all directories, sorted by depth (deepest first)
        all_dirs = []
        for dir_path in root.rglob("*"):
            if dir_path.is_dir() and not self._should_exclude_path(dir_path):
                all_dirs.append(dir_path)

        # Sort by depth (deepest first) to avoid removing parent of empty child
        all_dirs.sort(key=lambda p: len(p.parts), reverse=True)

        for dir_path in all_dirs:
            try:
                # Check if directory is empty
                if dir_path.exists() and not any(dir_path.iterdir()):
                    # Don't remove important directories
                    if dir_path.name not in {'scripts', 'docs', 'data', 'databases', 'documents', 'archives', 'assets'}:
                        dir_path.rmdir()
                        removed_count += 1
                        self.logger.info(f"Removed empty folder: {dir_path.name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {dir_path}: {e}")

        return removed_count

    async def _detect_duplicates(self, root: Path) -> List[Dict[str, Any]]:
        """Detect duplicate files by size and name."""
        file_hashes = {}
        duplicates = []

        for file_path in root.rglob("*"):
            if file_path.is_file() and not self._should_exclude_path(file_path):
                try:
                    file_size = file_path.stat().st_size
                    file_key = (file_path.name, file_size)

                    if file_key in file_hashes:
                        duplicates.append({
                            "name": file_path.name,
                            "size": file_size,
                            "paths": [str(file_hashes[file_key]), str(file_path)]
                        })
                    else:
                        file_hashes[file_key] = file_path
                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")

        return duplicates

    def _should_exclude_path(self, path: Path) -> bool:
        """Check if path should be excluded from organization."""
        # Check if any part of the path is in exclude list
        for part in path.parts:
            if part in self.exclude_dirs:
                return True
        return False

    async def _write_organization_report(self, results: Dict[str, Any], scan_task: ScanTask) -> Path:
        """Write organization report to disk."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"organization_report_{timestamp}.md"

        report_lines = [
            "# Project Organization Report",
            "",
            f"- Generated: {results['generated_at']}",
            f"- Root: `{results['root']}`",
            f"- Mode: {results['mode']}",
            "",
            "## Actions Performed",
        ]

        for action in results["actions_performed"]:
            report_lines.append(f"- [OK] {action}")

        if results["errors"]:
            report_lines.extend(["", "## Errors"])
            for error in results["errors"]:
                report_lines.append(f"- [ERROR] {error}")

        report_lines.extend([
            "",
            "## Summary",
            f"- Files renamed: {results['files_renamed']}",
            f"- Files moved: {results['files_moved']}",
            f"- Empty folders removed: {results['empty_folders_removed']}",
            f"- Duplicates found: {results['duplicates_found']}",
        ])

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: report_path.write_text("\n".join(report_lines))
        )

        return report_path

    async def _build_inventory(self, scan_task: ScanTask) -> Dict[str, Any]:
        """
        Walk the project tree and collect metadata.

        Returns:
            Dict containing file/directory metadata and aggregate stats.
        """
        files: List[Dict[str, Any]] = []
        directories: List[Dict[str, Any]] = []
        total_size = 0

        root = scan_task.root

        loop = asyncio.get_running_loop()
        for path in await loop.run_in_executor(
            None, lambda: list(root.rglob("*"))
        ):
            try:
                stat = path.stat()
            except (FileNotFoundError, PermissionError):
                continue

            rel_path = path.relative_to(root)
            entry = {
                "path": str(rel_path),
                "name": path.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "type": "directory" if path.is_dir() else "file",
            }

            if path.is_dir():
                directories.append(entry)
            else:
                files.append(entry)
                total_size += stat.st_size

        inventory = {
            "generated_at": datetime.utcnow().isoformat(),
            "root": str(root),
            "files": files,
            "directories": directories,
            "totals": {
                "size_bytes": total_size,
                "file_count": len(files),
                "directory_count": len(directories),
            },
        }

        return inventory

    async def _write_inventory_report(
        self, inventory: Dict[str, Any], scan_task: ScanTask
    ) -> Path:
        """Persist the inventory to disk (JSON + lightweight Markdown)."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_name = f"inventory_{scan_task.mode}_{timestamp}"
        json_path = self.output_dir / f"{base_name}.json"
        markdown_path = self.output_dir / f"{base_name}.md"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: json_path.write_text(json.dumps(inventory, indent=2))
        )

        summary_lines = [
            f"# Project Inventory Report ({scan_task.mode})",
            "",
            f"- Generated: {inventory['generated_at']}",
            f"- Root: `{inventory['root']}`",
            f"- Files: {inventory['totals']['file_count']}",
            f"- Directories: {inventory['totals']['directory_count']}",
            f"- Total Size: {inventory['totals']['size_bytes'] / (1024**2):.2f} MB",
            "",
            "## Sample Entries",
        ]

        # Include first few files for quick review
        for file_entry in inventory["files"][:10]:
            summary_lines.append(
                f"- `{file_entry['path']}` ({file_entry['size']} bytes, modified {file_entry['modified']})"
            )

        await loop.run_in_executor(
            None, lambda: markdown_path.write_text("\n".join(summary_lines))
        )

        return markdown_path
