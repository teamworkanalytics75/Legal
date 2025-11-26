#!/usr/bin/env python3
"""
Link Update Script for The Matrix Repository

This script automatically updates markdown links after repository reorganization
to ensure all internal references point to the correct new locations.

Usage:
    python scripts/update_links.py [--dry-run] [--verbose] [--file FILE]

Options:
    --dry-run    Show what would be changed without making changes
    --verbose    Show detailed output of each operation
    --file FILE  Update only the specified file
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class LinkUpdater:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.link_mappings = self._build_link_mappings()
        self.updated_files = []
        self.errors = []

    def _build_link_mappings(self) -> Dict[str, str]:
        """Build mappings from old paths to new paths based on reorganization."""
        return {
            # Export bundles moved to dist/
            "CHATGPT_UPLOAD_ATOMIC_AGENTS/": "dist/chatgpt_atomic_agents/",
            "CHATGPT_PROJECT_DOCS/": "dist/chatgpt_project_docs/",
            "chatgpt_export/": "dist/chatgpt_export/",

            # Documentation moved to docs/
            "ATOMIC_AGENT_ARCHITECTURE.md": "docs/architecture/ATOMIC_AGENT_ARCHITECTURE.md",
            "ATOMIC_AGENT_IMPLEMENTATION_STATUS.md": "docs/status/ATOMIC_AGENT_IMPLEMENTATION_STATUS.md",
            "ATOMIC_AGENTS_COMPLETE_SUMMARY.md": "docs/architecture/ATOMIC_AGENTS_COMPLETE_SUMMARY.md",
            "FINAL_PROJECT_STATUS_OCTOBER_10_2025.md": "docs/status/FINAL_PROJECT_STATUS_OCTOBER_10_2025.md",
            "DOCUMENTATION_UPDATE_SUMMARY_OCTOBER_11_2025.md": "docs/status/DOCUMENTATION_UPDATE_SUMMARY_OCTOBER_11_2025.md",
            "WITCHWEB_IMPLEMENTATION_COMPLETE.md": "docs/status/WITCHWEB_IMPLEMENTATION_COMPLETE.md",
            "MEMORY_SYSTEM_COMPLETE.md": "docs/status/MEMORY_SYSTEM_COMPLETE.md",
            "REORGANIZATION_COMPLETE.md": "docs/status/REORGANIZATION_COMPLETE.md",
            "CHATGPT_UPLOAD_FOLDER_CREATED.md": "docs/status/CHATGPT_UPLOAD_FOLDER_CREATED.md",
            "CLICKABLE_LINKS_FIXED.md": "docs/status/CLICKABLE_LINKS_FIXED.md",
            "LINKS_FIXED_SUMMARY.md": "docs/status/LINKS_FIXED_SUMMARY.md",
            "DOCUMENTATION_CONSOLIDATION_COMPLETE.md": "docs/status/DOCUMENTATION_CONSOLIDATION_COMPLETE.md",
            "FACTUALITY_FILTER_QUICKSTART.md": "docs/guides/FACTUALITY_FILTER_QUICKSTART.md",
            "FINANCIAL_SYSTEM_QUICKSTART.md": "docs/guides/FINANCIAL_SYSTEM_QUICKSTART.md",
            "QUICKSTART_NLP.md": "docs/guides/QUICKSTART_NLP.md",
            "STRATEGIC_MODULES_QUICKSTART.md": "docs/guides/STRATEGIC_MODULES_QUICKSTART.md",
            "WITCHWEB_UI_QUICK_START.md": "docs/guides/WITCHWEB_UI_QUICK_START.md",
            "WITCHWEB_CASE_LAW_GUIDE.md": "docs/guides/WITCHWEB_CASE_LAW_GUIDE.md",
            "BULK_DOWNLOAD_GUIDE.md": "docs/guides/BULK_DOWNLOAD_GUIDE.md",
            "AGENT_MEMORY_SYSTEM_README.md": "docs/guides/AGENT_MEMORY_SYSTEM_README.md",
            "NLP_SYSTEM_COMPARISON.md": "docs/architecture/NLP_SYSTEM_COMPARISON.md",
            "GRAPH_FIX_COMPARISON.md": "docs/architecture/GRAPH_FIX_COMPARISON.md",
            "WITCHWEB_2.0_RELEASE_NOTES.md": "docs/releases/WITCHWEB_2.0_RELEASE_NOTES.md",
            "WITCHWEB_2.1.1_COMPLETE_OCTOBER_11_2025.md": "docs/releases/WITCHWEB_2.1.1_COMPLETE_OCTOBER_11_2025.md",
            "OCTOBER_11_2025_ATOMIC_AGENTS_IMPLEMENTATION.md": "docs/releases/OCTOBER_11_2025_ATOMIC_AGENTS_IMPLEMENTATION.md",

            # Additional status files
            "ALWAYS_ON_EVIDENCE_COMPLETE.md": "docs/status/ALWAYS_ON_EVIDENCE_COMPLETE.md",
            "FACTUALITY_FILTER_IMPLEMENTATION.md": "docs/status/FACTUALITY_FILTER_IMPLEMENTATION.md",
            "FACTUALITY_FILTER_SUMMARY.md": "docs/status/FACTUALITY_FILTER_SUMMARY.md",
            "FINANCIAL_SYSTEM_IMPLEMENTATION_COMPLETE.md": "docs/status/FINANCIAL_SYSTEM_IMPLEMENTATION_COMPLETE.md",
            "GRAPH_INGESTION_TEST_REPORT.md": "docs/status/GRAPH_INGESTION_TEST_REPORT.md",
            "LLM_MEMORY_WRITING_IMPLEMENTATION.md": "docs/status/LLM_MEMORY_WRITING_IMPLEMENTATION.md",
            "MEMORY_SYSTEM_DEPLOYED.md": "docs/status/MEMORY_SYSTEM_DEPLOYED.md",
            "MEMORY_SYSTEM_LAWSUIT_INTEGRATION_COMPLETE.md": "docs/status/MEMORY_SYSTEM_LAWSUIT_INTEGRATION_COMPLETE.md",
            "NLP_SYSTEM_SUMMARY.md": "docs/status/NLP_SYSTEM_SUMMARY.md",
            "REORGANIZATION_LOG.md": "docs/status/REORGANIZATION_LOG.md",
            "STRATEGIC_MODULES_IMPLEMENTATION_SUMMARY.md": "docs/status/STRATEGIC_MODULES_IMPLEMENTATION_SUMMARY.md",
            "WITCHWEB_CASELAW_STATUS.md": "docs/status/WITCHWEB_CASELAW_STATUS.md",
            "WITCHWEB_CHECKPOINT_SYSTEM_SUMMARY.md": "docs/status/WITCHWEB_CHECKPOINT_SYSTEM_SUMMARY.md",

            # Additional guide files
            "REVIT_3D_IMPORT_FEATURE.md": "docs/guides/REVIT_3D_IMPORT_FEATURE.md",
            "REVIT_EXTERNAL_AUTOMATION_ADDED.md": "docs/guides/REVIT_EXTERNAL_AUTOMATION_ADDED.md",

            # Runtime artifacts moved
            "analysis_outputs/": "reports/analysis_outputs/",
            "results/": "reports/results/",
            "memory_snapshots/": "reports/memory_snapshots/",
            "jobs.db": "db/jobs.db",
            "case_law_pipeline.log": "logs/case_law_pipeline.log",
            "compile_all.log": "logs/compile_all.log",
            "download_all_ma.log": "logs/download_all_ma.log",
        }

    def update_all_files(self, dry_run: bool = False, verbose: bool = False) -> bool:
        """Update links in all markdown files."""
        success = True
        md_files = list(self.repo_root.rglob("*.md"))

        if verbose:
            print(f"Found {len(md_files)} markdown files to process")

        for md_file in md_files:
            try:
                file_success = self._update_file(md_file, dry_run, verbose)
                if not file_success:
                    success = False
            except Exception as e:
                error_msg = f"Error processing {md_file}: {str(e)}"
                self.errors.append(error_msg)
                print(f"❌ {error_msg}")
                success = False

        return success

    def update_single_file(self, file_path: str, dry_run: bool = False, verbose: bool = False) -> bool:
        """Update links in a single markdown file."""
        md_file = self.repo_root / file_path
        if not md_file.exists():
            print(f"❌ File not found: {file_path}")
            return False

        return self._update_file(md_file, dry_run, verbose)

    def _update_file(self, md_file: Path, dry_run: bool, verbose: bool) -> bool:
        """Update links in a single markdown file."""
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            changes_made = False

            # Update markdown links [text](path) and [text](path#anchor)
            for old_path, new_path in self.link_mappings.items():
                # Pattern for markdown links: [text](path) or [text](path#anchor)
                pattern = rf'(\[([^\]]+)\]\([^)]*{re.escape(old_path)}([^)]*)\)'

                def replace_link(match):
                    nonlocal changes_made
                    link_text = match.group(2)
                    anchor = match.group(3) if match.group(3) else ""
                    new_link = f"[{link_text}]({new_path}{anchor})"
                    changes_made = True
                    return new_link

                new_content = re.sub(pattern, replace_link, content)
                if new_content != content:
                    content = new_content
                    if verbose:
                        print(f"  🔗 Updated link: {old_path} → {new_path}")

            # Update relative path references
            for old_path, new_path in self.link_mappings.items():
                if old_path.endswith('/'):
                    # Directory references
                    pattern = rf'(\[([^\]]+)\]\([^)]*{re.escape(old_path)}([^)]*)\)'

                    def replace_dir_link(match):
                        nonlocal changes_made
                        link_text = match.group(2)
                        anchor = match.group(3) if match.group(3) else ""
                        new_link = f"[{link_text}]({new_path}{anchor})"
                        changes_made = True
                        return new_link

                    new_content = re.sub(pattern, replace_dir_link, content)
                    if new_content != content:
                        content = new_content
                        if verbose:
                            print(f"  📁 Updated directory link: {old_path} → {new_path}")

            if changes_made:
                if not dry_run:
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.updated_files.append(str(md_file.relative_to(self.repo_root)))

                if verbose:
                    print(f"✅ Updated: {md_file.relative_to(self.repo_root)}")

            return True

        except Exception as e:
            error_msg = f"Error updating {md_file}: {str(e)}"
            self.errors.append(error_msg)
            print(f"❌ {error_msg}")
            return False

    def generate_report(self) -> str:
        """Generate a report of the update operation."""
        report = f"""# Link Update Report

**Generated**: {datetime.now().isoformat()}
**Files Updated**: {len(self.updated_files)}
**Errors**: {len(self.errors)}

## Updated Files

"""

        for file_path in self.updated_files:
            report += f"- {file_path}\n"

        if self.errors:
            report += "\n## Errors\n\n"
            for error in self.errors:
                report += f"- {error}\n"

        return report


def main():
    parser = argparse.ArgumentParser(description="Update markdown links after repository reorganization")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--file", help="Update only the specified file")

    args = parser.parse_args()

    try:
        updater = LinkUpdater()

        print("🔗 Starting link update...")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")

        if args.file:
            success = updater.update_single_file(args.file, dry_run=args.dry_run, verbose=args.verbose)
        else:
            success = updater.update_all_files(dry_run=args.dry_run, verbose=args.verbose)

        if success:
            print(f"\n✅ Link update completed!")
            print(f"📋 Files updated: {len(updater.updated_files)}")
            if updater.errors:
                print(f"⚠️  Errors: {len(updater.errors)}")

            # Save report
            if not args.dry_run:
                report_file = Path("scripts/link_update_report.md")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(updater.generate_report())
                print(f"📄 Report saved to: {report_file}")
        else:
            print("\n⚠️  Link update completed with errors")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
