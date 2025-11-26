#!/usr/bin/env python3
"""
Export Sync Script for The Matrix Repository

This script synchronizes canonical source files to export bundles in dist/
to prevent drift between source and export versions.

Usage:
    python scripts/sync_exports.py [--dry-run] [--verbose]

Options:
    --dry-run    Show what would be done without making changes
    --verbose    Show detailed output of each operation
"""

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import argparse


class ExportSyncer:
    def __init__(self, config_path: str = "scripts/export_config.json"):
        self.config_path = Path(config_path)
        self.repo_root = Path(__file__).parent.parent
        self.dist_dir = self.repo_root / "dist"
        self.config = self._load_config()
        self.manifest = {
            "sync_time": datetime.now().isoformat(),
            "operations": [],
            "errors": []
        }

    def _load_config(self) -> Dict:
        """Load export configuration from JSON file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def sync_all(self, dry_run: bool = False, verbose: bool = False) -> bool:
        """Sync all configured export bundles."""
        success = True

        for bundle_name, bundle_config in self.config.items():
            if verbose:
                print(f"\n🔄 Syncing {bundle_name}...")

            try:
                bundle_success = self._sync_bundle(bundle_name, bundle_config, dry_run, verbose)
                if not bundle_success:
                    success = False
            except Exception as e:
                error_msg = f"Failed to sync {bundle_name}: {str(e)}"
                self.manifest["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                success = False

        # Save manifest
        if not dry_run:
            self._save_manifest()

        return success

    def _sync_bundle(self, bundle_name: str, bundle_config: Dict, dry_run: bool, verbose: bool) -> bool:
        """Sync a single export bundle."""
        bundle_dir = self.dist_dir / bundle_name
        sources = bundle_config.get("sources", {})

        # Create bundle directory
        if not dry_run:
            bundle_dir.mkdir(parents=True, exist_ok=True)

        success = True

        for category, file_list in sources.items():
            category_dir = bundle_dir / category
            if not dry_run:
                category_dir.mkdir(parents=True, exist_ok=True)

            for source_path in file_list:
                try:
                    source_file = self.repo_root / source_path
                    dest_file = category_dir / Path(source_path).name

                    if verbose:
                        print(f"  📄 {source_path} → {dest_file.relative_to(self.repo_root)}")

                    if not dry_run:
                        if source_file.exists():
                            shutil.copy2(source_file, dest_file)
                            self.manifest["operations"].append({
                                "action": "copy",
                                "source": str(source_path),
                                "destination": str(dest_file.relative_to(self.repo_root)),
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            error_msg = f"Source file not found: {source_path}"
                            self.manifest["errors"].append(error_msg)
                            print(f"⚠️  {error_msg}")
                            success = False
                    else:
                        if not source_file.exists():
                            print(f"⚠️  Source file not found: {source_path}")
                            success = False

                except Exception as e:
                    error_msg = f"Error processing {source_path}: {str(e)}"
                    self.manifest["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
                    success = False

        return success

    def _save_manifest(self):
        """Save sync manifest to dist directory."""
        manifest_file = self.dist_dir / "sync_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def generate_config_template(self):
        """Generate a template configuration file."""
        template = {
            "chatgpt_atomic_agents": {
                "sources": {
                    "Documentation": [
                        "ATOMIC_AGENT_ARCHITECTURE.md",
                        "ATOMIC_AGENT_IMPLEMENTATION_STATUS.md",
                        "ATOMIC_AGENTS_COMPLETE_SUMMARY.md",
                        "CHANGELOG.md",
                        "MODULE_INDEX.md",
                        "README.md"
                    ],
                    "Code/Infrastructure": [
                        "writer_agents/code/atomic_agent.py",
                        "writer_agents/code/job_persistence.py",
                        "writer_agents/code/master_supervisor.py",
                        "writer_agents/code/supervisors.py",
                        "writer_agents/code/task_dag.py",
                        "writer_agents/code/task_decomposer.py",
                        "writer_agents/code/worker_pool.py"
                    ],
                    "Code/Agents": [
                        "writer_agents/code/atomic_agents/citations.py",
                        "writer_agents/code/atomic_agents/drafting.py",
                        "writer_agents/code/atomic_agents/output.py",
                        "writer_agents/code/atomic_agents/research.py",
                        "writer_agents/code/atomic_agents/review.py"
                    ]
                }
            },
            "chatgpt_project_docs": {
                "sources": {
                    "Documentation": [
                        "PROJECT_DOCS/EXECUTIVE_SUMMARY.md",
                        "PROJECT_DOCS/SYSTEM_ARCHITECTURE.md",
                        "PROJECT_DOCS/TECHNICAL_REPORT.md",
                        "PROJECT_DOCS/COMPLETE_SYSTEM_INDEX.md"
                    ]
                }
            },
            "chatgpt_export": {
                "sources": {
                    "Consolidated": [
                        "README.md",
                        "CHANGELOG.md"
                    ]
                }
            }
        }

        template_file = self.repo_root / "scripts/export_config_template.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2)

        print(f"📝 Template created: {template_file}")


def main():
    parser = argparse.ArgumentParser(description="Sync canonical sources to export bundles")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--generate-template", action="store_true", help="Generate configuration template")

    args = parser.parse_args()

    try:
        syncer = ExportSyncer()

        if args.generate_template:
            syncer.generate_config_template()
            return

        if not syncer.config_path.exists():
            print(f"❌ Configuration file not found: {syncer.config_path}")
            print("Run with --generate-template to create a template")
            return 1

        print("🚀 Starting export sync...")
        if args.dry_run:
            print("🔍 DRY RUN MODE - No changes will be made")

        success = syncer.sync_all(dry_run=args.dry_run, verbose=args.verbose)

        if success:
            print("\n✅ Export sync completed successfully!")
            if not args.dry_run:
                print(f"📋 Manifest saved to: {syncer.dist_dir / 'sync_manifest.json'}")
        else:
            print("\n⚠️  Export sync completed with errors")
            return 1

        return 0

    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
