#!/usr/bin/env python3
"""
Recent Activity Digest

Generates a concise summary of recent work across the repository for agents to
consume. It aggregates:
- Recent git commits
- Recently modified files
- Background agent logs (if available)
- Recent agent/report outputs

Usage:
  python scripts/recent_activity_digest.py \
      --days 7 \
      --max-files 50 \
      --output-md reports/analysis_outputs/activity_digest.md \
      --output-json reports/analysis_outputs/activity_digest.json

Notes:
- Uses pathlib and avoids absolute paths.
- Degrades gracefully when git or logs are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------- Data Models ----------

@dataclass
class Commit:
    hash: str
    date: str
    author: str
    subject: str


@dataclass
class FileEntry:
    path: str
    mtime: str
    size_bytes: int


@dataclass
class LogEntry:
    timestamp: str
    logger: str
    level: str
    message: str


@dataclass
class ActivityDigest:
    generated_at: str
    window_days: int
    git_commits: List[Commit]
    recent_files: List[FileEntry]
    background_agent_logs: List[LogEntry]
    recent_outputs: List[FileEntry]


# ---------- Helpers ----------

def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a shell command and return (code, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", str(e)


def get_recent_git_commits(days: int, max_count: int = 50) -> List[Commit]:
    since = f"{days} days ago"
    fmt = "%h|%ad|%an|%s"
    code, out, _ = run([
        "git", "log", f"--since={since}", f"--max-count={max_count}",
        f"--pretty=format:{fmt}", "--date=iso"
    ])
    commits: List[Commit] = []
    if code != 0 or not out.strip():
        return commits
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        commits.append(Commit(hash=parts[0], date=parts[1], author=parts[2], subject=parts[3]))
    return commits


def list_recent_files(
    roots: List[Path],
    cutoff: datetime,
    max_files: int = 50,
    include_extensions: Optional[List[str]] = None,
    ignore_dirs: Optional[List[str]] = None,
) -> List[FileEntry]:
    include_ext = set((x.lower() for x in (include_extensions or [])))
    ignore_dirnames = set(ignore_dirs or [])

    entries: List[Tuple[float, FileEntry]] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            try:
                if p.is_dir():
                    # Skip ignored directories by name match
                    if p.name in ignore_dirnames:
                        # Skip walking children by continuing — Path.rglob doesn't allow prune,
                        # but we can at least avoid recording files from known dirs below
                        pass
                    continue
                if include_ext:
                    if p.suffix.lower() not in include_ext:
                        continue
                stat = p.stat()
                mtime_dt = datetime.fromtimestamp(stat.st_mtime)
                if mtime_dt < cutoff:
                    continue
                entry = FileEntry(path=str(p), mtime=mtime_dt.isoformat(timespec="seconds"), size_bytes=stat.st_size)
                entries.append((stat.st_mtime, entry))
            except (OSError, PermissionError):
                continue

    # Sort by most recent and trim
    entries.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in entries[:max_files]]


def parse_background_agent_logs(log_path: Path, cutoff: datetime, max_lines: int = 5000) -> List[LogEntry]:
    if not log_path.exists():
        return []
    entries: List[LogEntry] = []
    try:
        # Read tail-ish without loading massive files; this log is modest by design
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    for line in text.splitlines()[-max_lines:]:
        # Expected format: YYYY-MM-DD HH:MM:SS,ms - logger - LEVEL - message
        try:
            ts_str, rest = line.split(" - ", 1)
            # Remove comma millis for parsing
            ts_main = ts_str.split(",")[0]
            ts = datetime.strptime(ts_main, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue
        if ts < cutoff:
            continue
        # Split remaining
        parts = rest.split(" - ")
        if len(parts) < 3:
            continue
        logger_name = parts[0].strip()
        level = parts[1].strip()
        message = " - ".join(parts[2:]).strip()
        entries.append(LogEntry(timestamp=ts.isoformat(timespec="seconds"), logger=logger_name, level=level, message=message))
    return entries


def build_markdown(digest: ActivityDigest) -> str:
    lines: List[str] = []
    lines.append("# Activity Digest")
    lines.append("")
    lines.append(f"- Generated: {digest.generated_at}")
    lines.append(f"- Window: last {digest.window_days} days")
    lines.append("")

    # Commits
    lines.append("## Recent Commits")
    if digest.git_commits:
        for c in digest.git_commits[:20]:
            lines.append(f"- {c.date} [{c.hash}] {c.author}: {c.subject}")
    else:
        lines.append("- (no commits found in window or git unavailable)")
    lines.append("")

    # Files
    lines.append("## Recently Modified Files")
    if digest.recent_files:
        for f in digest.recent_files:
            lines.append(f"- {f.mtime} {f.path} ({f.size_bytes} bytes)")
    else:
        lines.append("- (no recent files in window)")
    lines.append("")

    # Agent Outputs
    lines.append("## Agent & Report Outputs")
    if digest.recent_outputs:
        for f in digest.recent_outputs:
            lines.append(f"- {f.mtime} {f.path} ({f.size_bytes} bytes)")
    else:
        lines.append("- (no recent outputs in window)")
    lines.append("")

    # Background Agent Logs
    lines.append("## Background Agent Logs")
    if digest.background_agent_logs:
        for e in digest.background_agent_logs[:50]:
            lines.append(f"- {e.timestamp} {e.logger} {e.level}: {e.message}")
        if len(digest.background_agent_logs) > 50:
            lines.append(f"- ... ({len(digest.background_agent_logs) - 50} more)")
    else:
        lines.append("- (no recent log entries or logs unavailable)")
    lines.append("")
    
    # Init Context Documentation
    init_context_index = Path("docs/init_context/INDEX.md")
    if init_context_index.exists():
        lines.append("## Init Context Documentation")
        lines.append(f"- Master index: [docs/init_context/INDEX.md](docs/init_context/INDEX.md)")
        lines.append("- Recent init sessions are documented at module and focus area levels")
        lines.append("- See index for links to all init context documentation")
        lines.append("")
    
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a recent activity digest for agents.")
    parser.add_argument("--days", type=int, default=7, help="Look back window in days (default: 7)")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum recent files to list (default: 50)")
    parser.add_argument("--output-md", type=str, default="reports/analysis_outputs/activity_digest.md", help="Output Markdown path")
    parser.add_argument("--output-json", type=str, default="reports/analysis_outputs/activity_digest.json", help="Output JSON path")
    args = parser.parse_args()

    cutoff = datetime.now() - timedelta(days=args.days)

    # Collect commits
    commits = get_recent_git_commits(args.days, max_count=100)

    # Select roots that are most relevant to agents and outputs
    roots = [
        Path("."),
        Path("background_agents"),
        Path("writer_agents"),
        Path("autogen_integration"),
        Path("reports"),
        Path("scripts"),
    ]

    # Recent files (code and configs)
    recent_files = list_recent_files(
        roots=roots,
        cutoff=cutoff,
        max_files=args.max_files,
        include_extensions=[
            ".py", ".md", ".json", ".txt", ".sql", ".yaml", ".yml",
            ".pdf",
        ],
        ignore_dirs=[".venv", "__pycache__", "node_modules", "mlruns", "tools"],
    )

    # Background agent logs
    log_path = Path("background_agents/logs/system.log")
    log_entries = parse_background_agent_logs(log_path, cutoff)

    # Outputs: focus on background_agents/outputs and reports/
    outputs = list_recent_files(
        roots=[Path("background_agents/outputs"), Path("reports")],
        cutoff=cutoff,
        max_files=max(100, args.max_files),
        include_extensions=[".json", ".md", ".txt", ".pdf"],
        ignore_dirs=[".venv", "__pycache__", "tools"],
    )
    
    # Init context documentation
    init_context_files = list_recent_files(
        roots=[Path("docs/init_context")],
        cutoff=cutoff,
        max_files=20,
        include_extensions=[".md"],
        ignore_dirs=[".venv", "__pycache__", "tools"],
    )
    outputs.extend(init_context_files)

    digest = ActivityDigest(
        generated_at=_now_iso(),
        window_days=args.days,
        git_commits=commits,
        recent_files=recent_files,
        background_agent_logs=log_entries,
        recent_outputs=outputs,
    )

    # Ensure output directories exist
    md_path = Path(args.output_md)
    json_path = Path(args.output_json)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path.write_text(json.dumps(asdict(digest), indent=2), encoding="utf-8")

    # Write Markdown
    md = build_markdown(digest)
    md_path.write_text(md, encoding="utf-8")

    # Print a brief pointer for CLI users
    print(f"Activity digest written to:\n  - {md_path}\n  - {json_path}")


if __name__ == "__main__":
    main()

