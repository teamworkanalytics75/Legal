#!/usr/bin/env python3
"""
Schedule Activity Digest

Runs the recent activity digest on a fixed interval, writing outputs to
reports/analysis_outputs/. Intended to be started alongside other
background agents or by a system scheduler.

Usage:
  python background_agents/schedule_activity_digest.py \
      --interval-hours 24 \
      --days-window 7
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def run_digest(days: int) -> None:
    md_out = Path("reports/analysis_outputs/activity_digest.md")
    json_out = Path("reports/analysis_outputs/activity_digest.json")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable or "python3",
        "scripts/recent_activity_digest.py",
        "--days", str(days),
        "--output-md", str(md_out),
        "--output-json", str(json_out),
    ]
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Generating activity digest...")
    subprocess.run(cmd, check=False)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Digest written to {md_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the activity digest on a schedule.")
    parser.add_argument("--interval-hours", type=float, default=24.0, help="Interval between runs in hours (default: 24)")
    parser.add_argument("--days-window", type=int, default=7, help="Lookback window in days (default: 7)")
    args = parser.parse_args()

    # First run immediately
    run_digest(args.days_window)

    # Loop forever
    sleep_s = max(60.0, args.interval_hours * 3600.0)
    while True:
        time.sleep(sleep_s)
        run_digest(args.days_window)


if __name__ == "__main__":
    main()

