#!/usr/bin/env python3
"""
Schedule Cursor Auto-Capture
=============================

Sets up automatic Cursor chat capture using Windows Task Scheduler.

Usage:
    python schedule_cursor_capture.py --install    # Install scheduled task
    python schedule_cursor_capture.py --remove     # Remove scheduled task
    python schedule_cursor_capture.py --status     # Check status
"""

import subprocess
import sys
import os
from pathlib import Path


def get_python_exe():
    """Get path to Python executable"""
    return sys.executable


def get_script_path():
    """Get path to working capture script"""
    script_dir = Path(__file__).parent
    return script_dir / "cursor_capture_working.py"


def create_task_xml(interval_minutes: int = 5):
    """Create Task Scheduler XML"""
    python_exe = get_python_exe()
    script_path = get_script_path()
    work_dir = Path(__file__).parent.parent

    xml = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Date>2025-01-01T00:00:00</Date>
    <Author>Cursor Auto-Capture</Author>
    <Description>Automatically captures Cursor chat conversations</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <Repetition>
        <Interval>PT{interval_minutes}M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
      <StartBoundary>2025-01-01T00:00:00</StartBoundary>
      <Enabled>true</Enabled>
    </TimeTrigger>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT15M</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{python_exe}"</Command>
      <Arguments>"{script_path}"</Arguments>
      <WorkingDirectory>{work_dir}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''

    return xml


def install_task(interval_minutes: int = 5):
    """Install scheduled task"""
    print("[INSTALL] Installing Cursor Auto-Capture scheduled task...")

    xml = create_task_xml(interval_minutes)
    xml_file = Path("data/cursor_capture_task.xml")
    xml_file.parent.mkdir(parents=True, exist_ok=True)

    with open(xml_file, 'w') as f:
        f.write(xml)

    try:
        # Delete existing task if it exists
        subprocess.run(
            ['schtasks', '/Delete', '/TN', 'CursorAutoCapture', '/F'],
            capture_output=True,
            check=False
        )

        # Create new task
        result = subprocess.run(
            ['schtasks', '/Create', '/XML', str(xml_file), '/TN', 'CursorAutoCapture'],
            capture_output=True,
            text=True,
            check=True
        )

        print("[SUCCESS] Task installed successfully!")
        print(f"   Task Name: CursorAutoCapture")
        print(f"   Interval: Every {interval_minutes} minutes")
        print(f"   Python: {get_python_exe()}")
        print(f"   Script: {get_script_path()}")
        print("\n[INFO] Next Steps:")
        print("   1. Open Task Scheduler to verify:")
        print("      Open 'Task Scheduler' and find 'CursorAutoCapture'")
        print("   2. Run a manual test:")
        print(f"      python {get_script_path()} --sync --test-run")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install task: {e}")
        print("\n[MANUAL] Manual Installation:")
        print(f"1. Open Task Scheduler")
        print(f"2. Create Basic Task named 'CursorAutoCapture'")
        print(f"3. Set trigger: Repeat every {interval_minutes} minutes")
        print(f"4. Action: Start a program")
        print(f"   Program: {get_python_exe()}")
        print(f"   Arguments: \"{get_script_path()}\"")
        print(f"   Start in: {get_script_path().parent}")
        return False

    return True


def remove_task():
    """Remove scheduled task"""
    print("[REMOVE] Removing Cursor Auto-Capture scheduled task...")

    try:
        result = subprocess.run(
            ['schtasks', '/Delete', '/TN', 'CursorAutoCapture', '/F'],
            capture_output=True,
            text=True,
            check=True
        )

        print("[SUCCESS] Task removed successfully!")

    except subprocess.CalledProcessError as e:
        if "does not exist" in e.stdout.lower():
            print("[INFO] Task does not exist")
        else:
            print(f"[ERROR] Failed to remove task: {e}")
            return False

    return True


def check_status():
    """Check if task is installed"""
    print("[STATUS] Checking Cursor Auto-Capture status...")

    try:
        result = subprocess.run(
            ['schtasks', '/Query', '/TN', 'CursorAutoCapture', '/FO', 'LIST'],
            capture_output=True,
            text=True,
            check=True
        )

        print("[OK] Task is installed and running")
        print("\n[DETAILS] Task Details:")
        print(result.stdout)

        print("\n[INFO] Commands:")
        print("  Run manually: schtasks /Run /TN CursorAutoCapture")
        print("  View history: schtasks /Query /TN CursorAutoCapture /V /FO LIST")

    except subprocess.CalledProcessError as e:
        print("[ERROR] Task is not installed")
        print("\n[INSTALL] To install:")
        print("  python schedule_cursor_capture.py --install")
        return False

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Schedule Cursor auto-capture')
    parser.add_argument('--install', action='store_true',
                       help='Install scheduled task')
    parser.add_argument('--remove', action='store_true',
                       help='Remove scheduled task')
    parser.add_argument('--status', action='store_true',
                       help='Check task status')
    parser.add_argument('--interval', type=int, default=5,
                       help='Check interval in minutes (default: 5)')

    args = parser.parse_args()

    if args.install:
        install_task(interval_minutes=args.interval)

    elif args.remove:
        remove_task()

    elif args.status:
        check_status()

    else:
        print("Cursor Auto-Capture Scheduler")
        print("=" * 50)
        print("\nUsage:")
        print("  --install    Install scheduled task")
        print("  --remove     Remove scheduled task")
        print("  --status     Check task status")
        print("\nExample:")
        print("  python schedule_cursor_capture.py --install --interval 10")


if __name__ == "__main__":
    main()

