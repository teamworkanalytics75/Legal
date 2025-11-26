#!/usr/bin/env python3
"""
Daily Cursor Tracker - Simple Daily Usage
==========================================

A simple script to quickly start/end sessions and track daily productivity.

Usage:
    python daily_tracker.py start "Project Name"
    python daily_tracker.py end
    python daily_tracker.py question "Your question here"
    python daily_tracker.py report
"""

import sys
import os
from datetime import datetime
from cursor_tracker import CursorTracker

def main():
    """Main function for daily tracking"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python daily_tracker.py start 'Project Name'")
        print("  python daily_tracker.py end")
        print("  python daily_tracker.py question 'Your question'")
        print("  python daily_tracker.py problem 'Problem solved'")
        print("  python daily_tracker.py features 2")
        print("  python daily_tracker.py bugs 1")
        print("  python daily_tracker.py report")
        print("  python daily_tracker.py status")
        return

    tracker = CursorTracker()
    command = sys.argv[1].lower()

    if command == "start":
        if len(sys.argv) < 3:
            project = "General Development"
        else:
            project = sys.argv[2]
        tracker.start_session(project)

    elif command == "end":
        tracker.end_session()

    elif command == "question":
        if len(sys.argv) < 3:
            print("Please provide a question")
            return
        question = sys.argv[2]
        tracker.add_question(question)

    elif command == "problem":
        if len(sys.argv) < 3:
            print("Please provide a problem description")
            return
        problem = sys.argv[2]
        tracker.add_problem_solved(problem)

    elif command == "features":
        if len(sys.argv) < 3:
            count = 1
        else:
            try:
                count = int(sys.argv[2])
            except ValueError:
                count = 1
        tracker.increment_features(count)

    elif command == "bugs":
        if len(sys.argv) < 3:
            count = 1
        else:
            try:
                count = int(sys.argv[2])
            except ValueError:
                count = 1
        tracker.increment_bugs_fixed(count)

    elif command == "report":
        report = tracker.daily_report()
        print(report)

    elif command == "status":
        # Show current session status
        active_session = None
        for session in reversed(tracker.data['sessions']):
            if session['end_time'] is None:
                active_session = session
                break

        if active_session:
            start_time = datetime.fromisoformat(active_session['start_time'])
            duration = (datetime.now() - start_time).total_seconds() / 60

            print(f"Active Session: {active_session['session_id']}")
            print(f"Project: {active_session['project_context']}")
            print(f"Started: {start_time.strftime('%H:%M')}")
            print(f"Duration: {duration:.1f} minutes")
            print(f"Questions: {len(active_session['questions'])}")
            print(f"Features: {active_session['features_implemented']}")
            print(f"Bugs Fixed: {active_session['bugs_fixed']}")
        else:
            print("No active session")

    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()

