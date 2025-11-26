#!/usr/bin/env python3
"""
Cursor Chat Tracker - Simple Version
====================================

A simplified tool to start tracking your Cursor usage immediately.
This version creates mock data for testing and can be extended to work with real Cursor data.

Usage:
    python cursor_tracker.py --start-session
    python cursor_tracker.py --end-session
    python cursor_tracker.py --daily-report
    python cursor_tracker.py --methodology
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import argparse

class CursorTracker:
    """Simple tracker for Cursor usage"""

    def __init__(self, data_file: str = "cursor_tracking.json"):
        self.data_file = data_file
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load tracking data from file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "sessions": [],
                "questions": [],
                "metrics": {},
                "created": datetime.now().isoformat()
            }

    def _save_data(self):
        """Save tracking data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

    def start_session(self, project_context: str = "General Development"):
        """Start a new coding session"""
        session_id = f"session_{len(self.data['sessions']) + 1}"
        session = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_minutes": 0,
            "project_context": project_context,
            "questions": [],
            "problems_solved": [],
            "features_implemented": 0,
            "bugs_fixed": 0,
            "code_generated": False,
            "documentation_created": False
        }

        self.data['sessions'].append(session)
        self._save_data()

        print(f"Started session: {session_id}")
        print(f"Project: {project_context}")
        print(f"Start time: {session['start_time']}")
        return session_id

    def end_session(self, session_id: str = None):
        """End the current or specified session"""
        if not session_id:
            # Find the most recent session without an end time
            for session in reversed(self.data['sessions']):
                if session['end_time'] is None:
                    session_id = session['session_id']
                    break

        if not session_id:
            print("No active session found")
            return

        # Find and update the session
        for session in self.data['sessions']:
            if session['session_id'] == session_id:
                session['end_time'] = datetime.now().isoformat()

                # Calculate duration
                start_time = datetime.fromisoformat(session['start_time'])
                end_time = datetime.fromisoformat(session['end_time'])
                duration = (end_time - start_time).total_seconds() / 60
                session['duration_minutes'] = duration

                self._save_data()

                print(f"Ended session: {session_id}")
                print(f"Duration: {duration:.1f} minutes")
                print(f"Questions asked: {len(session['questions'])}")
                print(f"Features implemented: {session['features_implemented']}")
                print(f"Bugs fixed: {session['bugs_fixed']}")
                return

        print(f"Session {session_id} not found")

    def add_question(self, question: str, category: str = "general", complexity: str = "medium"):
        """Add a question to the current session"""
        # Find the most recent active session
        active_session = None
        for session in reversed(self.data['sessions']):
            if session['end_time'] is None:
                active_session = session
                break

        if not active_session:
            print("No active session. Start a session first.")
            return

        question_data = {
            "question": question,
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "complexity": complexity
        }

        active_session['questions'].append(question_data)
        self.data['questions'].append(question_data)
        self._save_data()

        print(f"Added question: {question[:50]}...")

    def add_problem_solved(self, problem: str, solution: str = ""):
        """Add a problem solved to the current session"""
        active_session = None
        for session in reversed(self.data['sessions']):
            if session['end_time'] is None:
                active_session = session
                break

        if not active_session:
            print("No active session. Start a session first.")
            return

        problem_data = {
            "problem": problem,
            "solution": solution,
            "timestamp": datetime.now().isoformat()
        }

        active_session['problems_solved'].append(problem_data)
        self._save_data()

        print(f"Added problem solved: {problem[:50]}...")

    def increment_features(self, count: int = 1):
        """Increment features implemented count"""
        active_session = None
        for session in reversed(self.data['sessions']):
            if session['end_time'] is None:
                active_session = session
                break

        if not active_session:
            print("No active session. Start a session first.")
            return

        active_session['features_implemented'] += count
        active_session['code_generated'] = True
        self._save_data()

        print(f"Added {count} feature(s). Total: {active_session['features_implemented']}")

    def increment_bugs_fixed(self, count: int = 1):
        """Increment bugs fixed count"""
        active_session = None
        for session in reversed(self.data['sessions']):
            if session['end_time'] is None:
                active_session = session
                break

        if not active_session:
            print("No active session. Start a session first.")
            return

        active_session['bugs_fixed'] += count
        self._save_data()

        print(f"Fixed {count} bug(s). Total: {active_session['bugs_fixed']}")

    def calculate_metrics(self):
        """Calculate productivity metrics"""
        sessions = [s for s in self.data['sessions'] if s['end_time'] is not None]

        if not sessions:
            print("No completed sessions found")
            return

        total_time = sum(s['duration_minutes'] for s in sessions) / 60
        total_questions = sum(len(s['questions']) for s in sessions)
        total_features = sum(s['features_implemented'] for s in sessions)
        total_bugs = sum(s['bugs_fixed'] for s in sessions)

        # Question categories
        categories = {}
        for question in self.data['questions']:
            cat = question['category']
            categories[cat] = categories.get(cat, 0) + 1

        # Most productive hours
        hour_counts = {}
        for session in sessions:
            start_time = datetime.fromisoformat(session['start_time'])
            hour = start_time.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        most_productive_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        metrics = {
            "total_sessions": len(sessions),
            "total_time_hours": total_time,
            "total_questions": total_questions,
            "questions_per_hour": total_questions / total_time if total_time > 0 else 0,
            "average_session_duration": total_time / len(sessions),
            "total_features": total_features,
            "total_bugs_fixed": total_bugs,
            "features_per_hour": total_features / total_time if total_time > 0 else 0,
            "bugs_per_hour": total_bugs / total_time if total_time > 0 else 0,
            "question_categories": categories,
            "most_productive_hours": [h[0] for h in most_productive_hours]
        }

        self.data['metrics'] = metrics
        self._save_data()

        return metrics

    def daily_report(self, date: datetime = None):
        """Generate daily productivity report"""
        if not date:
            date = datetime.now()

        date_str = date.date().isoformat()

        # Filter sessions for the day
        day_sessions = []
        for session in self.data['sessions']:
            if session['end_time']:
                start_time = datetime.fromisoformat(session['start_time'])
                if start_time.date() == date.date():
                    day_sessions.append(session)

        if not day_sessions:
            return f"No sessions found for {date_str}"

        total_time = sum(s['duration_minutes'] for s in day_sessions) / 60
        total_questions = sum(len(s['questions']) for s in day_sessions)
        total_features = sum(s['features_implemented'] for s in day_sessions)
        total_bugs = sum(s['bugs_fixed'] for s in day_sessions)

        report = f"""
# Daily Productivity Report - {date_str}

## Summary
- **Total Sessions**: {len(day_sessions)}
- **Total Time**: {total_time:.1f} hours
- **Total Questions**: {total_questions}
- **Questions per Hour**: {total_questions/total_time:.1f}
- **Features Implemented**: {total_features}
- **Bugs Fixed**: {total_bugs}
- **Features per Hour**: {total_features/total_time:.1f}

## Session Breakdown
"""

        for i, session in enumerate(day_sessions, 1):
            start_time = datetime.fromisoformat(session['start_time'])
            end_time = datetime.fromisoformat(session['end_time'])

            report += f"""
### Session {i}: {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}
- **Duration**: {session['duration_minutes']:.1f} minutes
- **Questions**: {len(session['questions'])}
- **Project**: {session['project_context']}
- **Features**: {session['features_implemented']}
- **Bugs Fixed**: {session['bugs_fixed']}
- **Code Generated**: {'Yes' if session['code_generated'] else 'No'}
- **Documentation**: {'Yes' if session['documentation_created'] else 'No'}
"""

        return report

    def methodology_documentation(self):
        """Generate methodology documentation"""
        metrics = self.calculate_metrics()

        if not metrics:
            return "No metrics available. Complete some sessions first."

        doc = f"""
# AI-Assisted Development Methodology
## Based on {metrics['total_sessions']} coding sessions

## Overview
This methodology is based on {metrics['total_time_hours']:.1f} hours of AI-assisted development,
resulting in {metrics['total_questions']} questions and significant productivity gains.

## Key Productivity Metrics
- **Average Session Duration**: {metrics['average_session_duration']:.1f} hours
- **Questions per Hour**: {metrics['questions_per_hour']:.1f}
- **Features per Hour**: {metrics['features_per_hour']:.1f}
- **Bugs Fixed per Hour**: {metrics['bugs_per_hour']:.1f}
- **Most Productive Hours**: {', '.join(map(str, metrics['most_productive_hours']))}

## Question Categories (Professional Roles Covered)
"""

        for category, count in metrics['question_categories'].items():
            doc += f"- **{category.replace('_', ' ').title()}**: {count} questions\n"

        doc += f"""
## Best Practices for Team Implementation

### 1. Question Strategy
- Ask specific, focused questions
- Include code examples when possible
- Break complex problems into smaller parts
- Follow up with clarification questions

### 2. Session Management
- Work in focused sessions of {metrics['average_session_duration']:.1f} hours
- Take breaks between sessions
- Document solutions immediately
- Track time and questions

### 3. Learning Approach
- Start with simple questions, progress to complex
- Cover all professional roles (senior, mid, junior, QA, security, devops, product)
- Focus on hands-on implementation
- Learn through building, not just reading

### 4. Productivity Optimization
- Work during most productive hours: {', '.join(map(str, metrics['most_productive_hours']))}
- Maintain consistent daily coding practice
- Use AI assistance for all aspects of development
- Track and analyze your patterns

## Implementation Checklist for Team Members
- [ ] Set up Cursor with AI assistance
- [ ] Start tracking daily coding sessions
- [ ] Ask questions covering all professional roles
- [ ] Document solutions and methodology
- [ ] Analyze productivity patterns weekly
- [ ] Share insights with team

## Sample Questions by Category

### Senior Developer (Architecture)
- "What's the best architecture for this system?"
- "How should I structure this complex feature?"
- "What design patterns should I use here?"

### Mid-Level Developer (Implementation)
- "How do I implement this specific feature?"
- "What's the best way to integrate these APIs?"
- "How do I handle this data processing?"

### Junior Developer (Basics)
- "How do I fix this error?"
- "What's wrong with my code?"
- "How do I implement this simple feature?"

### QA/Testing
- "How do I test this thoroughly?"
- "What edge cases should I consider?"
- "How do I ensure this works under load?"

### Security
- "Is this secure?"
- "How do I handle sensitive data?"
- "What are the security implications?"

### DevOps
- "How do I deploy this?"
- "What about error handling?"
- "How do I handle production issues?"

### Product Management
- "What's the user experience like?"
- "How do users actually use this?"
- "What features are most important?"
"""

        return doc

    def create_sample_data(self):
        """Create sample data for testing"""
        print("Creating sample data for testing...")

        # Create sample sessions
        base_time = datetime.now() - timedelta(days=7)

        sample_sessions = [
            {
                "session_id": "session_1",
                "start_time": (base_time + timedelta(hours=9)).isoformat(),
                "end_time": (base_time + timedelta(hours=11)).isoformat(),
                "duration_minutes": 120,
                "project_context": "Matrix Legal AI System",
                "questions": [
                    {"question": "How do I structure a multi-agent system?", "timestamp": (base_time + timedelta(hours=9, minutes=15)).isoformat(), "category": "senior_dev", "complexity": "complex"},
                    {"question": "How do I implement Bayesian network integration?", "timestamp": (base_time + timedelta(hours=9, minutes=45)).isoformat(), "category": "mid_dev", "complexity": "medium"},
                    {"question": "How do I fix this SQLite error?", "timestamp": (base_time + timedelta(hours=10, minutes=30)).isoformat(), "category": "junior_dev", "complexity": "simple"}
                ],
                "problems_solved": [
                    {"problem": "Database connection issues", "solution": "Fixed connection string", "timestamp": (base_time + timedelta(hours=10, minutes=15)).isoformat()}
                ],
                "features_implemented": 2,
                "bugs_fixed": 1,
                "code_generated": True,
                "documentation_created": True
            },
            {
                "session_id": "session_2",
                "start_time": (base_time + timedelta(days=1, hours=14)).isoformat(),
                "end_time": (base_time + timedelta(days=1, hours=16)).isoformat(),
                "duration_minutes": 120,
                "project_context": "Vida DataHub Financial System",
                "questions": [
                    {"question": "How do I integrate Shopify API?", "timestamp": (base_time + timedelta(days=1, hours=14, minutes=10)).isoformat(), "category": "mid_dev", "complexity": "medium"},
                    {"question": "How do I parse PDF bank statements?", "timestamp": (base_time + timedelta(days=1, hours=14, minutes=45)).isoformat(), "category": "mid_dev", "complexity": "complex"},
                    {"question": "How do I test this API integration?", "timestamp": (base_time + timedelta(days=1, hours=15, minutes=30)).isoformat(), "category": "qa", "complexity": "medium"}
                ],
                "problems_solved": [
                    {"problem": "PDF parsing errors", "solution": "Used PyPDF2 library", "timestamp": (base_time + timedelta(days=1, hours=15)).isoformat()}
                ],
                "features_implemented": 3,
                "bugs_fixed": 2,
                "code_generated": True,
                "documentation_created": False
            }
        ]

        self.data['sessions'] = sample_sessions

        # Extract all questions
        all_questions = []
        for session in sample_sessions:
            all_questions.extend(session['questions'])
        self.data['questions'] = all_questions

        self._save_data()
        print("Sample data created successfully!")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Track Cursor usage and productivity')
    parser.add_argument('--start-session', type=str, help='Start a new session with project context')
    parser.add_argument('--end-session', type=str, help='End session with ID')
    parser.add_argument('--add-question', type=str, help='Add a question to current session')
    parser.add_argument('--add-problem', type=str, help='Add a problem solved to current session')
    parser.add_argument('--add-features', type=int, help='Increment features implemented count')
    parser.add_argument('--add-bugs', type=int, help='Increment bugs fixed count')
    parser.add_argument('--daily-report', action='store_true', help='Generate daily report')
    parser.add_argument('--methodology', action='store_true', help='Generate methodology documentation')
    parser.add_argument('--sample-data', action='store_true', help='Create sample data for testing')
    parser.add_argument('--metrics', action='store_true', help='Show current metrics')

    args = parser.parse_args()

    tracker = CursorTracker()

    if args.sample_data:
        tracker.create_sample_data()

    if args.start_session:
        tracker.start_session(args.start_session)

    if args.end_session:
        tracker.end_session(args.end_session)

    if args.add_question:
        tracker.add_question(args.add_question)

    if args.add_problem:
        tracker.add_problem_solved(args.add_problem)

    if args.add_features:
        tracker.increment_features(args.add_features)

    if args.add_bugs:
        tracker.increment_bugs_fixed(args.add_bugs)

    if args.daily_report:
        report = tracker.daily_report()
        print(report)

    if args.methodology:
        doc = tracker.methodology_documentation()
        print(doc)

    if args.metrics:
        metrics = tracker.calculate_metrics()
        if metrics:
            print("Current Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()

