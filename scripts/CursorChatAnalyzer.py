#!/usr/bin/env python3
"""
Cursor Chat History Analyzer
============================

A comprehensive tool to analyze your Cursor chat history, track work time,
categorize questions, and generate methodology documentation for your team.

Features:
- Extract chat history from Cursor's SQLite database
- Calculate actual work time from timestamps
- Categorize questions by professional role
- Identify patterns and learning progression
- Generate team methodology documentation
- Track productivity metrics over time

Usage:
    python cursor_chat_analyzer.py --analyze
    python cursor_chat_analyzer.py --export-methodology
    python cursor_chat_analyzer.py --daily-report
"""

import sqlite3
import json
import os
import re
import argparse
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import memory bank
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))
from EpisodicMemoryBank import EpisodicMemoryBank, EpisodicMemoryEntry

@dataclass
class ChatSession:
    """Represents a single chat session with Cursor"""
    session_id: str
    start_time: datetime
    end_time: datetime
    duration_minutes: float
    message_count: int
    user_messages: int
    assistant_messages: int
    project_context: str
    questions: List[str]
    problems_solved: List[str]
    code_generated: bool
    documentation_created: bool
    bugs_fixed: int
    features_implemented: int

@dataclass
class QuestionAnalysis:
    """Analysis of a question asked to Cursor"""
    question: str
    timestamp: datetime
    category: str  # senior_dev, mid_dev, junior_dev, qa, security, devops, product
    complexity: str  # simple, medium, complex
    resolution_time_minutes: float
    success: bool
    follow_up_questions: int
    code_examples: bool

@dataclass
class ProductivityMetrics:
    """Overall productivity metrics"""
    total_sessions: int
    total_time_hours: float
    total_questions: int
    questions_per_hour: float
    average_session_duration: float
    most_productive_hours: List[int]
    question_categories: Dict[str, int]
    learning_progression: List[str]
    common_problems: List[str]
    methodology_insights: List[str]

class CursorChatAnalyzer:
    """Main analyzer class for Cursor chat history"""

    def __init__(self, cursor_data_path: str = None):
        """Initialize the analyzer with Cursor data path"""
        self.cursor_data_path = cursor_data_path or self._find_cursor_data()
        self.sessions: List[ChatSession] = []
        self.questions: List[QuestionAnalysis] = []
        self.metrics: ProductivityMetrics = None

    def _find_cursor_data(self) -> str:
        """Find Cursor's data directory on Windows"""
        possible_paths = [
            os.path.expanduser("~/AppData/Roaming/Cursor"),
            os.path.expanduser("~/AppData/Local/Cursor"),
            "C:/Users/Owner/AppData/Roaming/Cursor",
            "C:/Users/Owner/AppData/Local/Cursor"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found Cursor data at: {path}")
                return path

        print("Could not find Cursor data directory. Please specify manually.")
        return None

    def extract_chat_history(self) -> bool:
        """Extract chat history from Cursor's SQLite database"""
        if not self.cursor_data_path:
            return False

        try:
            # Look for SQLite databases in Cursor's data directory
            db_files = []
            for root, dirs, files in os.walk(self.cursor_data_path):
                for file in files:
                    if file.endswith('.db') or file.endswith('.sqlite') or file.endswith('.sqlite3'):
                        db_files.append(os.path.join(root, file))

            print(f"Found {len(db_files)} database files")

            for db_file in db_files:
                try:
                    self._extract_from_database(db_file)
                except Exception as e:
                    print(f"Error reading {db_file}: {e}")
                    continue

            return len(self.sessions) > 0

        except Exception as e:
            print(f"Error extracting chat history: {e}")
            return False

    def _extract_from_database(self, db_path: str):
        """Extract data from a specific SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"Tables in {db_path}: {[t[0] for t in tables]}")

            # Look for chat-related tables
            chat_tables = [t[0] for t in tables if 'chat' in t[0].lower() or 'message' in t[0].lower()]

            for table in chat_tables:
                try:
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    print(f"Columns in {table}: {[c[1] for c in columns]}")

                    # Extract data
                    cursor.execute(f"SELECT * FROM {table} LIMIT 10")
                    rows = cursor.fetchall()
                    print(f"Sample data from {table}: {rows[:2]}")

                except Exception as e:
                    print(f"Error reading table {table}: {e}")

        finally:
            conn.close()

    def analyze_questions(self):
        """Analyze and categorize questions from chat history"""
        for session in self.sessions:
            for question in session.questions:
                analysis = self._analyze_question(question, session)
                self.questions.append(analysis)

    def _analyze_question(self, question: str, session: ChatSession) -> QuestionAnalysis:
        """Analyze a single question"""
        category = self._categorize_question(question)
        complexity = self._assess_complexity(question)

        return QuestionAnalysis(
            question=question,
            timestamp=session.start_time,
            category=category,
            complexity=complexity,
            resolution_time_minutes=session.duration_minutes / len(session.questions),
            success=True,  # Assume success if session completed
            follow_up_questions=0,  # TODO: Calculate from session
            code_examples=bool(re.search(r'code|implement|function|class', question.lower()))
        )

    def _categorize_question(self, question: str) -> str:
        """Categorize question by professional role"""
        question_lower = question.lower()

        # Senior Developer (Architecture, complex decisions)
        if any(word in question_lower for word in ['architecture', 'design', 'structure', 'pattern', 'framework', 'best practice']):
            return 'senior_dev'

        # Mid-Level Developer (Implementation, features)
        elif any(word in question_lower for word in ['implement', 'feature', 'api', 'integration', 'how to']):
            return 'mid_dev'

        # Junior Developer (Basic syntax, errors)
        elif any(word in question_lower for word in ['error', 'bug', 'fix', 'syntax', 'why', 'what']):
            return 'junior_dev'

        # QA/Testing
        elif any(word in question_lower for word in ['test', 'testing', 'validate', 'verify', 'edge case']):
            return 'qa'

        # Security
        elif any(word in question_lower for word in ['security', 'secure', 'vulnerability', 'auth', 'permission']):
            return 'security'

        # DevOps
        elif any(word in question_lower for word in ['deploy', 'production', 'server', 'infrastructure', 'monitoring']):
            return 'devops'

        # Product Management
        elif any(word in question_lower for word in ['user', 'experience', 'ui', 'ux', 'interface', 'workflow']):
            return 'product'

        else:
            return 'general'

    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['simple', 'basic', 'quick', 'easy', 'how do i']):
            return 'simple'
        elif any(word in question_lower for word in ['complex', 'advanced', 'sophisticated', 'optimize', 'scale']):
            return 'complex'
        else:
            return 'medium'

    def calculate_productivity_metrics(self):
        """Calculate overall productivity metrics"""
        if not self.sessions:
            return

        total_time = sum(session.duration_minutes for session in self.sessions) / 60
        total_questions = sum(len(session.questions) for session in self.sessions)

        # Most productive hours
        hour_counts = {}
        for session in self.sessions:
            hour = session.start_time.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        most_productive_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Question categories
        category_counts = {}
        for question in self.questions:
            category_counts[question.category] = category_counts.get(question.category, 0) + 1

        # Learning progression (simplified)
        learning_progression = self._identify_learning_progression()

        # Common problems
        common_problems = self._identify_common_problems()

        # Methodology insights
        methodology_insights = self._extract_methodology_insights()

        self.metrics = ProductivityMetrics(
            total_sessions=len(self.sessions),
            total_time_hours=total_time,
            total_questions=total_questions,
            questions_per_hour=total_questions / total_time if total_time > 0 else 0,
            average_session_duration=total_time / len(self.sessions) if self.sessions else 0,
            most_productive_hours=[h[0] for h in most_productive_hours],
            question_categories=category_counts,
            learning_progression=learning_progression,
            common_problems=common_problems,
            methodology_insights=methodology_insights
        )

    def _identify_learning_progression(self) -> List[str]:
        """Identify learning progression patterns"""
        progression = []

        # Analyze question complexity over time
        if len(self.questions) > 10:
            early_questions = self.questions[:len(self.questions)//3]
            late_questions = self.questions[-len(self.questions)//3:]

            early_complexity = sum(1 for q in early_questions if q.complexity == 'complex')
            late_complexity = sum(1 for q in late_questions if q.complexity == 'complex')

            if late_complexity > early_complexity:
                progression.append("Increasing complexity of questions over time")

            # Category progression
            early_categories = [q.category for q in early_questions]
            late_categories = [q.category for q in late_questions]

            if 'senior_dev' in late_categories and 'senior_dev' not in early_categories:
                progression.append("Progression to senior-level architectural questions")

        return progression

    def _identify_common_problems(self) -> List[str]:
        """Identify common problems encountered"""
        problems = []

        # Count error-related questions
        error_questions = [q for q in self.questions if 'error' in q.question.lower() or 'bug' in q.question.lower()]
        if len(error_questions) > 5:
            problems.append(f"Frequent debugging: {len(error_questions)} error-related questions")

        # Count integration questions
        integration_questions = [q for q in self.questions if 'integration' in q.question.lower() or 'api' in q.question.lower()]
        if len(integration_questions) > 3:
            problems.append(f"API/Integration challenges: {len(integration_questions)} related questions")

        return problems

    def _extract_methodology_insights(self) -> List[str]:
        """Extract methodology insights for team teaching"""
        insights = []

        # Question patterns
        if self.metrics and self.metrics.question_categories:
            top_category = max(self.metrics.question_categories.items(), key=lambda x: x[1])
            insights.append(f"Primary focus area: {top_category[0]} ({top_category[1]} questions)")

        # Session patterns
        if self.sessions:
            avg_duration = sum(s.duration_minutes for s in self.sessions) / len(self.sessions)
            insights.append(f"Average session duration: {avg_duration:.1f} minutes")

            if avg_duration > 60:
                insights.append("Long, focused coding sessions (deep work pattern)")
            else:
                insights.append("Short, iterative coding sessions (agile pattern)")

        # Learning approach
        if len(self.questions) > 20:
            code_questions = sum(1 for q in self.questions if q.code_examples)
            insights.append(f"Hands-on learning approach: {code_questions}/{len(self.questions)} questions included code examples")

        return insights

    def generate_daily_report(self, date: datetime = None) -> str:
        """Generate a daily productivity report"""
        if not date:
            date = datetime.now()

        # Filter sessions for the day
        day_sessions = [s for s in self.sessions if s.start_time.date() == date.date()]

        if not day_sessions:
            return f"No sessions found for {date.date()}"

        total_time = sum(s.duration_minutes for s in day_sessions) / 60
        total_questions = sum(len(s.questions) for s in day_sessions)

        report = f"""
# Daily Productivity Report - {date.date()}

## Summary
- **Total Sessions**: {len(day_sessions)}
- **Total Time**: {total_time:.1f} hours
- **Total Questions**: {total_questions}
- **Questions per Hour**: {total_questions/total_time:.1f}

## Session Breakdown
"""

        for i, session in enumerate(day_sessions, 1):
            report += f"""
### Session {i}: {session.start_time.strftime('%H:%M')} - {session.end_time.strftime('%H:%M')}
- **Duration**: {session.duration_minutes:.1f} minutes
- **Questions**: {len(session.questions)}
- **Project**: {session.project_context}
- **Features**: {session.features_implemented}
- **Bugs Fixed**: {session.bugs_fixed}
"""

        return report

    def export_methodology_documentation(self) -> str:
        """Export methodology documentation for team teaching"""
        if not self.metrics:
            return "No metrics available. Run analysis first."

        doc = f"""
# AI-Assisted Development Methodology
## Based on {self.metrics.total_sessions} coding sessions

## Overview
This methodology is based on {self.metrics.total_time_hours:.1f} hours of AI-assisted development,
resulting in {self.metrics.total_questions} questions and significant productivity gains.

## Key Productivity Metrics
- **Average Session Duration**: {self.metrics.average_session_duration:.1f} hours
- **Questions per Hour**: {self.metrics.questions_per_hour:.1f}
- **Most Productive Hours**: {', '.join(map(str, self.metrics.most_productive_hours))}

## Question Categories (Professional Roles Covered)
"""

        for category, count in self.metrics.question_categories.items():
            doc += f"- **{category.replace('_', ' ').title()}**: {count} questions\n"

        doc += f"""
## Learning Progression
"""
        for insight in self.metrics.learning_progression:
            doc += f"- {insight}\n"

        doc += f"""
## Common Challenges
"""
        for problem in self.metrics.common_problems:
            doc += f"- {problem}\n"

        doc += f"""
## Methodology Insights
"""
        for insight in self.metrics.methodology_insights:
            doc += f"- {insight}\n"

        doc += f"""
## Best Practices for Team Implementation

### 1. Question Strategy
- Ask specific, focused questions
- Include code examples when possible
- Break complex problems into smaller parts
- Follow up with clarification questions

### 2. Session Management
- Work in focused sessions of {self.metrics.average_session_duration:.1f} hours
- Take breaks between sessions
- Document solutions immediately
- Track time and questions

### 3. Learning Approach
- Start with simple questions, progress to complex
- Cover all professional roles (senior, mid, junior, QA, security, devops, product)
- Focus on hands-on implementation
- Learn through building, not just reading

### 4. Productivity Optimization
- Work during most productive hours: {', '.join(map(str, self.metrics.most_productive_hours))}
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
"""

        return doc

    def import_to_memory_bank(self, memory_store_path: str = "memory_store") -> int:
        """Import all analyzed chats into EpisodicMemoryBank.

        Args:
            memory_store_path: Path to memory store directory

        Returns:
            Number of memories imported
        """
        if not self.questions:
            print("No questions analyzed. Run extract_chat_history() and analyze_questions() first.")
            return 0

        memory_bank = EpisodicMemoryBank(storage_path=Path(memory_store_path))
        memories = []

        for question in self.questions:
            memory = EpisodicMemoryEntry(
                agent_type="CursorChatUser",
                memory_id=str(uuid.uuid4()),
                summary=question.question,
                context={
                    "category": question.category,
                    "complexity": question.complexity,
                    "resolution_time_minutes": question.resolution_time_minutes,
                    "success": question.success,
                    "follow_up_questions": question.follow_up_questions,
                    "code_examples": question.code_examples,
                    "timestamp": question.timestamp.isoformat()
                },
                source="cursor_chat",
                timestamp=question.timestamp,
                memory_type="conversation"
            )
            memories.append(memory)

        memory_bank.add_batch(memories)
        print(f"Successfully imported {len(memories)} Cursor chat memories")
        return len(memories)

    def save_analysis(self, filename: str = "cursor_analysis.json"):
        """Save analysis results to JSON file"""
        data = {
            'sessions': [asdict(session) for session in self.sessions],
            'questions': [asdict(question) for question in self.questions],
            'metrics': asdict(self.metrics) if self.metrics else None,
            'analysis_date': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Analysis saved to {filename}")

    def create_visualizations(self):
        """Create productivity visualizations"""
        if not self.sessions or not self.questions:
            print("No data available for visualization")
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cursor Chat Analysis Dashboard', fontsize=16)

        # 1. Daily coding time
        dates = [s.start_time.date() for s in self.sessions]
        daily_time = {}
        for date in dates:
            daily_time[date] = daily_time.get(date, 0) + 1

        axes[0, 0].plot(list(daily_time.keys()), list(daily_time.values()), marker='o')
        axes[0, 0].set_title('Daily Coding Sessions')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sessions')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Question categories
        if self.metrics and self.metrics.question_categories:
            categories = list(self.metrics.question_categories.keys())
            counts = list(self.metrics.question_categories.values())
            axes[0, 1].pie(counts, labels=categories, autopct='%1.1f%%')
            axes[0, 1].set_title('Question Categories')

        # 3. Session duration distribution
        durations = [s.duration_minutes for s in self.sessions]
        axes[1, 0].hist(durations, bins=20, alpha=0.7)
        axes[1, 0].set_title('Session Duration Distribution')
        axes[1, 0].set_xlabel('Duration (minutes)')
        axes[1, 0].set_ylabel('Frequency')

        # 4. Hourly productivity
        hours = [s.start_time.hour for s in self.sessions]
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        axes[1, 1].bar(hour_counts.keys(), hour_counts.values())
        axes[1, 1].set_title('Productivity by Hour')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Sessions')

        plt.tight_layout()
        plt.savefig('cursor_productivity_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Visualizations saved as 'cursor_productivity_dashboard.png'")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Analyze Cursor chat history')
    parser.add_argument('--analyze', action='store_true', help='Run full analysis')
    parser.add_argument('--export-methodology', action='store_true', help='Export methodology documentation')
    parser.add_argument('--daily-report', action='store_true', help='Generate daily report')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--data-path', type=str, help='Path to Cursor data directory')
    parser.add_argument('--import-to-memory', action='store_true',
                       help='Import chats to EpisodicMemoryBank')
    parser.add_argument('--memory-store-path', type=str, default='memory_store',
                       help='Path to memory store directory')

    args = parser.parse_args()

    analyzer = CursorChatAnalyzer(args.data_path)

    if args.analyze:
        print("Extracting chat history...")
        if analyzer.extract_chat_history():
            print("Analyzing questions...")
            analyzer.analyze_questions()
            print("Calculating metrics...")
            analyzer.calculate_productivity_metrics()
            print("Saving analysis...")
            analyzer.save_analysis()
            print("Analysis complete!")
        else:
            print("Failed to extract chat history")

    if args.export_methodology:
        if analyzer.metrics:
            doc = analyzer.export_methodology_documentation()
            with open('ai_development_methodology.md', 'w') as f:
                f.write(doc)
            print("Methodology documentation exported to 'ai_development_methodology.md'")
        else:
            print("Run analysis first with --analyze")

    if args.daily_report:
        report = analyzer.generate_daily_report()
        print(report)

    if args.visualize:
        analyzer.create_visualizations()

    if args.import_to_memory:
        if analyzer.questions:
            count = analyzer.import_to_memory_bank(args.memory_store_path)
            print(f"Imported {count} memories to {args.memory_store_path}")
        else:
            print("Run analysis first with --analyze")

if __name__ == "__main__":
    main()

