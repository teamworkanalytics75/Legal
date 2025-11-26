#!/usr/bin/env python3
"""
Cursor Chat Tracker Setup Script
================================

This script sets up the Cursor chat tracking system and creates sample data for testing.

Usage:
    python setup_cursor_tracker.py
"""

import os
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    packages = [
        'pandas',
        'matplotlib',
        'seaborn'
    ]

    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")

    # Import and run the tracker
    from cursor_tracker import CursorTracker

    tracker = CursorTracker()
    tracker.create_sample_data()

    print("✓ Sample data created")

def test_functionality():
    """Test the tracker functionality"""
    print("Testing functionality...")

    from cursor_tracker import CursorTracker

    tracker = CursorTracker()

    # Test metrics calculation
    metrics = tracker.calculate_metrics()
    if metrics:
        print("✓ Metrics calculation working")
        print(f"  Total sessions: {metrics['total_sessions']}")
        print(f"  Total time: {metrics['total_time_hours']:.1f} hours")
    else:
        print("✗ Metrics calculation failed")

    # Test daily report
    report = tracker.daily_report()
    if report and "Daily Productivity Report" in report:
        print("✓ Daily report generation working")
    else:
        print("✗ Daily report generation failed")

    # Test methodology documentation
    doc = tracker.methodology_documentation()
    if doc and "AI-Assisted Development Methodology" in doc:
        print("✓ Methodology documentation working")
    else:
        print("✗ Methodology documentation failed")

def create_usage_guide():
    """Create a usage guide"""
    guide = """
# Cursor Chat Tracker - Usage Guide

## Quick Start

### 1. Start Tracking Your Sessions
```bash
# Start a new coding session
python cursor_tracker.py --start-session "Matrix Legal AI System"

# Add questions as you work
python cursor_tracker.py --add-question "How do I implement Bayesian networks?"
python cursor_tracker.py --add-question "What's the best way to structure this API?"

# Add problems you solve
python cursor_tracker.py --add-problem "Database connection timeout" "Fixed connection string"

# Track features and bugs
python cursor_tracker.py --add-features 2
python cursor_tracker.py --add-bugs 1

# End the session
python cursor_tracker.py --end-session
```

### 2. Generate Reports
```bash
# Daily productivity report
python cursor_tracker.py --daily-report

# Methodology documentation for your team
python cursor_tracker.py --methodology

# Current metrics
python cursor_tracker.py --metrics
```

### 3. Advanced Analysis
```bash
# Full analysis with visualizations
python cursor_chat_analyzer.py --analyze --visualize

# Export methodology documentation
python cursor_chat_analyzer.py --export-methodology
```

## Question Categories

When adding questions, use these categories to track which professional roles you're covering:

- **senior_dev**: Architecture, design patterns, complex decisions
- **mid_dev**: Implementation, features, API integration
- **junior_dev**: Basic syntax, errors, simple features
- **qa**: Testing, validation, edge cases
- **security**: Security, authentication, vulnerabilities
- **devops**: Deployment, production, infrastructure
- **product**: User experience, interface, workflow

## Example Workflow

1. **Start your day**: `python cursor_tracker.py --start-session "Project Name"`
2. **As you code**: Add questions, problems, features, bugs
3. **End session**: `python cursor_tracker.py --end-session`
4. **Daily review**: `python cursor_tracker.py --daily-report`
5. **Weekly analysis**: `python cursor_tracker.py --methodology`

## Files Created

- `cursor_tracking.json`: Your session data
- `cursor_analysis.json`: Detailed analysis results
- `ai_development_methodology.md`: Team methodology documentation
- `cursor_productivity_dashboard.png`: Visualizations

## Tips for Effective Tracking

1. **Be consistent**: Track every coding session
2. **Ask diverse questions**: Cover all professional roles
3. **Document problems**: Record what you solve
4. **Track features**: Count what you build
5. **Review regularly**: Analyze your patterns weekly

## Team Implementation

1. **Share methodology**: Export documentation for your team
2. **Standardize questions**: Use consistent categories
3. **Track progress**: Monitor learning progression
4. **Share insights**: Discuss productivity patterns
5. **Iterate**: Improve methodology based on results
"""

    with open('CURSOR_TRACKER_USAGE.md', 'w') as f:
        f.write(guide)

    print("✓ Usage guide created: CURSOR_TRACKER_USAGE.md")

def main():
    """Main setup function"""
    print("Setting up Cursor Chat Tracker...")
    print("=" * 50)

    # Install requirements
    install_requirements()
    print()

    # Create sample data
    create_sample_data()
    print()

    # Test functionality
    test_functionality()
    print()

    # Create usage guide
    create_usage_guide()
    print()

    print("=" * 50)
    print("Setup complete!")
    print()
    print("Next steps:")
    print("1. Read CURSOR_TRACKER_USAGE.md for detailed instructions")
    print("2. Start tracking: python cursor_tracker.py --start-session 'My Project'")
    print("3. Generate reports: python cursor_tracker.py --daily-report")
    print("4. Create methodology: python cursor_tracker.py --methodology")

if __name__ == "__main__":
    main()

