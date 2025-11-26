#!/usr/bin/env python3
"""
Quick Summary of 1782 Analysis Results
=====================================

Generate a simple summary from the analysis results
"""

import json
from pathlib import Path
from collections import Counter

def generate_summary():
    """Generate a summary from the analysis results."""

    # Find the latest analysis file
    results_dir = Path("data/case_law/analysis_results")
    analysis_files = list(results_dir.glob("1782_basic_analysis_*.json"))

    if not analysis_files:
        print("❌ No analysis files found")
        return

    latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
    print(f"📄 Reading analysis from: {latest_file.name}")

    # Load results
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("\n" + "=" * 60)
    print("📊 1782 DISCOVERY CASES - ANALYSIS SUMMARY")
    print("=" * 60)

    # Overview
    print(f"\n📈 OVERVIEW:")
    print(f"  Total Files Analyzed: {results['analyzed_files']}")
    print(f"  Analysis Success Rate: {(results['analyzed_files']/results['total_files'])*100:.1f}%")

    # Summary Statistics
    stats = results.get('summary_stats', {})
    print(f"\n📊 STATISTICS:")
    print(f"  Average Text Length: {stats.get('average_text_length', 0):.0f} characters")
    print(f"  Average Word Count: {stats.get('average_word_count', 0):.0f} words")
    print(f"  Total Text Analyzed: {stats.get('total_text_length', 0):,} characters")

    # Outcome Predictions
    outcome_counts = results.get('outcome_summary', {})
    if outcome_counts:
        print(f"\n⚖️ CASE OUTCOMES:")
        total_outcomes = sum(outcome_counts.values())
        for outcome, count in sorted(outcome_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_outcomes) * 100
            print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    # Legal Pattern Analysis
    patterns = results.get('pattern_summary', {})
    if patterns:
        print(f"\n🔍 TOP LEGAL PATTERNS:")
        for pattern_type, keyword_counts in patterns.items():
            if keyword_counts:
                # Convert to Counter if it's a dict
                if isinstance(keyword_counts, dict):
                    counter = Counter(keyword_counts)
                else:
                    counter = keyword_counts

                top_keyword = counter.most_common(1)[0] if counter else ("none", 0)
                print(f"  {pattern_type.replace('_', ' ').title()}: {top_keyword[0]} ({top_keyword[1]} times)")

    # Entity Analysis
    entity_counts = stats.get('entity_counts', {})
    if entity_counts:
        print(f"\n🏷️ TOP ENTITIES:")
        if isinstance(entity_counts, dict):
            counter = Counter(entity_counts)
        else:
            counter = entity_counts

        for entity_type, count in counter.most_common(5):
            print(f"  {entity_type}: {count} occurrences")

    print(f"\n💾 Full results saved in: {latest_file}")
    print("🎯 Analysis complete! Ready for pattern analysis and knowledge graph construction.")

if __name__ == "__main__":
    generate_summary()
