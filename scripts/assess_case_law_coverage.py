#!/usr/bin/env python3
"""
Assess Case Law Coverage for Key Questions

Analyzes what we have and identifies gaps for:
1. Successful motion outlines/headers/structure
2. National security argument success patterns
3. Case similarity analysis
4. Other key questions
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import json

def analyze_database(db_path: Path) -> Dict[str, Any]:
    """Analyze a single database."""
    if not db_path.exists():
        return {"error": "Database not found"}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    stats = {}

    # Total cases
    cursor.execute("SELECT COUNT(*) FROM cases")
    stats["total_cases"] = cursor.fetchone()[0]

    # Check what columns exist
    cursor.execute("PRAGMA table_info(cases)")
    columns = [row[1] for row in cursor.fetchall()]

    # National security cases
    if "tag_national_security" in columns:
        cursor.execute("SELECT COUNT(*) FROM cases WHERE tag_national_security = 1")
        stats["ns_tagged"] = cursor.fetchone()[0]

    # Outcome data
    if "favorable_to_plaintiff" in columns:
        cursor.execute("SELECT COUNT(*) FROM cases WHERE favorable_to_plaintiff = 1")
        stats["granted"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cases WHERE favorable_to_defendant = 1")
        stats["denied"] = cursor.fetchone()[0]

    # Motion type data
    if "corpus_type" in columns:
        cursor.execute("SELECT corpus_type, COUNT(*) FROM cases GROUP BY corpus_type")
        stats["by_corpus_type"] = dict(cursor.fetchall())

    # Check for motion structure data
    if "cleaned_text" in columns:
        # Sample cases with structure
        cursor.execute("""
            SELECT COUNT(*) FROM cases
            WHERE cleaned_text LIKE '%I.%'
            OR cleaned_text LIKE '%1.%'
            OR cleaned_text LIKE '%INTRODUCTION%'
        """)
        stats["has_structure_indicators"] = cursor.fetchone()[0]

        # Cases with national security + outcome
        if "tag_national_security" in columns and "favorable_to_plaintiff" in columns:
            cursor.execute("""
                SELECT COUNT(*) FROM cases
                WHERE tag_national_security = 1
                AND favorable_to_plaintiff = 1
            """)
            stats["ns_granted"] = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM cases
                WHERE tag_national_security = 1
                AND favorable_to_defendant = 1
            """)
            stats["ns_denied"] = cursor.fetchone()[0]

    # Check for similarity features
    if "word_count" in columns and "citation_count" in columns:
        cursor.execute("""
            SELECT COUNT(*) FROM cases
            WHERE word_count > 0 AND citation_count > 0
        """)
        stats["has_similarity_features"] = cursor.fetchone()[0]

    conn.close()
    return stats

def assess_coverage() -> Dict[str, Any]:
    """Assess overall coverage for key questions."""

    db_paths = [
        Path("case_law_data/unified_corpus.db"),
        Path("case_law_data/ma_federal_motions.db"),
        Path("case_law_data/harvard_corpus.db"),
        Path("case_law_data/china_corpus.db"),
    ]

    results = {}
    for db_path in db_paths:
        if db_path.exists():
            results[db_path.name] = analyze_database(db_path)

    # Assess coverage for key questions
    coverage = {
        "motion_structure_analysis": {
            "question": "Can we analyze successful motion outlines/headers/structure?",
            "required": "Cases with structure (sections, headers) + outcome data",
            "status": "unknown"
        },
        "ns_argument_success": {
            "question": "Can we analyze national security argument success patterns?",
            "required": "NS-tagged cases with outcome data (granted/denied)",
            "status": "unknown"
        },
        "case_similarity": {
            "question": "Can we find cases similar to mine?",
            "required": "Cases with features (word count, citations, tags) + metadata",
            "status": "unknown"
        },
        "ns_sealing_patterns": {
            "question": "Can we analyze NS-based sealing motion patterns?",
            "required": "NS-tagged sealing motions with outcome data",
            "status": "unknown"
        },
    }

    # Evaluate coverage
    total_cases = sum(r.get("total_cases", 0) for r in results.values())
    total_ns = sum(r.get("ns_tagged", 0) for r in results.values())
    total_granted = sum(r.get("granted", 0) for r in results.values())
    total_ns_granted = sum(r.get("ns_granted", 0) for r in results.values())
    total_structure = sum(r.get("has_structure_indicators", 0) for r in results.values())

    # Motion structure analysis
    if total_structure > 50 and total_granted > 20:
        coverage["motion_structure_analysis"]["status"] = "adequate"
        coverage["motion_structure_analysis"]["count"] = f"{total_structure} cases with structure, {total_granted} with outcomes"
    elif total_structure > 20:
        coverage["motion_structure_analysis"]["status"] = "limited"
        coverage["motion_structure_analysis"]["count"] = f"{total_structure} cases with structure"
    else:
        coverage["motion_structure_analysis"]["status"] = "insufficient"
        coverage["motion_structure_analysis"]["count"] = f"{total_structure} cases with structure"

    # NS argument success
    total_ns_denied = sum(r.get("ns_denied", 0) for r in results.values())
    if total_ns_granted > 20:
        coverage["ns_argument_success"]["status"] = "adequate"
        coverage["ns_argument_success"]["count"] = f"{total_ns_granted} NS cases granted, {total_ns_denied} NS cases denied (total NS: {total_ns})"
    elif total_ns_granted > 5:
        coverage["ns_argument_success"]["status"] = "limited"
        coverage["ns_argument_success"]["count"] = f"{total_ns_granted} NS cases granted"
    else:
        coverage["ns_argument_success"]["status"] = "insufficient"
        coverage["ns_argument_success"]["count"] = f"{total_ns_granted} NS cases granted (total NS: {total_ns})"

    # Case similarity
    if total_cases > 500:
        coverage["case_similarity"]["status"] = "adequate"
        coverage["case_similarity"]["count"] = f"{total_cases} total cases"
    elif total_cases > 100:
        coverage["case_similarity"]["status"] = "limited"
        coverage["case_similarity"]["count"] = f"{total_cases} total cases"
    else:
        coverage["case_similarity"]["status"] = "insufficient"
        coverage["case_similarity"]["count"] = f"{total_cases} total cases"

    # NS sealing patterns
    if total_ns_granted > 10:
        coverage["ns_sealing_patterns"]["status"] = "adequate"
        coverage["ns_sealing_patterns"]["count"] = f"{total_ns_granted} NS motions granted"
    else:
        coverage["ns_sealing_patterns"]["status"] = "insufficient"
        coverage["ns_sealing_patterns"]["count"] = f"{total_ns_granted} NS motions granted"

    return {
        "database_stats": results,
        "coverage_assessment": coverage,
        "summary": {
            "total_cases": total_cases,
            "total_ns_tagged": total_ns,
            "total_granted": total_granted,
            "total_ns_granted": total_ns_granted,
            "cases_with_structure": total_structure
        }
    }

def generate_recommendations(assessment: Dict[str, Any]) -> List[str]:
    """Generate recommendations for improving coverage."""
    recommendations = []

    coverage = assessment["coverage_assessment"]
    summary = assessment["summary"]

    # Motion structure recommendations
    if coverage["motion_structure_analysis"]["status"] in ["limited", "insufficient"]:
        recommendations.append(
            "Download more motions with clear structure (sections, headers) from CourtListener. "
            f"Current: {summary['cases_with_structure']} cases with structure indicators. "
            "Target: 200+ motions with structure + outcome data."
        )

    # NS argument recommendations
    if coverage["ns_argument_success"]["status"] in ["limited", "insufficient"]:
        recommendations.append(
            "Download more national security sealing/pseudonym motions from CourtListener. "
            f"Current: {summary['total_ns_granted']} NS motions granted. "
            "Target: 50+ NS motions with outcome data (granted/denied). "
            "Search terms: 'national security' AND ('motion to seal' OR 'pseudonym' OR 'protective order')"
        )

    # Similarity recommendations
    if coverage["case_similarity"]["status"] == "insufficient":
        recommendations.append(
            f"Expand database with more diverse cases. Current: {summary['total_cases']} cases. "
            "Target: 1000+ cases for better similarity matching."
        )

    # NS sealing patterns
    if coverage["ns_sealing_patterns"]["status"] == "insufficient":
        recommendations.append(
            "Specifically target NS sealing motions: "
            "'national security' AND 'seal' AND ('granted' OR 'denied') "
            "Focus on federal district courts, especially D. Mass, D.D.C., S.D.N.Y."
        )

    return recommendations

if __name__ == "__main__":
    print("=" * 80)
    print("CASE LAW COVERAGE ASSESSMENT")
    print("=" * 80)
    print()

    assessment = assess_coverage()

    print("DATABASE STATISTICS:")
    print("-" * 80)
    for db_name, stats in assessment["database_stats"].items():
        print(f"\n{db_name}:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("COVERAGE ASSESSMENT FOR KEY QUESTIONS")
    print("=" * 80)
    print()

    for question_type, info in assessment["coverage_assessment"].items():
        status = info["status"]
        status_icon = "[OK]" if status == "adequate" else "[WARN]" if status == "limited" else "[LOW]"
        print(f"{status_icon} {info['question']}")
        print(f"   Status: {status.upper()}")
        print(f"   Required: {info['required']}")
        if "count" in info:
            print(f"   Current: {info['count']}")
        print()

    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()

    recommendations = generate_recommendations(assessment)
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("[OK] Coverage appears adequate for all key questions!")

    # Save detailed report
    output_path = Path("outputs/case_law_coverage_assessment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(assessment, f, indent=2, default=str)

    print(f"\n📄 Detailed report saved to: {output_path}")

