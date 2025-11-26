#!/usr/bin/env python3
"""
Manual Validation Script for §1782 Wishlist Cases

This script examines cases that were found but failed automated §1782 validation
to determine if they are actually valid §1782 cases that should be included.
"""

import json
import sys
from pathlib import Path

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient
from filters import is_actual_1782_case

def examine_case_content(cluster_id: int, case_name: str) -> dict:
    """Examine a case's content to determine if it's a valid §1782 case."""
    client = CourtListenerClient()

    print(f"\n{'='*80}")
    print(f"Examining: {case_name}")
    print(f"Cluster ID: {cluster_id}")
    print(f"{'='*80}")

    try:
        # Try to get the opinion by cluster ID
        result = client.search_opinions(
            keywords=[str(cluster_id)],
            limit=5
        )

        if not result or 'results' not in result:
            return {"status": "not_found", "reason": "No search results"}

        for opinion in result['results']:
            if opinion.get('cluster_id') == cluster_id:
                print(f"Found opinion: {opinion.get('caseName', 'Unknown')}")

                # Check automated validation
                is_valid = is_actual_1782_case(opinion)
                print(f"Automated validation: {'✓ PASS' if is_valid else '✗ FAIL'}")

                # Manual examination
                case_name_full = opinion.get('caseNameFull', '')
                snippet = opinion.get('snippet', '')

                print(f"\nCase Name Full: {case_name_full}")
                print(f"\nSnippet: {snippet[:500]}...")

                # Look for §1782 indicators
                text_to_examine = f"{case_name_full} {snippet}".lower()

                indicators = []
                if '1782' in text_to_examine:
                    indicators.append("Contains '1782'")
                if '28 u.s.c' in text_to_examine:
                    indicators.append("Contains '28 U.S.C.'")
                if 'judicial assistance' in text_to_examine:
                    indicators.append("Contains 'judicial assistance'")
                if 'foreign tribunal' in text_to_examine:
                    indicators.append("Contains 'foreign tribunal'")
                if 'letters rogatory' in text_to_examine:
                    indicators.append("Contains 'letters rogatory'")
                if 'discovery' in text_to_examine and 'foreign' in text_to_examine:
                    indicators.append("Contains 'foreign discovery'")
                if 'application' in text_to_examine and 'pursuant' in text_to_examine:
                    indicators.append("Contains 'application pursuant'")

                print(f"\n§1782 Indicators found: {len(indicators)}")
                for indicator in indicators:
                    print(f"  ✓ {indicator}")

                # Manual judgment
                manual_valid = len(indicators) >= 2 or '1782' in text_to_examine

                return {
                    "status": "found",
                    "automated_valid": is_valid,
                    "manual_valid": manual_valid,
                    "indicators": indicators,
                    "case_name_full": case_name_full,
                    "snippet": snippet[:200]
                }

        return {"status": "not_found", "reason": "Cluster ID not found in results"}

    except Exception as e:
        return {"status": "error", "reason": str(e)}

def main():
    """Examine high-priority cases that failed validation."""

    # High-priority cases to examine
    high_priority_cases = [
        (8421766, "In re Porsche Automobil Holding SE"),
        (9366884, "Abdul Latif Jameel Transp. Co. v. FedEx Corp."),
        (9353253, "In re Letter of Request from Supreme Ct. of Hong Kong"),
        (9484134, "In re Application of Peruvian Sporting Goods S.A.C."),
        (7326637, "In re Hand Held Prods., Inc."),
    ]

    results = []

    for cluster_id, case_name in high_priority_cases:
        result = examine_case_content(cluster_id, case_name)
        result['cluster_id'] = cluster_id
        result['case_name'] = case_name
        results.append(result)

    # Save results
    output_file = Path("data/case_law/manual_validation_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("MANUAL VALIDATION SUMMARY")
    print(f"{'='*80}")

    valid_cases = [r for r in results if r.get('manual_valid', False)]
    print(f"Cases that appear to be valid §1782 cases: {len(valid_cases)}")

    for result in results:
        if result.get('manual_valid', False):
            print(f"  ✓ {result['case_name']} - {len(result.get('indicators', []))} indicators")

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
