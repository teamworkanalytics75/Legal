#!/usr/bin/env python3
"""
Detailed Analysis of Judge Allison D. Burroughs' Rulings

Extracts detailed information about:
- Motion to seal rulings
- Pseudonym filings
- 1782 applications
- Ruling patterns and factors
"""

import sqlite3
from pathlib import Path
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any
import json

def extract_detailed_rulings(db_path: Path) -> Dict[str, Any]:
    """Extract detailed rulings information for Judge Burroughs."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all cases mentioning Burroughs
    cursor.execute("""
        SELECT
            cluster_id,
            case_name,
            court,
            docket_number,
            date_filed,
            cleaned_text
        FROM cases
        WHERE LOWER(cleaned_text) LIKE '%burroughs%'
        OR LOWER(case_name) LIKE '%burroughs%';
    """)

    all_cases = cursor.fetchall()

    print(f"Found {len(all_cases)} total cases mentioning Judge Burroughs")

    # Categorize by court
    district_court_cases = [c for c in all_cases if 'district' in c[2].lower() and 'mass' in c[2].lower()]
    circuit_court_cases = [c for c in all_cases if 'circuit' in c[2].lower()]

    print(f"  - District Court cases: {len(district_court_cases)}")
    print(f"  - Circuit Court cases: {len(circuit_court_cases)}")

    detailed_analysis = {
        'district_court_cases': [],
        'circuit_court_cases': [],
        'motions_to_seal': [],
        'pseudonym_motions': [],
        '1782_applications': [],
        'summary': {}
    }

    # Analyze each case
    for case_id, case_name, court, docket, date_filed, text in all_cases:
        text_lower = text.lower()

        case_info = {
            'case_id': case_id,
            'case_name': case_name,
            'court': court,
            'docket_number': docket,
            'date_filed': date_filed
        }

        # Check for motion to seal
        if any(term in text_lower for term in ['motion to seal', 'motion for seal', 'sealing', 'seal', 'impound']):
            case_info['has_seal_motion'] = True

            # Extract outcome
            if any(term in text_lower for term in ['motion granted', 'granted', 'seal granted']):
                case_info['seal_outcome'] = 'granted'
            elif any(term in text_lower for term in ['motion denied', 'denied']):
                case_info['seal_outcome'] = 'denied'
            else:
                case_info['seal_outcome'] = 'uncertain'

            # Extract reasoning
            reasoning = extract_reasoning(text, case_name)
            case_info['reasoning'] = reasoning

            detailed_analysis['motions_to_seal'].append(case_info)

        # Check for pseudonym
        if any(term in text_lower for term in ['pseudonym', 'doe', 'anonymous', 'proceed anonymously']):
            case_info['has_pseudonym_motion'] = True

            # Extract outcome
            if any(term in text_lower for term in ['may proceed under pseudonym', 'pseudonym granted', 'proceed anonymously']):
                case_info['pseudonym_outcome'] = 'granted'
            elif any(term in text_lower for term in ['pseudonym denied', 'identify', 'named']):
                case_info['pseudonym_outcome'] = 'denied'
            else:
                case_info['pseudonym_outcome'] = 'uncertain'

            detailed_analysis['pseudonym_motions'].append(case_info)

        # Check for 1782
        if any(term in text_lower for term in ['1782', 'section 1782', '28 u.s.c. 1782']):
            case_info['has_1782'] = True

            # Extract outcome
            if any(term in text_lower for term in ['granted', 'approved', 'permitted']):
                case_info['1782_outcome'] = 'granted'
            elif any(term in text_lower for term in ['denied', 'rejected']):
                case_info['1782_outcome'] = 'denied'
            else:
                case_info['1782_outcome'] = 'uncertain'

            detailed_analysis['1782_applications'].append(case_info)

        # Add to appropriate list
        if 'district' in court.lower():
            detailed_analysis['district_court_cases'].append(case_info)
        else:
            detailed_analysis['circuit_court_cases'].append(case_info)

    # Generate summary statistics
    detailed_analysis['summary'] = {
        'total_cases': len(all_cases),
        'district_court_cases': len(district_court_cases),
        'circuit_court_cases': len(circuit_court_cases),
        'motions_to_seal_count': len(detailed_analysis['motions_to_seal']),
        'pseudonym_motions_count': len(detailed_analysis['pseudonym_motions']),
        '1782_applications_count': len(detailed_analysis['1782_applications']),
        'seal_outcomes': Counter([c.get('seal_outcome') for c in detailed_analysis['motions_to_seal']]),
        'pseudonym_outcomes': Counter([c.get('pseudonym_outcome') for c in detailed_analysis['pseudonym_motions']]),
        '1782_outcomes': Counter([c.get('1782_outcome') for c in detailed_analysis['1782_applications']])
    }

    conn.close()

    return detailed_analysis

def extract_reasoning(text: str, case_name: str) -> Dict[str, Any]:
    """Extract reasoning from case text."""
    reasoning = {
        'privacy_concerns': [],
        'public_interest': [],
        'harassment': [],
        'safety': [],
        'other': []
    }

    text_lower = text.lower()

    # Look for privacy-related language
    if 'privacy' in text_lower:
        # Extract context around privacy mentions
        privacy_context = []
        for match in re.finditer(r'privacy', text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            privacy_context.append(text[start:end])
        reasoning['privacy_concerns'] = privacy_context[:3]  # Keep first 3

    # Look for harassment-related language
    if any(term in text_lower for term in ['harass', 'retaliate', 'threat']):
        reasoning['harassment'].append('Mentioned in case')

    # Look for safety concerns
    if any(term in text_lower for term in ['safety', 'danger', 'fear', 'risk of harm']):
        reasoning['safety'].append('Mentioned in case')

    # Look for public interest
    if 'public interest' in text_lower:
        # Extract context
        public_interest_context = []
        for match in re.finditer(r'public interest', text_lower):
            start = max(0, match.start() - 200)
            end = min(len(text), match.end() + 200)
            public_interest_context.append(text[start:end])
        reasoning['public_interest'] = public_interest_context[:3]

    return reasoning

def print_detailed_report(analysis: Dict[str, Any]):
    """Print a detailed report of findings."""

    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS: JUDGE ALLISON D. BURROUGHS")
    print("=" * 80)

    summary = analysis['summary']

    print(f"\nSUMMARY STATISTICS:")
    print(f"  Total cases: {summary['total_cases']}")
    print(f"  District Court cases: {summary['district_court_cases']}")
    print(f"  Circuit Court cases: {summary['circuit_court_cases']}")

    print(f"\nMOTION TYPES:")
    print(f"  Motions to Seal: {summary['motions_to_seal_count']}")
    print(f"  Pseudonym Motions: {summary['pseudonym_motions_count']}")
    print(f"  1782 Applications: {summary['1782_applications_count']}")

    # Outcomes
    print(f"\nMOTIONS TO SEAL OUTCOMES:")
    for outcome, count in summary['seal_outcomes'].items():
        if outcome:
            print(f"  {outcome.capitalize()}: {count}")

    print(f"\nPSEUDONYM MOTIONS OUTCOMES:")
    for outcome, count in summary['pseudonym_outcomes'].items():
        if outcome:
            print(f"  {outcome.capitalize()}: {count}")

    print(f"\n1782 APPLICATIONS OUTCOMES:")
    for outcome, count in summary['1782_outcomes'].items():
        if outcome:
            print(f"  {outcome.capitalize()}: {count}")

    # Detailed case information
    print("\n" + "=" * 80)
    print("DETAILED CASE INFORMATION")
    print("=" * 80)

    if analysis['motions_to_seal']:
        print("\nMOTIONS TO SEAL CASES:")
        for i, case in enumerate(analysis['motions_to_seal'][:5], 1):
            print(f"\n  Case {i}: {case['case_name']}")
            print(f"  Docket: {case.get('docket_number', 'N/A')}")
            print(f"  Date: {case.get('date_filed', 'N/A')}")
            print(f"  Outcome: {case.get('seal_outcome', 'N/A')}")

    if analysis['pseudonym_motions']:
        print("\nPSEUDONYM MOTIONS:")
        for i, case in enumerate(analysis['pseudonym_motions'][:5], 1):
            print(f"\n  Case {i}: {case['case_name']}")
            print(f"  Docket: {case.get('docket_number', 'N/A')}")
            print(f"  Date: {case.get('date_filed', 'N/A')}")
            print(f"  Outcome: {case.get('pseudonym_outcome', 'N/A')}")

    if analysis['1782_applications']:
        print("\n1782 APPLICATIONS:")
        for i, case in enumerate(analysis['1782_applications'][:5], 1):
            print(f"\n  Case {i}: {case['case_name']}")
            print(f"  Docket: {case.get('docket_number', 'N/A')}")
            print(f"  Date: {case.get('date_filed', 'N/A')}")
            print(f"  Outcome: {case.get('1782_outcome', 'N/A')}")

    # District Court cases (most relevant)
    print("\n" + "=" * 80)
    print("DISTRICT COURT CASES (Judge Burroughs' direct rulings)")
    print("=" * 80)

    for case in analysis['district_court_cases']:
        print(f"\n  {case['case_name']}")
        print(f"  Docket: {case.get('docket_number', 'N/A')}")
        print(f"  Date: {case.get('date_filed', 'N/A')}")

def save_json_report(analysis: Dict[str, Any], output_path: Path):
    """Save analysis as JSON for further processing."""

    # Convert Counter objects to regular dicts for JSON serialization
    serializable_analysis = {}
    for key, value in analysis.items():
        if key == 'summary':
            serializable_value = {}
            for k, v in value.items():
                if isinstance(v, Counter):
                    serializable_value[k] = dict(v)
                else:
                    serializable_value[k] = v
            serializable_analysis[key] = serializable_value
        else:
            serializable_analysis[key] = value

    with open(output_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)

    print(f"\nAnalysis saved to: {output_path}")

def main():
    """Main execution function."""

    # Set up paths
    script_root = Path(__file__).resolve().parent
    db_path = script_root / "case_law_data" / "ma_federal_motions.db"
    output_dir = script_root / "reports" / "judge_analysis"

    # Check database exists
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("JUDGE BURROUGHS DETAILED ANALYSIS")
    print("=" * 80)
    print(f"\nDatabase: {db_path}")
    print(f"Output directory: {output_dir}\n")

    # Extract detailed rulings
    print("Extracting detailed rulings...")
    analysis = extract_detailed_rulings(db_path)

    # Print report
    print_detailed_report(analysis)

    # Save JSON report
    output_path = output_dir / "burroughs_analysis.json"
    save_json_report(analysis, output_path)

    # Save text report
    text_output = output_dir / "burroughs_analysis.txt"
    with open(text_output, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("JUDGE ALLISON D. BURROUGHS RULING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total Cases Analyzed: {analysis['summary']['total_cases']}\n")
        f.write(f"District Court Cases: {analysis['summary']['district_court_cases']}\n")
        f.write(f"Circuit Court Cases: {analysis['summary']['circuit_court_cases']}\n\n")

        f.write("Motions to Seal:\n")
        f.write(f"  Count: {analysis['summary']['motions_to_seal_count']}\n")
        f.write(f"  Outcomes: {dict(analysis['summary']['seal_outcomes'])}\n\n")

        f.write("Pseudonym Motions:\n")
        f.write(f"  Count: {analysis['summary']['pseudonym_motions_count']}\n")
        f.write(f"  Outcomes: {dict(analysis['summary']['pseudonym_outcomes'])}\n\n")

        f.write("1782 Applications:\n")
        f.write(f"  Count: {analysis['summary']['1782_applications_count']}\n")
        f.write(f"  Outcomes: {dict(analysis['summary']['1782_outcomes'])}\n")

    print(f"\nDetailed text report saved to: {text_output}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

