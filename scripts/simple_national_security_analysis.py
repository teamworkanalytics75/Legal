#!/usr/bin/env python3
"""
Simple National Security Matter Analysis

Direct analysis of Statement of Claim for US national security classification
without full workflow dependencies.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

def analyze_national_security_indicators(statement_text: str) -> Dict[str, Any]:
    """Analyze statement for national security indicators."""

    statement_lower = statement_text.lower()

    # National security keywords
    natsec_keywords = {
        "national security": 3.0,
        "national defense": 3.0,
        "classified information": 4.0,
        "state secrets": 4.0,
        "foreign agent": 3.5,
        "foreign interference": 3.5,
        "cyber security": 2.5,
        "critical infrastructure": 3.0,
        "defense contractor": 2.5,
        "intelligence": 3.0,
        "espionage": 4.0,
        "terrorism": 4.0,
        "weapons of mass destruction": 4.0,
        "export control": 2.5,
        "cfius": 3.0,
        "foreign investment": 2.0,
        "prc": 2.5,
        "china": 2.0,
        "russia": 2.5,
        "iran": 2.5,
        "north korea": 3.0,
        "federal funding": 2.5,
        "defense grant": 3.0,
        "research security": 2.5,
        "xi jinping": 3.0,
        "xi mingze": 3.5,
        "harvard": 1.5,
        "harvard club": 2.0,
        "harvard-china": 2.5,
        "us citizen": 2.0,
        "endangerment": 2.5,
        "arrest": 2.0,
        "criminal prosecution": 2.0,
        "political": 1.5,
        "government crackdown": 2.5,
        "hostile government": 3.0
    }

    # Legal frameworks
    legal_frameworks = {
        "fara": 3.5,
        "foreign agents registration act": 3.5,
        "itar": 3.0,
        "international traffic in arms": 3.0,
        "ear": 2.5,
        "export administration regulations": 2.5,
        "cfius": 3.0,
        "committee on foreign investment": 3.0,
        "ndaa": 2.5,
        "national defense authorization": 2.5,
        "executive order": 2.0,
        "presidential directive": 2.0,
        "section 1782": 2.0,
        "28 u.s.c. 1782": 2.0
    }

    # Find all indicators
    indicators_found = []
    frameworks_found = []
    mention_counts = {}
    total_weighted_score = 0.0

    for keyword, weight in natsec_keywords.items():
        count = statement_lower.count(keyword)
        if count > 0:
            indicators_found.append(keyword)
            mention_counts[keyword] = count
            total_weighted_score += weight * min(count, 5)  # Cap at 5 mentions per keyword

    for framework, weight in legal_frameworks.items():
        count = statement_lower.count(framework)
        if count > 0:
            frameworks_found.append(framework)
            mention_counts[framework] = count
            total_weighted_score += weight * min(count, 5)

    # Calculate confidence based on weighted score
    # Maximum possible score ~150 (if all high-weight items appear)
    # Scale to 0-1 confidence
    max_possible_score = 100.0  # Approximate maximum
    raw_confidence = min(total_weighted_score / max_possible_score, 0.95)

    # Adjust based on number of unique indicators
    indicator_bonus = min(len(set(indicators_found + frameworks_found)) * 0.05, 0.20)
    confidence = min(raw_confidence + indicator_bonus, 0.95)

    # Determine classification strength
    if confidence >= 0.70:
        classification_strength = "strong"
    elif confidence >= 0.50:
        classification_strength = "moderate"
    elif confidence >= 0.35:
        classification_strength = "weak"
    else:
        classification_strength = "none"

    # Extract specific high-value mentions
    high_value_mentions = []
    if "xi jinping" in statement_lower or "xi mingze" in statement_lower:
        high_value_mentions.append("References to PRC leadership family (Xi Jinping/Xi Mingze)")
    if "harvard" in statement_lower and ("china" in statement_lower or "prc" in statement_lower):
        high_value_mentions.append("Harvard-China connections")
    if "us citizen" in statement_lower and ("endangerment" in statement_lower or "arrest" in statement_lower):
        high_value_mentions.append("US citizen endangerment in foreign jurisdiction")
    if "federal funding" in statement_lower or "defense grant" in statement_lower:
        high_value_mentions.append("Federal funding implications")
    if "arrest" in statement_lower and ("criminal" in statement_lower or "prosecution" in statement_lower):
        high_value_mentions.append("Criminal prosecution/arrest context")

    return {
        "classification_strength": classification_strength,
        "confidence": confidence,
        "weighted_score": total_weighted_score,
        "indicators": list(set(indicators_found)),
        "legal_frameworks": list(set(frameworks_found)),
        "mention_counts": mention_counts,
        "total_indicators": len(set(indicators_found)),
        "total_frameworks": len(set(frameworks_found)),
        "high_value_mentions": high_value_mentions,
        "assessment": _generate_assessment(classification_strength, confidence, high_value_mentions)
    }


def _generate_assessment(strength: str, confidence: float, high_value: List[str]) -> str:
    """Generate human-readable assessment."""

    if strength == "strong":
        verb = "DO"
        qualifier = "strongly"
    elif strength == "moderate":
        verb = "LIKELY DO"
        qualifier = "moderately"
    elif strength == "weak":
        verb = "MAY"
        qualifier = "potentially"
    else:
        verb = "DO NOT"
        qualifier = ""

    base = f"Based on analysis, the allegations {verb} constitute a matter of US national security (confidence: {confidence:.1%})"

    if high_value:
        reasons = "; ".join(high_value[:3])
        base += f". Key factors: {reasons}"

    return base


def main():
    """Main analysis function."""

    # Load statement of claim
    statement_path = Path(__file__).parent.parent / "case_law_data" / "tmp_corpus" / "Exhibit 2 — Certified Statement of Claim (Hong Kong, 2 Jun 2025).txt"

    if not statement_path.exists():
        print(f"ERROR: Statement file not found at {statement_path}")
        return

    with open(statement_path, 'r', encoding='utf-8') as f:
        statement_of_claim = f.read()

    print("="*70)
    print("NATIONAL SECURITY MATTER ANALYSIS")
    print("="*70)
    print(f"\nStatement of Claim: {len(statement_of_claim)} characters")
    print("Analyzing for US national security implications...\n")

    # Analyze
    results = analyze_national_security_indicators(statement_of_claim)

    # Print results
    print("="*70)
    print("RESULTS")
    print("="*70)

    print(f"\nClassification Strength: {results['classification_strength'].upper()}")
    print(f"Confidence Level: {results['confidence']:.1%}")
    print(f"Weighted Score: {results['weighted_score']:.1f}")

    print(f"\nAssessment:")
    print(f"  {results['assessment']}")

    print(f"\nNational Security Indicators Found: {results['total_indicators']}")
    if results['indicators']:
        print("  Top indicators:")
        for ind in sorted(results['indicators'], key=lambda x: results['mention_counts'].get(x, 0), reverse=True)[:10]:
            count = results['mention_counts'].get(ind, 0)
            print(f"    - {ind} ({count} mentions)")

    print(f"\nLegal Frameworks Identified: {results['total_frameworks']}")
    if results['legal_frameworks']:
        for fw in results['legal_frameworks']:
            count = results['mention_counts'].get(fw, 0)
            print(f"    - {fw} ({count} mentions)")

    print(f"\nHigh-Value Mentions:")
    for mention in results['high_value_mentions']:
        print(f"    - {mention}")

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "national_security_analysis_simple.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "analysis_type": "national_security_classification",
            "statement_length": len(statement_of_claim),
            **results
        }, f, indent=2)

    print(f"\n" + "="*70)
    print(f"Results saved to: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

