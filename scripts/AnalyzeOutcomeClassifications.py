#!/usr/bin/env python3
"""
Analyze Outcome Classifications
Check how many cases were classified vs unclassified in our NLP analysis.
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_outcome_classifications():
    """Analyze the outcome classifications from our NLP analysis."""

    # Find the latest analysis results file
    results_dir = Path("data/case_law/analysis_results")
    json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

    if not json_files:
        print("❌ No analysis results found!")
        return

    latest_file = json_files[0]
    print(f"📄 Reading analysis from: {latest_file.name}")

    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print("\n🔍 OUTCOME CLASSIFICATION ANALYSIS")
    print("=" * 60)

    # Analyze outcome predictions
    outcome_counts = Counter()
    confidence_scores = []
    unclear_cases = []
    classified_cases = []

    analysis_results = results.get('analysis_results', {})

    for filename, analysis in analysis_results.items():
        outcome_pred = analysis.get('outcome_prediction', {})
        prediction = outcome_pred.get('prediction', 'unknown')
        confidence = outcome_pred.get('confidence', 0.0)
        reasoning = outcome_pred.get('reasoning', '')

        outcome_counts[prediction] += 1
        confidence_scores.append(confidence)

        if prediction == 'unclear':
            unclear_cases.append({
                'filename': filename,
                'confidence': confidence,
                'reasoning': reasoning
            })
        else:
            classified_cases.append({
                'filename': filename,
                'prediction': prediction,
                'confidence': confidence,
                'reasoning': reasoning
            })

    # Summary statistics
    total_cases = len(analysis_results)
    unclear_count = outcome_counts['unclear']
    classified_count = total_cases - unclear_count

    print(f"\n📊 OUTCOME CLASSIFICATION SUMMARY:")
    print(f"  Total cases analyzed: {total_cases}")
    print(f"  Cases with clear outcomes: {classified_count}")
    print(f"  Cases with unclear outcomes: {unclear_count}")
    print(f"  Classification rate: {classified_count/total_cases*100:.1f}%")
    print(f"  Unclear rate: {unclear_count/total_cases*100:.1f}%")

    print(f"\n⚖️ OUTCOME BREAKDOWN:")
    for outcome, count in outcome_counts.most_common():
        percentage = count/total_cases*100
        print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    # Confidence analysis
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    high_confidence = sum(1 for c in confidence_scores if c >= 0.7)
    medium_confidence = sum(1 for c in confidence_scores if 0.3 <= c < 0.7)
    low_confidence = sum(1 for c in confidence_scores if c < 0.3)

    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"  Average confidence: {avg_confidence:.2f}")
    print(f"  High confidence (≥0.7): {high_confidence} cases")
    print(f"  Medium confidence (0.3-0.7): {medium_confidence} cases")
    print(f"  Low confidence (<0.3): {low_confidence} cases")

    # Show some examples of unclear cases
    print(f"\n❓ SAMPLE UNCLEAR CASES:")
    for i, case in enumerate(unclear_cases[:10]):
        print(f"  {i+1:2d}. {case['filename']}")
        print(f"      Confidence: {case['confidence']:.2f}")
        print(f"      Reasoning: {case['reasoning']}")
        print()

    if len(unclear_cases) > 10:
        print(f"  ... and {len(unclear_cases) - 10} more unclear cases")

    # Show some examples of classified cases
    print(f"\n✅ SAMPLE CLASSIFIED CASES:")
    for i, case in enumerate(classified_cases[:10]):
        print(f"  {i+1:2d}. {case['filename']}")
        print(f"      Prediction: {case['prediction']}")
        print(f"      Confidence: {case['confidence']:.2f}")
        print(f"      Reasoning: {case['reasoning']}")
        print()

    if len(classified_cases) > 10:
        print(f"  ... and {len(classified_cases) - 10} more classified cases")

    # Cross-reference with original petitions
    print(f"\n🔗 CROSS-REFERENCE WITH ORIGINAL PETITIONS:")

    # Load our case analysis to see which cases have petitions
    case_analysis_file = Path("data/case_law/analysis_results/complete_1782_analysis.json")
    if case_analysis_file.exists():
        with open(case_analysis_file, 'r', encoding='utf-8') as f:
            case_data = json.load(f)

        cases_with_petitions = set()
        case_details = case_data.get('case_details', {})
        for case_id, details in case_details.items():
            if details.get('has_petition', False):
                cases_with_petitions.add(case_id)

        # Check how many petition cases are classified vs unclear
        petition_classified = 0
        petition_unclear = 0

        for filename, analysis in analysis_results.items():
            case_id = filename.split('_')[0]
            if case_id in cases_with_petitions:
                prediction = analysis.get('outcome_prediction', {}).get('prediction', 'unknown')
                if prediction == 'unclear':
                    petition_unclear += 1
                else:
                    petition_classified += 1

        print(f"  Cases with original petitions:")
        print(f"    Classified: {petition_classified}")
        print(f"    Unclear: {petition_unclear}")
        print(f"    Total petition cases: {len(cases_with_petitions)}")

    print(f"\n" + "=" * 60)
    print(f"🎯 ANSWER TO YOUR QUESTION")
    print(f"=" * 60)
    print(f"\n📊 YES - We have {unclear_count} cases ({unclear_count/total_cases*100:.1f}%)")
    print(f"   that haven't been classified as granted/denied/mixed")
    print(f"   and don't yet feature in our model.")
    print(f"\n📈 BREAKDOWN:")
    print(f"   • {classified_count} cases classified ({classified_count/total_cases*100:.1f}%)")
    print(f"   • {unclear_count} cases unclassified ({unclear_count/total_cases*100:.1f}%)")
    print(f"\n🎯 These {unclear_count} unclassified cases represent")
    print(f"   potential training data for improving our model!")

if __name__ == "__main__":
    analyze_outcome_classifications()
