#!/usr/bin/env python3
"""
Classification Improvement Summary
Summarize the results of the enhanced classification.
"""

import json
from pathlib import Path
from collections import Counter

def summarize_classification_improvements():
    """Summarize the classification improvements."""

    print("🎯 CLASSIFICATION IMPROVEMENT SUMMARY")
    print("=" * 60)

    # Load enhanced classifications
    enhanced_file = Path("data/case_law/analysis_results/enhanced_classifications.json")
    if not enhanced_file.exists():
        print("❌ Enhanced classifications not found!")
        return

    with open(enhanced_file, 'r', encoding='utf-8') as f:
        enhanced_results = json.load(f)

    # Load original analysis for comparison
    results_dir = Path("data/case_law/analysis_results")
    json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

    if not json_files:
        print("❌ Original analysis not found!")
        return

    with open(json_files[0], 'r', encoding='utf-8') as f:
        original_results = json.load(f)

    # Analyze improvements
    original_analysis = original_results.get('analysis_results', {})

    # Count original classifications
    original_outcomes = Counter()
    for analysis in original_analysis.values():
        prediction = analysis.get('outcome_prediction', {}).get('prediction', 'unknown')
        original_outcomes[prediction] += 1

    # Count enhanced classifications
    enhanced_outcomes = Counter()
    confidence_scores = []

    for filename, classification in enhanced_results.items():
        prediction = classification.get('prediction', 'unknown')
        confidence = classification.get('confidence', 0.0)

        enhanced_outcomes[prediction] += 1
        confidence_scores.append(confidence)

    # Calculate improvements
    original_unclear = original_outcomes['unclear']
    enhanced_unclear = enhanced_outcomes['unclear']
    improvement_count = original_unclear - enhanced_unclear

    print(f"\n📊 BEFORE vs AFTER:")
    print(f"  Original unclassified: {original_unclear}")
    print(f"  Enhanced unclassified: {enhanced_unclear}")
    print(f"  Cases re-classified: {improvement_count}")
    print(f"  Improvement rate: {improvement_count/original_unclear*100:.1f}%")

    print(f"\n⚖️ ORIGINAL OUTCOME DISTRIBUTION:")
    total_original = sum(original_outcomes.values())
    for outcome, count in original_outcomes.most_common():
        percentage = count/total_original*100
        print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    print(f"\n⚖️ ENHANCED OUTCOME DISTRIBUTION:")
    total_enhanced = sum(enhanced_outcomes.values())
    for outcome, count in enhanced_outcomes.most_common():
        percentage = count/total_enhanced*100
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

    # Show some examples of re-classified cases
    print(f"\n✅ SAMPLE RE-CLASSIFIED CASES:")
    reclassified_examples = []

    for filename, classification in enhanced_results.items():
        if classification.get('prediction') != 'unclear':
            reclassified_examples.append({
                'filename': filename,
                'prediction': classification.get('prediction'),
                'confidence': classification.get('confidence', 0.0),
                'reasoning': classification.get('reasoning', '')
            })

    # Sort by confidence
    reclassified_examples.sort(key=lambda x: x['confidence'], reverse=True)

    for i, case in enumerate(reclassified_examples[:10]):
        print(f"  {i+1:2d}. {case['filename']}")
        print(f"      → {case['prediction'].title()} (confidence: {case['confidence']:.2f})")
        print(f"      Reasoning: {case['reasoning']}")
        print()

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🎯 FINAL IMPROVEMENT SUMMARY")
    print(f"=" * 60)

    print(f"\n📈 CLASSIFICATION IMPROVEMENTS:")
    print(f"  • Re-classified {improvement_count} cases ({improvement_count/original_unclear*100:.1f}% improvement)")
    print(f"  • Reduced unclear cases from {original_unclear} to {enhanced_unclear}")
    print(f"  • Average confidence: {avg_confidence:.2f}")

    print(f"\n📊 NEW CLASSIFICATION BREAKDOWN:")
    print(f"  • Granted: {enhanced_outcomes['granted']} cases")
    print(f"  • Denied: {enhanced_outcomes['denied']} cases")
    print(f"  • Mixed: {enhanced_outcomes['mixed']} cases")
    print(f"  • Unclear: {enhanced_outcomes['unclear']} cases")

    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Manual review of remaining {enhanced_unclear} unclear cases")
    print(f"  2. Focus on cases with confidence < 0.5")
    print(f"  3. Cross-reference multiple documents per case")
    print(f"  4. Use enhanced patterns for future cases")

if __name__ == "__main__":
    summarize_classification_improvements()
