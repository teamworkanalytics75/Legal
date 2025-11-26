#!/usr/bin/env python3
"""
Create Combined Analysis
Integrate enhanced classifications with original analysis.
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

def create_combined_analysis():
    """Create a combined analysis with enhanced classifications."""

    print("🔧 CREATING COMBINED ANALYSIS")
    print("=" * 50)

    # Load original analysis
    results_dir = Path("data/case_law/analysis_results")
    json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

    if not json_files:
        print("❌ Original analysis not found!")
        return

    with open(json_files[0], 'r', encoding='utf-8') as f:
        original_results = json.load(f)

    # Load enhanced classifications
    enhanced_file = Path("data/case_law/analysis_results/enhanced_classifications.json")
    if not enhanced_file.exists():
        print("❌ Enhanced classifications not found!")
        return

    with open(enhanced_file, 'r', encoding='utf-8') as f:
        enhanced_results = json.load(f)

    print(f"📄 Merging {len(original_results.get('analysis_results', {}))} original cases")
    print(f"📄 With {len(enhanced_results)} enhanced classifications")

    # Create combined analysis
    combined_results = original_results.copy()
    analysis_results = combined_results.get('analysis_results', {})

    # Update classifications
    updated_count = 0
    for filename, enhanced_classification in enhanced_results.items():
        if filename in analysis_results:
            # Update the outcome prediction
            analysis_results[filename]['outcome_prediction'] = {
                'prediction': enhanced_classification['prediction'],
                'confidence': enhanced_classification['confidence'],
                'reasoning': enhanced_classification['reasoning']
            }
            updated_count += 1

    print(f"✅ Updated {updated_count} classifications")

    # Recalculate statistics
    outcome_counts = Counter()
    confidence_scores = []

    for analysis in analysis_results.values():
        outcome_pred = analysis.get('outcome_prediction', {})
        prediction = outcome_pred.get('prediction', 'unknown')
        confidence = outcome_pred.get('confidence', 0.0)

        outcome_counts[prediction] += 1
        confidence_scores.append(confidence)

    # Update summary statistics
    total_cases = len(analysis_results)
    classified_cases = total_cases - outcome_counts['unclear']

    combined_results['summary'] = {
        'total_cases': total_cases,
        'classified_cases': classified_cases,
        'unclear_cases': outcome_counts['unclear'],
        'classification_rate': classified_cases/total_cases*100,
        'outcome_distribution': dict(outcome_counts),
        'average_confidence': sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0,
        'integration_date': datetime.now().isoformat(),
        'enhanced_classifications_integrated': True
    }

    # Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_dir / f"1782_combined_analysis_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n📊 COMBINED ANALYSIS STATISTICS:")
    print(f"  Total cases: {total_cases}")
    print(f"  Classified cases: {classified_cases}")
    print(f"  Unclear cases: {outcome_counts['unclear']}")
    print(f"  Classification rate: {classified_cases/total_cases*100:.1f}%")
    print(f"  Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}")

    print(f"\n📊 OUTCOME DISTRIBUTION:")
    for outcome, count in outcome_counts.most_common():
        percentage = count/total_cases*100
        print(f"  {outcome.title()}: {count} cases ({percentage:.1f}%)")

    print(f"\n💾 Combined analysis saved to: {output_file}")
    print(f"\n🎯 Model now incorporates enhanced classifications!")

    return output_file

if __name__ == "__main__":
    create_combined_analysis()
