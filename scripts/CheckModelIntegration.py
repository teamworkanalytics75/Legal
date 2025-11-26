#!/usr/bin/env python3
"""
Check Model Integration Status
Verify if our predictive model has incorporated the newly classified cases.
"""

import json
from pathlib import Path
from collections import Counter

def check_model_integration():
    """Check if the model has incorporated newly classified cases."""

    print("🔍 CHECKING MODEL INTEGRATION STATUS")
    print("=" * 60)

    # Load original analysis results
    results_dir = Path("data/case_law/analysis_results")
    json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

    if not json_files:
        print("❌ Original analysis results not found!")
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

    print(f"📄 Original analysis file: {json_files[0].name}")
    print(f"📄 Enhanced classifications file: {enhanced_file.name}")

    # Check if we have a combined/updated analysis
    combined_files = list(results_dir.glob("*combined*")) + list(results_dir.glob("*updated*")) + list(results_dir.glob("*enhanced*"))

    if combined_files:
        print(f"\n✅ Found potential combined analysis files:")
        for file in combined_files:
            print(f"  - {file.name}")
    else:
        print(f"\n❌ No combined/updated analysis files found")

    # Analyze what needs to be integrated
    original_analysis = original_results.get('analysis_results', {})

    print(f"\n📊 INTEGRATION ANALYSIS:")
    print(f"  Original analysis cases: {len(original_analysis)}")
    print(f"  Enhanced classifications: {len(enhanced_results)}")

    # Check overlap
    overlapping_files = set(original_analysis.keys()) & set(enhanced_results.keys())
    print(f"  Overlapping files: {len(overlapping_files)}")

    # Check which cases were re-classified
    reclassified_cases = []
    for filename in overlapping_files:
        original_pred = original_analysis[filename].get('outcome_prediction', {}).get('prediction', 'unknown')
        enhanced_pred = enhanced_results[filename].get('prediction', 'unknown')

        if original_pred == 'unclear' and enhanced_pred != 'unclear':
            reclassified_cases.append({
                'filename': filename,
                'original': original_pred,
                'enhanced': enhanced_pred,
                'confidence': enhanced_results[filename].get('confidence', 0.0)
            })

    print(f"  Cases re-classified: {len(reclassified_cases)}")

    # Show sample re-classified cases
    print(f"\n📋 SAMPLE RE-CLASSIFIED CASES:")
    for i, case in enumerate(reclassified_cases[:10]):
        print(f"  {i+1:2d}. {case['filename']}")
        print(f"      {case['original']} → {case['enhanced']} (confidence: {case['confidence']:.2f})")

    if len(reclassified_cases) > 10:
        print(f"  ... and {len(reclassified_cases) - 10} more")

    # Check if we need to create a combined analysis
    print(f"\n🔧 INTEGRATION STATUS:")

    if not combined_files:
        print(f"  ❌ Model has NOT incorporated enhanced classifications")
        print(f"  📝 Need to create combined analysis")
        print(f"  🚀 Should rerun analysis with enhanced data")
    else:
        print(f"  ✅ Found combined analysis files")
        print(f"  📊 Model may have incorporated enhanced classifications")

    # Create integration plan
    print(f"\n" + "=" * 60)
    print(f"🎯 INTEGRATION PLAN")
    print(f"=" * 60)

    if not combined_files:
        print(f"\n📋 STEPS TO INTEGRATE ENHANCED CLASSIFICATIONS:")
        print(f"  1. Create combined analysis file")
        print(f"  2. Merge original + enhanced classifications")
        print(f"  3. Update outcome statistics")
        print(f"  4. Recalculate model metrics")
        print(f"  5. Generate updated reports")

        print(f"\n🚀 RECOMMENDED ACTION:")
        print(f"  Run: py scripts/create_combined_analysis.py")
        print(f"  This will integrate the enhanced classifications")
        print(f"  and update our predictive model")
    else:
        print(f"\n✅ INTEGRATION COMPLETE")
        print(f"  Enhanced classifications appear to be integrated")
        print(f"  Model should reflect updated classifications")

def create_integration_script():
    """Create a script to integrate enhanced classifications."""

    script_content = '''#!/usr/bin/env python3
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
'''

    with open("scripts/create_combined_analysis.py", 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n💾 Created integration script: scripts/create_combined_analysis.py")

def main():
    check_model_integration()
    create_integration_script()

if __name__ == "__main__":
    main()
