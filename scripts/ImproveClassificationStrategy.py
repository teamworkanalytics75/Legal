#!/usr/bin/env python3
"""
Improved Classification Strategy
Analyze unclassified cases and develop better classification methods.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any, Tuple

def load_analysis_results():
    """Load the latest analysis results."""
    results_dir = Path("data/case_law/analysis_results")
    json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

    if not json_files:
        return None

    with open(json_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_unclassified_patterns():
    """Analyze patterns in unclassified cases to improve classification."""

    results = load_analysis_results()
    if not results:
        print("❌ No analysis results found!")
        return

    print("🔍 ANALYZING UNCLASSIFIED CASES FOR IMPROVEMENT")
    print("=" * 60)

    analysis_results = results.get('analysis_results', {})

    # Separate classified vs unclassified
    unclassified = []
    classified = []

    for filename, analysis in analysis_results.items():
        prediction = analysis.get('outcome_prediction', {}).get('prediction', 'unknown')
        if prediction == 'unclear':
            unclassified.append((filename, analysis))
        else:
            classified.append((filename, analysis))

    print(f"📊 Found {len(unclassified)} unclassified cases to analyze")

    # Analyze text content of unclassified cases
    print(f"\n🔍 ANALYZING UNCLASSIFIED CASE CONTENT:")

    # Load text files to analyze content
    text_dir = Path("data/case_law/extracted_text")

    unclassified_patterns = defaultdict(int)
    unclassified_keywords = defaultdict(int)
    unclassified_doc_types = defaultdict(int)

    sample_unclassified = []

    for i, (filename, analysis) in enumerate(unclassified[:20]):  # Analyze first 20
        text_file = text_dir / filename
        if text_file.exists():
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Extract case info
            case_info = filename.replace('.txt', '').split('_')
            if len(case_info) >= 3:
                doc_type = '_'.join(case_info[2:])
                unclassified_doc_types[doc_type] += 1

            # Look for patterns that might indicate outcomes
            text_lower = text.lower()

            # Check for common outcome indicators
            outcome_patterns = {
                'granted': ['granted', 'approve', 'allow', 'permit', 'authorize', 'order granting'],
                'denied': ['denied', 'deny', 'reject', 'refuse', 'dismiss', 'order denying'],
                'motion': ['motion', 'motion to', 'motion for'],
                'order': ['order', 'ordered', 'ordering'],
                'memorandum': ['memorandum', 'memorandum opinion'],
                'discovery': ['discovery', 'subpoena', 'deposition'],
                'ex parte': ['ex parte', 'ex-parte'],
                'application': ['application', 'applied', 'applying']
            }

            for pattern_type, keywords in outcome_patterns.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        unclassified_patterns[pattern_type] += 1
                        unclassified_keywords[keyword] += 1

            sample_unclassified.append({
                'filename': filename,
                'text_preview': text[:200] + "..." if len(text) > 200 else text,
                'text_length': len(text),
                'doc_type': doc_type if len(case_info) >= 3 else 'unknown'
            })

    print(f"\n📋 UNCLASSIFIED CASE PATTERNS:")
    for pattern, count in sorted(unclassified_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} cases")

    print(f"\n🔑 TOP KEYWORDS IN UNCLASSIFIED CASES:")
    for keyword, count in sorted(unclassified_keywords.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  '{keyword}': {count} cases")

    print(f"\n📄 DOCUMENT TYPES IN UNCLASSIFIED CASES:")
    for doc_type, count in sorted(unclassified_doc_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {doc_type}: {count} cases")

    print(f"\n📝 SAMPLE UNCLASSIFIED CASES:")
    for i, case in enumerate(sample_unclassified[:5]):
        print(f"\n  {i+1}. {case['filename']}")
        print(f"     Type: {case['doc_type']}")
        print(f"     Length: {case['text_length']} chars")
        print(f"     Preview: {case['text_preview']}")

def create_improved_classifier():
    """Create an improved classification strategy."""

    print(f"\n" + "=" * 60)
    print(f"🚀 IMPROVED CLASSIFICATION STRATEGY")
    print(f"=" * 60)

    print(f"\n📋 STRATEGY 1: ENHANCED KEYWORD MATCHING")
    print(f"  • Expand keyword lists for each outcome")
    print(f"  • Add context-aware pattern matching")
    print(f"  • Include legal terminology variations")
    print(f"  • Weight keywords by legal significance")

    print(f"\n📋 STRATEGY 2: DOCUMENT TYPE ANALYSIS")
    print(f"  • Classify by document type first")
    print(f"  • Orders → Look for 'granted'/'denied'")
    print(f"  • Petitions → Look for outcome indicators")
    print(f"  • Exhibits → May need different approach")

    print(f"\n📋 STRATEGY 3: MULTI-DOCUMENT ANALYSIS")
    print(f"  • Analyze cases with multiple documents together")
    print(f"  • Cross-reference petition + order")
    print(f"  • Look for outcome progression")

    print(f"\n📋 STRATEGY 4: MANUAL REVIEW PRIORITIZATION")
    print(f"  • Start with original petition cases")
    print(f"  • Focus on cases with 'order' documents")
    print(f"  • Prioritize cases with clear legal language")

    print(f"\n📋 STRATEGY 5: CONFIDENCE-BASED CLASSIFICATION")
    print(f"  • Use confidence scores to guide manual review")
    print(f"  • Auto-classify high-confidence cases")
    print(f"  • Flag medium-confidence for review")
    print(f"  • Manual review low-confidence cases")

def implement_enhanced_classifier():
    """Implement an enhanced classification system."""

    print(f"\n" + "=" * 60)
    print(f"🔧 IMPLEMENTING ENHANCED CLASSIFIER")
    print(f"=" * 60)

    # Enhanced keyword patterns
    enhanced_patterns = {
        'granted': {
            'strong': ['granted', 'approve', 'allow', 'permit', 'authorize', 'order granting', 'motion granted'],
            'medium': ['favor', 'support', 'uphold', 'sustain', 'affirm'],
            'weak': ['may', 'can', 'could', 'should']
        },
        'denied': {
            'strong': ['denied', 'deny', 'reject', 'refuse', 'dismiss', 'order denying', 'motion denied'],
            'medium': ['oppose', 'object', 'overrule', 'reverse'],
            'weak': ['cannot', 'may not', 'should not']
        },
        'mixed': {
            'strong': ['partially', 'in part', 'some', 'certain'],
            'medium': ['condition', 'subject to', 'provided that'],
            'weak': ['however', 'but', 'although']
        }
    }

    # Document type classification rules
    doc_type_rules = {
        'order': {
            'priority': 'high',
            'look_for': ['granted', 'denied', 'approved', 'rejected'],
            'context': 'Court decisions'
        },
        'petition': {
            'priority': 'high',
            'look_for': ['request', 'application', 'motion'],
            'context': 'Initial requests'
        },
        'memorandum': {
            'priority': 'medium',
            'look_for': ['opinion', 'analysis', 'recommendation'],
            'context': 'Legal analysis'
        },
        'exhibit': {
            'priority': 'low',
            'look_for': ['evidence', 'document', 'attachment'],
            'context': 'Supporting materials'
        }
    }

    print(f"\n📊 ENHANCED KEYWORD WEIGHTS:")
    for outcome, keywords in enhanced_patterns.items():
        print(f"\n  {outcome.upper()}:")
        for strength, words in keywords.items():
            print(f"    {strength}: {', '.join(words[:5])}{'...' if len(words) > 5 else ''}")

    print(f"\n📄 DOCUMENT TYPE PRIORITIES:")
    for doc_type, rules in doc_type_rules.items():
        print(f"  {doc_type}: {rules['priority']} priority - {rules['context']}")

    return enhanced_patterns, doc_type_rules

def create_classification_script():
    """Create a script to implement the improved classification."""

    script_content = '''#!/usr/bin/env python3
"""
Enhanced 1782 Case Classifier
Improved classification system for unclassified cases.
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

class Enhanced1782Classifier:
    def __init__(self):
        self.enhanced_patterns = {
            'granted': {
                'strong': ['granted', 'approve', 'allow', 'permit', 'authorize', 'order granting', 'motion granted', 'application granted'],
                'medium': ['favor', 'support', 'uphold', 'sustain', 'affirm', 'accept', 'endorse'],
                'weak': ['may', 'can', 'could', 'should', 'recommend']
            },
            'denied': {
                'strong': ['denied', 'deny', 'reject', 'refuse', 'dismiss', 'order denying', 'motion denied', 'application denied'],
                'medium': ['oppose', 'object', 'overrule', 'reverse', 'decline', 'refuse'],
                'weak': ['cannot', 'may not', 'should not', 'unable to']
            },
            'mixed': {
                'strong': ['partially', 'in part', 'some', 'certain', 'limited', 'conditional'],
                'medium': ['condition', 'subject to', 'provided that', 'however', 'but'],
                'weak': ['although', 'while', 'despite']
            }
        }

        self.doc_type_rules = {
            'order': {'priority': 'high', 'weight': 3.0},
            'petition': {'priority': 'high', 'weight': 2.5},
            'memorandum': {'priority': 'medium', 'weight': 2.0},
            'declaration': {'priority': 'medium', 'weight': 1.5},
            'exhibit': {'priority': 'low', 'weight': 1.0},
            'document': {'priority': 'low', 'weight': 1.0}
        }

    def classify_case(self, text: str, filename: str) -> Dict[str, any]:
        """Classify a case using enhanced methods."""

        # Extract document type
        doc_type = self._extract_doc_type(filename)
        doc_weight = self.doc_type_rules.get(doc_type, {}).get('weight', 1.0)

        # Score each outcome
        scores = {}
        for outcome, keywords in self.enhanced_patterns.items():
            score = 0
            text_lower = text.lower()

            for strength, words in keywords.items():
                strength_weight = {'strong': 3, 'medium': 2, 'weak': 1}[strength]

                for word in words:
                    count = text_lower.count(word)
                    score += count * strength_weight

            # Apply document type weighting
            scores[outcome] = score * doc_weight

        # Determine prediction
        if not any(scores.values()):
            prediction = 'unclear'
            confidence = 0.0
            reasoning = 'No outcome indicators found'
        else:
            max_score = max(scores.values())
            prediction = max(scores, key=scores.get)

            # Calculate confidence based on score distribution
            total_score = sum(scores.values())
            confidence = min(max_score / max(total_score, 1), 1.0)

            reasoning = f'Found {max_score:.1f} points for {prediction} (total: {total_score:.1f})'

        return {
            'prediction': prediction,
            'confidence': confidence,
            'reasoning': reasoning,
            'scores': scores,
            'doc_type': doc_type,
            'doc_weight': doc_weight
        }

    def _extract_doc_type(self, filename: str) -> str:
        """Extract document type from filename."""
        parts = filename.replace('.txt', '').split('_')
        if len(parts) >= 3:
            doc_type = '_'.join(parts[2:]).lower()

            # Map to standard types
            if 'order' in doc_type:
                return 'order'
            elif 'petition' in doc_type:
                return 'petition'
            elif 'memorandum' in doc_type:
                return 'memorandum'
            elif 'declaration' in doc_type:
                return 'declaration'
            elif 'exhibit' in doc_type or 'att' in doc_type:
                return 'exhibit'
            else:
                return 'document'
        return 'document'

    def batch_classify(self, text_dir: Path, output_file: Path):
        """Classify all unclassified cases."""

        # Load existing results
        results_dir = Path("data/case_law/analysis_results")
        json_files = sorted(results_dir.glob("1782_basic_analysis_*.json"), reverse=True)

        if not json_files:
            print("❌ No existing analysis found!")
            return

        with open(json_files[0], 'r', encoding='utf-8') as f:
            existing_results = json.load(f)

        analysis_results = existing_results.get('analysis_results', {})

        # Find unclassified cases
        unclassified_files = []
        for filename, analysis in analysis_results.items():
            prediction = analysis.get('outcome_prediction', {}).get('prediction', 'unknown')
            if prediction == 'unclear':
                unclassified_files.append(filename)

        print(f"🔍 Re-classifying {len(unclassified_files)} unclassified cases...")

        new_classifications = {}
        improved_count = 0

        for filename in unclassified_files:
            text_file = text_dir / filename
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                classification = self.classify_case(text, filename)
                new_classifications[filename] = classification

                if classification['prediction'] != 'unclear':
                    improved_count += 1
                    print(f"✅ {filename}: {classification['prediction']} (confidence: {classification['confidence']:.2f})")

        print(f"\n📊 IMPROVEMENT RESULTS:")
        print(f"  Cases re-classified: {improved_count}")
        print(f"  Remaining unclear: {len(unclassified_files) - improved_count}")
        print(f"  Improvement rate: {improved_count/len(unclassified_files)*100:.1f}%")

        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_classifications, f, indent=2)

        print(f"\n💾 Enhanced classifications saved to: {output_file}")

def main():
    classifier = Enhanced1782Classifier()
    text_dir = Path("data/case_law/extracted_text")
    output_file = Path("data/case_law/analysis_results/enhanced_classifications.json")

    classifier.batch_classify(text_dir, output_file)

if __name__ == "__main__":
    main()
'''

    with open("scripts/enhanced_1782_classifier.py", 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"\n💾 Created enhanced classifier script: scripts/enhanced_1782_classifier.py")

def main():
    analyze_unclassified_patterns()
    create_improved_classifier()
    implement_enhanced_classifier()
    create_classification_script()

    print(f"\n" + "=" * 60)
    print(f"🎯 NEXT STEPS TO IMPROVE CLASSIFICATION")
    print(f"=" * 60)
    print(f"\n1. Run enhanced classifier:")
    print(f"   py scripts/enhanced_1782_classifier.py")
    print(f"\n2. Manual review of remaining unclear cases")
    print(f"\n3. Focus on original petition cases first")
    print(f"\n4. Use document type prioritization")
    print(f"\n5. Cross-reference multiple documents per case")

if __name__ == "__main__":
    main()
