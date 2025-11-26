#!/usr/bin/env python3
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
