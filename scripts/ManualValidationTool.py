#!/usr/bin/env python3
"""
Manual Validation Tool for Heuristic Cases

This script helps you manually validate the 75 heuristic cases by:
1. Loading the advanced adjudicated cases
2. Presenting each case with its current label and reasoning
3. Allowing you to update the label based on manual review
4. Saving the validated results

Usage: python scripts/manual_validation_tool.py
"""

import json
import os
from typing import Dict, List, Any
from datetime import datetime

class ManualValidator:
    def __init__(self):
        self.adjudicated_file = "data/case_law/advanced_adjudicated_cases.json"
        self.validation_file = "data/case_law/manual_validation_results.json"
        self.cases = []
        self.validated_cases = []

    def load_adjudicated_cases(self):
        """Load the 75 heuristic cases."""
        print("🔍 Loading heuristic cases...")

        if not os.path.exists(self.adjudicated_file):
            print(f"❌ File not found: {self.adjudicated_file}")
            return False

        with open(self.adjudicated_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.cases = data['adjudicated_cases']
        print(f"✅ Loaded {len(self.cases)} heuristic cases")
        return True

    def display_case_summary(self):
        """Display summary of cases by outcome."""
        print("\n📊 Case Summary by Heuristic Label:")

        outcome_counts = {}
        for case in self.cases:
            outcome = case['adjudicated_outcome']
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        for outcome, count in outcome_counts.items():
            print(f"  {outcome}: {count} cases")

    def get_case_text(self, case: Dict[str, Any]) -> str:
        """Get the text content for a case."""
        # Try to find the actual case file
        file_name = case['file_name']
        case_file = f"data/case_law/1782_discovery/{file_name}.json"

        if os.path.exists(case_file):
            try:
                with open(case_file, 'r', encoding='utf-8') as f:
                    case_data = json.load(f)

                # Combine all available text
                text_parts = []

                if case_data.get('opinion_text'):
                    text_parts.append(f"OPINION TEXT:\n{case_data['opinion_text'][:2000]}...")

                if case_data.get('caseNameFull'):
                    text_parts.append(f"CASE NAME: {case_data['caseNameFull']}")

                if case_data.get('attorney_text'):
                    text_parts.append(f"ATTORNEY TEXT:\n{case_data['attorney_text'][:1000]}...")

                if case_data.get('extracted_text'):
                    text_parts.append(f"EXTRACTED TEXT:\n{case_data['extracted_text'][:2000]}...")

                return "\n\n".join(text_parts) if text_parts else "No text available"

            except Exception as e:
                return f"Error loading case file: {e}"
        else:
            return f"Case file not found: {case_file}"

    def validate_case(self, case: Dict[str, Any], case_index: int) -> Dict[str, Any]:
        """Validate a single case."""
        print(f"\n{'='*80}")
        print(f"📋 CASE {case_index + 1} of {len(self.cases)}")
        print(f"{'='*80}")

        # Basic case info
        print(f"File: {case['file_name']}")
        print(f"Cluster ID: {case['cluster_id']}")
        print(f"Court: {case['court_id']}")
        print(f"Date Filed: {case['date_filed']}")

        # Current heuristic label
        print(f"\n🤖 HEURISTIC LABEL: {case['adjudicated_outcome']}")
        print(f"Confidence: {case['confidence_score']:.3f}")
        print(f"Reasoning: {case['reasoning']}")

        # Pattern matches
        patterns = case['pattern_matches']
        print(f"\n📊 Pattern Matches:")
        print(f"  Strong Success: {patterns['strong_success']}")
        print(f"  Strong Failure: {patterns['strong_failure']}")
        print(f"  Contextual Success: {patterns['contextual_success']}")
        print(f"  Contextual Failure: {patterns['contextual_failure']}")

        # Get case text
        print(f"\n📄 CASE TEXT:")
        case_text = self.get_case_text(case)
        print(case_text)

        # Manual validation
        print(f"\n{'='*80}")
        print("🔍 MANUAL VALIDATION")
        print(f"{'='*80}")
        print("Based on your review of the case text, what is the actual outcome?")
        print("\nOptions:")
        print("  SUCCESS - Application/motion was granted")
        print("  FAILURE - Application/motion was denied")
        print("  MIXED   - Partially granted/denied")
        print("  UNCLEAR - Cannot determine from available text")
        print("  SKIP    - Skip this case for now")

        while True:
            choice = input("\nEnter your validation (SUCCESS/FAILURE/MIXED/UNCLEAR/SKIP): ").strip().upper()

            if choice in ['SUCCESS', 'FAILURE', 'MIXED', 'UNCLEAR', 'SKIP']:
                break
            else:
                print("❌ Invalid choice. Please enter SUCCESS, FAILURE, MIXED, UNCLEAR, or SKIP")

        if choice == 'SKIP':
            return None

        # Get additional notes
        notes = input("\n📝 Additional notes (optional): ").strip()

        # Create validated case
        validated_case = case.copy()
        validated_case['manual_validation'] = {
            'validated_outcome': choice,
            'validation_date': datetime.now().isoformat(),
            'validator_notes': notes,
            'original_heuristic_outcome': case['adjudicated_outcome'],
            'original_confidence': case['confidence_score']
        }

        return validated_case

    def run_validation(self):
        """Run the manual validation process."""
        print("🚀 Starting Manual Validation Process")
        print("="*80)

        if not self.load_adjudicated_cases():
            return

        self.display_case_summary()

        print(f"\n📋 You will be reviewing {len(self.cases)} cases.")
        print("Press Ctrl+C at any time to save progress and exit.")

        try:
            for i, case in enumerate(self.cases):
                validated_case = self.validate_case(case, i)

                if validated_case:
                    self.validated_cases.append(validated_case)
                    print(f"✅ Case {i+1} validated as: {validated_case['manual_validation']['validated_outcome']}")
                else:
                    print(f"⏭️  Case {i+1} skipped")

                # Ask if user wants to continue
                if i < len(self.cases) - 1:
                    continue_choice = input(f"\nContinue to next case? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        break

        except KeyboardInterrupt:
            print(f"\n\n⏹️  Validation interrupted. Saving progress...")

        self.save_validation_results()

    def save_validation_results(self):
        """Save the validation results."""
        if not self.validated_cases:
            print("❌ No cases validated. Nothing to save.")
            return

        results = {
            'validation_date': datetime.now().isoformat(),
            'total_cases_reviewed': len(self.validated_cases),
            'total_heuristic_cases': len(self.cases),
            'validation_summary': self.get_validation_summary(),
            'validated_cases': self.validated_cases
        }

        with open(self.validation_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Validation results saved to: {self.validation_file}")
        print(f"📊 Validated {len(self.validated_cases)} cases")

        # Display summary
        summary = self.get_validation_summary()
        print(f"\n📈 Validation Summary:")
        for outcome, count in summary.items():
            print(f"  {outcome}: {count} cases")

    def get_validation_summary(self) -> Dict[str, int]:
        """Get summary of validation results."""
        summary = {}
        for case in self.validated_cases:
            outcome = case['manual_validation']['validated_outcome']
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

def main():
    """Main function."""
    print("🔍 Manual Validation Tool for Heuristic Cases")
    print("="*80)

    validator = ManualValidator()
    validator.run_validation()

    print("\n🎉 Manual validation process complete!")
    print("Next steps:")
    print("1. Review the validation results")
    print("2. Retrain the model with validated cases")
    print("3. Compare model performance before/after validation")

if __name__ == "__main__":
    main()
