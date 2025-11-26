#!/usr/bin/env python3
"""
Petition Writing Guidance Generator

Generates practical petition writing guidance based on model feature importance.
Creates actionable drafting checklist and recommendations.

Usage: python scripts/generate_writing_guidance.py
"""

import json
import pickle
import os
from typing import Dict, List, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WritingGuidanceGenerator:
    def __init__(self):
        self.model_dir = "data/case_law/petition_model"
        self.feature_importance_file = os.path.join(self.model_dir, "feature_importance.json")
        self.model_results_file = os.path.join(self.model_dir, "model_results.json")
        self.output_file = "data/case_law/petition_writing_guidance.md"

    def load_feature_importance(self) -> Dict[str, Any]:
        """Load feature importance data."""
        logger.info("Loading feature importance data...")

        if not os.path.exists(self.feature_importance_file):
            logger.error(f"Feature importance file not found: {self.feature_importance_file}")
            return {}

        with open(self.feature_importance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded feature importance for {len(data)} models")
        return data

    def load_model_results(self) -> Dict[str, Any]:
        """Load model results."""
        logger.info("Loading model results...")

        if not os.path.exists(self.model_results_file):
            logger.error(f"Model results file not found: {self.model_results_file}")
            return {}

        with open(self.model_results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info("Loaded model results")
        return data

    def categorize_features(self, feature_importance: Dict[str, Any]) -> Dict[str, List[tuple]]:
        """Categorize features by type."""
        logger.info("Categorizing features...")

        categories = {
            'citations': [],
            'legal_terms': [],
            'procedural_elements': [],
            'jurisdiction': [],
            'outcome_indicators': [],
            'text_metrics': [],
            'intel_factors': [],
            'sentiment': [],
            'complexity': []
        }

        # Get combined feature importance across all models
        combined_importance = {}
        for model_name, importance in feature_importance.items():
            for feature, imp in importance.items():
                combined_importance[feature] = combined_importance.get(feature, 0) + imp

        # Categorize features
        for feature, importance in combined_importance.items():
            if feature.startswith('citation_'):
                categories['citations'].append((feature, importance))
            elif feature.startswith('legal_'):
                categories['legal_terms'].append((feature, importance))
            elif feature.startswith('procedural_'):
                categories['procedural_elements'].append((feature, importance))
            elif feature.startswith('jurisdiction_'):
                categories['jurisdiction'].append((feature, importance))
            elif feature.startswith('outcome_'):
                categories['outcome_indicators'].append((feature, importance))
            elif feature.startswith('intel_'):
                categories['intel_factors'].append((feature, importance))
            elif feature.startswith('sentiment_'):
                categories['sentiment'].append((feature, importance))
            elif feature.startswith('complexity_'):
                categories['complexity'].append((feature, importance))
            elif feature in ['text_length', 'word_count', 'sentence_count', 'avg_sentence_length', 'avg_word_length']:
                categories['text_metrics'].append((feature, importance))

        # Sort each category by importance
        for category in categories:
            categories[category].sort(key=lambda x: x[1], reverse=True)

        return categories

    def generate_citation_guidance(self, citations: List[tuple]) -> List[str]:
        """Generate guidance for citations."""
        guidance = []

        if not citations:
            return guidance

        guidance.append("## 📚 Citation Strategy")
        guidance.append("")

        # Top positive citations
        positive_citations = [c for c in citations if c[1] > 0.1]
        if positive_citations:
            guidance.append("### ✅ Include These Citations")
            guidance.append("")
            for citation, importance in positive_citations[:5]:
                citation_name = citation.replace('citation_', '').replace('_', ' ').title()
                guidance.append(f"- **{citation_name}**: High importance ({importance:.3f})")
            guidance.append("")

        # Citation diversity
        guidance.append("### 📊 Citation Diversity")
        guidance.append("")
        guidance.append("- **Include multiple case citations** to show comprehensive research")
        guidance.append("- **Balance old and new authorities** to demonstrate legal evolution")
        guidance.append("- **Cite both circuit and Supreme Court cases** when applicable")
        guidance.append("")

        return guidance

    def generate_legal_term_guidance(self, legal_terms: List[tuple]) -> List[str]:
        """Generate guidance for legal terminology."""
        guidance = []

        if not legal_terms:
            return guidance

        guidance.append("## ⚖️ Legal Terminology")
        guidance.append("")

        # Top legal terms
        top_terms = legal_terms[:10]
        guidance.append("### 🔑 Key Legal Terms to Include")
        guidance.append("")
        for term, importance in top_terms:
            term_name = term.replace('legal_', '').replace('_', ' ').title()
            guidance.append(f"- **{term_name}**: {importance:.3f} importance")
        guidance.append("")

        # Legal term strategy
        guidance.append("### 📝 Legal Writing Strategy")
        guidance.append("")
        guidance.append("- **Use precise legal terminology** throughout the petition")
        guidance.append("- **Define technical terms** when first introduced")
        guidance.append("- **Maintain consistent terminology** across all sections")
        guidance.append("- **Include relevant statutory language** where applicable")
        guidance.append("")

        return guidance

    def generate_procedural_guidance(self, procedural: List[tuple]) -> List[str]:
        """Generate guidance for procedural elements."""
        guidance = []

        if not procedural:
            return guidance

        guidance.append("## ⚡ Procedural Elements")
        guidance.append("")

        # Top procedural elements
        top_procedural = procedural[:8]
        guidance.append("### 🎯 Important Procedural Considerations")
        guidance.append("")
        for proc, importance in top_procedural:
            proc_name = proc.replace('procedural_', '').replace('_', ' ').title()
            guidance.append(f"- **{proc_name}**: {importance:.3f} importance")
        guidance.append("")

        # Procedural strategy
        guidance.append("### 📋 Procedural Strategy")
        guidance.append("")
        guidance.append("- **Address procedural requirements** explicitly")
        guidance.append("- **Include protective order language** when appropriate")
        guidance.append("- **Consider intervenor rights** and third-party interests")
        guidance.append("- **Address confidentiality concerns** proactively")
        guidance.append("")

        return guidance

    def generate_jurisdiction_guidance(self, jurisdiction: List[tuple]) -> List[str]:
        """Generate guidance for jurisdiction considerations."""
        guidance = []

        if not jurisdiction:
            return guidance

        guidance.append("## 🏛️ Jurisdiction Considerations")
        guidance.append("")

        # Top jurisdictions
        top_jurisdictions = jurisdiction[:5]
        guidance.append("### 🌍 Jurisdiction-Specific Factors")
        guidance.append("")
        for juris, importance in top_jurisdictions:
            juris_name = juris.replace('jurisdiction_', '').replace('_', ' ').title()
            guidance.append(f"- **{juris_name}**: {importance:.3f} importance")
        guidance.append("")

        # Jurisdiction strategy
        guidance.append("### 🎯 Jurisdiction Strategy")
        guidance.append("")
        guidance.append("- **Research circuit-specific precedents**")
        guidance.append("- **Address local rules and practices**")
        guidance.append("- **Consider venue-specific factors**")
        guidance.append("- **Cite relevant local authorities**")
        guidance.append("")

        return guidance

    def generate_intel_factor_guidance(self, intel_factors: List[tuple]) -> List[str]:
        """Generate guidance for Intel factors."""
        guidance = []

        if not intel_factors:
            return guidance

        guidance.append("## 🧠 Intel Factors Analysis")
        guidance.append("")

        # Intel factors
        guidance.append("### 📊 Factor Analysis Structure")
        guidance.append("")
        guidance.append("Structure your petition around the four Intel factors:")
        guidance.append("")
        guidance.append("1. **Factor 1: Foreign Tribunal**")
        guidance.append("   - Establish that the foreign proceeding is a 'tribunal'")
        guidance.append("   - Show the proceeding is 'adjudicative' in nature")
        guidance.append("   - Demonstrate the proceeding is 'pending' or 'reasonably contemplated'")
        guidance.append("")
        guidance.append("2. **Factor 2: Interested Person**")
        guidance.append("   - Establish that the petitioner is an 'interested person'")
        guidance.append("   - Show participation rights in the foreign proceeding")
        guidance.append("   - Demonstrate standing to seek discovery")
        guidance.append("")
        guidance.append("3. **Factor 3: Discovery Scope**")
        guidance.append("   - Show discovery is 'for use' in the foreign proceeding")
        guidance.append("   - Demonstrate relevance to the foreign case")
        guidance.append("   - Address proportionality and burden concerns")
        guidance.append("")
        guidance.append("4. **Factor 4: Discretionary Factors**")
        guidance.append("   - Address any applicable discretionary factors")
        guidance.append("   - Consider burden, cost, and scope")
        guidance.append("   - Address confidentiality and privilege concerns")
        guidance.append("")

        return guidance

    def generate_text_metrics_guidance(self, text_metrics: List[tuple]) -> List[str]:
        """Generate guidance for text metrics."""
        guidance = []

        if not text_metrics:
            return guidance

        guidance.append("## 📝 Writing Style and Structure")
        guidance.append("")

        # Text metrics analysis
        guidance.append("### 📊 Optimal Text Characteristics")
        guidance.append("")
        for metric, importance in text_metrics:
            metric_name = metric.replace('_', ' ').title()
            guidance.append(f"- **{metric_name}**: {importance:.3f} importance")
        guidance.append("")

        # Writing style guidance
        guidance.append("### ✍️ Writing Style Recommendations")
        guidance.append("")
        guidance.append("- **Use clear, concise language**")
        guidance.append("- **Structure arguments logically**")
        guidance.append("- **Include sufficient detail** without being verbose")
        guidance.append("- **Use headings and subheadings** for organization")
        guidance.append("- **Include relevant case law** and statutory citations")
        guidance.append("")

        return guidance

    def generate_checklist(self, categories: Dict[str, List[tuple]]) -> List[str]:
        """Generate a practical drafting checklist."""
        guidance = []

        guidance.append("## ✅ §1782 Petition Drafting Checklist")
        guidance.append("")

        # Pre-drafting
        guidance.append("### 📋 Pre-Drafting Preparation")
        guidance.append("")
        guidance.append("- [ ] **Research foreign proceeding** and applicable law")
        guidance.append("- [ ] **Identify relevant case law** and authorities")
        guidance.append("- [ ] **Gather supporting documents** and evidence")
        guidance.append("- [ ] **Review local rules** and procedural requirements")
        guidance.append("")

        # Drafting structure
        guidance.append("### 📝 Drafting Structure")
        guidance.append("")
        guidance.append("- [ ] **Introduction** - Brief overview of the request")
        guidance.append("- [ ] **Factual Background** - Relevant facts and foreign proceeding")
        guidance.append("- [ ] **Legal Analysis** - Intel factors and applicable law")
        guidance.append("- [ ] **Discovery Request** - Specific documents and information sought")
        guidance.append("- [ ] **Conclusion** - Summary and relief requested")
        guidance.append("")

        # Content requirements
        guidance.append("### 📄 Content Requirements")
        guidance.append("")
        guidance.append("- [ ] **Include protective order language**")
        guidance.append("- [ ] **Address confidentiality concerns**")
        guidance.append("- [ ] **Cite relevant authorities**")
        guidance.append("- [ ] **Address burden and proportionality**")
        guidance.append("- [ ] **Include foreign proceeding details**")
        guidance.append("")

        # Review and revision
        guidance.append("### 🔍 Review and Revision")
        guidance.append("")
        guidance.append("- [ ] **Check for completeness** of all required elements")
        guidance.append("- [ ] **Verify citations** and legal authorities")
        guidance.append("- [ ] **Review for clarity** and organization")
        guidance.append("- [ ] **Check formatting** and procedural requirements")
        guidance.append("")

        return guidance

    def generate_writing_guidance(self):
        """Generate comprehensive writing guidance."""
        logger.info("Generating petition writing guidance...")

        # Load data
        feature_importance = self.load_feature_importance()
        model_results = self.load_model_results()

        if not feature_importance or not model_results:
            logger.error("Failed to load required data")
            return

        # Categorize features
        categories = self.categorize_features(feature_importance)

        # Generate guidance sections
        guidance_sections = []

        # Header
        guidance_sections.extend([
            "# 📝 §1782 Petition Writing Guidance",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Based on**: {model_results.get('total_samples', 0)} petition samples",
            f"**Model Performance**: {model_results.get('model_performance', {}).get('best_model', 'N/A')}",
            "",
            "This guidance is based on machine learning analysis of successful §1782 petitions.",
            "",
        ])

        # Generate guidance for each category
        guidance_sections.extend(self.generate_citation_guidance(categories['citations']))
        guidance_sections.extend(self.generate_legal_term_guidance(categories['legal_terms']))
        guidance_sections.extend(self.generate_procedural_guidance(categories['procedural_elements']))
        guidance_sections.extend(self.generate_jurisdiction_guidance(categories['jurisdiction']))
        guidance_sections.extend(self.generate_intel_factor_guidance(categories['intel_factors']))
        guidance_sections.extend(self.generate_text_metrics_guidance(categories['text_metrics']))

        # Generate checklist
        guidance_sections.extend(self.generate_checklist(categories))

        # Additional recommendations
        guidance_sections.extend([
            "## 🚀 Additional Recommendations",
            "",
            "### 💡 Pro Tips",
            "",
            "- **Start with a strong introduction** that clearly states the request",
            "- **Use concrete examples** to illustrate your points",
            "- **Address potential objections** proactively",
            "- **Include relevant foreign law** when applicable",
            "- **Consider timing** and urgency factors",
            "",
            "### ⚠️ Common Pitfalls to Avoid",
            "",
            "- **Vague or overly broad discovery requests**",
            "- **Insufficient foreign proceeding details**",
            "- **Missing protective order language**",
            "- **Inadequate burden analysis**",
            "- **Poor organization** and unclear structure",
            "",
            "### 📚 Resources",
            "",
            "- **Intel Corp. v. Advanced Micro Devices, Inc.**, 542 U.S. 241 (2004)",
            "- **28 U.S.C. § 1782** - Discovery in aid of foreign proceedings",
            "- **Local court rules** and procedural requirements",
            "- **Circuit-specific precedents** and case law",
            "",
        ])

        # Save guidance
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(guidance_sections))

        logger.info(f"✓ Writing guidance saved to: {self.output_file}")

    def run_generation(self):
        """Run the complete guidance generation process."""
        logger.info("🚀 Starting Writing Guidance Generation")
        logger.info("="*80)

        self.generate_writing_guidance()

        logger.info("🎉 Writing guidance generation complete!")

        print(f"\n📊 Summary:")
        print(f"  Writing guidance generated")
        print(f"  Based on machine learning analysis")
        print(f"  Includes practical drafting checklist")
        print(f"  Covers all major petition elements")

def main():
    """Main function."""
    print("📝 Petition Writing Guidance Generator")
    print("="*80)

    generator = WritingGuidanceGenerator()
    generator.run_generation()

    print("\n✅ Writing guidance generation complete!")
    print("Check petition_writing_guidance.md for results.")

if __name__ == "__main__":
    main()
