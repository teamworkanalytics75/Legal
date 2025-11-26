#!/usr/bin/env python3
"""
User Draft Validator

Applies the trained petition model to the user's Harvard/EsuWiki petition draft
and provides feedback on predicted outcome and feature analysis.

Usage: python scripts/validate_user_draft.py
"""

import json
import pickle
import os
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserDraftValidator:
    def __init__(self):
        self.model_dir = "data/case_law/petition_model"
        self.scaler_file = os.path.join(self.model_dir, "scaler.pkl")
        self.best_model_file = os.path.join(self.model_dir, "gradient_boosting_model.pkl")
        self.feature_importance_file = os.path.join(self.model_dir, "feature_importance.json")
        self.output_file = "data/case_law/harvard_case_prediction.json"

        # Feature patterns (same as training)
        self.citation_patterns = {
            'intel_corp': re.compile(r'intel\s+corp', re.IGNORECASE),
            'chevron': re.compile(r'chevron', re.IGNORECASE),
            'euromepa': re.compile(r'euromepa', re.IGNORECASE),
            'hegna': re.compile(r'hegna', re.IGNORECASE),
            'advanced_micro': re.compile(r'advanced\s+micro', re.IGNORECASE),
            'mees': re.compile(r'mees', re.IGNORECASE),
            'esmerian': re.compile(r'esmerian', re.IGNORECASE),
            'naranjo': re.compile(r'naranjo', re.IGNORECASE),
            'buiter': re.compile(r'buiter', re.IGNORECASE),
            'hourani': re.compile(r'hourani', re.IGNORECASE),
            'delano_farms': re.compile(r'delano\s+farms', re.IGNORECASE),
            'posco': re.compile(r'posco', re.IGNORECASE)
        }

        self.legal_patterns = {
            'protective_order': re.compile(r'protective\s+order', re.IGNORECASE),
            'confidentiality': re.compile(r'confidentiality', re.IGNORECASE),
            'privilege': re.compile(r'privilege', re.IGNORECASE),
            'burden': re.compile(r'burden', re.IGNORECASE),
            'relevance': re.compile(r'relevance', re.IGNORECASE),
            'proportionality': re.compile(r'proportionality', re.IGNORECASE),
            'cost': re.compile(r'cost', re.IGNORECASE),
            'time': re.compile(r'time', re.IGNORECASE),
            'scope': re.compile(r'scope', re.IGNORECASE),
            'foreign_tribunal': re.compile(r'foreign\s+tribunal', re.IGNORECASE),
            'international_arbitration': re.compile(r'international\s+arbitration', re.IGNORECASE),
            'discovery': re.compile(r'discovery', re.IGNORECASE),
            'subpoena': re.compile(r'subpoena', re.IGNORECASE),
            'deposition': re.compile(r'deposition', re.IGNORECASE),
            'document_production': re.compile(r'document\s+production', re.IGNORECASE)
        }

        self.intel_factors = {
            'factor_1': re.compile(r'factor\s+1|first\s+factor', re.IGNORECASE),
            'factor_2': re.compile(r'factor\s+2|second\s+factor', re.IGNORECASE),
            'factor_3': re.compile(r'factor\s+3|third\s+factor', re.IGNORECASE),
            'factor_4': re.compile(r'factor\s+4|fourth\s+factor', re.IGNORECASE),
            'intel_factors': re.compile(r'intel\s+factors', re.IGNORECASE)
        }

        self.procedural_patterns = {
            'ex_parte': re.compile(r'ex\s+parte', re.IGNORECASE),
            'intervenor': re.compile(r'intervenor', re.IGNORECASE),
            'amicus': re.compile(r'amicus', re.IGNORECASE),
            'motion_to_quash': re.compile(r'motion\s+to\s+quash', re.IGNORECASE),
            'motion_to_compel': re.compile(r'motion\s+to\s+compel', re.IGNORECASE),
            'seal': re.compile(r'seal', re.IGNORECASE),
            'redact': re.compile(r'redact', re.IGNORECASE)
        }

    def load_model_and_scaler(self) -> tuple:
        """Load the trained model and scaler."""
        logger.info("Loading trained model and scaler...")

        if not os.path.exists(self.best_model_file):
            logger.error(f"Model file not found: {self.best_model_file}")
            return None, None

        if not os.path.exists(self.scaler_file):
            logger.error(f"Scaler file not found: {self.scaler_file}")
            return None, None

        with open(self.best_model_file, 'rb') as f:
            model = pickle.load(f)

        with open(self.scaler_file, 'rb') as f:
            scaler = pickle.load(f)

        logger.info("✓ Model and scaler loaded successfully")
        return model, scaler

    def load_feature_importance(self) -> Dict[str, Any]:
        """Load feature importance data."""
        logger.info("Loading feature importance...")

        if not os.path.exists(self.feature_importance_file):
            logger.error(f"Feature importance file not found: {self.feature_importance_file}")
            return {}

        with open(self.feature_importance_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info("✓ Feature importance loaded")
        return data

    def get_user_draft_text(self) -> str:
        """Get the user's draft petition text."""
        logger.info("Getting user draft text...")

        # For now, we'll create a sample text based on the user's Harvard/EsuWiki case
        # In practice, this would be loaded from a file or input by the user

        sample_draft = """
        IN THE UNITED STATES DISTRICT COURT
        FOR THE DISTRICT OF MASSACHUSETTS

        IN RE APPLICATION OF HARVARD UNIVERSITY
        FOR JUDICIAL ASSISTANCE PURSUANT TO 28 U.S.C. § 1782

        MOTION FOR DISCOVERY IN AID OF FOREIGN PROCEEDING

        TO THE HONORABLE COURT:

        Harvard University respectfully moves this Court for an order pursuant to 28 U.S.C. § 1782
        authorizing discovery for use in a foreign proceeding before the International Court of Justice
        regarding the EsuWiki intellectual property dispute.

        I. FACTUAL BACKGROUND

        This application arises from an ongoing intellectual property dispute between Harvard University
        and EsuWiki Corporation regarding the unauthorized use of Harvard's proprietary research data
        and methodologies in EsuWiki's artificial intelligence systems.

        The foreign proceeding is currently pending before the International Court of Justice (ICJ)
        in The Hague, Netherlands, where Harvard seeks to establish its intellectual property rights
        and obtain injunctive relief against EsuWiki's continued unauthorized use of its research.

        II. LEGAL ANALYSIS

        A. Factor 1: Foreign Tribunal

        The ICJ constitutes a "foreign tribunal" within the meaning of 28 U.S.C. § 1782. The ICJ is
        an international judicial body established by the United Nations Charter with the authority
        to render binding decisions on disputes between states and international organizations.

        B. Factor 2: Interested Person

        Harvard University is an "interested person" in the foreign proceeding. As the plaintiff
        in the ICJ action, Harvard has standing to seek discovery in aid of its foreign proceeding.

        C. Factor 3: Discovery Scope

        The discovery sought is "for use" in the foreign proceeding. The requested documents and
        information are directly relevant to establishing Harvard's intellectual property rights
        and demonstrating EsuWiki's unauthorized use of Harvard's research.

        D. Factor 4: Discretionary Factors

        The discovery request is proportional and not unduly burdensome. Harvard seeks only
        documents and information directly relevant to the intellectual property dispute.

        III. DISCOVERY REQUEST

        Harvard seeks discovery from EsuWiki Corporation, including:

        1. All documents relating to EsuWiki's development and implementation of AI systems
        2. Source code and algorithms used in EsuWiki's AI technology
        3. Communications between EsuWiki and Harvard researchers
        4. Documents relating to EsuWiki's knowledge of Harvard's research methodologies

        IV. PROTECTIVE ORDER

        Harvard requests that this Court enter a protective order to ensure the confidentiality
        of any sensitive information produced in response to this discovery request.

        V. CONCLUSION

        For the foregoing reasons, Harvard respectfully requests that this Court grant its
        application for discovery pursuant to 28 U.S.C. § 1782.

        Respectfully submitted,

        HARVARD UNIVERSITY

        By: [Attorney Name]
        [Attorney Information]
        """

        logger.info("✓ Sample draft text loaded")
        return sample_draft

    def extract_features_from_draft(self, text: str) -> Dict[str, Any]:
        """Extract features from the user's draft petition."""
        logger.info("Extracting features from draft petition...")

        features = {}

        # Basic text features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['paragraph_count'] = len(text.split('\n\n'))

        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0

        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0

        # Citation features
        for citation, pattern in self.citation_patterns.items():
            matches = pattern.findall(text)
            features[f'citation_{citation}'] = len(matches)

        features['total_citations'] = sum(features[f'citation_{citation}'] for citation in self.citation_patterns.keys())

        # Legal term features
        for term, pattern in self.legal_patterns.items():
            matches = pattern.findall(text)
            features[f'legal_{term}'] = len(matches)

        # Intel factor features
        for factor, pattern in self.intel_factors.items():
            matches = pattern.findall(text)
            features[f'intel_{factor}'] = len(matches)

        # Procedural features
        for proc, pattern in self.procedural_patterns.items():
            matches = pattern.findall(text)
            features[f'procedural_{proc}'] = len(matches)

        # Jurisdiction features (assuming Massachusetts)
        features['jurisdiction_massachusetts'] = 1
        features['jurisdiction_washington'] = 0
        features['jurisdiction_california'] = 0
        features['jurisdiction_new_york'] = 0
        features['jurisdiction_texas'] = 0
        features['jurisdiction_florida'] = 0
        features['jurisdiction_nebraska'] = 0
        features['jurisdiction_maryland'] = 0
        features['jurisdiction_wisconsin'] = 0

        # Sentiment features (simplified)
        positive_words = ['grant', 'approve', 'allow', 'permit', 'favor', 'support', 'benefit', 'help', 'assist']
        negative_words = ['deny', 'reject', 'refuse', 'oppose', 'burden', 'cost', 'harm', 'prejudice', 'unfair']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        features['sentiment_positive'] = positive_count / max(features['word_count'], 1)
        features['sentiment_negative'] = negative_count / max(features['word_count'], 1)
        features['sentiment_polarity'] = (positive_count - negative_count) / max(features['word_count'], 1)

        # Complexity features
        legal_terms = sum(len(pattern.findall(text)) for pattern in self.legal_patterns.values())
        features['complexity_legal_density'] = legal_terms / max(features['word_count'], 1)

        citations = sum(len(pattern.findall(text)) for pattern in self.citation_patterns.values())
        features['complexity_citation_density'] = citations / max(features['word_count'], 1)

        # Additional features (matching training features)
        features['citation_count'] = features['total_citations']
        features['complexity_avg_sentence_length'] = features['avg_sentence_length']
        features['court_id'] = 0  # Default value
        features['has_extracted_text'] = 1  # Assume we have extracted text
        features['has_opinion_text'] = 0  # This is a petition, not an opinion
        features['docket_number'] = 0  # Default value
        features['date_filed'] = 0  # Default value
        features['is_success'] = 0  # This is what we're predicting

        logger.info(f"✓ Extracted {len(features)} features from draft")
        return features

    def predict_outcome(self, features: Dict[str, Any], model, scaler) -> Dict[str, Any]:
        """Predict the outcome using the trained model."""
        logger.info("Predicting petition outcome...")

        # Convert features to DataFrame format
        import pandas as pd
        import numpy as np

        # Create a DataFrame with the features
        feature_df = pd.DataFrame([features])

        # Handle missing values
        feature_df = feature_df.fillna(0)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)

        # Convert categorical variables to numeric
        for col in feature_df.columns:
            if feature_df[col].dtype == 'object':
                feature_df[col] = pd.Categorical(feature_df[col]).codes

        # Scale features
        X_scaled = scaler.transform(feature_df)

        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        # Get prediction confidence
        confidence = max(probability)

        result = {
            'predicted_outcome': 'SUCCESS' if prediction == 1 else 'FAILURE',
            'success_probability': probability[1],
            'failure_probability': probability[0],
            'confidence': confidence,
            'prediction': prediction
        }

        logger.info(f"✓ Prediction: {result['predicted_outcome']} (confidence: {confidence:.3f})")
        return result

    def analyze_feature_impact(self, features: Dict[str, Any], feature_importance: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze which features help or hurt the prediction."""
        logger.info("Analyzing feature impact...")

        # Get combined feature importance
        combined_importance = {}
        for model_name, importance in feature_importance.items():
            for feature, imp in importance.items():
                combined_importance[feature] = combined_importance.get(feature, 0) + imp

        # Analyze feature impact
        helpful_features = []
        harmful_features = []

        for feature, importance in combined_importance.items():
            if feature in features:
                feature_value = features[feature]
                if feature_value > 0 and importance > 0.01:  # Significant positive impact
                    helpful_features.append({
                        'feature': feature,
                        'value': feature_value,
                        'importance': importance,
                        'impact': feature_value * importance
                    })
                elif feature_value == 0 and importance > 0.01:  # Missing helpful feature
                    harmful_features.append({
                        'feature': feature,
                        'value': 0,
                        'importance': importance,
                        'impact': -importance,
                        'reason': 'Missing helpful feature'
                    })

        # Sort by impact
        helpful_features.sort(key=lambda x: x['impact'], reverse=True)
        harmful_features.sort(key=lambda x: abs(x['impact']), reverse=True)

        analysis = {
            'helpful_features': helpful_features[:10],  # Top 10 helpful
            'harmful_features': harmful_features[:10],  # Top 10 harmful
            'total_features_analyzed': len(combined_importance),
            'significant_features': len([f for f in combined_importance.values() if f > 0.01])
        }

        logger.info(f"✓ Analyzed {analysis['total_features_analyzed']} features")
        return analysis

    def generate_recommendations(self, prediction: Dict[str, Any], feature_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improving the petition."""
        recommendations = []

        # Overall assessment
        if prediction['success_probability'] > 0.7:
            recommendations.append("🎉 **Strong Petition**: Your draft shows high potential for success!")
        elif prediction['success_probability'] > 0.5:
            recommendations.append("✅ **Good Foundation**: Your petition has a solid foundation with room for improvement.")
        else:
            recommendations.append("⚠️ **Needs Improvement**: Consider revising key elements to strengthen your petition.")

        # Specific recommendations based on feature analysis
        recommendations.append("\n## 📈 Specific Recommendations")
        recommendations.append("")

        # Missing helpful features
        if feature_analysis['harmful_features']:
            recommendations.append("### ✅ Add These Elements")
            recommendations.append("")
            for feature in feature_analysis['harmful_features'][:5]:
                feature_name = feature['feature'].replace('_', ' ').title()
                recommendations.append(f"- **{feature_name}**: Missing element with {feature['importance']:.3f} importance")
            recommendations.append("")

        # Strengths
        if feature_analysis['helpful_features']:
            recommendations.append("### 💪 Your Strengths")
            recommendations.append("")
            for feature in feature_analysis['helpful_features'][:5]:
                feature_name = feature['feature'].replace('_', ' ').title()
                recommendations.append(f"- **{feature_name}**: Strong element with {feature['impact']:.3f} impact")
            recommendations.append("")

        # General recommendations
        recommendations.append("### 🎯 General Recommendations")
        recommendations.append("")
        recommendations.append("- **Include protective order language** to address confidentiality concerns")
        recommendations.append("- **Cite relevant authorities** to support your legal arguments")
        recommendations.append("- **Address all Intel factors** explicitly in your analysis")
        recommendations.append("- **Use clear, concise language** throughout the petition")
        recommendations.append("- **Include specific discovery requests** with clear scope and relevance")
        recommendations.append("")

        return recommendations

    def validate_user_draft(self):
        """Run the complete validation process."""
        logger.info("🚀 Starting User Draft Validation")
        logger.info("="*80)

        # Load model and scaler
        model, scaler = self.load_model_and_scaler()
        if model is None or scaler is None:
            return

        # Load feature importance
        feature_importance = self.load_feature_importance()

        # Get user draft
        draft_text = self.get_user_draft_text()

        # Extract features
        features = self.extract_features_from_draft(draft_text)

        # Predict outcome
        prediction = self.predict_outcome(features, model, scaler)

        # Analyze feature impact
        feature_analysis = self.analyze_feature_impact(features, feature_importance)

        # Generate recommendations
        recommendations = self.generate_recommendations(prediction, feature_analysis)

        # Create comprehensive results
        results = {
            'validation_date': datetime.now().isoformat(),
            'draft_analysis': {
                'text_length': features['text_length'],
                'word_count': features['word_count'],
                'sentence_count': features['sentence_count'],
                'total_citations': features['total_citations'],
                'has_protective_order': features['has_protective_order'],
                'has_intel_factors': features['has_intel_factors'],
                'has_foreign_tribunal': features['has_foreign_tribunal']
            },
            'prediction': prediction,
            'feature_analysis': feature_analysis,
            'recommendations': recommendations
        }

        # Save results
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Validation results saved to: {self.output_file}")

        # Print summary
        print(f"\n📊 Harvard/EsuWiki Petition Analysis:")
        print(f"  Predicted Outcome: {prediction['predicted_outcome']}")
        print(f"  Success Probability: {prediction['success_probability']:.2%}")
        print(f"  Confidence: {prediction['confidence']:.2%}")
        print(f"  Text Length: {features['text_length']} characters")
        print(f"  Word Count: {features['word_count']} words")
        print(f"  Citations: {features['total_citations']}")
        print(f"  Has Protective Order: {'Yes' if features['has_protective_order'] else 'No'}")
        print(f"  Has Intel Factors: {'Yes' if features['has_intel_factors'] else 'No'}")

        return results

def main():
    """Main function."""
    print("🔍 User Draft Validator")
    print("="*80)

    validator = UserDraftValidator()
    results = validator.validate_user_draft()

    print("\n✅ User draft validation complete!")
    print("Check harvard_case_prediction.json for detailed results.")

if __name__ == "__main__":
    main()
