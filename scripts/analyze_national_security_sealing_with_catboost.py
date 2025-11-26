#!/usr/bin/env python3
"""
Analyze National Security Sealing Cases with CatBoost

This script:
1. Searches CourtListener for cases involving national security + sealing/pseudonym/protective orders
2. Downloads relevant cases
3. Extracts features related to national security definitions
4. Uses CatBoost to analyze which features predict successful sealing
5. Generates a report on national security definitions and sealing patterns
"""

import sys
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "case_law_data" / "scripts"))
sys.path.insert(0, str(project_root / "document_ingestion"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import CourtListener tools
try:
    from DownloadCaseLaw import CourtListenerClient
    COURTLISTENER_AVAILABLE = True
except ImportError:
    logger.warning("CourtListenerClient not available")
    COURTLISTENER_AVAILABLE = False

# Import CatBoost tools
try:
    from catboost import CatBoostClassifier
    import pandas as pd
    import numpy as np
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost not available")
    CATBOOST_AVAILABLE = False


class NationalSecuritySealingAnalyzer:
    """Analyze national security sealing cases with CatBoost."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize analyzer."""
        self.config_path = config_path or (project_root / "document_ingestion" / "courtlistener_config.json")
        self.output_dir = project_root / "outputs" / "national_security_sealing_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if COURTLISTENER_AVAILABLE:
            try:
                self.cl_client = CourtListenerClient(config_path=str(self.config_path))
                logger.info("CourtListener client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CourtListener client: {e}")
                self.cl_client = None
        else:
            self.cl_client = None

    def search_local_database(self) -> List[Dict[str, Any]]:
        """Search local database for national security + sealing cases."""
        import sqlite3

        db_paths = [
            project_root / "case_law_data" / "unified_corpus.db",
            project_root / "case_law_data" / "ma_federal_motions.db",
            project_root / "case_law_data" / "harvard_corpus.db",
        ]

        all_cases = []
        seen_cluster_ids = set()

        logger.info("Searching local databases for national security sealing cases...")

        for db_path in db_paths:
            if not db_path.exists():
                continue

            try:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Search for cases with national security tag AND sealing keywords
                query = """
                    SELECT
                        cluster_id,
                        case_name,
                        court,
                        date_filed,
                        cleaned_text,
                        tag_national_security,
                        tag_academic_institution,
                        tag_foreign_government,
                        tag_defamation,
                        favorable_to_plaintiff,
                        favorable_to_defendant,
                        corpus_type,
                        corpus_subset
                    FROM cases
                    WHERE (
                        tag_national_security = 1
                        OR cleaned_text LIKE '%national security%'
                    )
                    AND (
                        cleaned_text LIKE '%motion to seal%'
                        OR cleaned_text LIKE '%sealing%'
                        OR cleaned_text LIKE '%pseudonym%'
                        OR cleaned_text LIKE '%protective order%'
                        OR cleaned_text LIKE '%file under seal%'
                        OR cleaned_text LIKE '%impound%'
                    )
                    AND cleaned_text IS NOT NULL
                    AND LENGTH(cleaned_text) > 500
                    ORDER BY tag_national_security DESC, date_filed DESC
                    LIMIT 100
                """

                cursor.execute(query)
                rows = cursor.fetchall()

                for row in rows:
                    cluster_id = row['cluster_id']
                    if cluster_id not in seen_cluster_ids:
                        seen_cluster_ids.add(cluster_id)
                        all_cases.append({
                            'cluster_id': cluster_id,
                            'case_name': row['case_name'],
                            'court': row['court'],
                            'date_filed': row['date_filed'],
                            'plain_text': row['cleaned_text'],
                            'html': row['cleaned_text'],
                            'source': 'local_database',
                            'database': db_path.name,
                            'tag_national_security': row['tag_national_security'],
                            'tag_academic_institution': row['tag_academic_institution'],
                            'tag_foreign_government': row['tag_foreign_government'],
                            'tag_defamation': row['tag_defamation'],
                            'favorable_to_plaintiff': row['favorable_to_plaintiff'],
                            'favorable_to_defendant': row['favorable_to_defendant'],
                        })

                conn.close()
                logger.info(f"  Found {len(rows)} cases in {db_path.name}")

            except Exception as e:
                logger.error(f"Error searching {db_path}: {e}")
                continue

        logger.info(f"\nTotal unique cases from local database: {len(all_cases)}")
        return all_cases

    def search_courtlistener(self) -> List[Dict[str, Any]]:
        """
        Search CourtListener for additional cases involving national security + sealing.
        """
        if not self.cl_client:
            logger.warning("CourtListener client not available - skipping CourtListener search")
            return []

        queries = [
            ('"national security" AND ("motion to seal" OR "sealing")', 'National Security + Sealing'),
            ('"national security" AND pseudonym', 'National Security + Pseudonym'),
            ('"national security" AND "protective order"', 'National Security + Protective Order'),
        ]

        all_cases = []
        seen_case_ids = set()

        logger.info("Searching CourtListener for additional cases...")

        for query, description in queries[:3]:  # Limit to 3 queries to avoid rate limits
            logger.info(f"Searching: {description}")
            try:
                # Use keywords parameter for search_opinions
                keywords = query.replace('"', '').replace(' AND ', ' ').replace(' OR ', ' ').split()
                keywords = [kw.strip() for kw in keywords if len(kw) > 2][:5]  # Limit keywords

                results = self.cl_client.search_opinions(
                    keywords=keywords,
                    page_size=20,
                    include_non_precedential=False
                )

                # Extract results from response
                if results and 'results' in results:
                    results = results['results']
                elif isinstance(results, list):
                    results = results
                else:
                    results = []

                if results:
                    logger.info(f"  Found {len(results)} cases for: {description}")
                    for case in results:
                        case_id = case.get('id') or case.get('resource_uri', '')
                        if case_id and case_id not in seen_case_ids:
                            seen_case_ids.add(case_id)
                            case['source'] = 'courtlistener'
                            case['search_query'] = description
                            all_cases.append(case)

            except Exception as e:
                logger.error(f"Error searching for '{description}': {e}")
                continue

        logger.info(f"Total cases from CourtListener: {len(all_cases)}")
        return all_cases

    def extract_national_security_features(self, case_text: str) -> Dict[str, Any]:
        """
        Extract features related to national security definitions and sealing justifications.

        Features to extract:
        - National security definition phrases
        - Sealing justification language
        - Protective measures mentioned
        - Foreign government involvement
        - Intelligence/classified information
        - Balancing test language
        """
        text_lower = case_text.lower()

        features = {}

        # National Security Definition Phrases
        ns_definitions = [
            "matter of national security",
            "national security interest",
            "national security concern",
            "national security implications",
            "national security risk",
            "national security threat",
            "national security classification",
            "national security exception",
            "national security exemption",
            "national security privilege",
        ]

        for phrase in ns_definitions:
            features[f"ns_definition_{phrase.replace(' ', '_')}"] = int(phrase in text_lower)

        # Sealing Justification Language
        sealing_justifications = [
            "justify sealing",
            "warrant sealing",
            "support sealing",
            "sealing is appropriate",
            "sealing is necessary",
            "sealing is warranted",
            "good cause for sealing",
            "compelling reason for sealing",
            "sealing serves",
            "sealing protects",
        ]

        for phrase in sealing_justifications:
            features[f"sealing_just_{phrase.replace(' ', '_')}"] = int(phrase in text_lower)

        # Protective Measures
        protective_measures = [
            "protective order",
            "confidentiality order",
            "file under seal",
            "impound",
            "proceed under pseudonym",
            "redact",
            "sealed filing",
            "ex parte",
        ]

        for measure in protective_measures:
            features[f"protective_{measure.replace(' ', '_')}"] = int(measure in text_lower)

        # Foreign Government Involvement
        foreign_gov_indicators = [
            "foreign government",
            "foreign state",
            "foreign agent",
            "foreign interference",
            "foreign influence",
            "sovereign immunity",
            "foreign sovereign",
            "foreign entity",
        ]

        for indicator in foreign_gov_indicators:
            features[f"foreign_{indicator.replace(' ', '_')}"] = int(indicator in text_lower)

        # Intelligence/Classified Information
        intel_indicators = [
            "classified information",
            "classified material",
            "state secret",
            "intelligence",
            "intelligence source",
            "intelligence method",
            "sensitive compartmented",
            "top secret",
            "secret",
            "confidential",
        ]

        for indicator in intel_indicators:
            features[f"intel_{indicator.replace(' ', '_')}"] = int(indicator in text_lower)

        # Balancing Test Language
        balancing_language = [
            "balance",
            "balancing test",
            "weigh",
            "outweigh",
            "competing interests",
            "countervailing interest",
            "presumption of public access",
            "public interest",
            "compelling interest",
            "significant interest",
        ]

        for phrase in balancing_language:
            features[f"balancing_{phrase.replace(' ', '_')}"] = int(phrase in text_lower)

        # Count totals
        features['total_ns_definitions'] = sum(1 for k, v in features.items() if k.startswith('ns_definition_') and v)
        features['total_sealing_justifications'] = sum(1 for k, v in features.items() if k.startswith('sealing_just_') and v)
        features['total_protective_measures'] = sum(1 for k, v in features.items() if k.startswith('protective_') and v)
        features['total_foreign_gov'] = sum(1 for k, v in features.items() if k.startswith('foreign_') and v)
        features['total_intel'] = sum(1 for k, v in features.items() if k.startswith('intel_') and v)
        features['total_balancing'] = sum(1 for k, v in features.items() if k.startswith('balancing_') and v)

        return features

    def classify_outcome(self, case_text: str) -> str:
        """
        Classify case outcome: granted, denied, or mixed.
        """
        text_lower = case_text.lower()

        granted_phrases = [
            "motion to seal granted",
            "motion granted",
            "sealing granted",
            "protective order granted",
            "may proceed under pseudonym",
            "may file under seal",
            "impounded",
            "sealed",
        ]

        denied_phrases = [
            "motion to seal denied",
            "motion denied",
            "sealing denied",
            "protective order denied",
            "unseal",
            "public access",
        ]

        granted_count = sum(1 for phrase in granted_phrases if phrase in text_lower)
        denied_count = sum(1 for phrase in denied_phrases if phrase in text_lower)

        if granted_count > denied_count:
            return "granted"
        elif denied_count > granted_count:
            return "denied"
        else:
            return "mixed"

    def analyze_with_catboost(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use CatBoost to analyze which national security features predict successful sealing.
        """
        if not CATBOOST_AVAILABLE:
            logger.error("CatBoost not available")
            return {}

        logger.info("\nExtracting features from cases...")

        # Extract features and outcomes
        feature_rows = []
        for case in cases:
            case_text = case.get('plain_text', '') or case.get('html', '') or case.get('html_with_citations', '') or ''

            if not case_text or len(case_text) < 500:
                continue

            features = self.extract_national_security_features(case_text)
            outcome = self.classify_outcome(case_text)

            row = {
                'case_id': case.get('id', '') or case.get('cluster_id', ''),
                'case_name': case.get('caseName', '') or case.get('case_name', 'Unknown'),
                'outcome': outcome,
                **features
            }
            feature_rows.append(row)

        if len(feature_rows) < 10:
            logger.warning(f"Too few cases with text ({len(feature_rows)}) for CatBoost analysis")
            return {}

        logger.info(f"Extracted features from {len(feature_rows)} cases")

        # Convert to DataFrame
        df = pd.DataFrame(feature_rows)

        # Filter to binary outcomes (granted vs denied)
        binary_df = df[df['outcome'].isin(['granted', 'denied'])].copy()

        if len(binary_df) < 10:
            logger.warning(f"Too few binary cases ({len(binary_df)}) for CatBoost analysis")
            return {}

        logger.info(f"Binary cases: {len(binary_df)} (granted: {len(binary_df[binary_df['outcome'] == 'granted'])}, denied: {len(binary_df[binary_df['outcome'] == 'denied'])})")

        # Prepare features
        feature_cols = [c for c in binary_df.columns if c not in ['case_id', 'case_name', 'outcome']]
        X = binary_df[feature_cols]
        y = binary_df['outcome'].map({'granted': 1, 'denied': 0})

        # Train CatBoost model
        logger.info("\nTraining CatBoost model...")
        model = CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            verbose=False,
            random_state=42
        )

        model.fit(X, y)

        # Get feature importance
        feature_importance = model.get_feature_importance()
        importance_dict = {feature_cols[i]: float(feature_importance[i]) for i in range(len(feature_cols))}
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # Predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        accuracy = (predictions == y).mean()

        logger.info(f"\nCatBoost Results:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Top 10 Features:")
        for i, (feature, importance) in enumerate(list(importance_dict.items())[:10], 1):
            logger.info(f"    {i}. {feature}: {importance:.4f}")

        return {
            'model': model,
            'feature_importance': importance_dict,
            'accuracy': float(accuracy),
            'feature_cols': feature_cols,
            'cases_analyzed': len(binary_df),
            'cases_granted': len(binary_df[binary_df['outcome'] == 'granted']),
            'cases_denied': len(binary_df[binary_df['outcome'] == 'denied']),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
        }

    def generate_report(self, cases: List[Dict[str, Any]], analysis: Dict[str, Any]) -> str:
        """Generate comprehensive report."""
        report_lines = [
            "# National Security Sealing Analysis with CatBoost",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"- **Total cases found:** {len(cases)}",
            f"- **Cases analyzed with CatBoost:** {analysis.get('cases_analyzed', 0)}",
            f"- **Cases granted:** {analysis.get('cases_granted', 0)}",
            f"- **Cases denied:** {analysis.get('cases_denied', 0)}",
            f"- **Model accuracy:** {analysis.get('accuracy', 0):.1%}",
            "",
            "---",
            "",
            "## Top Features Predicting Successful Sealing",
            "",
        ]

        if analysis.get('feature_importance'):
            report_lines.append("Based on CatBoost feature importance analysis:")
            report_lines.append("")
            for i, (feature, importance) in enumerate(list(analysis['feature_importance'].items())[:20], 1):
                report_lines.append(f"{i}. **{feature}**: {importance:.4f}")

        report_lines.extend([
            "",
            "---",
            "",
            "## National Security Definitions Found",
            "",
            "The following phrases were used to define or justify national security interests:",
            "",
        ])

        # Extract NS definition patterns from cases
        ns_definitions_found = set()
        for case in cases[:20]:  # Analyze first 20 cases
            text = case.get('plain_text', '') or case.get('html', '') or ''
            if 'national security' in text.lower():
                # Extract sentences with "national security"
                import re
                sentences = re.split(r'[.!?]+', text)
                for sent in sentences:
                    if 'national security' in sent.lower() and len(sent) < 500:
                        ns_definitions_found.add(sent.strip()[:200])

        for i, definition in enumerate(list(ns_definitions_found)[:10], 1):
            report_lines.append(f"{i}. {definition}...")

        report_lines.extend([
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "1. **Emphasize top features**: Focus on the national security definition phrases",
            "   and sealing justifications that are most predictive of success.",
            "",
            "2. **Use balancing test language**: Include balancing test language to show",
            "   the court considered competing interests.",
            "",
            "3. **Highlight protective measures**: Specify which protective measures are",
            "   being requested and why they are necessary.",
            "",
        ])

        return "\n".join(report_lines)

    def run(self):
        """Run the full analysis pipeline."""
        logger.info("=" * 80)
        logger.info("NATIONAL SECURITY SEALING ANALYSIS WITH CATBOOST")
        logger.info("=" * 80)

        # Step 1: Search local database first (faster, no API limits)
        cases = self.search_local_database()

        # Step 2: Supplement with CourtListener if we need more cases
        if len(cases) < 50:
            logger.info(f"\nOnly {len(cases)} cases found locally. Searching CourtListener for additional cases...")
            cl_cases = self.search_courtlistener()
            cases.extend(cl_cases)

        if not cases:
            logger.error("No cases found. Cannot proceed with analysis.")
            return

        # Save raw cases
        cases_file = self.output_dir / "cases_found.json"
        with open(cases_file, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, default=str)
        logger.info(f"\nSaved {len(cases)} cases to: {cases_file}")

        # Step 2: Analyze with CatBoost
        if CATBOOST_AVAILABLE:
            analysis = self.analyze_with_catboost(cases)

            if analysis:
                # Save analysis
                analysis_file = self.output_dir / "catboost_analysis.json"
                # Convert model to dict (can't serialize model directly)
                analysis_serializable = {k: v for k, v in analysis.items() if k != 'model'}
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_serializable, f, indent=2, default=str)
                logger.info(f"Saved analysis to: {analysis_file}")

                # Step 3: Generate report
                report = self.generate_report(cases, analysis)
                report_file = self.output_dir / "analysis_report.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"Saved report to: {report_file}")

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)


if __name__ == "__main__":
    analyzer = NationalSecuritySealingAnalyzer()
    analyzer.run()

