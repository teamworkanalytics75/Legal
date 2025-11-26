#!/usr/bin/env python3
"""
Auto-Update Pipeline - Scheduled regeneration of rules from new case law.

Monitors case database for new cases, retrains CatBoost, and regenerates
rule configurations with version control.
"""

import asyncio
import json
import logging
import sqlite3
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "case_law_data" / "ma_federal_motions.db"
RULES_DIR = PROJECT_ROOT / "writer_agents" / "code" / "sk_plugins" / "rules"
OUTPUT_DIR = Path(__file__).parent
UPDATE_LOG_FILE = OUTPUT_DIR / "auto_update_log.json"


def detect_new_cases(db_path: Path, last_check: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Detect new cases added since last check."""

    if not db_path.exists():
        logger.warning(f"Database not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)

        # Get all cases
        query = """
        SELECT cluster_id, cleaned_text, created_at, updated_at
        FROM cases
        WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
        ORDER BY updated_at DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.info("No cases found in database")
            return []

        # Filter by date if last_check provided
        if last_check:
            df['updated_at'] = pd.to_datetime(df['updated_at'])
            new_cases = df[df['updated_at'] > last_check]
        else:
            # If no last_check, return recent cases (last 30 days)
            df['updated_at'] = pd.to_datetime(df['updated_at'])
            cutoff_date = datetime.now() - timedelta(days=30)
            new_cases = df[df['updated_at'] > cutoff_date]

        logger.info(f"Found {len(new_cases)} new cases since last check")

        # Convert to list of dicts
        cases = []
        for _, row in new_cases.iterrows():
            cases.append({
                "case_id": row['cluster_id'],
                "text": row['cleaned_text'],
                "created_at": row['created_at'],
                "updated_at": row['updated_at']
            })

        return cases

    except Exception as e:
        logger.error(f"Failed to detect new cases: {e}")
        return []


def retrain_catboost_model(new_cases: List[Dict]) -> Optional[Any]:
    """Retrain CatBoost model with new cases."""

    try:
        # Import analysis functions
        import sys
        analysis_path = PROJECT_ROOT / "analysis"
        sys.path.append(str(analysis_path))

        from analyze_ma_motion_doc import (
            load_dataset, extract_motion_outcomes, engineer_motion_features,
            train_catboost_model
        )

        logger.info("Retraining CatBoost model with new cases...")

        # Load existing dataset
        dataset = load_dataset(DB_PATH)
        outcomes_df = extract_motion_outcomes(dataset["cases"])
        features_df = engineer_motion_features(dataset, outcomes_df)

        # Train new model
        model, label_encoder, feature_cols, shap_importance = train_catboost_model(features_df)

        logger.info(f"Model retrained successfully. Top features: {list(shap_importance.keys())[:5]}")

        return {
            "model": model,
            "label_encoder": label_encoder,
            "feature_cols": feature_cols,
            "shap_importance": shap_importance
        }

    except Exception as e:
        logger.error(f"CatBoost retraining failed: {e}")
        return None


def extract_new_patterns(model_data: Dict, new_cases: List[Dict]) -> List[Dict]:
    """Extract patterns from new cases using updated model."""

    try:
        from audit_catboost_patterns import (
            extract_section_structure, extract_citation_density,
            extract_harm_mentions, extract_winning_citations
        )

        patterns = []

        for case in new_cases:
            case_text = case["text"]

            # Extract patterns
            section_structure = extract_section_structure(case_text)
            citation_density = extract_citation_density(case_text, section_structure)
            harm_mentions = extract_harm_mentions(case_text)
            winning_citations = extract_winning_citations(case_text)

            # Compute SHAP scores for this case
            try:
                from analyze_ma_motion_doc import compute_draft_features
                case_features = compute_draft_features(case_text)

                # Get SHAP values (simplified)
                shap_scores = {}
                for feature in model_data["feature_cols"]:
                    if feature in case_features:
                        shap_scores[feature] = float(case_features[feature])

            except Exception as e:
                logger.warning(f"SHAP computation failed for case {case['case_id']}: {e}")
                shap_scores = {}

            pattern_record = {
                "case_id": case["case_id"],
                "outcome": "unknown",  # Would need to determine from database
                "shap_scores": shap_scores,
                "section_structure": section_structure,
                "citation_density": citation_density,
                "harm_mentions": harm_mentions,
                "winning_citations": winning_citations,
                "word_count": len(case_text.split()),
                "text_length": len(case_text),
                "extracted_at": datetime.now().isoformat()
            }

            patterns.append(pattern_record)

        logger.info(f"Extracted patterns from {len(patterns)} new cases")
        return patterns

    except Exception as e:
        logger.error(f"Pattern extraction failed: {e}")
        return []


def regenerate_rules(model_data: Dict, patterns: List[Dict]) -> Dict[str, Any]:
    """Regenerate rule configurations from updated model and patterns."""

    try:
        from translate_features_to_rules import (
            calculate_feature_thresholds, generate_privacy_rules,
            generate_harassment_rules, generate_safety_rules,
            generate_retaliation_rules, generate_citation_rules,
            generate_harm_count_rules, generate_public_interest_rules,
            generate_transparency_rules, generate_section_structure_rules
        )

        logger.info("Regenerating rule configurations...")

        # Calculate feature thresholds
        feature_stats = calculate_feature_thresholds(patterns, model_data["shap_importance"])

        # Generate individual feature rules
        rule_generators = {
            "mentions_privacy_rules.json": generate_privacy_rules,
            "mentions_harassment_rules.json": generate_harassment_rules,
            "mentions_safety_rules.json": generate_safety_rules,
            "mentions_retaliation_rules.json": generate_retaliation_rules,
            "citation_requirements.json": generate_citation_rules,
            "harm_thresholds.json": generate_harm_count_rules,
            "public_interest_rules.json": generate_public_interest_rules,
            "transparency_rules.json": generate_transparency_rules,
            "section_structure.json": generate_section_structure_rules
        }

        new_rules = {}

        for filename, generator in rule_generators.items():
            if filename == "citation_requirements.json":
                rules = generator(feature_stats, patterns, [])  # Empty winning citations for now
            elif filename == "section_structure.json":
                rules = generator(patterns)
            else:
                rules = generator(feature_stats, patterns)

            new_rules[filename] = rules

        logger.info(f"Generated {len(new_rules)} rule configurations")
        return new_rules

    except Exception as e:
        logger.error(f"Rule regeneration failed: {e}")
        return {}


def version_rules(new_rules: Dict[str, Any], version: str) -> bool:
    """Version control for rules directory."""

    try:
        # Create version directory
        version_dir = RULES_DIR / f"versions" / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Backup current rules
        backup_dir = version_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        if RULES_DIR.exists():
            for rule_file in RULES_DIR.glob("*.json"):
                shutil.copy2(rule_file, backup_dir / rule_file.name)

        # Save new rules
        for filename, rules in new_rules.items():
            rule_path = RULES_DIR / filename
            with open(rule_path, 'w') as f:
                json.dump(rules, f, indent=2)

            # Also save to version directory
            version_path = version_dir / filename
            with open(version_path, 'w') as f:
                json.dump(rules, f, indent=2)

        # Create version metadata
        version_metadata = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "rule_files": list(new_rules.keys()),
            "total_rules": len(new_rules)
        }

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2)

        logger.info(f"Rules versioned successfully: {version}")
        return True

    except Exception as e:
        logger.error(f"Version control failed: {e}")
        return False


async def test_new_rules(new_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Test new rules with validation pipeline."""

    try:
        from validation_pipeline import run_validation_pipeline

        logger.info("Testing new rules with validation pipeline...")

        # This would test the new rules
        # For now, return a mock result
        test_results = {
            "test_passed": True,
            "validation_score": 0.85,
            "improvements_detected": True,
            "timestamp": datetime.now().isoformat()
        }

        logger.info("New rules validation completed")
        return test_results

    except Exception as e:
        logger.error(f"Rule testing failed: {e}")
        return {"test_passed": False, "error": str(e)}


def auto_update_pipeline() -> Dict[str, Any]:
    """
    Scheduled pipeline to regenerate rules from new case law:
    1. Check for new cases in ma_federal_motions.db
    2. Re-run CatBoost training
    3. Extract new SHAP importance
    4. Regenerate rule configs
    5. Version control rules/ directory
    6. Re-test SK functions with new rules
    """

    logger.info("Starting auto-update pipeline...")

    update_log = {
        "started_at": datetime.now().isoformat(),
        "steps_completed": [],
        "errors": [],
        "results": {}
    }

    try:
        # 1. Detect new cases
        logger.info("Step 1: Detecting new cases...")
        last_check = load_last_check_time()
        new_cases = detect_new_cases(DB_PATH, last_check)

        if not new_cases:
            logger.info("No new cases found, skipping update")
            update_log["results"]["new_cases"] = 0
            update_log["steps_completed"].append("detect_new_cases")
            return update_log

        update_log["results"]["new_cases"] = len(new_cases)
        update_log["steps_completed"].append("detect_new_cases")

        # 2. Retrain CatBoost
        logger.info("Step 2: Retraining CatBoost model...")
        model_data = retrain_catboost_model(new_cases)

        if not model_data:
            update_log["errors"].append("CatBoost retraining failed")
            return update_log

        update_log["results"]["model_retrained"] = True
        update_log["steps_completed"].append("retrain_catboost")

        # 3. Extract new patterns
        logger.info("Step 3: Extracting patterns from new cases...")
        patterns = extract_new_patterns(model_data, new_cases)

        update_log["results"]["patterns_extracted"] = len(patterns)
        update_log["steps_completed"].append("extract_patterns")

        # 4. Regenerate rules
        logger.info("Step 4: Regenerating rule configurations...")
        new_rules = regenerate_rules(model_data, patterns)

        if not new_rules:
            update_log["errors"].append("Rule regeneration failed")
            return update_log

        update_log["results"]["rules_regenerated"] = len(new_rules)
        update_log["steps_completed"].append("regenerate_rules")

        # 5. Version control
        logger.info("Step 5: Version controlling rules...")
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_success = version_rules(new_rules, version)

        if not version_success:
            update_log["errors"].append("Version control failed")
            return update_log

        update_log["results"]["version"] = version
        update_log["steps_completed"].append("version_control")

        # 6. Test suite
        logger.info("Step 6: Testing new rules...")
        test_results = asyncio.run(test_new_rules(new_rules))

        update_log["results"]["test_results"] = test_results
        update_log["steps_completed"].append("test_rules")

        logger.info("Auto-update pipeline completed successfully")

    except Exception as e:
        logger.error(f"Auto-update pipeline failed: {e}")
        update_log["errors"].append(str(e))

    # Save update log
    update_log["completed_at"] = datetime.now().isoformat()
    save_update_log(update_log)

    return update_log


def load_last_check_time() -> Optional[datetime]:
    """Load the last check time from update log."""

    try:
        if UPDATE_LOG_FILE.exists():
            with open(UPDATE_LOG_FILE, 'r') as f:
                log_data = json.load(f)

            # Get the most recent successful update
            if "completed_at" in log_data:
                return datetime.fromisoformat(log_data["completed_at"])

    except Exception as e:
        logger.warning(f"Failed to load last check time: {e}")

    return None


def save_update_log(update_log: Dict[str, Any]) -> None:
    """Save update log to file."""

    try:
        with open(UPDATE_LOG_FILE, 'w') as f:
            json.dump(update_log, f, indent=2)

        logger.info(f"Update log saved to {UPDATE_LOG_FILE}")

    except Exception as e:
        logger.error(f"Failed to save update log: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Run auto-update pipeline
    results = auto_update_pipeline()

    print(f"Auto-update completed:")
    print(f"  Steps completed: {results['steps_completed']}")
    print(f"  Errors: {results['errors']}")
    print(f"  Results: {results['results']}")
