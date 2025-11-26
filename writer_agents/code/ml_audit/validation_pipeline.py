#!/usr/bin/env python3
"""
Validation Pipeline - Test rule effectiveness against sample cases.

Implements feedback loop testing to validate that atomic plugins
actually improve CatBoost scores on sample cases.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent
VALIDATION_RESULTS_FILE = OUTPUT_DIR / "validation_results.json"


async def validate_rule_effectiveness(
    orchestrator,  # FeatureOrchestrator instance
    test_cases: List[str],
    catboost_model=None
) -> Dict[str, Any]:
    """
    Test rule configs against sample cases:
    1. Generate draft using plugins
    2. Score with CatBoost
    3. Log improvements/regressions
    4. Suggest rule adjustments
    """
    logger.info(f"Starting validation pipeline with {len(test_cases)} test cases")

    results = []

    # Prepare feature extraction function
    compute_draft_features = None
    try:
        import sys
        analysis_path = PROJECT_ROOT / "analysis"
        sys.path.append(str(analysis_path))
        from analyze_ma_motion_doc import compute_draft_features as _compute_draft_features
        compute_draft_features = _compute_draft_features
    except Exception:
        logger.warning("analyze_ma_motion_doc not available; using fallback feature extractor")

        def compute_draft_features(text: str) -> Dict[str, float]:
            text_lower = text.lower()
            return {
                "mentions_privacy": float(text_lower.count("privacy")),
                "mentions_harassment": float(text_lower.count("harass")),
                "mentions_safety": float(text_lower.count("safety")),
                "mentions_retaliation": float(text_lower.count("retaliat")),
                "citation_count": float(text_lower.count(" v. ")),
            }

    for i, draft in enumerate(test_cases):
        logger.info(f"Processing test case {i+1}/{len(test_cases)}")

        try:
            # Baseline score (if CatBoost model available)
            baseline_score = None
            if catboost_model:
                try:
                    features = compute_draft_features(draft)

                    # Convert to DataFrame format
                    feature_df = pd.DataFrame([features])

                    # Ensure all required columns exist
                    if hasattr(catboost_model, 'feature_names_'):
                        for col in catboost_model.feature_names_:
                            if col not in feature_df.columns:
                                feature_df[col] = 0
                        feature_df = feature_df[catboost_model.feature_names_]

                    # Get prediction probabilities
                    proba = catboost_model.predict_proba(feature_df)[0]
                    baseline_score = float(np.max(proba))

                except Exception as e:
                    logger.warning(f"Baseline scoring failed for case {i+1}: {e}")
                    baseline_score = 0.5  # Default neutral score

            # Apply plugins
            weak_features = await orchestrator.analyze_draft(draft)
            improved_draft = await orchestrator.strengthen_draft(draft, weak_features)

            # New score (if CatBoost model available)
            improved_score = None
            if catboost_model and baseline_score is not None:
                try:
                    features = compute_draft_features(improved_draft)
                    feature_df = pd.DataFrame([features])

                    # Ensure all required columns exist
                    if hasattr(catboost_model, 'feature_names_'):
                        for col in catboost_model.feature_names_:
                            if col not in feature_df.columns:
                                feature_df[col] = 0
                        feature_df = feature_df[catboost_model.feature_names_]

                    proba = catboost_model.predict_proba(feature_df)[0]
                    improved_score = float(np.max(proba))

                except Exception as e:
                    logger.warning(f"Improved scoring failed for case {i+1}: {e}")
                    improved_score = baseline_score

            # Calculate improvement metrics
            delta = 0.0
            improvement_percent = 0.0
            if baseline_score is not None and improved_score is not None:
                delta = improved_score - baseline_score
                improvement_percent = (delta / baseline_score) * 100 if baseline_score > 0 else 0

            # Analyze plugin effectiveness
            plugin_analysis = {}
            for feature_name, analysis in weak_features.items():
                plugin_analysis[feature_name] = {
                    "gap": analysis.get("gap", 0),
                    "current": analysis.get("current", 0),
                    "target": analysis.get("target", 0),
                    "priority": analysis.get("priority", "medium")
                }

            result = {
                "case_id": f"test_case_{i+1}",
                "baseline_score": baseline_score,
                "improved_score": improved_score,
                "delta": delta,
                "improvement_percent": improvement_percent,
                "plugins_invoked": list(weak_features.keys()),
                "plugin_analysis": plugin_analysis,
                "original_length": len(draft),
                "improved_length": len(improved_draft),
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

            baseline_str = f"{baseline_score:.3f}" if baseline_score is not None else "N/A"
            improved_str = f"{improved_score:.3f}" if improved_score is not None else "N/A"
            logger.info(f"Case {i+1}: Baseline={baseline_str}, Improved={improved_str}, Delta={delta:.3f}")

        except Exception as e:
            logger.error(f"Validation failed for case {i+1}: {e}")
            results.append({
                "case_id": f"test_case_{i+1}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

    # Analyze results
    summary = analyze_plugin_effectiveness(results)

    # Save results
    validation_data = {
        "summary": summary,
        "detailed_results": results,
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(test_cases)
    }

    with open(VALIDATION_RESULTS_FILE, 'w') as f:
        json.dump(validation_data, f, indent=2)

    logger.info(f"Validation results saved to {VALIDATION_RESULTS_FILE}")
    logger.info(f"Summary: {summary}")

    return validation_data


def analyze_plugin_effectiveness(results: List[Dict]) -> Dict[str, Any]:
    """Analyze which plugins consistently improve scores."""

    # Filter out error cases
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {"error": "No valid results to analyze"}

    # Calculate overall statistics
    total_cases = len(valid_results)
    improved_cases = len([r for r in valid_results if r.get("delta", 0) > 0])
    avg_improvement = np.mean([r.get("delta", 0) for r in valid_results])
    avg_improvement_percent = np.mean([r.get("improvement_percent", 0) for r in valid_results])

    # Analyze plugin effectiveness
    plugin_stats = {}
    for result in valid_results:
        for plugin in result.get("plugins_invoked", []):
            if plugin not in plugin_stats:
                plugin_stats[plugin] = {
                    "invocations": 0,
                    "total_delta": 0,
                    "cases": []
                }

            plugin_stats[plugin]["invocations"] += 1
            plugin_stats[plugin]["total_delta"] += result.get("delta", 0)
            plugin_stats[plugin]["cases"].append({
                "case_id": result["case_id"],
                "delta": result.get("delta", 0),
                "improvement_percent": result.get("improvement_percent", 0)
            })

    # Calculate plugin averages
    for plugin, stats in plugin_stats.items():
        stats["avg_delta"] = stats["total_delta"] / stats["invocations"]
        stats["success_rate"] = len([c for c in stats["cases"] if c["delta"] > 0]) / stats["invocations"]

    # Identify most effective plugins
    effective_plugins = sorted(
        plugin_stats.items(),
        key=lambda x: x[1]["avg_delta"],
        reverse=True
    )

    # Identify problematic plugins
    problematic_plugins = [
        (plugin, stats) for plugin, stats in plugin_stats.items()
        if stats["avg_delta"] < -0.05  # 5% regression threshold
    ]

    summary = {
        "overall_stats": {
            "total_cases": total_cases,
            "improved_cases": improved_cases,
            "improvement_rate": improved_cases / total_cases if total_cases > 0 else 0,
            "avg_improvement": avg_improvement,
            "avg_improvement_percent": avg_improvement_percent
        },
        "plugin_effectiveness": plugin_stats,
        "most_effective_plugins": effective_plugins[:5],  # Top 5
        "problematic_plugins": problematic_plugins,
        "recommendations": generate_recommendations(plugin_stats, avg_improvement)
    }

    return summary


def generate_recommendations(plugin_stats: Dict, avg_improvement: float) -> List[str]:
    """Generate recommendations based on plugin effectiveness analysis."""
    recommendations = []

    # Overall performance
    if avg_improvement > 0.1:
        recommendations.append("✅ Overall plugin system is performing well with significant improvements")
    elif avg_improvement > 0.05:
        recommendations.append("🟡 Plugin system shows moderate improvements, consider rule tuning")
    else:
        recommendations.append("❌ Plugin system shows minimal improvements, review rule configurations")

    # Plugin-specific recommendations
    for plugin, stats in plugin_stats.items():
        if stats["avg_delta"] > 0.1:
            recommendations.append(f"✅ {plugin} is highly effective (avg improvement: {stats['avg_delta']:.3f})")
        elif stats["avg_delta"] < -0.05:
            recommendations.append(f"❌ {plugin} shows regression (avg change: {stats['avg_delta']:.3f}), review rules")
        elif stats["success_rate"] < 0.5:
            recommendations.append(f"🟡 {plugin} has low success rate ({stats['success_rate']:.1%}), consider threshold adjustments")

    return recommendations


async def run_validation_pipeline(orchestrator, test_cases: List[str] = None) -> Dict[str, Any]:
    """Run the complete validation pipeline."""

    if test_cases is None:
        # Generate sample test cases
        test_cases = generate_sample_test_cases()

    logger.info(f"Running validation pipeline with {len(test_cases)} test cases")

    # Run validation
    results = await validate_rule_effectiveness(orchestrator, test_cases)

    return results


def generate_sample_test_cases() -> List[str]:
    """Generate sample test cases for validation."""

    sample_cases = [
        """
        MOTION TO SEAL

        This case involves a dispute between two parties regarding personal information.
        The plaintiff seeks to seal certain documents to protect privacy.

        The documents contain sensitive information that should not be public.
        """,

        """
        MOTION FOR PSEUDONYM

        Plaintiff requests permission to proceed under a pseudonym due to privacy concerns.

        This matter involves personal information that could cause harm if disclosed.
        The plaintiff fears retaliation and harassment if their identity is revealed.
        """,

        """
        MOTION TO SEAL COURT RECORDS

        Defendant moves to seal certain court records in this case.

        The records contain confidential business information and personal details.
        Public disclosure would cause significant harm to the defendant's business interests.
        """,

        """
        MOTION FOR PROTECTIVE ORDER

        Plaintiff seeks a protective order to prevent disclosure of sensitive information.

        The information at issue includes personal addresses, phone numbers, and financial data.
        Disclosure would create safety concerns for the plaintiff and their family.
        """,

        """
        MOTION TO PROCEED ANONYMOUSLY

        Plaintiff requests to proceed anonymously in this litigation.

        This case involves allegations of harassment and intimidation.
        The plaintiff has a reasonable fear of retaliation if their identity is disclosed.
        """
    ]

    return sample_cases


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    async def main():
        # This would be used with a real orchestrator
        # orchestrator = FeatureOrchestrator(plugins, catboost_model)
        # results = await run_validation_pipeline(orchestrator)
        # print(f"Validation completed: {results['summary']}")
        pass

    asyncio.run(main())
