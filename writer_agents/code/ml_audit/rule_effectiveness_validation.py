#!/usr/bin/env python3
"""
Rule Effectiveness Validation - Test rule effectiveness against sample cases.

Validates that atomic plugins and their rules actually improve motion quality
when applied to sample cases.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent
VALIDATION_RESULTS_FILE = OUTPUT_DIR / "rule_effectiveness_results.json"


class RuleEffectivenessValidator:
    """Validates rule effectiveness against sample cases."""

    def __init__(self, orchestrator, sample_cases: List[Dict[str, Any]]):
        self.orchestrator = orchestrator
        self.sample_cases = sample_cases
        self.results = []

    async def validate_all_rules(self) -> Dict[str, Any]:
        """Validate all rules against sample cases."""
        logger.info(f"Starting rule effectiveness validation with {len(self.sample_cases)} sample cases")

        for i, case in enumerate(self.sample_cases):
            logger.info(f"Validating case {i+1}/{len(self.sample_cases)}: {case.get('case_id', 'unknown')}")

            try:
                result = await self._validate_single_case(case)
                self.results.append(result)

            except Exception as e:
                logger.error(f"Validation failed for case {i+1}: {e}")
                self.results.append({
                    "case_id": case.get("case_id", f"case_{i+1}"),
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        # Analyze results
        analysis = self._analyze_results()

        # Save results
        self._save_results(analysis)

        return analysis

    async def _validate_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single case against all rules."""

        case_id = case.get("case_id", "unknown")
        draft_text = case.get("text", "")
        expected_outcome = case.get("outcome", "unknown")

        # Get baseline analysis
        baseline_features = await self._get_baseline_features(draft_text)

        # Apply each plugin individually to measure effectiveness
        plugin_results = {}

        for plugin_name, plugin in self.orchestrator.plugins.items():
            try:
                plugin_result = await self._test_plugin_effectiveness(plugin, draft_text, baseline_features)
                plugin_results[plugin_name] = plugin_result

            except Exception as e:
                logger.warning(f"Plugin {plugin_name} failed for case {case_id}: {e}")
                plugin_results[plugin_name] = {"error": str(e)}

        # Test combined plugin effectiveness
        combined_result = await self._test_combined_effectiveness(draft_text, baseline_features)

        return {
            "case_id": case_id,
            "expected_outcome": expected_outcome,
            "baseline_features": baseline_features,
            "plugin_results": plugin_results,
            "combined_result": combined_result,
            "timestamp": datetime.now().isoformat()
        }

    async def _get_baseline_features(self, draft_text: str) -> Dict[str, Any]:
        """Get baseline feature analysis for the draft."""
        try:
            # Use orchestrator's analyze_draft method
            weak_features = await self.orchestrator.analyze_draft(draft_text)

            # Extract feature values
            baseline_features = {}
            for feature_name, analysis in weak_features.items():
                baseline_features[feature_name] = {
                    "current_value": analysis.get("current", 0),
                    "target_value": analysis.get("target", 0),
                    "gap": analysis.get("gap", 0),
                    "priority": analysis.get("priority", "medium")
                }

            return baseline_features

        except Exception as e:
            logger.error(f"Baseline feature analysis failed: {e}")
            return {}

    async def _test_plugin_effectiveness(self, plugin, draft_text: str, baseline_features: Dict) -> Dict[str, Any]:
        """Test effectiveness of a single plugin."""

        plugin_name = plugin.feature_name

        try:
            # Query Chroma for patterns
            results = await plugin.query_chroma(draft_text[:500])

            # Extract patterns
            patterns = await plugin.extract_patterns(results)

            # Generate argument
            argument = await plugin.generate_argument(patterns, draft_text)

            # Validate the generated argument
            validation = await plugin.validate_draft(argument)

            # Measure improvement
            improvement_metrics = self._calculate_improvement_metrics(
                baseline_features.get(plugin_name, {}),
                validation.value if validation.success else {}
            )

            return {
                "plugin_name": plugin_name,
                "chroma_results_count": len(results),
                "patterns_extracted": len(patterns.get("common_phrases", [])),
                "argument_length": len(argument),
                "validation_passed": validation.success,
                "validation_score": validation.value.get("score", 0) if validation.success else 0,
                "improvement_metrics": improvement_metrics,
                "generated_argument": argument[:200] + "..." if len(argument) > 200 else argument
            }

        except Exception as e:
            logger.error(f"Plugin effectiveness test failed for {plugin_name}: {e}")
            return {"error": str(e)}

    async def _test_combined_effectiveness(self, draft_text: str, baseline_features: Dict) -> Dict[str, Any]:
        """Test effectiveness of all plugins combined."""

        try:
            # Run full orchestrator workflow
            weak_features = await self.orchestrator.analyze_draft(draft_text)

            if not weak_features:
                return {
                    "no_weak_features": True,
                    "improvement_percent": 0,
                    "plugins_applied": 0
                }

            # Strengthen draft
            improved_draft = await self.orchestrator.strengthen_draft(draft_text, weak_features)

            # Validate with CatBoost if available
            validation = await self.orchestrator.validate_with_catboost(improved_draft)

            return {
                "original_length": len(draft_text),
                "improved_length": len(improved_draft),
                "weak_features_count": len(weak_features),
                "plugins_applied": len(weak_features),
                "catboost_improved": validation.get("improved", False),
                "improvement_percent": validation.get("improvement_percent", 0),
                "confidence": validation.get("confidence", 0),
                "validation_success": validation.get("improved", False)
            }

        except Exception as e:
            logger.error(f"Combined effectiveness test failed: {e}")
            return {"error": str(e)}

    def _calculate_improvement_metrics(self, baseline: Dict, validation: Dict) -> Dict[str, Any]:
        """Calculate improvement metrics for a plugin."""

        if not baseline or not validation:
            return {"improvement_detected": False, "metrics": {}}

        current_value = baseline.get("current_value", 0)
        target_value = baseline.get("target_value", 0)
        validation_score = validation.get("score", 0)

        # Calculate improvement
        if target_value > 0:
            improvement_percent = ((validation_score - current_value) / target_value) * 100
        else:
            improvement_percent = 0

        improvement_detected = validation_score > current_value

        return {
            "improvement_detected": improvement_detected,
            "improvement_percent": improvement_percent,
            "current_value": current_value,
            "target_value": target_value,
            "validation_score": validation_score,
            "gap_reduction": max(0, current_value - validation_score)
        }

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze validation results to determine rule effectiveness."""

        if not self.results:
            return {"error": "No results to analyze"}

        # Filter out error cases
        valid_results = [r for r in self.results if "error" not in r]

        if not valid_results:
            return {"error": "No valid results to analyze"}

        # Analyze plugin effectiveness
        plugin_stats = defaultdict(lambda: {
            "total_tests": 0,
            "successful_tests": 0,
            "avg_improvement": 0,
            "validation_pass_rate": 0,
            "cases": []
        })

        # Analyze combined effectiveness
        combined_stats = {
            "total_cases": len(valid_results),
            "cases_with_improvements": 0,
            "avg_improvement_percent": 0,
            "avg_plugins_applied": 0,
            "catboost_improvements": 0
        }

        total_improvement = 0
        total_plugins_applied = 0
        catboost_improvements = 0

        for result in valid_results:
            # Analyze plugin results
            for plugin_name, plugin_result in result.get("plugin_results", {}).items():
                if "error" not in plugin_result:
                    plugin_stats[plugin_name]["total_tests"] += 1

                    if plugin_result.get("validation_passed", False):
                        plugin_stats[plugin_name]["successful_tests"] += 1

                    improvement = plugin_result.get("improvement_metrics", {})
                    if improvement.get("improvement_detected", False):
                        plugin_stats[plugin_name]["avg_improvement"] += improvement.get("improvement_percent", 0)

                    plugin_stats[plugin_name]["cases"].append({
                        "case_id": result["case_id"],
                        "validation_passed": plugin_result.get("validation_passed", False),
                        "improvement_percent": improvement.get("improvement_percent", 0)
                    })

            # Analyze combined results
            combined_result = result.get("combined_result", {})
            if combined_result.get("validation_success", False):
                combined_stats["cases_with_improvements"] += 1

            improvement_percent = combined_result.get("improvement_percent", 0)
            total_improvement += improvement_percent

            plugins_applied = combined_result.get("plugins_applied", 0)
            total_plugins_applied += plugins_applied

            if combined_result.get("catboost_improved", False):
                catboost_improvements += 1

        # Calculate averages
        for plugin_name, stats in plugin_stats.items():
            if stats["total_tests"] > 0:
                stats["validation_pass_rate"] = stats["successful_tests"] / stats["total_tests"]
                stats["avg_improvement"] = stats["avg_improvement"] / stats["total_tests"]

        combined_stats["avg_improvement_percent"] = total_improvement / len(valid_results)
        combined_stats["avg_plugins_applied"] = total_plugins_applied / len(valid_results)
        combined_stats["catboost_improvements"] = catboost_improvements

        # Generate recommendations
        recommendations = self._generate_recommendations(plugin_stats, combined_stats)

        return {
            "summary": {
                "total_cases_tested": len(valid_results),
                "total_cases_with_errors": len(self.results) - len(valid_results),
                "overall_success_rate": combined_stats["cases_with_improvements"] / len(valid_results),
                "avg_improvement_percent": combined_stats["avg_improvement_percent"],
                "catboost_improvement_rate": catboost_improvements / len(valid_results)
            },
            "plugin_effectiveness": dict(plugin_stats),
            "combined_effectiveness": combined_stats,
            "recommendations": recommendations,
            "detailed_results": self.results
        }

    def _generate_recommendations(self, plugin_stats: Dict, combined_stats: Dict) -> List[str]:
        """Generate recommendations based on validation results."""

        recommendations = []

        # Overall system performance
        success_rate = combined_stats["cases_with_improvements"] / combined_stats["total_cases"]
        if success_rate > 0.8:
            recommendations.append("✅ Overall system performance is excellent (>80% success rate)")
        elif success_rate > 0.6:
            recommendations.append("🟡 Overall system performance is good (>60% success rate)")
        else:
            recommendations.append("❌ Overall system performance needs improvement (<60% success rate)")

        # Plugin-specific recommendations
        for plugin_name, stats in plugin_stats.items():
            if stats["total_tests"] > 0:
                pass_rate = stats["validation_pass_rate"]
                avg_improvement = stats["avg_improvement"]

                if pass_rate > 0.8 and avg_improvement > 10:
                    recommendations.append(f"✅ {plugin_name} is highly effective (pass rate: {pass_rate:.1%}, avg improvement: {avg_improvement:.1f}%)")
                elif pass_rate < 0.5:
                    recommendations.append(f"❌ {plugin_name} has low pass rate ({pass_rate:.1%}), consider rule adjustments")
                elif avg_improvement < 5:
                    recommendations.append(f"🟡 {plugin_name} shows minimal improvement ({avg_improvement:.1f}%), review thresholds")

        # CatBoost integration
        catboost_rate = combined_stats["catboost_improvements"] / combined_stats["total_cases"]
        if catboost_rate > 0.7:
            recommendations.append("✅ CatBoost integration is working well")
        else:
            recommendations.append("🟡 CatBoost integration could be improved")

        return recommendations

    def _save_results(self, analysis: Dict[str, Any]) -> None:
        """Save validation results to file."""

        try:
            with open(VALIDATION_RESULTS_FILE, 'w') as f:
                json.dump(analysis, f, indent=2)

            logger.info(f"Rule effectiveness results saved to {VALIDATION_RESULTS_FILE}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def generate_sample_cases() -> List[Dict[str, Any]]:
    """Generate sample cases for rule validation."""

    sample_cases = [
        {
            "case_id": "sample_privacy_weak",
            "outcome": "granted",
            "text": """
            MOTION TO SEAL

            This case involves some privacy concerns.
            The plaintiff seeks to seal certain documents.
            """
        },
        {
            "case_id": "sample_citation_weak",
            "outcome": "granted",
            "text": """
            MOTION FOR PSEUDONYM

            Plaintiff requests permission to proceed under a pseudonym.
            This matter involves personal information.
            """
        },
        {
            "case_id": "sample_harassment_weak",
            "outcome": "granted",
            "text": """
            MOTION TO SEAL COURT RECORDS

            Defendant moves to seal certain court records.
            The records contain confidential information.
            """
        },
        {
            "case_id": "sample_multiple_weak",
            "outcome": "granted",
            "text": """
            MOTION FOR PROTECTIVE ORDER

            Plaintiff seeks a protective order.
            The information includes personal details.
            """
        },
        {
            "case_id": "sample_strong_case",
            "outcome": "granted",
            "text": """
            MOTION TO PROCEED ANONYMOUSLY

            Plaintiff requests to proceed anonymously due to significant privacy harm.
            This case involves substantial personal information disclosure that would cause
            harassment and retaliation. The privacy interests outweigh any public interest.

            The case of 605 F. Supp. 3 establishes the standard for privacy harm analysis.
            Additionally, 353 Mass. 614 provides guidance on personal information protection.
            The expectation of privacy is substantial in this context.
            """
        }
    ]

    return sample_cases


async def run_rule_effectiveness_validation(orchestrator) -> Dict[str, Any]:
    """Run complete rule effectiveness validation."""

    logger.info("🔍 Starting Rule Effectiveness Validation")

    # Generate sample cases
    sample_cases = generate_sample_cases()

    # Create validator
    validator = RuleEffectivenessValidator(orchestrator, sample_cases)

    # Run validation
    results = await validator.validate_all_rules()

    logger.info("✅ Rule effectiveness validation completed")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    async def main():
        # This would be used with a real orchestrator
        # orchestrator = FeatureOrchestrator(plugins, catboost_model)
        # results = await run_rule_effectiveness_validation(orchestrator)
        # print(f"Validation completed: {results['summary']}")
        pass

    asyncio.run(main())
