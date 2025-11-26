#!/usr/bin/env python3
"""
Refinement Loop - Sub-coordinator for quality improvement (specialized component, NOT the main orchestrator).

SYSTEM HIERARCHY & MENTAL MODEL:
┌─────────────────────────────────────────────────────────┐
│ Conductor                                               │
│ = THE MAIN GLUE that orchestrates everything           │
│ - Coordinates AutoGen (writing team)                   │
│ - Coordinates SK Kernel (quality control framework)    │
│ - Manages workflow phases (EXPLORE → PLAN → DRAFT →   │
│   VALIDATE → REVIEW → REFINE → COMMIT)                 │
└─────────────────────────────────────────────────────────┘
                    │ calls during VALIDATE/REFINE phases
                    ↓
┌─────────────────────────────────────────────────────────┐
│ RefinementLoop (this class)                            │
│ = SUB-COORDINATOR for iterative quality enhancement    │
│ - Coordinates CatBoost research + SK quality plugins   │
│ - Specialized for feature analysis & improvement       │
│ - Runs iterative refinement feedback loops             │
└─────────────────────────────────────────────────────────┘

TEAMS:
- CatBoost = Research/Quant team (ML predictions, feature analysis)
- Semantic Kernel = Quality Control framework (plugins enforce standards)
- AutoGen = Writing team (generates initial drafts)

WHAT IS DRAFTENHANCER BUILT WITH?
- Custom Python sub-coordinator (NOT AutoGen, NOT Agents SDK, NOT main orchestrator)
- Directly calls CatBoost library (Python: catboost.CatBoostClassifier)
- Coordinates SK plugins (plugins use SK Kernel internally)
- Pure Python coordination logic with asyncio

NOTE: Conductor is the REAL orchestrator/glue.
RefinementLoop is a specialized sub-component for iterative quality enhancement.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os
import hashlib

from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, EditResult, DocumentLocation
from .document_structure import parse_document_structure, apply_edit as apply_single_edit, DocumentStructure
from .feature_extractor import FeatureExtractor
from .catboost_predictor import CatBoostPredictor
from .shap_analyzer import SHAPAnalyzer
from .memory_manager import MemoryManager

# Import outline management (package-safe: try absolute then fallback)
try:
    from writer_agents.code.outline_manager import OutlineManager, load_outline_manager  # type: ignore
    from writer_agents.code.plugin_calibrator import PluginCalibrator  # type: ignore
    OUTLINE_AVAILABLE = True
except Exception:
    try:
        # Fallback: adjust path to include writer_agents/code
        code_dir = Path(__file__).parent.parent.parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from outline_manager import OutlineManager, load_outline_manager  # type: ignore
        from plugin_calibrator import PluginCalibrator  # type: ignore
        OUTLINE_AVAILABLE = True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Outline management not available: {e}")
        OutlineManager = None
        PluginCalibrator = None
        OUTLINE_AVAILABLE = False

logger = logging.getLogger(__name__)


class RefinementLoop:
    """
    Sub-coordinator for iterative quality enhancement (NOT the main orchestrator).

    Position: Called BY Conductor during VALIDATE/REFINE phases
    Main Orchestrator: Conductor (the real glue that holds everything together)

    Built with: Custom Python sub-coordinator
    - Directly calls CatBoost library for research/analysis
    - Coordinates SK plugins (which use SK Kernel internally)
    - Pure Python coordination logic with asyncio
    - Runs iterative refinement feedback loops

    Mental Model:
    - CatBoost = Research/Quant team (direct Python library calls)
    - SK Plugins = Quality Control team (plugins use SK Kernel)
    - AutoGen = Writing team (generates drafts) - coordinated by Conductor

    This sub-coordinator:
    - Calls CatBoost directly to identify weak features & predict success
    - Coordinates SK quality control plugins to generate improvements
    - Uses custom logic to prioritize, merge, and enhance drafts
    - Runs iterative feedback loops until quality improves

    NOTE: Conductor is the main orchestrator.
    RefinementLoop is specialized for iterative quality enhancement coordination.
    """

    def __init__(
        self,
        plugins: Dict[str, BaseFeaturePlugin],
        catboost_model=None,
        model_path: Optional[Path] = None,
        shap_importance: Optional[Dict[str, float]] = None,
        debug_mode: bool = False,
        outline_manager: Optional[Any] = None,
        plugin_calibrator: Optional[Any] = None
    ):
        self.plugins = plugins
        self.catboost_model = catboost_model
        self.model_path = model_path
        self.shap_importance = shap_importance or {}
        self.baseline_score = None
        self.feature_targets = {}
        self.label_encoder = None
        self.debug_mode = debug_mode

        # ✅ Initialize outline management (perfect outline structure)
        if OUTLINE_AVAILABLE:
            self.outline_manager = outline_manager or load_outline_manager()
            self.plugin_calibrator = plugin_calibrator or PluginCalibrator(self.outline_manager)
            logger.info("✅ Outline Manager initialized - plugins will be organized by perfect outline structure")
            logger.info(f"   Perfect outline has {len(self.outline_manager.sections)} sections")
        else:
            self.outline_manager = None
            self.plugin_calibrator = None
            logger.warning("Outline management not available - using legacy plugin organization")

        # Initialize specialized components
        self.feature_extractor = FeatureExtractor()
        self.catboost_predictor = CatBoostPredictor(
            model_path=model_path if not catboost_model else None,
            shap_importance=shap_importance
        )
        # Use provided model if available
        if catboost_model:
            self.catboost_predictor.model = catboost_model
            self.catboost_predictor.shap_importance = shap_importance or {}
        self.shap_analyzer = SHAPAnalyzer(self.catboost_predictor)
        
        # Find memory store from plugins
        memory_store = None
        for plugin in plugins.values():
            if hasattr(plugin, 'memory_store') and plugin.memory_store:
                memory_store = plugin.memory_store
                break
        self.memory_manager = MemoryManager(memory_store)

        # Keep backward compatibility - expose model for legacy code
        self.catboost_model = self.catboost_predictor.model
        self.shap_importance = self.catboost_predictor.shap_importance

        # Pre-computed lookup tables for performance
        self._feature_to_plugin_map: Dict[str, str] = {}
        self._feature_target_lookup: Dict[str, Dict[str, Any]] = {}
        self._successful_case_averages: Dict[str, float] = {}

        # Debug output directory
        self._debug_output_dir = Path(__file__).parents[2] / "debug_output" if debug_mode else None
        if self._debug_output_dir and debug_mode:
            self._debug_output_dir.mkdir(exist_ok=True)

        # Load CatBoost model unless explicitly skipped
        skip_catboost = str(os.environ.get("MATRIX_SKIP_CATBOOST", "")).strip().lower() in {"1", "true", "yes", "on"}
        if skip_catboost:
            # Clear model if skipping
            self.catboost_predictor.model = None
            self.catboost_model = None

        # SHAP importance is loaded by CatBoostPredictor during initialization
        # Update shap_importance reference to use predictor's value
        if self.catboost_predictor.shap_importance:
            self.shap_importance = self.catboost_predictor.shap_importance

        # Generate feature targets from SHAP values if available
        if self.shap_importance:
            self._generate_feature_targets()

        # ✅ Recalibrate plugin targets based on perfect outline if available
        if self.plugin_calibrator:
            self._recalibrate_plugin_targets_from_outline()

        # Pre-compute lookup tables after feature targets are generated
        self._build_lookup_tables()

        logger.info(f"RefinementLoop initialized with {len(plugins)} plugins")
        if self.catboost_model:
            logger.info("CatBoost model loaded and ready")
        if self.outline_manager:
            logger.info("✅ Perfect outline structure loaded - plugins organized by section")

    # Note: Model loading and SHAP importance extraction are now handled by CatBoostPredictor
    # These methods have been moved to maintain single responsibility principle

    def _recalibrate_plugin_targets_from_outline(self) -> None:
        """Recalibrate plugin targets based on perfect outline structure."""
        if not self.plugin_calibrator:
            return

        logger.info("🔄 Recalibrating plugin targets based on perfect outline structure...")

        # Get all calibrations
        calibrations = self.plugin_calibrator.get_all_calibrations()

        # Update feature_targets with outline-based targets
        for plugin_name, calibration in calibrations.items():
            section_name = calibration.get("section")
            target = self.plugin_calibrator.get_target_value(plugin_name, section_name)

            if target is not None:
                # Update or add target
                # Handle dict-shaped current_target (from _generate_feature_targets)
                current_target_raw = self.feature_targets.get(plugin_name, 0)
                if isinstance(current_target_raw, dict):
                    current_target = current_target_raw.get('target', 0.0)
                else:
                    current_target = float(current_target_raw) if current_target_raw else 0.0
                
                updated_target = self.plugin_calibrator.update_plugin_targets(
                    plugin_name, current_target, section_name
                )
                
                # Preserve dict structure if it exists, otherwise store as float
                if isinstance(current_target_raw, dict):
                    # Update the 'target' field in the dict
                    current_target_raw['target'] = updated_target
                    self.feature_targets[plugin_name] = current_target_raw
                else:
                    self.feature_targets[plugin_name] = updated_target

                if updated_target != current_target:
                    logger.info(f"   📊 {plugin_name}: {current_target} → {updated_target} (section: {section_name})")

        # Count targets robustly (handle dict-shaped targets)
        def _is_positive(val: Any) -> bool:
            try:
                if isinstance(val, dict):
                    return float(val.get('target', 0)) > 0
                return float(val) > 0
            except Exception:
                return False
        count = sum(1 for t in self.feature_targets.values() if _is_positive(t))
        logger.info(f"✅ Recalibrated {count} plugin targets")

    def _generate_feature_targets(self) -> None:
        """Generate feature targets based on SHAP importance and successful case patterns."""
        try:
            # Load successful case averages for Section 1782 discovery cases
            successful_averages = self._load_successful_case_averages()

            # Feature mapping: CatBoost feature names -> SK plugin names (for legacy plugins)
            feature_plugin_map = {
                'mentions_privacy': 'mentions_privacy',
                'mentions_harassment': 'mentions_harassment',
                'mentions_safety': 'mentions_safety',
                'mentions_retaliation': 'mentions_retaliation',
                'citation_count': 'citation_retrieval',
                'privacy_harm_count': 'privacy_harm_count',
                'mentions_public_interest': 'public_interest',
                'mentions_transparency': 'transparency_argument',
            }

            # Legacy successful case averages (fallback for MA federal motions)
            legacy_averages = {
                'mentions_privacy': 1.91,
                'mentions_safety': 0.70,
                'privacy_harm_count': 6.49,
                'citation_count': 5.0,
                'mentions_harassment': 1.0,
                'mentions_retaliation': 1.0,
                'mentions_public_interest': 2.0,
                'mentions_transparency': 1.0,
            }

            # Generate targets based on SHAP importance
            # Calculate max_shap safely (handle dict values)
            max_shap = 1.0
            if self.shap_importance:
                shap_values = []
                for val in self.shap_importance.values():
                    if isinstance(val, dict):
                        val = val.get('mean_abs_shap') or val.get('value') or 0.0
                    try:
                        shap_values.append(float(val))
                    except (TypeError, ValueError):
                        pass
                if shap_values:
                    max_shap = max(shap_values)
                if max_shap <= 0:
                    max_shap = 1.0

            for feature_name, shap_value in self.shap_importance.items():
                # Handle dict-shaped SHAP entries
                if isinstance(shap_value, dict):
                    shap_value = shap_value.get('mean_abs_shap') or shap_value.get('value') or 0.0
                try:
                    shap_value = float(shap_value)
                except Exception:
                    shap_value = 0.0
                # Map to plugin name (if mapped, otherwise use feature name directly)
                plugin_name = feature_plugin_map.get(feature_name, feature_name)

                # Get base target from successful case averages (prefer Section 1782, fallback to legacy)
                base_target = successful_averages.get(feature_name) or legacy_averages.get(feature_name)
                
                # Handle dict-shaped targets gracefully
                if isinstance(base_target, dict):
                    base_target = base_target.get('target', 1.0)
                
                # If no base target found, use a reasonable default based on feature type
                if base_target is None:
                    base_target = self._estimate_default_target(feature_name, successful_averages)
                
                # Ensure base_target is a float (handle all edge cases)
                # Check for dict first (could come from successful_averages or _estimate_default_target)
                if isinstance(base_target, dict):
                    base_target = base_target.get('target', 1.0)
                
                # Handle None case
                if base_target is None:
                    base_target = 1.0
                
                # Convert to float
                try:
                    base_target = float(base_target)
                except (TypeError, ValueError):
                    base_target = 1.0
                
                # Final safety check - ensure it's a valid positive number
                if not isinstance(base_target, (int, float)) or base_target <= 0:
                    base_target = 1.0

                # Normalize SHAP to 0-1 range
                normalized_shap = shap_value / max_shap if max_shap > 0 else 0.5

                # Scale target: min 80% of base, max 150% of base based on importance
                scaled_target = base_target * (0.8 + 0.7 * normalized_shap)

                self.feature_targets[plugin_name] = {
                    'target': scaled_target,
                    'base_target': base_target,
                    'shap_importance': shap_value,
                    'normalized_shap': normalized_shap,
                    'feature_name': feature_name
                }

            logger.info(f"Generated {len(self.feature_targets)} feature targets from SHAP values")
            logger.debug(f"Top 5 targets: {sorted(self.feature_targets.items(), key=lambda x: x[1]['shap_importance'], reverse=True)[:5]}")

        except Exception as e:
            logger.error(f"Failed to generate feature targets: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _load_successful_case_averages(self) -> Dict[str, float]:
        """Load successful case averages from unified_features.csv for Section 1782 discovery.
        
        OPTIMIZED: Skip loading 3M-row CSV (310MB, 10+ minutes) - system uses legacy_averages fallback instead.
        The fallback chain in _generate_feature_targets() will use hardcoded defaults (see line 274).
        """
        # Skip CSV loading - it's too slow (3M rows, 310MB, 10+ minutes)
        # System will use legacy_averages hardcoded defaults (see _generate_feature_targets line 274)
        # Top 20 citations are already loaded from top_motion_citations.json via CitationRetrievalPlugin
        logger.info("Skipping unified_features.csv loading (3M rows too slow) - using legacy_averages fallback")
        return {}

    def _estimate_default_target(self, feature_name: str, successful_averages: Dict[str, float]) -> float:
        """Estimate a default target value for a feature based on its type."""
        # Structural sentence features - aim for moderate sentence length
        if 'sentence_word_count' in feature_name:
            if 'max' in feature_name:
                return 45.0  # Maximum sentence length target
            elif 'median' in feature_name:
                return 22.0  # Median sentence length target
            elif 'avg' in feature_name:
                return 20.0  # Average sentence length target

        # Paragraph features
        if 'paragraph' in feature_name:
            if 'count' in feature_name:
                return 15.0  # Paragraph count target
            elif 'word_count' in feature_name:
                if 'avg' in feature_name or 'median' in feature_name:
                    return 120.0  # Words per paragraph target
                elif 'intro' in feature_name:
                    return 80.0  # Introduction paragraph
                elif 'conclusion' in feature_name:
                    return 60.0  # Conclusion paragraph

        # Ratio features - typically 0-1 range
        if 'ratio' in feature_name or 'density' in feature_name:
            return 0.15  # 15% target ratio

        # Intel factor features
        if 'intel' in feature_name.lower():
            if 'focus_score' in feature_name:
                return 0.25  # Composite focus score
            elif 'token_ratio' in feature_name or 'density' in feature_name:
                return 0.10  # 10% token share/density

        # Enumeration density
        if 'enumeration' in feature_name:
            return 0.20  # 20% enumeration density

        # Default fallback
        return 1.0

    async def orchestrate_improvements(self, draft_text: str) -> Dict[str, Any]:
        """Enhanced orchestration: analyze draft, prioritize improvements based on CatBoost features."""
        try:
            # Compute CatBoost features
            analysis_result = await self.analyze_draft(draft_text)
            # Handle both old format (dict of weak_features) and new format (dict with 'weak_features' key)
            if isinstance(analysis_result, dict) and "weak_features" in analysis_result:
                weak_features = analysis_result["weak_features"]
                shap_insights = analysis_result.get("shap_insights")
                shap_recommendations = analysis_result.get("shap_recommendations", [])
            else:
                # Backward compatibility: old format
                weak_features = analysis_result if isinstance(analysis_result, dict) else {}
                shap_insights = None
                shap_recommendations = []

            # Prioritize based on SHAP importance if available
            prioritized_features = self._prioritize_features(weak_features)

            # Generate improvement suggestions
            improvements = []
            for feature_name, analysis in prioritized_features.items():
                plugin = analysis["plugin"]

                # Get target from generated targets if available
                if feature_name in self.feature_targets:
                    target_info = self.feature_targets[feature_name]
                    analysis['target'] = target_info['target']
                    analysis['shap_importance'] = target_info['shap_importance']
                    analysis['priority_score'] = target_info['normalized_shap']

                improvements.append({
                    "feature": feature_name,
                    "current_value": analysis["current"],
                    "target_value": analysis["target"],
                    "gap": analysis["gap"],
                    "priority": self._calculate_priority(analysis),
                    "shap_importance": analysis.get("shap_importance", 0.0),
                    "plugin": plugin
                })

            # Sort by priority
            improvements.sort(key=lambda x: (
                0 if x["priority"] == "high" else 1 if x["priority"] == "medium" else 2,
                -x.get("shap_importance", 0)
            ))

            return {
                "weak_features_count": len(prioritized_features),
                "improvements": improvements,
                "recommendations": self._generate_recommendations(improvements)
            }

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return {"error": str(e), "improvements": []}

    def _prioritize_features(self, weak_features: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize weak features based on SHAP importance."""
        prioritized = {}

        for feature_name, analysis in weak_features.items():
            # Check if we have SHAP importance for this feature
            if feature_name in self.feature_targets:
                target_info = self.feature_targets[feature_name]
                analysis['shap_importance'] = target_info['shap_importance']
                analysis['normalized_shap'] = target_info['normalized_shap']

            prioritized[feature_name] = analysis

        return prioritized

    def _calculate_priority(self, analysis: Dict[str, Any]) -> str:
        """Calculate priority level based on gap and SHAP importance."""
        gap = analysis.get("gap", 0)
        target = analysis.get("target", 1.0)
        if isinstance(target, dict):
            try:
                target = float(target.get('target', 1.0) or 1.0)
            except Exception:
                target = 1.0
        shap_importance = analysis.get("shap_importance", 0.0)

        # High priority if: large gap (>50% of target) AND high SHAP importance
        if gap > target * 0.5 and shap_importance > 0.05:
            return "high"
        # Medium priority if: moderate gap OR high SHAP importance
        elif gap > target * 0.3 or shap_importance > 0.03:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(self, improvements: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable recommendations."""
        recommendations = []

        high_priority = [i for i in improvements if i["priority"] == "high"]
        if high_priority:
            recommendations.append(f"Focus on {len(high_priority)} high-priority features with strong predictive value.")

        for improvement in improvements[:5]:  # Top 5 recommendations
            feat_name = improvement["feature"].replace("_", " ").title()
            current = improvement["current_value"]
            target = improvement["target_value"]
            if isinstance(target, dict):
                try:
                    target = float(target.get('target', 0) or 0)
                except Exception:
                    target = 0.0

            recommendations.append(
                f"{feat_name}: Increase from {current:.1f} to {target:.1f} "
                f"(gap: {improvement['gap']:.1f})"
            )

        return recommendations

    def _sanitize_weak_features(self, weak_features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Ensure weak feature analyses contain required keys with numeric values."""
        sanitized: Dict[str, Dict[str, Any]] = {}

        for feature_name, analysis in (weak_features or {}).items():
            if not isinstance(analysis, dict):
                logger.warning("Skipping weak feature %s: expected dict, got %s", feature_name, type(analysis).__name__)
                continue

            current = analysis.get("current")
            target = analysis.get("target")
            gap = analysis.get("gap")

            try:
                current_num = float(current)
            except (TypeError, ValueError):
                logger.warning("Skipping weak feature %s: invalid current value %s", feature_name, current)
                continue

            try:
                target_num = float(target)
            except (TypeError, ValueError):
                logger.warning("Skipping weak feature %s: invalid target value %s", feature_name, target)
                continue

            try:
                gap_num = float(gap) if gap is not None else target_num - current_num
            except (TypeError, ValueError):
                gap_num = target_num - current_num

            sanitized_analysis = dict(analysis)
            sanitized_analysis["current"] = current_num
            sanitized_analysis["target"] = target_num
            sanitized_analysis["gap"] = gap_num
            sanitized[feature_name] = sanitized_analysis

        missing_count = len(weak_features or {}) - len(sanitized)
        if missing_count:
            logger.warning("Skipped %d weak feature entries due to missing/invalid keys", missing_count)

        return sanitized

    async def analyze_draft(self, draft_text: str) -> Dict[str, Any]:
        """Compute CatBoost features for draft, identify weak areas."""
        try:
            # Query past CatBoost analyses before running new analysis
            past_analyses = self.memory_manager.retrieve_similar_analyses(draft_text, k=3)

            # Extract features using FeatureExtractor
            features = self.feature_extractor.extract_features(draft_text)

            # Compare to targets using pre-computed lookup tables
            weak_features = {}
            for feature_name, value in features.items():
                # Use pre-computed mapping for faster lookup
                plugin_name = self._feature_to_plugin_map.get(feature_name, feature_name)

                # Get target using optimized lookup
                target = None
                plugin = None

                # Try lookup tables first (fastest)
                if plugin_name in self._feature_target_lookup:
                    target_info = self._feature_target_lookup[plugin_name]
                    target = target_info.get('target') if isinstance(target_info, dict) else target_info
                elif feature_name in self._feature_target_lookup:
                    # Direct feature_name lookup
                    target_info = self._feature_target_lookup[feature_name]
                    target = target_info.get('target') if isinstance(target_info, dict) else target_info
                elif plugin_name in self.plugins:
                    # Fallback to plugin rules (slower)
                    plugin = self.plugins[plugin_name]
                    rule = plugin.rules if hasattr(plugin, 'rules') else {}
                    target = rule.get("successful_case_average", 0)

                # Normalize target to numeric if dict and flag when below threshold
                target_num = None
                if isinstance(target, dict):
                    try:
                        target_num = float(target.get('target', 0) or 0)
                    except Exception:
                        target_num = None
                elif target is not None:
                    try:
                        target_num = float(target)
                    except Exception:
                        target_num = None

                if target_num is not None and value < target_num * 0.8:
                    # Get plugin reference if available
                    if plugin is None and plugin_name in self.plugins:
                        plugin = self.plugins[plugin_name]

                    weak_features[plugin_name] = {
                        "current": value,
                        "target": target_num,
                        "gap": target_num - value,
                        "plugin": plugin,  # May be None for structural features
                        "feature_name": feature_name,
                        "has_plugin": plugin is not None
                    }

            logger.info(f"Identified {len(weak_features)} weak features: {list(weak_features.keys())}")

            # Debug mode: log all feature values, not just weak ones
            if self.debug_mode:
                logger.debug("=" * 60)
                logger.debug("DEBUG: All Feature Values")
                logger.debug("=" * 60)
                for feature_name, value in sorted(features.items()):
                    target = None
                    plugin_name = self._feature_to_plugin_map.get(feature_name, feature_name)
                    if plugin_name in self._feature_target_lookup:
                        target_info = self._feature_target_lookup[plugin_name]
                        target = target_info.get('target') if isinstance(target_info, dict) else target_info

                    status = "WEAK" if plugin_name in weak_features else "OK"
                    target_str = f" (target: {target:.2f})" if target is not None else ""
                    logger.debug(f"  {feature_name}: {value:.3f}{target_str} [{status}]")
                logger.debug("=" * 60)

            # Run CatBoost prediction using CatBoostPredictor
            prediction_result = self.catboost_predictor.predict(features)
            success_prob = prediction_result.get("success_probability")
            prediction = prediction_result.get("prediction")
            confidence = prediction_result.get("confidence")

            # Compute SHAP insights using SHAPAnalyzer
            shap_insights = self.shap_analyzer.compute_insights(features, top_n=10)

            # Run corpus validation to check if draft matches patterns from successful motions
            corpus_validation = None
            try:
                from .corpus_validation_plugin import CorpusValidationPlugin
                from .document_structure import DocumentStructure
                
                # Create document structure from draft text
                doc_structure = DocumentStructure(draft_text)
                
                # Try to get corpus validator from plugins or create new instance
                corpus_validator = None
                for plugin in self.plugins.values():
                    if isinstance(plugin, CorpusValidationPlugin):
                        corpus_validator = plugin
                        break
                
                # If not found, try to create one (requires chroma_store and rules_dir)
                if not corpus_validator and hasattr(self, 'plugins') and self.plugins:
                    # Use first plugin's chroma_store and rules_dir if available
                    first_plugin = list(self.plugins.values())[0]
                    if hasattr(first_plugin, 'chroma_store') and hasattr(first_plugin, 'rules_dir'):
                        try:
                            from semantic_kernel import Kernel
                            # Create minimal kernel for corpus validator
                            kernel = Kernel() if not hasattr(self, 'kernel') else self.kernel
                            # Get database access from first plugin if available
                            # Plugins store db_paths after initialization
                            db_paths = getattr(first_plugin, 'db_paths', None)
                            corpus_validator = CorpusValidationPlugin(
                                kernel=kernel,
                                chroma_store=first_plugin.chroma_store,
                                rules_dir=first_plugin.rules_dir,
                                memory_store=getattr(first_plugin, 'memory_store', None),
                                db_paths=db_paths,
                                enable_langchain=True,  # Enable for corpus search
                                enable_courtlistener=False,  # Not needed for corpus search
                                enable_storm=False  # Not needed for corpus search
                            )
                        except Exception as e:
                            logger.debug(f"Could not create corpus validator: {e}")
                
                # Run corpus validation if available
                if corpus_validator:
                    corpus_result = await corpus_validator.validate_draft_against_corpus(
                        doc_structure,
                        context={"section_validation_results": {}}
                    )
                    if corpus_result:
                        corpus_validation = {
                            "matches_patterns": corpus_result.data.get('matches_patterns', False) if corpus_result.data else False,
                            "overall_similarity": corpus_result.data.get('overall_similarity', 0.0) if corpus_result.data else 0.0,
                            "recommendations": corpus_result.data.get('recommendations', []) if corpus_result.data else [],
                            "corpus_available": True
                        }
            except Exception as e:
                logger.debug(f"Corpus validation failed in RefinementLoop: {e}")
                corpus_validation = {"corpus_available": False, "error": str(e)}

            # Store analysis in memory using MemoryManager
            analysis_summary = {
                "weak_features_count": len(weak_features),
                "weak_features": list(weak_features.keys()),
                "features_analyzed": len(features),
                "success_probability": success_prob,
                "prediction": int(prediction) if prediction is not None else None,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "shap_available": shap_insights is not None and shap_insights.get("shap_available", False)
            }
            self.memory_manager.store_analysis(
                analysis_data=analysis_summary,
                draft_text=draft_text,
                weak_features=weak_features,
                features=features,
                success_prob=success_prob,
                prediction=prediction,
                confidence=confidence,
                shap_insights=shap_insights
            )

            # Return weak_features with SHAP insights and corpus validation
            result = {
                "weak_features": weak_features,
                "features": features,
                "success_probability": success_prob,
                "prediction": int(prediction) if prediction is not None else None,
                "confidence": confidence,
                "shap_insights": shap_insights if shap_insights else None,
                "corpus_validation": corpus_validation
            }
            
            # If draft doesn't match corpus patterns, flag for revision
            if corpus_validation and corpus_validation.get("corpus_available") and not corpus_validation.get("matches_patterns"):
                logger.info(f"⚠️ Draft doesn't match corpus patterns (similarity: {corpus_validation.get('overall_similarity', 0):.2f}). Flagging for revision.")
                # Add corpus recommendations to weak_features if available
                corpus_recs = corpus_validation.get("recommendations", [])
                if corpus_recs:
                    for rec in corpus_recs:
                        section = rec.get('section', 'unknown')
                        # Add as a weak feature indicator
                        weak_features[f"corpus_pattern_{section}"] = {
                            "feature_name": f"corpus_pattern_{section}",
                            "current_value": 0,
                            "target": 1,
                            "gap": 1,
                            "priority": rec.get('priority', 'medium'),
                            "message": rec.get('message', ''),
                            "corpus_recommendation": True,
                            "corpus_examples": rec.get('examples', [])
                        }

            # ✅ Add SHAP-based recommendations to weak_features if available
            if shap_insights and shap_insights.get("shap_available"):
                # Prioritize features that are both weak AND hurting (negative SHAP)
                hurting_features = list(shap_insights.get("top_hurting_features", {}).keys())

                # Add SHAP context to weak_features that are also hurting
                for feat_name in hurting_features:
                    # Try to match hurting feature to weak_features
                    for weak_key, weak_info in weak_features.items():
                        if weak_info.get("feature_name") == feat_name or weak_key == feat_name:
                            weak_info["shap_impact"] = shap_insights["top_hurting_features"].get(feat_name, 0)
                            weak_info["shap_priority"] = "high"  # High priority: weak AND hurting
                            break

                result["shap_recommendations"] = shap_insights.get("recommendations", [])
                result["top_helping_features"] = shap_insights.get("top_helping_features", {})
                result["top_hurting_features"] = shap_insights.get("top_hurting_features", {})

            return result

        except Exception as e:
            logger.error(f"Draft analysis failed: {e}")
            return {}

    def _build_lookup_tables(self) -> None:
        """Pre-compute lookup tables for faster feature target access."""
        try:
            # Build feature_name -> plugin_name mapping
            # Use the mapping logic from _map_feature_to_plugin
            mapping = {
                'mentions_privacy': 'mentions_privacy',
                'mentions_harassment': 'mentions_harassment',
                'mentions_safety': 'mentions_safety',
                'mentions_retaliation': 'mentions_retaliation',
                'citation_count': 'citation_retrieval',
                'privacy_harm_count': 'privacy_harm_count',
                'mentions_public_interest': 'public_interest',
                'mentions_transparency': 'transparency_argument',
                'max_sentence_word_count': 'sentence_structure',
                'median_sentence_word_count': 'sentence_structure',
                'avg_sentence_word_count': 'sentence_structure',
                'sentence_complex_ratio': 'sentence_structure',
                'sentence_long_ratio': 'sentence_structure',
                'paragraph_count': 'paragraph_structure',
                'paragraph_avg_word_count': 'paragraph_structure',
                'paragraph_median_word_count': 'paragraph_structure',
                'paragraph_max_word_count': 'paragraph_structure',
                'paragraph_avg_sentence_count': 'paragraph_structure',
                'paragraph_sentence_std': 'paragraph_structure',
                'enumeration_density': 'enumeration_density',
            }
            self._feature_to_plugin_map = mapping

            # Build feature target lookup (combine feature_targets with plugin rules)
            self._feature_target_lookup = {}
            for plugin_name, target_info in self.feature_targets.items():
                self._feature_target_lookup[plugin_name] = target_info

            # Also add direct feature_name lookups for structural features
            for feature_name in self._feature_to_plugin_map.keys():
                plugin_name = self._feature_to_plugin_map[feature_name]
                if plugin_name in self.feature_targets:
                    self._feature_target_lookup[feature_name] = self.feature_targets[plugin_name]

            # Cache successful case averages
            self._successful_case_averages = self._load_successful_case_averages()

            logger.debug(f"Built lookup tables: {len(self._feature_to_plugin_map)} feature mappings, {len(self._feature_target_lookup)} target lookups")
        except Exception as e:
            logger.warning(f"Failed to build lookup tables: {e}")

    # Note: Document hashing and cache management are now handled by FeatureExtractor
    # These methods have been moved to maintain single responsibility principle

    def _map_feature_to_plugin(self, feature_name: str) -> str:
        """Map CatBoost feature name to SK plugin name.

        Structural features now have dedicated plugins for enforcement.
        """
        mapping = {
            # Legacy plugin mappings
            'mentions_privacy': 'mentions_privacy',
            'mentions_harassment': 'mentions_harassment',
            'mentions_safety': 'mentions_safety',
            'mentions_retaliation': 'mentions_retaliation',
            'citation_count': 'citation_retrieval',
            'privacy_harm_count': 'privacy_harm_count',
            'mentions_public_interest': 'public_interest',
            'mentions_transparency': 'transparency_argument',

            # Structural feature mappings (new atomic plugins)
            'max_sentence_word_count': 'sentence_structure',
            'median_sentence_word_count': 'sentence_structure',
            'avg_sentence_word_count': 'sentence_structure',
            'sentence_complex_ratio': 'sentence_structure',
            'sentence_long_ratio': 'sentence_structure',

            'paragraph_count': 'paragraph_structure',
            'paragraph_avg_word_count': 'paragraph_structure',
            'paragraph_median_word_count': 'paragraph_structure',
            'paragraph_max_word_count': 'paragraph_structure',
            'paragraph_avg_sentence_count': 'paragraph_structure',
            'paragraph_sentence_std': 'paragraph_structure',

            'enumeration_density': 'enumeration_density',
        }
        return mapping.get(feature_name, feature_name)

    def _detect_sections(self, text: str) -> Dict[str, int]:
        """
        Detect which sections are present in the draft and their positions.

        Uses pattern matching with fuzzy matching fallback for robustness.
        Handles variations in section headers (e.g., "Legal Standard", "I. Legal Standard", "LEGAL STANDARD").

        Returns:
            Dictionary mapping section names to their position (0-indexed)
        """
        if not self.outline_manager:
            return {}

        section_positions = {}
        text_lower = text.lower()

        # Section detection patterns (expanded for robustness)
        section_patterns = {
            "introduction": ["introduction", "intro", "preliminary", "preface", "overview"],
            "legal_standard": ["legal standard", "legal framework", "legal test", "standard of review",
                              "applicable law", "legal authority", "standard"],
            "factual_background": ["factual background", "facts", "background", "statement of facts",
                                  "factual", "statement", "recitation"],
            "privacy_harm": ["privacy harm", "privacy", "good cause", "privacy concerns",
                            "privacy interests", "privacy rights", "privacy interests"],
            "danger_safety": ["danger", "safety", "threat", "harm", "risk", "safety concerns",
                             "threats", "risks", "harassment"],
            "public_interest": ["public interest", "public access", "transparency", "public's interest",
                               "public right", "public access right"],
            "balancing_test": ["balancing", "balance", "weigh", "outweigh", "balancing test",
                              "balancing factors", "weighing", "competing interests"],
            "protective_measures": ["protective measures", "protective order", "proposed measures",
                                   "proposed order", "measures", "relief requested"],
            "conclusion": ["conclusion", "wherefore", "respectfully submitted", "requested relief",
                          "prayer", "relief", "requested"]
        }

        # Find section positions with improved detection
        lines = text.split('\n')
        for line_idx, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Skip empty lines
            if not line_lower:
                continue

            # Check if line looks like a section header (more flexible patterns)
            is_header = (
                # Markdown headers
                line_lower.startswith(('#', '##', '###', '####')) or
                # Numbered sections (roman or arabic)
                any(line_lower.startswith(prefix) for prefix in ['i.', 'ii.', 'iii.', 'iv.', 'v.', 'vi.', 'vii.', 'viii.', 'ix.', 'x.',
                                                                  '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.',
                                                                  'a.', 'b.', 'c.', 'd.', 'e.', 'f.', 'g.', 'h.']) or
                # Section keywords
                any(keyword in line_lower for keyword in ['section', 'part', 'subsection']) or
                # ALL CAPS (common in legal documents)
                (len(line_lower) < 100 and line_lower.isupper() and len(line_lower) > 5) or
                # Bold/underlined patterns (common in legal formatting)
                (len(line_lower) < 100 and any(marker in line_lower for marker in ['___', '***', '===']))
            )

            if is_header:
                # Check for section patterns with fuzzy matching
                for section_name, patterns in section_patterns.items():
                    # Exact match
                    if any(pattern in line_lower for pattern in patterns):
                        if section_name not in section_positions:
                            section_positions[section_name] = line_idx
                            logger.debug(f"Detected section '{section_name}' at line {line_idx}: {line[:80]}")
                    # Fuzzy match: check if most words from pattern are present
                    elif len(patterns) > 0:
                        # Check if key words from first pattern are present
                        key_words = patterns[0].split()
                        if len(key_words) >= 2:
                            # If at least 2 key words are present, it's likely this section
                            if sum(1 for word in key_words if word in line_lower) >= 2:
                                if section_name not in section_positions:
                                    section_positions[section_name] = line_idx
                                    logger.debug(f"Detected section '{section_name}' (fuzzy) at line {line_idx}: {line[:80]}")

        return section_positions

    def _organize_plugins_by_section(
        self,
        detected_sections: Dict[str, int],
        weak_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Organize plugins by section based on perfect outline structure.

        Returns:
            Dictionary mapping section names to lists of plugin names to use
        """
        if not self.outline_manager:
            # Fallback: use all plugins
            return {"all": list(self.plugins.keys())}

        organized = {}
        perfect_order = self.outline_manager.get_section_order()

        # Organize by detected sections
        for section_name in perfect_order:
            required, optional = self.outline_manager.get_plugins_for_section(section_name)

            # Prioritize required plugins
            plugins_for_section = required.copy()

            # Add optional plugins if section is present
            if section_name in detected_sections:
                plugins_for_section.extend(optional)

            # Add plugins for weak features in this section
            if weak_features:
                for weak_key, weak_info in weak_features.items():
                    plugin_name = weak_info.get("feature_name", weak_key)
                    # Check if plugin belongs to this section
                    if plugin_name in required or plugin_name in optional:
                        if plugin_name not in plugins_for_section:
                            plugins_for_section.append(plugin_name)

            if plugins_for_section:
                organized[section_name] = plugins_for_section

        return organized

    async def collect_edit_requests(
        self,
        text: str,
        weak_features: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Collect edit requests from all plugins that support the new edit system.

        Args:
            text: Full document text
            weak_features: Optional dict of weak features from CatBoost analysis
            context: Optional context dict (research_results, validation_state, etc.)

        Returns:
            List of EditRequest objects from all plugins
        """
        try:
            # Parse document structure once
            structure = parse_document_structure(text)
            context = context or {}

            all_requests = []

            # ✅ Detect sections in draft and organize plugins by perfect outline
            detected_sections = {}
            plugins_by_section = {}
            if self.outline_manager:
                detected_sections = self._detect_sections(text)
                plugins_by_section = self._organize_plugins_by_section(detected_sections, weak_features)

                # Validate section order
                validation = self.outline_manager.validate_section_order(list(detected_sections.keys()))
                if not validation["valid"]:
                    logger.warning("⚠️ Section order issues detected:")
                    for issue in validation["issues"]:
                        logger.warning(f"   {issue['message']}")
                else:
                    logger.info("✅ Section order validated against perfect outline structure")

                logger.info(f"📋 Organized plugins by {len(plugins_by_section)} sections from perfect outline")

            # Prepare context for plugins
            plugin_context = {
                "weak_features": weak_features or {},
                "research_results": context.get("research_results"),
                "validation_state": context.get("validation_state"),
                "detected_sections": detected_sections,  # ✅ Add section info
                "plugins_by_section": plugins_by_section,  # ✅ Add plugin organization
                "quality_constraints": context.get("quality_constraints"),
                "constraint_validation": context.get("constraint_validation"),
                "constraint_version": context.get("constraint_version"),
                "structured_facts": context.get("structured_facts"),
                "filtered_evidence": context.get("filtered_evidence") or context.get("evidence"),
                "fact_filter_stats": context.get("fact_filter_stats"),
            }

            # ✅ Collect requests organized by section (if outline available)
            if plugins_by_section:
                # Use outline-based organization
                for section_name, plugin_names in plugins_by_section.items():
                    logger.debug(f"Collecting edit requests for section '{section_name}' ({len(plugin_names)} plugins)")
                    for plugin_name in plugin_names:
                        if plugin_name not in self.plugins:
                            continue
                        plugin = self.plugins[plugin_name]
                        try:
                            if hasattr(plugin, 'generate_edit_requests'):
                                # Add section context
                                plugin_context["current_section"] = section_name
                                if self.plugin_calibrator:
                                    plugin_context["section_requirements"] = self.plugin_calibrator.get_enumeration_requirements(section_name)

                                # Try new signature with context (most plugins should support this now)
                                try:
                                    requests = await plugin.generate_edit_requests(text, structure, plugin_context)
                                except TypeError:
                                    # Plugin doesn't accept context parameter, try without
                                    requests = await plugin.generate_edit_requests(text, structure)

                                if requests:
                                    logger.debug(f"Plugin {plugin_name} (section: {section_name}) generated {len(requests)} edit requests")
                                    all_requests.extend(requests)
                        except Exception as e:
                            logger.warning(f"Plugin {plugin_name} (section: {section_name}) failed: {e}")
                            continue
            else:
                # Fallback: collect from all plugins (legacy behavior)
                for plugin_name, plugin in self.plugins.items():
                    try:
                        if hasattr(plugin, 'generate_edit_requests'):
                            # Try new signature with context (most plugins should support this now)
                            try:
                                requests = await plugin.generate_edit_requests(text, structure, plugin_context)
                            except TypeError:
                                # Plugin doesn't accept context parameter, try without
                                requests = await plugin.generate_edit_requests(text, structure)

                            if requests:
                                logger.debug(f"Plugin {plugin_name} generated {len(requests)} edit requests")
                                all_requests.extend(requests)
                    except Exception as e:
                        logger.warning(f"Plugin {plugin_name} failed to generate edit requests: {e}")
                        continue

            logger.info(f"Collected {len(all_requests)} total edit requests from {len(plugins_by_section) if plugins_by_section else len(self.plugins)} plugins")

            # Debug mode: log all edit requests before conflict resolution
            if self.debug_mode:
                logger.debug("=" * 60)
                logger.debug(f"DEBUG: Edit Requests Before Conflict Resolution ({len(all_requests)} requests)")
                logger.debug("=" * 60)
                for i, req in enumerate(all_requests, 1):
                    logger.debug(f"  Request {i}: {req.plugin_name} - {req.edit_type} at para {req.location.paragraph_index}, priority {req.priority:.2f}")
                    logger.debug(f"    Reason: {req.reason[:100] if req.reason else 'N/A'}")
                logger.debug("=" * 60)

            return all_requests

        except Exception as e:
            logger.error(f"Failed to collect edit requests: {e}")
            return []

    def resolve_edit_conflicts(self, requests: List[EditRequest]) -> List[EditRequest]:
        """
        Resolve conflicts between edit requests.

        Strategy:
        1. Sort by priority (high priority first)
        2. Detect overlapping locations
        3. Merge compatible edits (e.g., multiple insertions in same paragraph)
        4. Reject conflicting edits (e.g., replace vs insert at same location)
        5. Adjust locations for previously applied edits

        Args:
            requests: List of edit requests, potentially conflicting

        Returns:
            Resolved list of edit requests to apply
        """
        if not requests:
            return []

        # Sort by priority (descending)
        sorted_requests = sorted(requests, key=lambda r: r.priority, reverse=True)

        resolved = []
        applied_locations = {}  # Track locations that have been edited

        for request in sorted_requests:
            # Create location key for conflict detection
            location_key = self._location_key(request.location)

            # Check if location already has an edit
            if location_key in applied_locations:
                existing = applied_locations[location_key]

                # Check if compatible (e.g., both are insertions at same location)
                if self._are_compatible(existing, request):
                    # Merge compatible edits
                    merged = self._merge_edits(existing, request)
                    if merged:
                        resolved.remove(existing)
                        resolved.append(merged)
                        applied_locations[location_key] = merged
                        logger.debug(f"Merged compatible edits at {location_key}")
                else:
                    # Reject conflicting edit (lower priority)
                    logger.debug(f"Rejected conflicting edit at {location_key} (priority: {request.priority} vs {existing.priority})")
                    continue
            else:
                # No conflict, add request
                resolved.append(request)
                applied_locations[location_key] = request

        logger.info(f"Resolved {len(requests)} edit requests to {len(resolved)} non-conflicting requests")
        return resolved

    def _location_key(self, location: DocumentLocation) -> str:
        """Create a unique key for a location to detect conflicts."""
        parts = []
        if location.paragraph_index is not None:
            parts.append(f"para_{location.paragraph_index}")
        if location.sentence_index is not None:
            parts.append(f"sent_{location.sentence_index}")
        if location.character_offset is not None:
            parts.append(f"char_{location.character_offset}")
        parts.append(location.position_type)
        return "_".join(parts)

    def _are_compatible(self, req1: EditRequest, req2: EditRequest) -> bool:
        """Check if two edit requests are compatible (can be merged)."""
        # Same location required
        if self._location_key(req1.location) != self._location_key(req2.location):
            return False

        # Both must be insertions to be compatible
        if req1.edit_type == "insert" and req2.edit_type == "insert":
            return True

        # Replace and delete are incompatible with each other
        return False

    def _merge_edits(self, req1: EditRequest, req2: EditRequest) -> Optional[EditRequest]:
        """Merge two compatible edit requests."""
        if not self._are_compatible(req1, req2):
            return None

        # Merge content (combine text)
        merged_content = req1.content + " " + req2.content

        # Use higher priority
        merged_priority = max(req1.priority, req2.priority)

        # Merge affected plugins
        merged_affected = list(set(req1.affected_plugins + req2.affected_plugins))

        # Merge metadata
        merged_metadata = {**req1.metadata, **req2.metadata}
        merged_metadata['merged_from'] = [req1.plugin_name, req2.plugin_name]

        # Combine reasons
        merged_reason = f"{req1.reason}; {req2.reason}"

        # Create merged request (use higher priority plugin name or combine)
        merged_plugin_name = req1.plugin_name if req1.priority >= req2.priority else req2.plugin_name

        return EditRequest(
            plugin_name=f"{merged_plugin_name}_merged",
            location=req1.location,
            edit_type=req1.edit_type,
            content=merged_content,
            priority=merged_priority,
            affected_plugins=merged_affected,
            metadata=merged_metadata,
            reason=merged_reason
        )

    async def apply_edits(self, text: str, requests: List[EditRequest]) -> EditResult:
        """
        Apply edit requests to document.

        Args:
            text: Original document text
            requests: List of resolved edit requests

        Returns:
            EditResult with applied edits and metrics impact
        """
        if not requests:
            return EditResult(
                success=True,
                original_text=text,
                modified_text=text,
                applied_edits=[],
                failed_edits=[],
                metrics_impact={}
            )

        try:
            # Parse initial structure
            structure = parse_document_structure(text)
            current_text = text
            applied = []
            failed = []

            # Apply edits in order (already sorted by priority)
            for request in requests:
                try:
                    modified_text, updated_structure = apply_single_edit(current_text, structure, request)
                    current_text = modified_text
                    structure = updated_structure
                    applied.append(request)
                    logger.debug(f"Applied edit from {request.plugin_name} at {self._location_key(request.location)}")
                except Exception as e:
                    logger.error(f"Failed to apply edit from {request.plugin_name}: {e}")
                    failed.append(request)

            # Calculate metrics impact
            metrics_impact = self._calculate_metrics_impact(text, current_text)

            return EditResult(
                success=len(failed) == 0,
                original_text=text,
                modified_text=current_text,
                applied_edits=applied,
                failed_edits=failed,
                metrics_impact=metrics_impact
            )

        except Exception as e:
            logger.error(f"Failed to apply edits: {e}")
            return EditResult(
                success=False,
                original_text=text,
                modified_text=text,
                applied_edits=[],
                failed_edits=requests,
                metrics_impact={}
            )

    def _calculate_metrics_impact(self, original: str, modified: str) -> Dict[str, Any]:
        """Calculate impact of edits on document metrics."""
        orig_structure = parse_document_structure(original)
        mod_structure = parse_document_structure(modified)

        # Count words (simple approximation)
        orig_words = len(original.split())
        mod_words = len(modified.split())

        return {
            "word_count_change": mod_words - orig_words,
            "paragraph_count_change": len(mod_structure.paragraphs) - len(orig_structure.paragraphs),
            "sentence_count_change": sum(len(p.sentences) for p in mod_structure.paragraphs) - sum(len(p.sentences) for p in orig_structure.paragraphs),
            "character_count_change": len(modified) - len(original),
            "original_length": len(original),
            "modified_length": len(modified)
        }

    def calculate_affected_plugins(self, edit: EditRequest, all_plugins: Dict[str, BaseFeaturePlugin]) -> List[str]:
        """
        Calculate which plugins are affected by an edit.

        Args:
            edit: Edit request
            all_plugins: Dictionary of all available plugins

        Returns:
            List of plugin names that need re-validation
        """
        # Start with explicitly listed affected plugins
        affected = set(edit.affected_plugins)

        # Add plugins that are affected by any text change
        always_affected = ["word_count", "character_count", "sentence_count"]
        for plugin_name in always_affected:
            if plugin_name in all_plugins:
                affected.add(plugin_name)

        # Add structural plugins if paragraph-level edit
        if edit.location.paragraph_index is not None:
            paragraph_affected = ["paragraph_structure", "per_paragraph", "paragraph_monitor"]
            for plugin_name in paragraph_affected:
                if plugin_name in all_plugins:
                    affected.add(plugin_name)

        # Add sentence plugins if sentence-level edit
        if edit.location.sentence_index is not None:
            sentence_affected = ["sentence_structure", "sentence_length"]
            for plugin_name in sentence_affected:
                if plugin_name in all_plugins:
                    affected.add(plugin_name)

        return list(affected)

    async def _store_edit_results_in_memory(
        self,
        original_text: str,
        edit_result: EditResult,
        weak_features: Dict[str, Any]
    ) -> None:
        """
        Store edit results in EpisodicMemoryBank for learning.

        Args:
            original_text: Original draft text before edits
            edit_result: Result of applying edits
            weak_features: Weak features that triggered the edits
        """
        try:
            if not self.memory_manager.memory_store:
                logger.debug("No memory store available, skipping edit result storage")
                return

            # Prepare memory entry
            edit_summary = {
                "applied_edits_count": len(edit_result.applied_edits),
                "failed_edits_count": len(edit_result.failed_edits),
                "success": edit_result.success,
                "metrics_impact": edit_result.metrics_impact,
                "weak_features_addressed": list(weak_features.keys()) if weak_features else [],
                "edit_plugins": list(set(e.plugin_name for e in edit_result.applied_edits)),
                "timestamp": datetime.now().isoformat()
            }

            # Store as execution memory
            memory_context = {
                "agent_type": "RefinementLoop",
                "operation": "edit_coordination",
                "edit_summary": edit_summary,
                "applied_edits": [
                    {
                        "plugin": e.plugin_name,
                        "edit_type": e.edit_type,
                        "location": {
                            "paragraph": e.location.paragraph_index,
                            "sentence": e.location.sentence_index,
                        },
                        "reason": e.reason
                    }
                    for e in edit_result.applied_edits[:10]  # Limit to first 10
                ]
            }

            # Create memory entry
            query_text = f"Applied {len(edit_result.applied_edits)} edits to strengthen weak features: {', '.join(edit_summary['weak_features_addressed'][:5])}"

            # Use memory_manager's underlying store
            try:
                from EpisodicMemoryBank import EpisodicMemoryEntry
                memory_entry = EpisodicMemoryEntry(
                    agent_type="RefinementLoop",
                    memory_id=f"edit_{datetime.now().isoformat()}",
                    summary=query_text,
                    context=memory_context,
                    source="refinement_loop",
                    timestamp=datetime.now(),
                    memory_type="execution"
                )
                self.memory_manager.memory_store.add(memory_entry)
            except ImportError:
                # Fallback to store_memory if available
                if hasattr(self.memory_manager.memory_store, 'store_memory'):
                    self.memory_manager.memory_store.store_memory(
                        agent_type="RefinementLoop",
                        query=query_text,
                        result=json.dumps(edit_summary, indent=2),
                        context=memory_context,
                        memory_type="execution"
                    )

            logger.debug(f"Stored edit results in memory: {len(edit_result.applied_edits)} edits applied")

        except Exception as e:
            logger.warning(f"Failed to store edit results in memory: {e}")

    async def revalidate_affected_plugins(
        self,
        text: str,
        affected_plugin_names: List[str]
    ) -> Dict[str, FunctionResult]:
        """
        Re-validate plugins that were affected by edits.

        Args:
            text: Modified document text
            affected_plugin_names: List of plugin names to re-validate

        Returns:
            Dictionary mapping plugin names to their validation results
        """
        results = {}

        for plugin_name in affected_plugin_names:
            if plugin_name in self.plugins:
                plugin = self.plugins[plugin_name]
                try:
                    # Use validate method if available
                    if hasattr(plugin, 'validate'):
                        result = await plugin.validate(text)
                    elif hasattr(plugin, 'validate_draft'):
                        result = await plugin.validate_draft(text)
                    else:
                        logger.warning(f"Plugin {plugin_name} has no validate method")
                        continue

                    results[plugin_name] = result
                    logger.debug(f"Re-validated {plugin_name}: success={result.success}")
                except Exception as e:
                    logger.error(f"Failed to re-validate {plugin_name}: {e}")
                    results[plugin_name] = FunctionResult(
                        success=False,
                        value=None,
                        error=str(e)
                    )

        return results

    async def strengthen_draft(
        self,
        draft_text: str,
        weak_features: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Invoke plugins to strengthen weak features.

        Now uses location-specific edit requests when available, with fallback to legacy system.

        Args:
            draft_text: Current draft text
            weak_features: Dict of weak features from CatBoost analysis
            context: Optional context (research_results, validation_state, etc.)
        """
        try:
            # Try new edit request system first
            edit_requests = await self.collect_edit_requests(draft_text, weak_features, context)

            if edit_requests:
                logger.info(f"Using new edit coordination system: {len(edit_requests)} edit requests collected")

                # Resolve conflicts
                resolved_requests = self.resolve_edit_conflicts(edit_requests)

                # Apply edits
                edit_result = await self.apply_edits(draft_text, resolved_requests)

                if edit_result.success and edit_result.applied_edits:
                    # Collect all affected plugins
                    all_affected = set()
                    for edit in edit_result.applied_edits:
                        affected = self.calculate_affected_plugins(edit, self.plugins)
                        all_affected.update(affected)

                    # Re-validate affected plugins
                    if all_affected:
                        revalidation_results = await self.revalidate_affected_plugins(
                            edit_result.modified_text,
                            list(all_affected)
                        )
                        logger.info(f"Re-validated {len(revalidation_results)} affected plugins after edits")

                    # Store edit results in memory for learning
                    await self._store_edit_results_in_memory(
                        draft_text,
                        edit_result,
                        weak_features
                    )

                    logger.info(f"Applied {len(edit_result.applied_edits)} edits successfully")
                    return edit_result.modified_text
                else:
                    logger.warning(f"Edit application had issues: {len(edit_result.failed_edits)} failed edits")
                    # Fall through to legacy system if edits failed

            # Fallback to legacy improvement system
            logger.info("Using legacy improvement system (no edit requests generated)")
            sanitized_features = self._sanitize_weak_features(weak_features or {})
            improvements = []
            structural_features_reported = []

            for feature_name, analysis in sanitized_features.items():
                plugin = analysis.get("plugin")
                has_plugin = analysis.get("has_plugin", bool(plugin))
                current_value = analysis.get("current", 0.0)
                target_value = analysis.get("target", current_value)
                gap_value = analysis.get("gap", target_value - current_value)

                logger.info(
                    "Strengthening %s (current: %s, target: %s)",
                    feature_name,
                    current_value,
                    target_value,
                )

                # Handle features with plugins (including structural plugins)
                if has_plugin and plugin:
                    try:
                        # Structural plugins have different optimization methods
                        if feature_name in ['sentence_structure', 'paragraph_structure', 'enumeration_density']:
                            # Use specialized optimization methods for structural plugins
                            if feature_name == 'sentence_structure':
                                result = await plugin.optimize_sentences(
                                    draft_text, {"max_sentence_word_count": target_value}
                                )
                            elif feature_name == 'paragraph_structure':
                                result = await plugin.optimize_paragraphs(
                                    draft_text, {"paragraph_avg_word_count": target_value}
                                )
                            elif feature_name == 'enumeration_density':
                                result = await plugin.generate_enumeration_suggestions(draft_text)
                            else:
                                result = None

                            if result and result.success:
                                opt_data = result.value
                                guidance = opt_data.get("optimization_guidance", []) or opt_data.get("suggestions", [])
                                recommendations = opt_data.get("specific_recommendations", []) or opt_data.get("formatting_suggestions", [])

                                improvements.append({
                                    "feature": feature_name,
                                    "argument": "\n".join(guidance + recommendations),
                                    "gap": gap_value,
                                    "priority": "high" if gap_value > target_value * 0.5 else "medium",
                                    "type": "structural"
                                })
                            else:
                                # Fallback to analysis method
                                if feature_name == 'sentence_structure':
                                    result = await plugin.analyze_sentence_strength(draft_text)
                                elif feature_name == 'paragraph_structure':
                                    result = await plugin.analyze_paragraph_strength(draft_text)
                                elif feature_name == 'enumeration_density':
                                    result = await plugin.analyze_enumeration_strength(draft_text)
                                else:
                                    result = None

                                if result and result.success:
                                    recs = result.value.get("recommendations", [])
                                    improvements.append({
                                        "feature": feature_name,
                                        "argument": "\n".join(recs),
                                        "gap": analysis["gap"],
                                        "priority": "high" if analysis["gap"] > analysis["target"] * 0.5 else "medium",
                                        "type": "structural"
                                    })
                        elif feature_name == 'required_case_citations':
                            # Required case citation plugin - enforce citations
                            result = await plugin.enforce_required_citations(draft_text)
                            if result.success and result.value:
                                missing = result.value.get('missing_cases', [])
                                if missing:
                                    # Get suggestions for missing citations
                                    suggestions_result = await plugin.get_citation_suggestions(draft_text)
                                    suggestions = suggestions_result.value.get('suggestions', []) if suggestions_result.success else []

                                    argument_parts = []
                                    argument_parts.append(f"REQUIRED: Add citations to {len(missing)} required cases:")
                                    for sugg in suggestions[:5]:  # Top 5 suggestions
                                        if isinstance(sugg, dict):
                                            argument_parts.append(f"  • {sugg.get('suggestion', '')}")
                                        else:
                                            argument_parts.append(f"  • {str(sugg)}")

                                    improvements.append({
                                        "feature": feature_name,
                                        "argument": "\n".join(argument_parts),
                                        "gap": len(missing),
                                        "priority": "high",
                                        "type": "citation_enforcement",
                                        "missing_count": len(missing),
                                        "compliance_score": result.value.get('overall_compliance_score', 0)
                                    })
                        else:
                            # Legacy plugins - use standard pattern
                            results = await plugin.query_chroma(draft_text[:500])
                            patterns = await plugin.extract_patterns(results)
                            argument = await plugin.generate_argument(patterns, draft_text)

                            improvements.append({
                                "feature": feature_name,
                                "argument": argument,
                                "gap": gap_value,
                                "priority": "high" if gap_value > target_value * 0.5 else "medium",
                                "type": "content"
                            })

                    except Exception as e:
                        logger.error(f"Plugin {feature_name} failed: {e}")
                        # Fallback to recommendation
                        structural_features_reported.append({
                            "feature": feature_name,
                            "current": current_value,
                            "target": target_value,
                            "gap": gap_value,
                            "recommendation": self._generate_structural_recommendation(feature_name, analysis)
                        })
                        continue
                else:
                    # No plugin - generate recommendation
                    structural_features_reported.append({
                        "feature": feature_name,
                        "current": current_value,
                        "target": target_value,
                        "gap": gap_value,
                        "recommendation": self._generate_structural_recommendation(feature_name, analysis)
                    })

            # Integrate improvements into draft (now async)
            improved_draft = await self._merge_improvements(draft_text, improvements, structural_features_reported)
            structural_count = sum(1 for i in improvements if i.get('type') == 'structural')
            logger.info(f"Generated {len(improvements)} improvements ({structural_count} structural), "
                       f"{len(structural_features_reported)} features reported without plugins")

            # Store legacy improvements in memory so plugins can reuse insights
            if improvements or structural_features_reported:
                applied_edits: List[EditRequest] = []
                failed_edits: List[EditRequest] = []

                def _legacy_location() -> DocumentLocation:
                    # Legacy system lacks precise pointers; use best-effort placeholder
                    return DocumentLocation(paragraph_index=0, sentence_index=0, position_type="replace")

                priority_lookup = {"high": 85, "medium": 60, "low": 40}

                for imp in improvements:
                    try:
                        edit_req = EditRequest(
                            plugin_name=imp.get("feature", "legacy_improvement"),
                            location=_legacy_location(),
                            edit_type="replace",
                            content=imp.get("argument", ""),
                            priority=priority_lookup.get(imp.get("priority"), 50),
                            affected_plugins=[imp.get("feature")] if imp.get("feature") else [],
                            metadata={
                                "legacy_mode": True,
                                "improvement_type": imp.get("type"),
                                "gap": imp.get("gap"),
                            },
                            reason=f"Legacy improvement for {imp.get('feature', 'unknown')} (gap={imp.get('gap', 0):.2f})"
                        )
                        applied_edits.append(edit_req)
                    except Exception as exc:
                        logger.debug(f"Could not record legacy improvement for {imp.get('feature')}: {exc}")

                for report in structural_features_reported:
                    try:
                        failed_request = EditRequest(
                            plugin_name=report.get("feature", "legacy_recommendation"),
                            location=_legacy_location(),
                            edit_type="insert",
                            content=report.get("recommendation", ""),
                            priority=45,
                            affected_plugins=[report.get("feature")] if report.get("feature") else [],
                            metadata={
                                "legacy_mode": True,
                                "structural_report": True,
                                "gap": report.get("gap"),
                            },
                            reason=f"No plugin available for {report.get('feature', 'unknown')}"
                        )
                        failed_edits.append(failed_request)
                    except Exception as exc:
                        logger.debug(f"Could not record structural recommendation for {report.get('feature')}: {exc}")

                metrics_impact = self._calculate_metrics_impact(draft_text, improved_draft) if draft_text != improved_draft else {}
                edit_result = EditResult(
                    success=bool(applied_edits),
                    original_text=draft_text,
                    modified_text=improved_draft,
                    applied_edits=applied_edits,
                    failed_edits=failed_edits,
                    metrics_impact=metrics_impact
                )

                await self._store_edit_results_in_memory(
                    draft_text,
                    edit_result,
                    weak_features
                )
                logger.debug(f"Stored {len(applied_edits)} legacy improvements (and {len(failed_edits)} recommendations) in memory")

            return improved_draft

        except Exception as e:
            logger.error(f"Draft strengthening failed: {e}")
            return draft_text

    def _generate_structural_recommendation(self, feature_name: str, analysis: Dict[str, Any]) -> str:
        """Generate human-readable recommendation for structural features."""
        current = analysis["current"]
        target = analysis["target"]

        if 'sentence_word_count' in feature_name:
            if 'max' in feature_name:
                return f"Reduce maximum sentence length from {current:.1f} to {target:.1f} words by splitting long sentences."
            elif 'median' in feature_name or 'avg' in feature_name:
                return f"Aim for {target:.1f} words per sentence (current: {current:.1f}). Keep sentences concise and clear."

        elif 'paragraph' in feature_name:
            if 'count' in feature_name:
                return f"Adjust paragraph count to around {target:.0f} paragraphs (current: {current:.0f})."
            elif 'word_count' in feature_name:
                return f"Aim for {target:.0f} words per paragraph (current: {current:.0f}). Ensure paragraphs are well-structured."

        elif 'enumeration' in feature_name:
            return f"Increase enumeration density to {target:.2%} (current: {current:.2%}) by using more structured lists and numbered items."

        elif 'intel' in feature_name.lower():
            return f"Increase Intel factor focus (target: {target:.2f}, current: {current:.2f}). Emphasize relevant Intel test factors."

        else:
            return f"Improve {feature_name} from {current:.2f} to {target:.2f} to match successful case patterns."

    async def _merge_improvements(self, original_draft: str, improvements: List[Dict], structural_features: List[Dict] = None) -> str:
        """Merge plugin improvements into the original draft using LLM integration."""
        try:
            structural_features = structural_features or []

            # If no improvements, return original
            if not improvements and not structural_features:
                return original_draft

            # Sort improvements by priority
            improvements.sort(key=lambda x: 0 if x.get("priority") == "high" else 1)

            # Build improvement instructions for LLM
            improvement_instructions = "Based on CatBoost analysis of successful cases, integrate the following improvements into the draft:\n\n"

            # Add plugin-based improvements
            for i, improvement in enumerate(improvements, 1):
                feature_name = improvement['feature'].replace('_', ' ').title()
                improvement_instructions += f"{i}. **{feature_name} Enhancement**: {improvement['argument']}\n\n"

            # Add structural feature recommendations
            if structural_features:
                improvement_instructions += "\n### Structural Feature Recommendations\n\n"
                for feat in structural_features:
                    improvement_instructions += f"- **{feat['feature'].replace('_', ' ').title()}**: {feat['recommendation']}\n"
                improvement_instructions += "\n"

            # Use LLM to integrate improvements into the draft
            # Get kernel from plugins (they all have access to it)
            kernel = None
            if self.plugins:
                for plugin in self.plugins.values():
                    if hasattr(plugin, 'kernel') and plugin.kernel:
                        kernel = plugin.kernel
                        break
            
            if kernel:
                try:
                    # Create a prompt function to integrate improvements
                    integration_prompt = f"""You are a legal document editor. Integrate the following improvements into the provided draft motion.

IMPROVEMENTS TO INTEGRATE:
{improvement_instructions}

ORIGINAL DRAFT:
{original_draft}

INSTRUCTIONS:
- Integrate the improvements naturally into the existing text
- Do NOT append a new section - modify the existing content
- Maintain the document's structure and flow
- Ensure improvements are contextually appropriate
- Keep all existing content unless explicitly told to replace it
- Return the complete, improved draft

IMPROVED DRAFT:"""

                    # Try to get chat service from kernel
                    from semantic_kernel.connectors.ai import ChatCompletionClientBase
                    chat_service = None
                    
                    # Try different methods to get chat service
                    try:
                        if hasattr(kernel, 'get_service'):
                            try:
                                chat_service = kernel.get_service(type_id="chat_completion")
                            except (AttributeError, KeyError, ValueError, TypeError):
                                pass
                        
                        if not chat_service:
                            services = getattr(kernel, 'services', {})
                            if services:
                                if isinstance(services, dict):
                                    chat_service = list(services.values())[0]
                                elif hasattr(services, '__iter__'):
                                    chat_service = next(iter(services)) if services else None
                    except Exception:
                        pass
                    
                    if chat_service and isinstance(chat_service, ChatCompletionClientBase):
                        # Use chat service to integrate improvements
                        # Use helper function for version compatibility
                        try:
                            from writer_agents.code.sk_compat import get_chat_components
                        except ImportError:
                            try:
                                from sk_compat import get_chat_components
                            except ImportError:
                                from semantic_kernel.contents import ChatHistory
                                get_chat_components = None
                        
                        if get_chat_components:
                            ChatHistory, _, _ = get_chat_components()
                        else:
                            from semantic_kernel.contents import ChatHistory
                        
                        chat_history = ChatHistory()
                        chat_history.add_system_message("You are a legal document editor that integrates improvements into existing drafts.")
                        chat_history.add_user_message(integration_prompt)
                        
                        # Get execution settings if available
                        settings = None
                        if hasattr(kernel, 'get_prompt_execution_settings_from_service_id'):
                            try:
                                settings = kernel.get_prompt_execution_settings_from_service_id(
                                    service_id=chat_service.service_id
                                )
                            except Exception:
                                pass
                        
                        response = await chat_service.get_chat_message_contents(
                            chat_history=chat_history,
                            settings=settings
                        )
                        
                        # Extract content from response
                        integrated_draft = None
                        if isinstance(response, list) and len(response) > 0:
                            message = response[0]
                            if hasattr(message, 'content'):
                                integrated_draft = str(message.content).strip()
                            elif hasattr(message, 'text'):
                                integrated_draft = str(message.text).strip()
                            else:
                                integrated_draft = str(message).strip()
                        elif hasattr(response, 'content'):
                            integrated_draft = str(response.content).strip()
                        elif hasattr(response, 'value'):
                            integrated_draft = str(response.value).strip()
                        else:
                            integrated_draft = str(response).strip()
                        
                        if integrated_draft and len(integrated_draft) > len(original_draft) * 0.5:
                            logger.info(f"Successfully integrated {len(improvements)} improvements using LLM")
                            return integrated_draft

                except Exception as e:
                    logger.warning(f"LLM integration failed, falling back to append method: {e}")

            # Fallback: append improvements if LLM integration fails
            logger.info("Using fallback: appending improvements as suggestions")
            improvement_section = "\n\n## AI-Generated Improvements\n\n"
            improvement_section += "Based on analysis of successful cases, the following improvements are suggested:\n\n"
            for i, improvement in enumerate(improvements, 1):
                improvement_section += f"### {i}. {improvement['feature'].replace('_', ' ').title()} Enhancement\n\n"
                improvement_section += f"{improvement['argument']}\n\n"
            if structural_features:
                improvement_section += "\n### Structural Feature Recommendations\n\n"
                for feat in structural_features:
                    improvement_section += f"- **{feat['feature'].replace('_', ' ').title()}**: {feat['recommendation']}\n"
                improvement_section += "\n"
            return original_draft + improvement_section

        except Exception as e:
            logger.error(f"Improvement merging failed: {e}")
            return original_draft

    async def validate_with_catboost(self, improved_draft: str) -> Dict[str, Any]:
        """Score improved draft with CatBoost, validate improvements."""
        try:
            if not self.catboost_model:
                logger.warning("No CatBoost model available for validation")
                return {"prediction": None, "improved": False, "features": {}}

            # Import compute_draft_features
            import sys
            code_dir = Path(__file__).parents[3]
            if str(code_dir) not in sys.path:
                sys.path.append(str(code_dir))
            # Legacy fallback: include analysis/ if present
            analysis_dir = code_dir / "analysis"
            if analysis_dir.exists() and str(analysis_dir) not in sys.path:
                sys.path.append(str(analysis_dir))

            from analyze_ma_motion_doc import compute_draft_features
            features = compute_draft_features(improved_draft)

            # Score with CatBoost
            import pandas as pd
            import numpy as np

            # Convert features to DataFrame format expected by model
            feature_df = pd.DataFrame([features])

            # Ensure all required columns exist
            if hasattr(self.catboost_model, 'feature_names_'):
                for col in self.catboost_model.feature_names_:
                    if col not in feature_df.columns:
                        feature_df[col] = 0
                feature_df = feature_df[self.catboost_model.feature_names_]

            # Get prediction probabilities
            proba = self.catboost_model.predict_proba(feature_df)[0]
            prediction = self.catboost_model.predict(feature_df)[0]
            confidence = float(np.max(proba))

            # Check if improved
            improved = False
            improvement_percent = 0.0
            if self.baseline_score is not None:
                improved = confidence > self.baseline_score
                improvement_percent = ((confidence - self.baseline_score) / self.baseline_score) * 100

            # Log feedback for rule adjustment
            await self._log_validation_feedback(features, confidence, improved)

            return {
                "prediction": prediction,
                "confidence": confidence,
                "improved": improved,
                "improvement_percent": improvement_percent,
                "features": features,
                "probability_distribution": proba.tolist(),
                "baseline_score": self.baseline_score
            }

        except Exception as e:
            logger.error(f"CatBoost validation failed: {e}")
            return {"prediction": None, "improved": False, "features": {}, "error": str(e)}

    async def _log_validation_feedback(self, features: Dict, confidence: float, improved: bool) -> None:
        """Log validation feedback for rule adjustment."""
        try:
            feedback_log = {
                "timestamp": datetime.now().isoformat(),
                "features": features,
                "confidence": confidence,
                "improved": improved,
                "baseline_score": self.baseline_score
            }

            # Save to feedback log file
            feedback_file = Path(__file__).parents[2] / "ml_audit" / "validation_feedback.jsonl"
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback_log) + '\n')

            logger.info(f"Validation feedback logged: improved={improved}, confidence={confidence:.3f}")

        except Exception as e:
            logger.error(f"Failed to log validation feedback: {e}")

    async def run_feedback_loop(
        self,
        draft_text: str,
        max_iterations: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run complete feedback loop: analyze -> strengthen -> validate -> adjust.

        Args:
            draft_text: Initial draft text
            max_iterations: Maximum number of iterations
            context: Optional context dict (research_results, validation_state, etc.)

        Returns:
            Dict with iteration results, final draft, and improvement metrics
        """
        try:
            logger.info(f"Starting feedback loop with max {max_iterations} iterations")

            current_draft = draft_text
            iteration_results = []
            baseline_length = len(draft_text)
            previous_confidence = None
            previous_improvement = None
            consecutive_low_improvements = 0
            stop_reason = None

            # Set baseline score from initial analysis
            if self.catboost_model:
                initial_validation = await self.validate_with_catboost(draft_text)
                self.baseline_score = initial_validation.get("confidence", 0.0)
                previous_confidence = self.baseline_score
                logger.info(f"Baseline confidence: {self.baseline_score:.3f}")

            for iteration in range(max_iterations):
                logger.info(f"Feedback loop iteration {iteration + 1}/{max_iterations}")

                # Analyze current draft
                analysis_result = await self.analyze_draft(current_draft)
                
                # Extract weak_features from analysis result (analyze_draft returns dict with 'weak_features' key)
                if isinstance(analysis_result, dict) and "weak_features" in analysis_result:
                    weak_features = analysis_result["weak_features"]
                else:
                    # Backward compatibility: old format
                    weak_features = analysis_result if isinstance(analysis_result, dict) else {}

                # Stop if no weak features found
                if not weak_features:
                    stop_reason = "No weak features found"
                    logger.info(f"Stopping feedback loop: {stop_reason}")
                    break

                # Strengthen draft with context
                improved_draft = await self.strengthen_draft(current_draft, weak_features, context=context)

                # Validate with CatBoost
                validation = await self.validate_with_catboost(improved_draft)
                confidence = validation.get("confidence", 0)
                improvement_percent = validation.get("improvement_percent", 0)

                # Record iteration results
                iteration_result = {
                    "iteration": iteration + 1,
                    "weak_features": weak_features,
                    "improved": validation.get("improved", False),
                    "confidence": confidence,
                    "improvement_percent": improvement_percent,
                    "draft_length": len(improved_draft),
                    "draft_length_change": len(improved_draft) - baseline_length
                }

                iteration_results.append(iteration_result)

                # Check stopping criteria
                # 1. High confidence threshold (> 0.85)
                if confidence > 0.85:
                    stop_reason = f"High confidence achieved: {confidence:.3f}"
                    logger.info(stop_reason)
                    break

                # 2. Draft length bloat (> 50% increase)
                length_increase_pct = ((len(improved_draft) - baseline_length) / baseline_length) * 100 if baseline_length > 0 else 0
                if length_increase_pct > 50:
                    stop_reason = f"Draft length increased by {length_increase_pct:.1f}% (bloat detected)"
                    logger.warning(stop_reason)
                    break

                # 3. Diminishing returns (< 1% improvement for 2 consecutive iterations)
                if previous_improvement is not None:
                    if improvement_percent < 1.0:
                        consecutive_low_improvements += 1
                        if consecutive_low_improvements >= 2:
                            stop_reason = f"Diminishing returns: {consecutive_low_improvements} consecutive iterations with < 1% improvement"
                            logger.info(stop_reason)
                            break
                    else:
                        consecutive_low_improvements = 0

                # 4. Sufficient improvement (> 10%)
                if improvement_percent > 10:
                    stop_reason = f"Sufficient improvement achieved: {improvement_percent:.1f}%"
                    logger.info(stop_reason)
                    break

                # Update for next iteration
                current_draft = improved_draft
                previous_confidence = confidence
                previous_improvement = improvement_percent

                logger.info(f"Iteration {iteration + 1} complete: confidence={confidence:.3f}, improvement={improvement_percent:.1f}%")

                # Debug mode: save intermediate drafts
                if self.debug_mode and self._debug_output_dir:
                    draft_file = self._debug_output_dir / f"iteration_{iteration + 1}_draft.txt"
                    with open(draft_file, 'w', encoding='utf-8') as f:
                        f.write(improved_draft)
                    logger.debug(f"Saved intermediate draft to {draft_file}")

            # Compile final results
            final_confidence = iteration_results[-1]["confidence"] if iteration_results else previous_confidence
            final_improvement = iteration_results[-1]["improvement_percent"] if iteration_results else 0

            final_results = {
                "iterations_completed": len(iteration_results),
                "final_draft": current_draft,
                "iteration_results": iteration_results,
                "total_improvement": final_improvement,
                "final_confidence": final_confidence,
                "baseline_confidence": self.baseline_score,
                "stop_reason": stop_reason if stop_reason else f"Reached max iterations ({max_iterations})",
                "success": iteration_results[-1]["improved"] if iteration_results else False
            }

            logger.info(f"Feedback loop completed: {final_results['iterations_completed']} iterations, {final_results['total_improvement']:.1f}% improvement")

            return final_results

        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "iterations_completed": 0
            }

    async def run_full_analysis(self, draft_text: str) -> Dict[str, Any]:
        """Run complete analysis: analyze -> strengthen -> validate."""
        try:
            logger.info("Starting full draft analysis")

            # Step 1: Analyze draft for weak features
            analysis_result = await self.analyze_draft(draft_text)
            # Handle both old format (dict of weak_features) and new format (dict with 'weak_features' key)
            if isinstance(analysis_result, dict) and "weak_features" in analysis_result:
                weak_features = analysis_result["weak_features"]
                shap_insights = analysis_result.get("shap_insights")
                shap_recommendations = analysis_result.get("shap_recommendations", [])
            else:
                # Backward compatibility: old format
                weak_features = analysis_result if isinstance(analysis_result, dict) else {}
                shap_insights = None
                shap_recommendations = []

            # Step 2: Strengthen draft using plugins
            improved_draft = await self.strengthen_draft(draft_text, weak_features)

            # Step 3: Validate improvements with CatBoost
            validation_result = await self.validate_with_catboost(improved_draft)

            # Compile results
            results = {
                "original_draft_length": len(draft_text),
                "improved_draft_length": len(improved_draft),
                "weak_features_identified": len(weak_features),
                "weak_features": weak_features,
                "improved_draft": improved_draft,
                "validation_result": validation_result,
                "success": validation_result.get("improved", False)
            }

            logger.info(f"Full analysis complete. Success: {results['success']}")
            return results

        except Exception as e:
            logger.error(f"Full analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "original_draft": draft_text
            }

    def set_baseline_score(self, score: float) -> None:
        """Set baseline score for comparison."""
        self.baseline_score = score
        logger.info(f"Baseline score set to {score}")

    async def get_plugin_recommendations(self, draft_text: str) -> Dict[str, Any]:
        """Get specific recommendations from each plugin."""
        try:
            recommendations = {}

            for feature_name, plugin in self.plugins.items():
                try:
                    # Get plugin-specific analysis
                    if hasattr(plugin, 'analyze_privacy_strength'):
                        result = await plugin.analyze_privacy_strength(draft_text)
                    elif hasattr(plugin, 'enforce_case_citation'):
                        # Individual case enforcement plugin
                        result = await plugin.enforce_case_citation(draft_text)
                        if result.success and result.value:
                            case_name = result.value.get('case_name', 'unknown')
                            if 'individual_case_citations' not in analysis:
                                analysis['individual_case_citations'] = {}
                            analysis['individual_case_citations'][case_name] = result.value
                    elif hasattr(plugin, 'enforce_required_citations'):
                        # Required case citation enforcement
                        result = await plugin.enforce_required_citations(draft_text)
                        if result.success and result.value:
                            analysis['required_case_citations'] = result.value
                    elif hasattr(plugin, 'analyze_citation_strength'):
                        result = await plugin.analyze_citation_strength(draft_text)
                    elif hasattr(plugin, 'analyze_harassment_risk'):
                        result = await plugin.analyze_harassment_risk(draft_text)
                    elif hasattr(plugin, 'analyze_safety_concerns'):
                        result = await plugin.analyze_safety_concerns(draft_text)
                    elif hasattr(plugin, 'analyze_retaliation_risk'):
                        result = await plugin.analyze_retaliation_risk(draft_text)
                    elif hasattr(plugin, 'analyze_harm_diversity'):
                        result = await plugin.analyze_harm_diversity(draft_text)
                    elif hasattr(plugin, 'analyze_public_interest_balance'):
                        result = await plugin.analyze_public_interest_balance(draft_text)
                    elif hasattr(plugin, 'analyze_transparency_arguments'):
                        result = await plugin.analyze_transparency_arguments(draft_text)
                    else:
                        # Fallback to generic validation
                        result = await plugin.validate_draft(draft_text)

                    recommendations[feature_name] = result.value if result.success else None

                except Exception as e:
                    logger.error(f"Plugin {feature_name} recommendation failed: {e}")
                    recommendations[feature_name] = None

            return recommendations

        except Exception as e:
            logger.error(f"Plugin recommendations failed: {e}")
            return {}
