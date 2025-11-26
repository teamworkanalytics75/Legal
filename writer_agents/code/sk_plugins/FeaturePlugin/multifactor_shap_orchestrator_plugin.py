#!/usr/bin/env python3
"""
Multi-Factor SHAP Orchestrator Plugin - Coordinates all important multi-factor SHAP features.

This plugin orchestrates enforcement of:
1. Perfect Outline Features:
   - transition_legal_standard_to_factual_background (#1 feature, 64.80 importance)
   - has_bullet_points (#2 feature, 31.56 importance)

2. Threshold Features:
   - max_enumeration_depth (#1 feature for seal motions, 27.27 importance)
   - enumeration_density
   - enumeration_count

3. Paragraph Structure Features (critical for pseudonym motions):
   - paragraph_count (#1 feature, 24.78 importance)
   - avg_words_per_paragraph (#2 feature, 8.13 importance)

4. Interaction Features:
   - word_count × balancing_test_position (0.337 strength - #1 interaction)
   - sentence_count × danger_safety_position (0.332 strength - #2 interaction)
   - char_count × danger_safety_position (0.107 strength)
   - word_count × danger_safety_position (0.093 strength)
   - word_count × char_count
   - word_count × sentence_count

This plugin coordinates all these features to ensure optimal motion structure.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult, EditRequest, DocumentLocation

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class MultifactorShapOrchestratorPlugin(BaseFeaturePlugin):
    """Orchestrator plugin for all multi-factor SHAP features."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, plugins: Dict[str, BaseFeaturePlugin] = None, **kwargs):
        super().__init__(kernel, "multifactor_shap_orchestrator", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("MultifactorShapOrchestratorPlugin initialized")

        # Use provided plugins dict or try to get from plugin registry
        self.plugins = plugins or {}
        
        # Try to get plugins from registry if not provided
        if not self.plugins:
            try:
                from .. import plugin_registry
                # Get all registered plugins by iterating through list
                plugin_names = plugin_registry.list_plugins()
                for name in plugin_names:
                    plugin = plugin_registry.get_plugin(name)
                    if plugin and isinstance(plugin, BaseFeaturePlugin):
                        self.plugins[name] = plugin
            except Exception as e:
                logger.debug(f"Could not access plugin registry: {e}")

    def _get_plugin_from_registry(self, plugin_name: str) -> Optional[BaseFeaturePlugin]:
        """
        Get plugin from registry or plugins dict.
        
        Args:
            plugin_name: Name of the plugin to retrieve (e.g., "introduction_word_count")
            
        Returns:
            Plugin instance if found, None otherwise
        """
        # Try plugins dict first
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]
        
        # Try plugin registry
        try:
            from .. import plugin_registry
            # First try exact match
            plugin = plugin_registry.get_plugin(plugin_name)
            if plugin and isinstance(plugin, BaseFeaturePlugin):
                return plugin
            
            # If not found, try converting feature name to plugin name format
            # BaseFeaturePlugin uses feature_name.title() + "Plugin"
            # e.g., "introduction_word_count" -> "Introduction_Word_CountPlugin"
            plugin_name_parts = plugin_name.split('_')
            # Try title() format (what BaseFeaturePlugin actually uses)
            plugin_name_title = plugin_name.title() + "Plugin"
            plugin = plugin_registry.get_plugin(plugin_name_title)
            if plugin and isinstance(plugin, BaseFeaturePlugin):
                return plugin
            
            # Also try without underscores (class name format)
            plugin_class_name = ''.join(word.capitalize() for word in plugin_name_parts) + "Plugin"
            plugin = plugin_registry.get_plugin(plugin_class_name)
            if plugin and isinstance(plugin, BaseFeaturePlugin):
                return plugin
            
            # Also try with underscores
            plugin_name_alt = '_'.join(word.capitalize() for word in plugin_name_parts) + "Plugin"
            plugin = plugin_registry.get_plugin(plugin_name_alt)
            if plugin and isinstance(plugin, BaseFeaturePlugin):
                return plugin
            
            # Last resort: search all plugins by feature_name attribute
            all_plugin_names = plugin_registry.list_plugins()
            for reg_name in all_plugin_names:
                plugin = plugin_registry.get_plugin(reg_name)
                if plugin and isinstance(plugin, BaseFeaturePlugin):
                    # Check if this plugin's feature_name matches
                    if hasattr(plugin, 'feature_name') and plugin.feature_name == plugin_name:
                        return plugin
        except Exception as e:
            logger.debug(f"Could not get plugin {plugin_name} from registry: {e}")
        
        return None

    async def validate_all_features(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """
        Validate all multi-factor SHAP features.

        Coordinates validation across:
        1. Perfect Outline Features
        2. Threshold Features
        3. Paragraph Structure Features
        4. Interaction Features
        """
        results = {}
        all_issues = []
        all_recommendations = []
        feature_scores = {}

        # 1. Perfect Outline Features
        logger.info("Validating perfect outline features...")

        # Transition Legal Standard -> Factual Background
        transition_plugin = self._get_plugin_from_registry("transition_legal_to_factual")
        if transition_plugin:
            transition_result = await transition_plugin.validate_transition(document, context)
            results["transition"] = transition_result.data if transition_result.success else None
            if not transition_result.success and transition_result.data:
                all_issues.extend(transition_result.data.get("issues", []))
                all_recommendations.extend(transition_result.data.get("recommendations", []))
            feature_scores["transition"] = 64.80  # #1 feature

        # Bullet Points
        bullet_plugin = self._get_plugin_from_registry("bullet_points")
        if bullet_plugin:
            bullet_result = await bullet_plugin.validate_bullet_points(document, context)
            results["bullet_points"] = bullet_result.data if bullet_result.success else None
            if not bullet_result.success and bullet_result.data:
                all_issues.extend(bullet_result.data.get("issues", []))
                all_recommendations.extend(bullet_result.data.get("recommendations", []))
            feature_scores["bullet_points"] = 31.56  # #2 feature

        # 2. Threshold Features
        logger.info("Validating threshold features...")

        # Max Enumeration Depth (critical for seal motions)
        enum_depth_plugin = self._get_plugin_from_registry("max_enumeration_depth")
        if enum_depth_plugin:
            enum_result = await enum_depth_plugin.validate_enumeration_depth(document, context)
            results["max_enumeration_depth"] = enum_result.data if enum_result.success else None
            if not enum_result.success and enum_result.data:
                all_issues.extend(enum_result.data.get("issues", []))
                all_recommendations.extend(enum_result.data.get("recommendations", []))
            feature_scores["max_enumeration_depth"] = 27.27  # #1 for seal motions

        # 3. Paragraph Structure Features (critical for pseudonym motions)
        logger.info("Validating paragraph structure features...")

        # Validate atomic paragraph features so scoring can key off exact feature names
        paragraph_count_plugin = self._get_plugin_from_registry("paragraph_count")
        if paragraph_count_plugin and hasattr(paragraph_count_plugin, "validate_paragraph_count"):
            pc_result = await paragraph_count_plugin.validate_paragraph_count(document, context)
            results["paragraph_count"] = pc_result.data if pc_result.success else pc_result.data
            if pc_result.data:
                all_issues.extend(pc_result.data.get("issues", []))
                all_recommendations.extend(pc_result.data.get("recommendations", []))

        avg_words_plugin = self._get_plugin_from_registry("avg_words_per_paragraph")
        if avg_words_plugin and hasattr(avg_words_plugin, "validate_avg_words_per_paragraph"):
            aw_result = await avg_words_plugin.validate_avg_words_per_paragraph(document, context)
            results["avg_words_per_paragraph"] = aw_result.data if aw_result.success else aw_result.data
            if aw_result.data:
                all_issues.extend(aw_result.data.get("issues", []))
                all_recommendations.extend(aw_result.data.get("recommendations", []))

        # Preserve interaction readout for paragraph structure interactions
        para_plugin = self._get_plugin_from_registry("paragraph_structure_interaction")
        if para_plugin and hasattr(para_plugin, "validate_paragraph_interactions"):
            para_result = await para_plugin.validate_paragraph_interactions(document, context)
            results["paragraph_structure"] = para_result.data if para_result.success else para_result.data
            if para_result.data:
                all_issues.extend(para_result.data.get("issues", []))
                all_recommendations.extend(para_result.data.get("recommendations", []))

        # Assign weights for paragraph metrics used in overall score
        feature_scores["paragraph_count"] = 24.78  # #1 for pseudonym motions
        feature_scores["avg_words_per_paragraph"] = 8.13  # #2 for pseudonym motions

        # 4. Interaction Features
        logger.info("Validating interaction features...")

        # Section Position Interactions
        section_plugin = self._get_plugin_from_registry("section_position_interaction")
        if section_plugin:
            section_result = await section_plugin.validate_section_positions(document, context)
            results["section_positions"] = section_result.data if section_result.success else None
            if not section_result.success and section_result.data:
                all_issues.extend(section_result.data.get("issues", []))
                all_recommendations.extend(section_result.data.get("recommendations", []))

        # Document Length Interactions
        length_plugin = self._get_plugin_from_registry("document_length_interaction")
        if length_plugin:
            length_result = await length_plugin.validate_document_length(document, context)
            results["document_length"] = length_result.data if length_result.success else None
            if not length_result.success and length_result.data:
                all_issues.extend(length_result.data.get("issues", []))
                all_recommendations.extend(length_result.data.get("recommendations", []))

        # Additional Interaction Plugins (wired up to enforce top-ranked interactions)
        # 1) word_count × balancing_test_position (strongest interaction)
        wc_bt_plugin = self._get_plugin_from_registry("word_count_balancing_test_interaction")
        if wc_bt_plugin and hasattr(wc_bt_plugin, "validate_word_count_balancing_test_interaction"):
            wc_bt_result = await wc_bt_plugin.validate_word_count_balancing_test_interaction(document, context)
            results["word_count_balancing_test_interaction"] = wc_bt_result.data if wc_bt_result.success else wc_bt_result.data
            if wc_bt_result.data:
                all_issues.extend(wc_bt_result.data.get("issues", []))
                all_recommendations.extend(wc_bt_result.data.get("recommendations", []))

        # 2) sentence_count × danger_safety_position
        sc_ds_plugin = self._get_plugin_from_registry("sentence_count_danger_safety_interaction")
        if sc_ds_plugin and hasattr(sc_ds_plugin, "validate_sentence_count_danger_safety_interaction"):
            sc_ds_result = await sc_ds_plugin.validate_sentence_count_danger_safety_interaction(document, context)
            results["sentence_count_danger_safety_interaction"] = sc_ds_result.data if sc_ds_result.success else sc_ds_result.data
            if sc_ds_result.data:
                all_issues.extend(sc_ds_result.data.get("issues", []))
                all_recommendations.extend(sc_ds_result.data.get("recommendations", []))

        # 3) char_count × danger_safety_position
        cc_ds_plugin = self._get_plugin_from_registry("char_count_danger_safety_interaction")
        if cc_ds_plugin and hasattr(cc_ds_plugin, "validate_char_count_danger_safety_interaction"):
            cc_ds_result = await cc_ds_plugin.validate_char_count_danger_safety_interaction(document, context)
            results["char_count_danger_safety_interaction"] = cc_ds_result.data if cc_ds_result.success else cc_ds_result.data
            if cc_ds_result.data:
                all_issues.extend(cc_ds_result.data.get("issues", []))
                all_recommendations.extend(cc_ds_result.data.get("recommendations", []))

        # 4) word_count × danger_safety_position
        wc_ds_plugin = self._get_plugin_from_registry("word_count_danger_safety_interaction")
        if wc_ds_plugin and hasattr(wc_ds_plugin, "validate_word_count_danger_safety_interaction"):
            wc_ds_result = await wc_ds_plugin.validate_word_count_danger_safety_interaction(document, context)
            results["word_count_danger_safety_interaction"] = wc_ds_result.data if wc_ds_result.success else wc_ds_result.data
            if wc_ds_result.data:
                all_issues.extend(wc_ds_result.data.get("issues", []))
                all_recommendations.extend(wc_ds_result.data.get("recommendations", []))

        # 5) word_count × char_count
        wc_cc_plugin = self._get_plugin_from_registry("word_count_char_count_interaction")
        if wc_cc_plugin and hasattr(wc_cc_plugin, "validate_word_count_char_count_interaction"):
            wc_cc_result = await wc_cc_plugin.validate_word_count_char_count_interaction(document, context)
            results["word_count_char_count_interaction"] = wc_cc_result.data if wc_cc_result.success else wc_cc_result.data
            if wc_cc_result.data:
                all_issues.extend(wc_cc_result.data.get("issues", []))
                all_recommendations.extend(wc_cc_result.data.get("recommendations", []))

        # 6) avg_words_per_paragraph × enumeration_density
        aw_ed_plugin = self._get_plugin_from_registry("avg_words_enumeration_density_interaction")
        if aw_ed_plugin and hasattr(aw_ed_plugin, "validate_avg_words_enumeration_density_interaction"):
            aw_ed_result = await aw_ed_plugin.validate_avg_words_enumeration_density_interaction(document, context)
            results["avg_words_enumeration_density_interaction"] = aw_ed_result.data if aw_ed_result.success else aw_ed_result.data
            if aw_ed_result.data:
                all_issues.extend(aw_ed_result.data.get("issues", []))
                all_recommendations.extend(aw_ed_result.data.get("recommendations", []))

        # 5. Section-Specific Features (from hierarchical sub-outline analysis)
        logger.info("Validating section-specific features...")

        # Legal Standard Section
        ls_wc_plugin = self._get_plugin_from_registry("legal_standard_word_count")
        if ls_wc_plugin and hasattr(ls_wc_plugin, "validate_legal_standard_word_count"):
            ls_wc_result = await ls_wc_plugin.validate_legal_standard_word_count(document, context)
            results["legal_standard_word_count"] = ls_wc_result.data if ls_wc_result.success else ls_wc_result.data
            if ls_wc_result.data:
                all_issues.extend(ls_wc_result.data.get("issues", []))
                all_recommendations.extend(ls_wc_result.data.get("recommendations", []))

        ls_ed_plugin = self._get_plugin_from_registry("legal_standard_enumeration_depth")
        if ls_ed_plugin and hasattr(ls_ed_plugin, "validate_legal_standard_enumeration_depth"):
            ls_ed_result = await ls_ed_plugin.validate_legal_standard_enumeration_depth(document, context)
            results["legal_standard_enumeration_depth"] = ls_ed_result.data if ls_ed_result.success else ls_ed_result.data
            if ls_ed_result.data:
                all_issues.extend(ls_ed_result.data.get("issues", []))
                all_recommendations.extend(ls_ed_result.data.get("recommendations", []))

        ls_ps_plugin = self._get_plugin_from_registry("legal_standard_paragraph_structure")
        if ls_ps_plugin and hasattr(ls_ps_plugin, "validate_legal_standard_paragraph_structure"):
            ls_ps_result = await ls_ps_plugin.validate_legal_standard_paragraph_structure(document, context)
            results["legal_standard_paragraph_structure"] = ls_ps_result.data if ls_ps_result.success else ls_ps_result.data
            if ls_ps_result.data:
                all_issues.extend(ls_ps_result.data.get("issues", []))
                all_recommendations.extend(ls_ps_result.data.get("recommendations", []))

        ls_so_plugin = self._get_plugin_from_registry("legal_standard_sub_outline")
        if ls_so_plugin and hasattr(ls_so_plugin, "validate_legal_standard_sub_outline"):
            ls_so_result = await ls_so_plugin.validate_legal_standard_sub_outline(document, context)
            results["legal_standard_sub_outline"] = ls_so_result.data if ls_so_result.success else ls_so_result.data
            if ls_so_result.data:
                all_issues.extend(ls_so_result.data.get("issues", []))
                all_recommendations.extend(ls_so_result.data.get("recommendations", []))

        # Balancing Test Section
        bt_wc_plugin = self._get_plugin_from_registry("balancing_test_word_count")
        if bt_wc_plugin and hasattr(bt_wc_plugin, "validate_balancing_test_word_count"):
            bt_wc_result = await bt_wc_plugin.validate_balancing_test_word_count(document, context)
            results["balancing_test_word_count"] = bt_wc_result.data if bt_wc_result.success else bt_wc_result.data
            if bt_wc_result.data:
                all_issues.extend(bt_wc_result.data.get("issues", []))
                all_recommendations.extend(bt_wc_result.data.get("recommendations", []))

        bt_ed_plugin = self._get_plugin_from_registry("balancing_test_enumeration_depth")
        if bt_ed_plugin and hasattr(bt_ed_plugin, "validate_balancing_test_enumeration_depth"):
            bt_ed_result = await bt_ed_plugin.validate_balancing_test_enumeration_depth(document, context)
            results["balancing_test_enumeration_depth"] = bt_ed_result.data if bt_ed_result.success else bt_ed_result.data
            if bt_ed_result.data:
                all_issues.extend(bt_ed_result.data.get("issues", []))
                all_recommendations.extend(bt_ed_result.data.get("recommendations", []))

        bt_ps_plugin = self._get_plugin_from_registry("balancing_test_paragraph_structure")
        if bt_ps_plugin and hasattr(bt_ps_plugin, "validate_balancing_test_paragraph_structure"):
            bt_ps_result = await bt_ps_plugin.validate_balancing_test_paragraph_structure(document, context)
            results["balancing_test_paragraph_structure"] = bt_ps_result.data if bt_ps_result.success else bt_ps_result.data
            if bt_ps_result.data:
                all_issues.extend(bt_ps_result.data.get("issues", []))
                all_recommendations.extend(bt_ps_result.data.get("recommendations", []))

        bt_so_plugin = self._get_plugin_from_registry("balancing_test_sub_outline")
        if bt_so_plugin and hasattr(bt_so_plugin, "validate_balancing_test_sub_outline"):
            bt_so_result = await bt_so_plugin.validate_balancing_test_sub_outline(document, context)
            results["balancing_test_sub_outline"] = bt_so_result.data if bt_so_result.success else bt_so_result.data
            if bt_so_result.data:
                all_issues.extend(bt_so_result.data.get("issues", []))
                all_recommendations.extend(bt_so_result.data.get("recommendations", []))

        # Danger/Safety Section
        ds_wc_plugin = self._get_plugin_from_registry("danger_safety_word_count")
        if ds_wc_plugin and hasattr(ds_wc_plugin, "validate_danger_safety_word_count"):
            ds_wc_result = await ds_wc_plugin.validate_danger_safety_word_count(document, context)
            results["danger_safety_word_count"] = ds_wc_result.data if ds_wc_result.success else ds_wc_result.data
            if ds_wc_result.data:
                all_issues.extend(ds_wc_result.data.get("issues", []))
                all_recommendations.extend(ds_wc_result.data.get("recommendations", []))

        ds_ed_plugin = self._get_plugin_from_registry("danger_safety_enumeration_depth")
        if ds_ed_plugin and hasattr(ds_ed_plugin, "validate_danger_safety_enumeration_depth"):
            ds_ed_result = await ds_ed_plugin.validate_danger_safety_enumeration_depth(document, context)
            results["danger_safety_enumeration_depth"] = ds_ed_result.data if ds_ed_result.success else ds_ed_result.data
            if ds_ed_result.data:
                all_issues.extend(ds_ed_result.data.get("issues", []))
                all_recommendations.extend(ds_ed_result.data.get("recommendations", []))

        # Additional sections (privacy_harm, factual_background, protective_measures, public_interest)
        # Word count plugins for other sections
        ph_wc_plugin = self._get_plugin_from_registry("privacy_harm_word_count")
        if ph_wc_plugin and hasattr(ph_wc_plugin, "validate_privacy_harm_word_count"):
            ph_wc_result = await ph_wc_plugin.validate_privacy_harm_word_count(document, context)
            results["privacy_harm_word_count"] = ph_wc_result.data if ph_wc_result.success else ph_wc_result.data
            if ph_wc_result.data:
                all_issues.extend(ph_wc_result.data.get("issues", []))
                all_recommendations.extend(ph_wc_result.data.get("recommendations", []))

        fb_wc_plugin = self._get_plugin_from_registry("factual_background_word_count")
        if fb_wc_plugin and hasattr(fb_wc_plugin, "validate_factual_background_word_count"):
            fb_wc_result = await fb_wc_plugin.validate_factual_background_word_count(document, context)
            results["factual_background_word_count"] = fb_wc_result.data if fb_wc_result.success else fb_wc_result.data
            if fb_wc_result.data:
                all_issues.extend(fb_wc_result.data.get("issues", []))
                all_recommendations.extend(fb_wc_result.data.get("recommendations", []))

        pm_wc_plugin = self._get_plugin_from_registry("protective_measures_word_count")
        if pm_wc_plugin and hasattr(pm_wc_plugin, "validate_protective_measures_word_count"):
            pm_wc_result = await pm_wc_plugin.validate_protective_measures_word_count(document, context)
            results["protective_measures_word_count"] = pm_wc_result.data if pm_wc_result.success else pm_wc_result.data
            if pm_wc_result.data:
                all_issues.extend(pm_wc_result.data.get("issues", []))
                all_recommendations.extend(pm_wc_result.data.get("recommendations", []))

        pi_wc_plugin = self._get_plugin_from_registry("public_interest_word_count")
        if pi_wc_plugin and hasattr(pi_wc_plugin, "validate_public_interest_word_count"):
            pi_wc_result = await pi_wc_plugin.validate_public_interest_word_count(document, context)
            results["public_interest_word_count"] = pi_wc_result.data if pi_wc_result.success else pi_wc_result.data
            if pi_wc_result.data:
                all_issues.extend(pi_wc_result.data.get("issues", []))
                all_recommendations.extend(pi_wc_result.data.get("recommendations", []))

        # Introduction and Conclusion word count plugins
        intro_wc_plugin = self._get_plugin_from_registry("introduction_word_count")
        if intro_wc_plugin and hasattr(intro_wc_plugin, "validate_introduction_word_count"):
            intro_wc_result = await intro_wc_plugin.validate_introduction_word_count(document, context)
            results["introduction_word_count"] = intro_wc_result.data if intro_wc_result.success else intro_wc_result.data
            if intro_wc_result.data:
                all_issues.extend(intro_wc_result.data.get("issues", []))
                all_recommendations.extend(intro_wc_result.data.get("recommendations", []))

        concl_wc_plugin = self._get_plugin_from_registry("conclusion_word_count")
        if concl_wc_plugin and hasattr(concl_wc_plugin, "validate_conclusion_word_count"):
            concl_wc_result = await concl_wc_plugin.validate_conclusion_word_count(document, context)
            results["conclusion_word_count"] = concl_wc_result.data if concl_wc_result.success else concl_wc_result.data
            if concl_wc_result.data:
                all_issues.extend(concl_wc_result.data.get("issues", []))
                all_recommendations.extend(concl_wc_result.data.get("recommendations", []))

        # 6. Paragraph Structure Plugins (for all 9 sections)
        logger.info("Validating paragraph structure for all sections...")
        para_structure_sections = [
            'introduction', 'factual_background', 'privacy_harm', 'danger_safety',
            'public_interest', 'protective_measures', 'conclusion'
        ]
        
        for section_name in para_structure_sections:
            plugin_name = f"{section_name}_paragraph_structure"
            plugin = self._get_plugin_from_registry(plugin_name)
            if plugin and hasattr(plugin, f"validate_{section_name}_paragraph_structure"):
                try:
                    result = await getattr(plugin, f"validate_{section_name}_paragraph_structure")(document, context)
                    results[plugin_name] = result.data if result.success else result.data
                    if result.data:
                        all_issues.extend(result.data.get("issues", []))
                        all_recommendations.extend(result.data.get("recommendations", []))
                except Exception as e:
                    logger.debug(f"Error validating {plugin_name}: {e}")

        # 7. Enumeration Depth Plugins (for all 9 sections)
        logger.info("Validating enumeration depth for all sections...")
        enum_depth_sections = [
            'introduction', 'factual_background', 'privacy_harm', 'public_interest',
            'protective_measures', 'conclusion'
        ]
        
        for section_name in enum_depth_sections:
            plugin_name = f"{section_name}_enumeration_depth"
            plugin = self._get_plugin_from_registry(plugin_name)
            if plugin and hasattr(plugin, f"validate_{section_name}_enumeration_depth"):
                try:
                    result = await getattr(plugin, f"validate_{section_name}_enumeration_depth")(document, context)
                    results[plugin_name] = result.data if result.success else result.data
                    if result.data:
                        all_issues.extend(result.data.get("issues", []))
                        all_recommendations.extend(result.data.get("recommendations", []))
                except Exception as e:
                    logger.debug(f"Error validating {plugin_name}: {e}")

        # 8. Sentence Count Plugins (for all 9 sections)
        logger.info("Validating sentence count per paragraph for all sections...")
        sentence_count_sections = [
            'introduction', 'legal_standard', 'factual_background', 'privacy_harm',
            'danger_safety', 'public_interest', 'balancing_test', 'protective_measures', 'conclusion'
        ]
        
        for section_name in sentence_count_sections:
            plugin_name = f"{section_name}_sentence_count"
            plugin = self._get_plugin_from_registry(plugin_name)
            if plugin and hasattr(plugin, f"validate_{section_name}_sentence_count"):
                try:
                    result = await getattr(plugin, f"validate_{section_name}_sentence_count")(document, context)
                    results[plugin_name] = result.data if result.success else result.data
                    if result.data:
                        all_issues.extend(result.data.get("issues", []))
                        all_recommendations.extend(result.data.get("recommendations", []))
                except Exception as e:
                    logger.debug(f"Error validating {plugin_name}: {e}")

        # 9. Words Per Sentence Plugins (for all 9 sections)
        logger.info("Validating words per sentence for all sections...")
        words_per_sentence_sections = [
            'introduction', 'legal_standard', 'factual_background', 'privacy_harm',
            'danger_safety', 'public_interest', 'balancing_test', 'protective_measures', 'conclusion'
        ]
        
        for section_name in words_per_sentence_sections:
            plugin_name = f"{section_name}_words_per_sentence"
            plugin = self._get_plugin_from_registry(plugin_name)
            if plugin and hasattr(plugin, f"validate_{section_name}_words_per_sentence"):
                try:
                    result = await getattr(plugin, f"validate_{section_name}_words_per_sentence")(document, context)
                    results[plugin_name] = result.data if result.success else result.data
                    if result.data:
                        all_issues.extend(result.data.get("issues", []))
                        all_recommendations.extend(result.data.get("recommendations", []))
                except Exception as e:
                    logger.debug(f"Error validating {plugin_name}: {e}")

        # Calculate overall score
        critical_issues = [issue for issue in all_issues if issue.get("severity") == "critical"]
        high_issues = [issue for issue in all_issues if issue.get("severity") == "high"]
        overall_success = len(critical_issues) == 0

        # 10. Constraint Resolution
        logger.info("Resolving constraint conflicts...")
        constraint_resolver = self._get_plugin_from_registry("constraint_resolver")
        if constraint_resolver and hasattr(constraint_resolver, "resolve_all_constraints"):
            try:
                resolution_result = await constraint_resolver.resolve_all_constraints(document, context)
                if resolution_result and resolution_result.data:
                    # Add resolved constraints to results
                    results["constraint_resolution"] = resolution_result.data
                    
                    # Add flagged conflicts to issues
                    flagged_conflicts = resolution_result.data.get("flagged_conflicts", [])
                    for conflict in flagged_conflicts:
                        all_issues.append({
                            "type": "constraint_conflict",
                            "severity": "medium",
                            "message": conflict.get("message", "Constraint conflict detected"),
                            "conflict_details": conflict
                        })
                    
                    # Add resolution recommendations
                    resolution_recommendations = resolution_result.data.get("recommendations", [])
                    all_recommendations.extend(resolution_recommendations)
            except Exception as e:
                logger.warning(f"Error in constraint resolution: {e}")

        # Prioritize recommendations by feature importance
        prioritized_recommendations = sorted(
            all_recommendations,
            key=lambda r: feature_scores.get(r.get("type", "").split("_")[0], 0),
            reverse=True
        )

        return FunctionResult(
            success=overall_success,
            data={
                "feature_results": results,
                "feature_scores": feature_scores,
                "all_issues": all_issues,
                "critical_issues": critical_issues,
                "high_issues": high_issues,
                "prioritized_recommendations": prioritized_recommendations[:10],  # Top 10
                "total_issues": len(all_issues),
                "total_recommendations": len(all_recommendations),
                "overall_score": self._calculate_overall_score(results, feature_scores)
            },
            message=f"Multi-factor validation complete: {len(critical_issues)} critical, {len(high_issues)} high priority issues"
        )

    def _calculate_overall_score(self, results: Dict[str, Any], feature_scores: Dict[str, float]) -> float:
        """Calculate overall feature score based on validation results."""
        total_score = 0.0
        total_weight = 0.0

        for feature_name, score in feature_scores.items():
            # Check if feature passed validation
            passed = False
            for result_key, result_data in results.items():
                if result_data and feature_name in str(result_key):
                    if isinstance(result_data, dict):
                        passed = result_data.get("meets_target", False) or result_data.get("transition_present", False) or result_data.get("has_bullet_points", False)
                    break

            weight = score
            total_weight += weight
            if passed:
                total_score += weight

        return (total_score / total_weight * 100) if total_weight > 0 else 0.0

    async def generate_comprehensive_report(self, document: "DocumentStructure", context: Dict[str, Any] = None) -> FunctionResult:
        """Generate comprehensive report on all multi-factor features."""
        validation_result = await self.validate_all_features(document, context)

        report = {
            "summary": {
                "overall_score": validation_result.data.get("overall_score", 0),
                "total_issues": validation_result.data.get("total_issues", 0),
                "critical_issues": len(validation_result.data.get("critical_issues", [])),
                "high_issues": len(validation_result.data.get("high_issues", []))
            },
            "perfect_outline_features": {
                "transition_legal_to_factual": validation_result.data.get("feature_results", {}).get("transition"),
                "bullet_points": validation_result.data.get("feature_results", {}).get("bullet_points")
            },
            "threshold_features": {
                "max_enumeration_depth": validation_result.data.get("feature_results", {}).get("max_enumeration_depth")
            },
            "paragraph_structure_features": {
                "paragraph_structure": validation_result.data.get("feature_results", {}).get("paragraph_structure")
            },
            "interaction_features": {
                "section_positions": validation_result.data.get("feature_results", {}).get("section_positions"),
                "document_length": validation_result.data.get("feature_results", {}).get("document_length")
            },
            "prioritized_recommendations": validation_result.data.get("prioritized_recommendations", [])
        }

        return FunctionResult(
            success=True,
            data=report,
            message="Comprehensive multi-factor report generated"
        )

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate edit requests from all multi-factor plugins.

        Coordinates edit requests across all plugins, prioritizing by feature importance.
        """
        all_requests = []

        # Get edit requests from each plugin
        plugins = [
            ("max_enumeration_depth", self._get_plugin_from_registry("max_enumeration_depth")),
            ("paragraph_structure_interaction", self._get_plugin_from_registry("paragraph_structure_interaction")),
            ("transition_legal_to_factual", self._get_plugin_from_registry("transition_legal_to_factual")),
            ("bullet_points", self._get_plugin_from_registry("bullet_points"))
        ]

        for plugin_name, plugin in plugins:
            if plugin and hasattr(plugin, "generate_edit_requests"):
                try:
                    requests = await plugin.generate_edit_requests(text, structure, context)
                    all_requests.extend(requests)
                except Exception as e:
                    logger.warning(f"Failed to get edit requests from {plugin_name}: {e}")

        # Sort by priority (higher priority first)
        all_requests.sort(key=lambda r: r.priority, reverse=True)

        # Limit to top 20 requests to avoid overwhelming
        return all_requests[:20]
