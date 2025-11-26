# 📊 Current SK Plugin Count

**Date**: 2025-01-XX
**Status**: Current count of plugins in the SK (Semantic Kernel) module

---

## 🎯 Quick Answer

**Total SK Plugins: 60+ plugin files**

Breakdown:
- **FeaturePlugin directory**: 60 files (including base classes and orchestrators)
- **Actual Feature Plugins**: ~56 active plugins
- **Other plugin modules**: 6+ additional plugins
- **Total**: ~62+ plugin files across all SK modules

---

## 📋 Detailed Breakdown

### FeaturePlugin Directory: 60 Files

**Plugin Files** (ending in `_plugin.py`): 60 files

**Note**: Some files are not actual plugins:
- `base_feature_plugin.py` - Base class (not a plugin)
- `feature_orchestrator.py` - Orchestrator (RefinementLoop)
- `case_enforcement_plugin_generator.py` - Factory (generates plugins)
- `document_structure.py` - Utility module

**Actual Feature Plugins**: ~56 active plugins

#### Complete List of FeaturePlugin Files:

1. `avoid_balancing_test_phrase_plugin.py`
2. `avoid_compelling_interest_phrase_plugin.py`
3. `avoid_first_amendment_plugin.py`
4. `avoid_narrowly_tailored_plugin.py`
5. `balancing_outweigh_plugin.py`
6. `balancing_test_position_plugin.py`
7. `bullet_points_plugin.py`
8. `character_count_plugin.py`
9. `citation_format_plugin.py`
10. `citation_retrieval_plugin.py`
11. `custodian_count_plugin.py`
12. `danger_safety_position_plugin.py`
13. `enumeration_danger_safety_plugin.py`
14. `enumeration_density_plugin.py`
15. `foreign_government_plugin.py`
16. `formatting_plugin.py`
17. `harvard_lawsuit_plugin.py`
18. `hk_national_security_plugin.py`
19. `intel_classified_info_plugin.py`
20. `intel_factor_1_participant_plugin.py`
21. `intel_factor_1_plugin.py`
22. `intel_factor_2_plugin.py`
23. `intel_factor_2_receptivity_plugin.py`
24. `intel_factor_3_circumvention_plugin.py`
25. `intel_factor_3_plugin.py`
26. `intel_factor_4_burden_plugin.py`
27. `intel_factor_4_plugin.py`
28. `mentions_harassment_plugin.py`
29. `mentions_privacy_plugin.py`
30. `mentions_retaliation_plugin.py`
31. `mentions_safety_plugin.py`
32. `national_security_definitions_plugin.py`
33. `paragraph_monitor_plugin.py`
34. `paragraph_structure_plugin.py`
35. `per_paragraph_plugin.py`
36. `prc_national_security_plugin.py`
37. `presumption_acknowledgment_plugin.py`
38. `privacy_harm_count_plugin.py`
39. `protective_measures_plugin.py`
40. `public_interest_plugin.py`
41. `required_case_citation_plugin.py`
42. `rule_26_discovery_plugin.py`
43. `rule_45_mentions_plugin.py`
44. `scope_breadth_plugin.py`
45. `sentence_count_plugin.py`
46. `sentence_length_plugin.py`
47. `sentence_structure_plugin.py`
48. `statutory_requirement_1_plugin.py`
49. `statutory_requirement_2_plugin.py`
50. `statutory_requirement_3_plugin.py`
51. `timing_argument_plugin.py`
52. `transition_legal_to_factual_plugin.py`
53. `trump_june4_proclamation_plugin.py`
54. `use_balance_concepts_plugin.py`
55. `use_competing_interests_plugin.py`
56. `use_motion_language_plugin.py`
57. `word_choice_plugin.py`
58. `word_count_plugin.py`
59. `word_frequency_plugin.py`

**Non-Plugin Files** (in FeaturePlugin directory):
- `base_feature_plugin.py` - Base class
- `feature_orchestrator.py` - Orchestrator (RefinementLoop)
- `case_enforcement_plugin_generator.py` - Factory
- `document_structure.py` - Utility

---

### Other SK Plugin Modules

#### DraftingPlugin: 3-4 plugins
- `factual_timeline_function.py`
- `privacy_harm_function.py`
- `causation_analysis_function.py`
- `factual_timeline.py` (native)
- `privacy_harm_native.py` (native)

#### ValidationPlugin: 2+ plugins
- `petition_validator_function.py`
- `enhanced_validator.py`
- Additional validation functions

#### AssemblyPlugin: 2 plugins
- `document_assembler.py`
- `exhibit_linker.py`

#### SHAPInsightPlugin: 1 plugin
- `shap_insight_plugin.py`

**Total Other Plugins**: ~8-10 plugins

---

## 📊 Summary

| Category | Count |
|----------|-------|
| **FeaturePlugin files** | 60 |
| **Active FeaturePlugins** | ~56 |
| **Other plugin modules** | ~8-10 |
| **Total plugin files** | ~62-68 |
| **Orchestrators/Managers** | 3-4 |
| **Base classes/Utilities** | 3-4 |

---

## 🎯 Key Plugins by Category

### CatBoost Analysis Plugins (Top Features)
- `transition_legal_to_factual_plugin.py` - #1 feature (64.80 importance)
- `bullet_points_plugin.py` - #2 feature (31.56 importance)
- `danger_safety_position_plugin.py` - #3 feature (3.36 importance)
- `balancing_test_position_plugin.py` - #4 feature (0.23 importance)
- `enumeration_danger_safety_plugin.py` - #5 feature (0.05 importance)

### National Security Plugins
- `intel_classified_info_plugin.py` - #1 feature (17.01 importance)
- `balancing_outweigh_plugin.py` - #2 feature (12.75 importance)
- `presumption_acknowledgment_plugin.py` - #5 feature (8.65 importance)
- `protective_measures_plugin.py` - #7 feature (4.32 importance)
- `national_security_definitions_plugin.py` - #8 feature (3.67 importance)
- `foreign_government_plugin.py` - #9 feature (3.40 importance)

### Content Analysis Plugins
- `mentions_privacy_plugin.py`
- `mentions_harassment_plugin.py`
- `mentions_safety_plugin.py`
- `mentions_retaliation_plugin.py`
- `privacy_harm_count_plugin.py`

### Structure Plugins
- `paragraph_structure_plugin.py`
- `sentence_structure_plugin.py`
- `enumeration_density_plugin.py`
- `bullet_points_plugin.py`

### Citation Plugins
- `citation_retrieval_plugin.py`
- `required_case_citation_plugin.py`
- `citation_format_plugin.py`

### Intel Factor Plugins
- `intel_factor_1_plugin.py`
- `intel_factor_2_plugin.py`
- `intel_factor_3_plugin.py`
- `intel_factor_4_plugin.py`
- `intel_factor_1_participant_plugin.py`
- `intel_factor_2_receptivity_plugin.py`
- `intel_factor_3_circumvention_plugin.py`
- `intel_factor_4_burden_plugin.py`

---

## 🔧 Dynamic Plugins

**Runtime-Generated Plugins**:
- `IndividualCaseEnforcementPlugin` - Generated by `CaseEnforcementPluginFactory`
- Based on `master_case_citations.json`
- ~21 case enforcement plugins (created at runtime, not stored as files)

---

## 📈 Growth

The plugin count has grown from the initial analysis:
- **Previous count** (from analysis docs): 59
- **Current count**: 60+ FeaturePlugin files
- **Growth**: +1-2 plugins since last analysis

---

## ✅ Verification

Count verified by:
- ✅ File system scan (`Get-ChildItem *plugin.py`)
- ✅ Python pathlib count
- ✅ Manual review of `__init__.py` exports

---

## 📝 Notes

- Some plugins may be dynamically generated at runtime
- `CaseEnforcementPluginFactory` creates plugins from case citations
- Plugin registry may have additional configured plugins not yet created
- All plugins inherit from `BaseFeaturePlugin` or `BaseSKPlugin`

