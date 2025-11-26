#!/usr/bin/env python3
"""
Feature Plugin Module - Atomic SK plugins for CatBoost features.
"""

from .base_feature_plugin import BaseFeaturePlugin
from .mentions_privacy_plugin import PrivacyPlugin
from .mentions_harassment_plugin import HarassmentPlugin
from .mentions_safety_plugin import SafetyPlugin
from .mentions_retaliation_plugin import RetaliationPlugin
from .citation_retrieval_plugin import CitationRetrievalPlugin
from .required_case_citation_plugin import RequiredCaseCitationPlugin
from .case_enforcement_plugin_generator import (
    IndividualCaseEnforcementPlugin,
    CaseEnforcementPluginFactory
)
from .privacy_harm_count_plugin import PrivacyHarmCountPlugin
from .public_interest_plugin import PublicInterestPlugin
from .transparency_argument_plugin import TransparencyArgumentPlugin
from .sentence_structure_plugin import SentenceStructurePlugin
from .paragraph_structure_plugin import ParagraphStructurePlugin
from .enumeration_density_plugin import EnumerationDensityPlugin
from .per_paragraph_plugin import PerParagraphPlugin
from .paragraph_monitor_plugin import ParagraphMonitorPlugin
from .statutory_requirement_1_plugin import StatutoryRequirement1Plugin
from .statutory_requirement_2_plugin import StatutoryRequirement2Plugin
from .statutory_requirement_3_plugin import StatutoryRequirement3Plugin
from .intel_factor_1_plugin import IntelFactor1Plugin
from .intel_factor_2_plugin import IntelFactor2Plugin
from .intel_factor_3_plugin import IntelFactor3Plugin
from .intel_factor_4_plugin import IntelFactor4Plugin
from .intel_factor_1_participant_plugin import IntelFactor1ParticipantPlugin
from .intel_factor_2_receptivity_plugin import IntelFactor2ReceptivityPlugin
from .intel_factor_3_circumvention_plugin import IntelFactor3CircumventionPlugin
from .intel_factor_4_burden_plugin import IntelFactor4BurdenPlugin
from .rule_45_mentions_plugin import Rule45MentionsPlugin
from .custodian_count_plugin import CustodianCountPlugin
from .scope_breadth_plugin import ScopeBreadthPlugin
from .sentence_count_plugin import SentenceCountPlugin
from .sentence_length_plugin import SentenceLengthPlugin
from .word_count_plugin import WordCountPlugin
from .word_choice_plugin import WordChoicePlugin
from .word_frequency_plugin import WordFrequencyPlugin
from .character_count_plugin import CharacterCountPlugin
from .formatting_plugin import FormattingPlugin
from .citation_format_plugin import CitationFormatPlugin
from .rule_26_discovery_plugin import Rule26DiscoveryPlugin
from .balancing_test_position_plugin import BalancingTestPositionPlugin
from .transition_legal_to_factual_plugin import TransitionLegalToFactualPlugin
from .feature_orchestrator import RefinementLoop
from .national_security_definitions_plugin import NationalSecurityDefinitionsPlugin
from .balancing_outweigh_plugin import BalancingOutweighPlugin
from .presumption_acknowledgment_plugin import PresumptionAcknowledgmentPlugin
from .protective_measures_plugin import ProtectiveMeasuresPlugin
from .foreign_government_plugin import ForeignGovernmentPlugin
from .intel_classified_info_plugin import IntelClassifiedInfoPlugin
from .hk_national_security_plugin import HKNationalSecurityPlugin
from .prc_national_security_plugin import PRCNationalSecurityPlugin
from .trump_june4_proclamation_plugin import TrumpJune4ProclamationPlugin
from .harvard_lawsuit_plugin import HarvardLawsuitPlugin
from .timing_argument_plugin import TimingArgumentPlugin
from .avoid_first_amendment_plugin import AvoidFirstAmendmentPlugin
from .avoid_balancing_test_phrase_plugin import AvoidBalancingTestPhrasePlugin
from .use_balance_concepts_plugin import UseBalanceConceptsPlugin
from .use_motion_language_plugin import UseMotionLanguagePlugin
from .avoid_narrowly_tailored_plugin import AvoidNarrowlyTailoredPlugin
from .avoid_compelling_interest_phrase_plugin import AvoidCompellingInterestPhrasePlugin
from .use_competing_interests_plugin import UseCompetingInterestsPlugin
from .document_structure import (
    DocumentStructure,
    Paragraph,
    Sentence,
    parse_document_structure,
    apply_edit as apply_document_edit
)
# Section-specific word count plugins
from .introduction_word_count_plugin import IntroductionWordCountPlugin
from .legal_standard_word_count_plugin import LegalStandardWordCountPlugin
from .factual_background_word_count_plugin import FactualBackgroundWordCountPlugin
from .privacy_harm_word_count_plugin import PrivacyHarmWordCountPlugin
from .danger_safety_word_count_plugin import DangerSafetyWordCountPlugin
from .public_interest_word_count_plugin import PublicInterestWordCountPlugin
from .balancing_test_word_count_plugin import BalancingTestWordCountPlugin
from .protective_measures_word_count_plugin import ProtectiveMeasuresWordCountPlugin
from .conclusion_word_count_plugin import ConclusionWordCountPlugin
# Section-specific paragraph structure plugins
from .introduction_paragraph_structure_plugin import IntroductionParagraphStructurePlugin
from .legal_standard_paragraph_structure_plugin import LegalStandardParagraphStructurePlugin
from .factual_background_paragraph_structure_plugin import FactualBackgroundParagraphStructurePlugin
from .privacy_harm_paragraph_structure_plugin import PrivacyHarmParagraphStructurePlugin
from .danger_safety_paragraph_structure_plugin import DangerSafetyParagraphStructurePlugin
from .public_interest_paragraph_structure_plugin import PublicInterestParagraphStructurePlugin
from .balancing_test_paragraph_structure_plugin import BalancingTestParagraphStructurePlugin
from .protective_measures_paragraph_structure_plugin import ProtectiveMeasuresParagraphStructurePlugin
from .conclusion_paragraph_structure_plugin import ConclusionParagraphStructurePlugin
# Section-specific enumeration depth plugins
from .introduction_enumeration_depth_plugin import IntroductionEnumerationDepthPlugin
from .legal_standard_enumeration_depth_plugin import LegalStandardEnumerationDepthPlugin
from .factual_background_enumeration_depth_plugin import FactualBackgroundEnumerationDepthPlugin
from .privacy_harm_enumeration_depth_plugin import PrivacyHarmEnumerationDepthPlugin
from .public_interest_enumeration_depth_plugin import PublicInterestEnumerationDepthPlugin
from .balancing_test_enumeration_depth_plugin import BalancingTestEnumerationDepthPlugin
from .danger_safety_enumeration_depth_plugin import DangerSafetyEnumerationDepthPlugin
from .protective_measures_enumeration_depth_plugin import ProtectiveMeasuresEnumerationDepthPlugin
from .conclusion_enumeration_depth_plugin import ConclusionEnumerationDepthPlugin
# Section-specific sentence count plugins
from .introduction_sentence_count_plugin import IntroductionSentenceCountPlugin
from .legal_standard_sentence_count_plugin import LegalStandardSentenceCountPlugin
from .factual_background_sentence_count_plugin import FactualBackgroundSentenceCountPlugin
from .privacy_harm_sentence_count_plugin import PrivacyHarmSentenceCountPlugin
from .danger_safety_sentence_count_plugin import DangerSafetySentenceCountPlugin
from .public_interest_sentence_count_plugin import PublicInterestSentenceCountPlugin
from .balancing_test_sentence_count_plugin import BalancingTestSentenceCountPlugin
from .protective_measures_sentence_count_plugin import ProtectiveMeasuresSentenceCountPlugin
from .conclusion_sentence_count_plugin import ConclusionSentenceCountPlugin
# Section-specific words per sentence plugins
from .introduction_words_per_sentence_plugin import IntroductionWordsPerSentencePlugin
from .legal_standard_words_per_sentence_plugin import LegalStandardWordsPerSentencePlugin
from .factual_background_words_per_sentence_plugin import FactualBackgroundWordsPerSentencePlugin
from .privacy_harm_words_per_sentence_plugin import PrivacyHarmWordsPerSentencePlugin
from .danger_safety_words_per_sentence_plugin import DangerSafetyWordsPerSentencePlugin
from .public_interest_words_per_sentence_plugin import PublicInterestWordsPerSentencePlugin
from .balancing_test_words_per_sentence_plugin import BalancingTestWordsPerSentencePlugin
from .protective_measures_words_per_sentence_plugin import ProtectiveMeasuresWordsPerSentencePlugin
from .conclusion_words_per_sentence_plugin import ConclusionWordsPerSentencePlugin
# Constraint resolver
from .constraint_resolver_plugin import ConstraintResolverPlugin

# Backwards compatibility aliases
DraftEnhancer = RefinementLoop          # Previous name
DraftQualityController = RefinementLoop # Previous name
FeatureOrchestrator = RefinementLoop    # Original name
MentionsPrivacyPlugin = PrivacyPlugin      # Backwards compatibility
MentionsHarassmentPlugin = HarassmentPlugin  # Backwards compatibility
MentionsSafetyPlugin = SafetyPlugin          # Backwards compatibility
MentionsRetaliationPlugin = RetaliationPlugin  # Backwards compatibility

__all__ = [
    "BaseFeaturePlugin",
    "PrivacyPlugin",
    "HarassmentPlugin",
    "SafetyPlugin",
    "RetaliationPlugin",
    "CitationRetrievalPlugin",
    "RequiredCaseCitationPlugin",
    "IndividualCaseEnforcementPlugin",
    "CaseEnforcementPluginFactory",
    "PrivacyHarmCountPlugin",
    "PublicInterestPlugin",
    "TransparencyArgumentPlugin",
    "SentenceStructurePlugin",
    "ParagraphStructurePlugin",
    "EnumerationDensityPlugin",
    "PerParagraphPlugin",
    "ParagraphMonitorPlugin",
    "StatutoryRequirement1Plugin",
    "StatutoryRequirement2Plugin",
    "StatutoryRequirement3Plugin",
    "IntelFactor1Plugin",
    "IntelFactor2Plugin",
    "IntelFactor3Plugin",
    "IntelFactor4Plugin",
    "IntelFactor1ParticipantPlugin",
    "IntelFactor2ReceptivityPlugin",
    "IntelFactor3CircumventionPlugin",
    "IntelFactor4BurdenPlugin",
    "Rule45MentionsPlugin",
    "CustodianCountPlugin",
    "ScopeBreadthPlugin",
    "SentenceCountPlugin",
    "SentenceLengthPlugin",
    "WordCountPlugin",
    "WordChoicePlugin",
    "WordFrequencyPlugin",
    "CharacterCountPlugin",
    "FormattingPlugin",
    "CitationFormatPlugin",
    "Rule26DiscoveryPlugin",
    "BalancingTestPositionPlugin",
    "TransitionLegalToFactualPlugin",
    "RefinementLoop",
    "NationalSecurityDefinitionsPlugin",
    "BalancingOutweighPlugin",
    "PresumptionAcknowledgmentPlugin",
    "ProtectiveMeasuresPlugin",
    "ForeignGovernmentPlugin",
    "IntelClassifiedInfoPlugin",
    "HKNationalSecurityPlugin",
    "PRCNationalSecurityPlugin",
    "TrumpJune4ProclamationPlugin",
    "HarvardLawsuitPlugin",
    "TimingArgumentPlugin",
    "AvoidFirstAmendmentPlugin",
    "AvoidBalancingTestPhrasePlugin",
    "UseBalanceConceptsPlugin",
    "UseMotionLanguagePlugin",
    "AvoidNarrowlyTailoredPlugin",
    "AvoidCompellingInterestPhrasePlugin",
    "UseCompetingInterestsPlugin",
    "DocumentStructure",
    "Paragraph",
    "Sentence",
    "parse_document_structure",
    "apply_document_edit",
    "DraftEnhancer",            # Backwards compatibility
    "DraftQualityController",   # Backwards compatibility
    "FeatureOrchestrator",      # Backwards compatibility
    "MentionsPrivacyPlugin",    # Backwards compatibility
    "MentionsHarassmentPlugin", # Backwards compatibility
    "MentionsSafetyPlugin",     # Backwards compatibility
    "MentionsRetaliationPlugin", # Backwards compatibility
    # Section-specific word count plugins
    "IntroductionWordCountPlugin",
    "LegalStandardWordCountPlugin",
    "FactualBackgroundWordCountPlugin",
    "PrivacyHarmWordCountPlugin",
    "DangerSafetyWordCountPlugin",
    "PublicInterestWordCountPlugin",
    "BalancingTestWordCountPlugin",
    "ProtectiveMeasuresWordCountPlugin",
    "ConclusionWordCountPlugin",
    # Section-specific paragraph structure plugins
    "IntroductionParagraphStructurePlugin",
    "LegalStandardParagraphStructurePlugin",
    "FactualBackgroundParagraphStructurePlugin",
    "PrivacyHarmParagraphStructurePlugin",
    "DangerSafetyParagraphStructurePlugin",
    "PublicInterestParagraphStructurePlugin",
    "BalancingTestParagraphStructurePlugin",
    "ProtectiveMeasuresParagraphStructurePlugin",
    "ConclusionParagraphStructurePlugin",
    # Section-specific enumeration depth plugins
    "IntroductionEnumerationDepthPlugin",
    "LegalStandardEnumerationDepthPlugin",
    "FactualBackgroundEnumerationDepthPlugin",
    "PrivacyHarmEnumerationDepthPlugin",
    "PublicInterestEnumerationDepthPlugin",
    "BalancingTestEnumerationDepthPlugin",
    "DangerSafetyEnumerationDepthPlugin",
    "ProtectiveMeasuresEnumerationDepthPlugin",
    "ConclusionEnumerationDepthPlugin",
    # Section-specific sentence count plugins
    "IntroductionSentenceCountPlugin",
    "LegalStandardSentenceCountPlugin",
    "FactualBackgroundSentenceCountPlugin",
    "PrivacyHarmSentenceCountPlugin",
    "DangerSafetySentenceCountPlugin",
    "PublicInterestSentenceCountPlugin",
    "BalancingTestSentenceCountPlugin",
    "ProtectiveMeasuresSentenceCountPlugin",
    "ConclusionSentenceCountPlugin",
    # Section-specific words per sentence plugins
    "IntroductionWordsPerSentencePlugin",
    "LegalStandardWordsPerSentencePlugin",
    "FactualBackgroundWordsPerSentencePlugin",
    "PrivacyHarmWordsPerSentencePlugin",
    "DangerSafetyWordsPerSentencePlugin",
    "PublicInterestWordsPerSentencePlugin",
    "BalancingTestWordsPerSentencePlugin",
    "ProtectiveMeasuresWordsPerSentencePlugin",
    "ConclusionWordsPerSentencePlugin",
    # Constraint resolver
    "ConstraintResolverPlugin"
]
