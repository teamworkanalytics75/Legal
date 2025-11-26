#!/usr/bin/env python3
"""
Analyze User Draft Documents for Motion to Seal/Pseudonym.

This script extracts text from the user's draft DOCX files and runs them through
the writer_agents system to analyze and suggest improvements.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import zipfile
from xml.etree import ElementTree as ET

# Add writer_agents to path
sys.path.append(str(Path(__file__).parent / "writer_agents" / "code"))
sys.path.append(str(Path(__file__).parent / "writer_agents" / "code" / "sk_plugins"))

# Import the analysis modules
from analysis.analyze_ma_motion_doc import read_docx_text, compute_draft_features, extract_citations_from_text
from sk_plugins.FeaturePlugin.feature_orchestrator import FeatureOrchestrator
from sk_plugins.FeaturePlugin.mentions_privacy_plugin import MentionsPrivacyPlugin
from sk_plugins.FeaturePlugin.citation_retrieval_plugin import CitationRetrievalPlugin
from sk_plugins.FeaturePlugin.mentions_safety_plugin import MentionsSafetyPlugin
from sk_plugins.FeaturePlugin.mentions_harassment_plugin import MentionsHarassmentPlugin
from sk_plugins.FeaturePlugin.mentions_retaliation_plugin import MentionsRetaliationPlugin
from sk_plugins.FeaturePlugin.mentions_transparency_plugin import MentionsTransparencyPlugin
from sk_plugins.FeaturePlugin.privacy_harm_count_plugin import PrivacyHarmCountPlugin
from sk_plugins.FeaturePlugin.public_interest_plugin import PublicInterestPlugin

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DraftAnalyzer:
    """Analyzes user draft documents using the writer_agents system."""

    def __init__(self):
        self.rules_dir = Path(__file__).parent / "writer_agents" / "code" / "sk_plugins" / "rules"
        self.plugins = {}
        self.orchestrator = None

    def setup_plugins(self):
        """Initialize all feature plugins."""
        logger.info("Setting up feature plugins...")

        # Mock kernel and chroma store for testing
        from unittest.mock import Mock
        mock_kernel = Mock()
        mock_chroma_store = Mock()

        # Initialize all plugins
        self.plugins = {
            "mentions_privacy": MentionsPrivacyPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "citation_retrieval": CitationRetrievalPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "mentions_safety": MentionsSafetyPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "mentions_harassment": MentionsHarassmentPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "mentions_retaliation": MentionsRetaliationPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "mentions_transparency": MentionsTransparencyPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "privacy_harm_count": PrivacyHarmCountPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
            "public_interest": PublicInterestPlugin(mock_kernel, mock_chroma_store, self.rules_dir),
        }

        # Initialize orchestrator
        self.orchestrator = FeatureOrchestrator(self.plugins, None)
        logger.info(f"Initialized {len(self.plugins)} plugins")

    def extract_text_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            return read_docx_text(docx_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {docx_path}: {e}")
            return ""

    def analyze_draft_features(self, text: str) -> Dict[str, Any]:
        """Analyze draft using CatBoost features."""
        try:
            features = compute_draft_features(text)
            citations = extract_citations_from_text(text)

            return {
                "features": features,
                "citations": citations,
                "word_count": len(text.split()),
                "char_count": len(text)
            }
        except Exception as e:
            logger.error(f"Failed to analyze draft features: {e}")
            return {}

    async def analyze_draft_with_plugins(self, text: str, doc_name: str) -> Dict[str, Any]:
        """Analyze draft using all feature plugins."""
        logger.info(f"Analyzing {doc_name} with feature plugins...")

        results = {
            "document_name": doc_name,
            "analysis_results": {},
            "improvement_suggestions": {},
            "overall_assessment": {}
        }

        # Analyze with each plugin
        for plugin_name, plugin in self.plugins.items():
            try:
                logger.info(f"Running {plugin_name} analysis...")

                # Get the appropriate analysis method for each plugin
                if hasattr(plugin, 'analyze_privacy_strength'):
                    analysis_result = await plugin.analyze_privacy_strength(text)
                elif hasattr(plugin, 'analyze_citation_strength'):
                    analysis_result = await plugin.analyze_citation_strength(text)
                elif hasattr(plugin, 'analyze_safety_concerns'):
                    analysis_result = await plugin.analyze_safety_concerns(text)
                elif hasattr(plugin, 'analyze_harassment_concerns'):
                    analysis_result = await plugin.analyze_harassment_concerns(text)
                elif hasattr(plugin, 'analyze_retaliation_concerns'):
                    analysis_result = await plugin.analyze_retaliation_concerns(text)
                elif hasattr(plugin, 'analyze_transparency_concerns'):
                    analysis_result = await plugin.analyze_transparency_concerns(text)
                elif hasattr(plugin, 'analyze_privacy_harm_count'):
                    analysis_result = await plugin.analyze_privacy_harm_count(text)
                elif hasattr(plugin, 'analyze_public_interest'):
                    analysis_result = await plugin.analyze_public_interest(text)
                else:
                    logger.warning(f"No analysis method found for {plugin_name}")
                    continue

                if analysis_result.success:
                    results["analysis_results"][plugin_name] = analysis_result.value

                    # Get improvement suggestions
                    if hasattr(plugin, 'suggest_privacy_improvements'):
                        suggestions = await plugin.suggest_privacy_improvements(text)
                    elif hasattr(plugin, 'suggest_citation_improvements'):
                        suggestions = await plugin.suggest_citation_improvements(text)
                    elif hasattr(plugin, 'suggest_safety_improvements'):
                        suggestions = await plugin.suggest_safety_improvements(text)
                    elif hasattr(plugin, 'suggest_harassment_improvements'):
                        suggestions = await plugin.suggest_harassment_improvements(text)
                    elif hasattr(plugin, 'suggest_retaliation_improvements'):
                        suggestions = await plugin.suggest_retaliation_improvements(text)
                    elif hasattr(plugin, 'suggest_transparency_improvements'):
                        suggestions = await plugin.suggest_transparency_improvements(text)
                    elif hasattr(plugin, 'suggest_privacy_harm_improvements'):
                        suggestions = await plugin.suggest_privacy_harm_improvements(text)
                    elif hasattr(plugin, 'suggest_public_interest_improvements'):
                        suggestions = await plugin.suggest_public_interest_improvements(text)
                    else:
                        suggestions = None

                    if suggestions and suggestions.success:
                        results["improvement_suggestions"][plugin_name] = suggestions.value

            except Exception as e:
                logger.error(f"Error analyzing {plugin_name}: {e}")
                results["analysis_results"][plugin_name] = {"error": str(e)}

        return results

    async def run_orchestrator_analysis(self, text: str, doc_name: str) -> Dict[str, Any]:
        """Run the full orchestrator analysis."""
        logger.info(f"Running orchestrator analysis for {doc_name}...")

        try:
            # Analyze draft for weak features
            weak_features = await self.orchestrator.analyze_draft(text)

            # Strengthen draft if weak features found
            improved_draft = None
            if weak_features:
                improved_draft = await self.orchestrator.strengthen_draft(text, weak_features)

            return {
                "weak_features": weak_features,
                "improved_draft": improved_draft,
                "has_improvements": bool(weak_features)
            }
        except Exception as e:
            logger.error(f"Orchestrator analysis failed: {e}")
            return {"error": str(e)}

    def generate_report(self, all_results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("DRAFT DOCUMENT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        for doc_name, results in all_results.items():
            report.append(f"📄 DOCUMENT: {doc_name}")
            report.append("-" * 60)

            # Basic stats
            if "basic_analysis" in results:
                basic = results["basic_analysis"]
                report.append(f"Word Count: {basic.get('word_count', 'N/A')}")
                report.append(f"Character Count: {basic.get('char_count', 'N/A')}")
                report.append(f"Citations Found: {len(basic.get('citations', []))}")
                report.append("")

            # Feature analysis
            if "feature_analysis" in results:
                features = results["feature_analysis"].get("features", {})
                report.append("🔍 FEATURE ANALYSIS:")
                for feature, value in features.items():
                    if isinstance(value, (int, float)) and value > 0:
                        report.append(f"  • {feature}: {value}")
                report.append("")

            # Plugin analysis results
            if "plugin_analysis" in results:
                plugin_results = results["plugin_analysis"].get("analysis_results", {})
                report.append("🧩 PLUGIN ANALYSIS RESULTS:")
                for plugin_name, analysis in plugin_results.items():
                    if "error" not in analysis:
                        report.append(f"  • {plugin_name}:")
                        for key, value in analysis.items():
                            if isinstance(value, (int, float)) and value > 0:
                                report.append(f"    - {key}: {value}")
                report.append("")

            # Improvement suggestions
            if "plugin_analysis" in results:
                suggestions = results["plugin_analysis"].get("improvement_suggestions", {})
                if suggestions:
                    report.append("💡 IMPROVEMENT SUGGESTIONS:")
                    for plugin_name, suggestion_data in suggestions.items():
                        report.append(f"  • {plugin_name}:")
                        if isinstance(suggestion_data, dict):
                            for key, value in suggestion_data.items():
                                if isinstance(value, list):
                                    for item in value:
                                        report.append(f"    - {item}")
                                else:
                                    report.append(f"    - {value}")
                        report.append("")

            # Orchestrator results
            if "orchestrator_analysis" in results:
                orch_results = results["orchestrator_analysis"]
                if "weak_features" in orch_results and orch_results["weak_features"]:
                    report.append("⚠️  WEAK FEATURES IDENTIFIED:")
                    for feature, data in orch_results["weak_features"].items():
                        report.append(f"  • {feature}: Current={data.get('current', 'N/A')}, Target={data.get('target', 'N/A')}")
                    report.append("")

                if orch_results.get("has_improvements"):
                    report.append("✅ DRAFT IMPROVEMENTS GENERATED")
                    report.append("")

            report.append("=" * 80)
            report.append("")

        return "\n".join(report)

async def main():
    """Main analysis function."""
    logger.info("Starting draft document analysis...")

    # Initialize analyzer
    analyzer = DraftAnalyzer()
    analyzer.setup_plugins()

    # Find user draft documents
    draft_paths = [
        Path("Lawsuit Data Analysis/Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Initial Filing/9-25/Word Docs/04 Ex Parte Motion.docx"),
        Path("Lawsuit Data Analysis/Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Initial Filing/9-25/Word Docs/05 Memorandum of Law.docx"),
        Path("Lawsuit Data Analysis/Harvard - The Art of War-20251007T134844Z-1-001/Harvard - The Art of War/Initial Filing/Combined.docx"),
    ]

    all_results = {}

    for doc_path in draft_paths:
        if doc_path.exists():
            logger.info(f"Processing {doc_path.name}...")

            # Extract text
            text = analyzer.extract_text_from_docx(doc_path)
            if not text:
                logger.warning(f"No text extracted from {doc_path.name}")
                continue

            # Basic analysis
            basic_analysis = analyzer.analyze_draft_features(text)

            # Plugin analysis
            plugin_analysis = await analyzer.analyze_draft_with_plugins(text, doc_path.name)

            # Orchestrator analysis
            orchestrator_analysis = await analyzer.run_orchestrator_analysis(text, doc_path.name)

            all_results[doc_path.name] = {
                "basic_analysis": basic_analysis,
                "plugin_analysis": plugin_analysis,
                "orchestrator_analysis": orchestrator_analysis
            }
        else:
            logger.warning(f"Document not found: {doc_path}")

    # Generate report
    report = analyzer.generate_report(all_results)

    # Save report
    report_path = Path("draft_analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Analysis complete! Report saved to {report_path}")
    print("\n" + report)

if __name__ == "__main__":
    asyncio.run(main())
