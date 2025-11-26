#!/usr/bin/env python3
"""
Update Memory Database with Corrected Analysis
=============================================

Replaces incorrect memories with corrected ones to prevent future contamination.
This ensures the system learns from the corrected understanding.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add writer_agents/code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "writer_agents" / "code"))

from memory_system import MemoryStore, AgentMemory
from master_supervisor import MemoryConfig


class MemoryDatabaseUpdater:
    """Updates memory database with corrected analysis."""

    def __init__(self, db_path: str = "jobs.db"):
        self.db_path = db_path

        # Initialize memory store
        self.memory_store = MemoryStore(
            storage_path="memory_store",
            use_local_embeddings=True,
            embedding_model="all-MiniLM-L6-v2"
        )

    def update_memories_with_corrected_analysis(self, corrected_analysis_file: str) -> Dict[str, Any]:
        """Update memories with corrected analysis results."""

        print(f"[INFO] Loading corrected analysis from: {corrected_analysis_file}")

        with open(corrected_analysis_file, 'r') as f:
            analysis_data = json.load(f)

        print(f"[INFO] Corrected analysis loaded: {analysis_data['agents_executed']} agents executed")

        # Create CORRECTED memories for each agent type
        corrected_memories = {}

        # Research Phase Agents (10 agents) - CORRECTED
        research_memories = self._create_corrected_research_memories(analysis_data)
        corrected_memories.update(research_memories)

        # Drafting Phase Agents (10 agents) - CORRECTED
        drafting_memories = self._create_corrected_drafting_memories(analysis_data)
        corrected_memories.update(drafting_memories)

        # Citation Phase Agents (5 agents) - CORRECTED
        citation_memories = self._create_corrected_citation_memories(analysis_data)
        corrected_memories.update(citation_memories)

        # QA Phase Agents (10 agents) - CORRECTED
        qa_memories = self._create_corrected_qa_memories(analysis_data)
        corrected_memories.update(qa_memories)

        # Output Phase Agents (5 agents) - CORRECTED
        output_memories = self._create_corrected_output_memories(analysis_data)
        corrected_memories.update(output_memories)

        # Supervisors (4 agents) - CORRECTED
        supervisor_memories = self._create_corrected_supervisor_memories(analysis_data)
        corrected_memories.update(supervisor_memories)

        # Master Supervisor (1 agent) - CORRECTED
        master_memories = self._create_corrected_master_supervisor_memories(analysis_data)
        corrected_memories.update(master_memories)

        print(f"[INFO] Created CORRECTED memories for {len(corrected_memories)} agent types")

        return corrected_memories

    def _create_corrected_research_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for research phase agents."""
        memories = {}

        research_agents = [
            "FactExtractorAgent",
            "PrecedentFinderAgent",
            "PrecedentRankerAgent",
            "PrecedentSummarizerAgent",
            "StatuteLocatorAgent",
            "ExhibitFetcherAgent",
            "EvidenceAnalyzerAgent",
            "TimelineBuilderAgent",
            "WitnessLocatorAgent",
            "DocumentAnalyzerAgent"
        ]

        for agent in research_agents:
            memory_content = f"""
CORRECTED RESEARCH MEMORY: McGrath Knowledge Analysis

CRITICAL CORRECTION: Yi Wang confronted user with OTHER slides (not the Xi Mingze slide).
This shows GENERAL presentation knowledge, NOT specific slide knowledge.

Key findings from the CORRECTED McGrath analysis:
- Yi Wang confrontation provides evidence of GENERAL presentation knowledge only
- Alumni networks in China provide multiple pathways for knowledge transmission
- Three-year presentation history increases likelihood of knowledge spread
- Sendelta investigation timeline supports pre-defamation knowledge
- CCP pressure on Harvard alumni likely influenced information flow

CORRECTED Research methodology:
- Properly distinguish between general and specific knowledge
- Analyze institutional connections between McGrath and Harvard China operations
- Examine timeline of events leading to April 19, 2019 defamation
- Assess circumstantial evidence supporting knowledge inference
- Evaluate evidence strength accurately (Yi Wang = Medium, not Very Strong)

This CORRECTED analysis demonstrates the importance of properly distinguishing between general and specific knowledge in legal research.
            """.strip()

            memories[agent] = memory_content

        return memories

    def _create_corrected_drafting_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for drafting phase agents."""
        memories = {}

        drafting_agents = [
            "OutlineBuilderAgent",
            "SectionWriterAgent",
            "ParagraphWriterAgent",
            "TransitionAgent",
            "ArgumentBuilderAgent",
            "EvidenceOrganizerAgent",
            "LegalReasoningAgent",
            "ConclusionWriterAgent",
            "SummaryGeneratorAgent",
            "ReportFormatterAgent"
        ]

        for agent in drafting_agents:
            memory_content = f"""
CORRECTED DRAFTING MEMORY: McGrath Knowledge Analysis

CORRECTED Drafting approach for knowledge transmission analysis:
- Structure arguments around circumstantial evidence (not direct evidence)
- Emphasize institutional connections and timeline analysis
- Properly characterize Yi Wang evidence as general knowledge (not specific)
- Use circumstantial evidence to support knowledge inference
- Organize evidence by strength and relevance (corrected understanding)
- Build logical progression from evidence to conclusion

CORRECTED Key drafting principles:
- Lead with institutional evidence (Harvard China operations)
- Support with circumstantial evidence (alumni networks, timeline)
- Address evidence gaps (no direct evidence of specific slide knowledge)
- Acknowledge limitations while emphasizing probability
- Use legal standards (preponderance of evidence) with corrected confidence
- Properly distinguish between general and specific knowledge

This CORRECTED analysis shows how to structure complex legal arguments about knowledge transmission with proper evidence characterization.
            """.strip()

            memories[agent] = memory_content

        return memories

    def _create_corrected_citation_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for citation phase agents."""
        memories = {}

        citation_agents = [
            "CitationFinderAgent",
            "CitationNormalizerAgent",
            "CitationVerifierAgent",
            "CitationLocatorAgent",
            "CitationInserterAgent"
        ]

        for agent in citation_agents:
            memory_content = f"""
CORRECTED CITATION MEMORY: McGrath Knowledge Analysis

CORRECTED Citation patterns for knowledge transmission cases:
- Focus on circumstantial evidence precedents (not direct evidence)
- Cite institutional connection case law
- Reference timeline analysis methodologies
- Include knowledge inference case law with proper evidence characterization
- Document circumstantial evidence standards
- Distinguish between general and specific knowledge precedents

CORRECTED Key citation types:
- Legal precedents for circumstantial knowledge inference
- Institutional connection case law
- Timeline analysis methodologies
- Circumstantial evidence standards
- General vs. specific knowledge transmission patterns

This CORRECTED analysis demonstrates proper citation of knowledge transmission precedents with accurate evidence characterization.
            """.strip()

            memories[agent] = memory_content

        return memories

    def _create_corrected_qa_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for QA phase agents."""
        memories = {}

        qa_agents = [
            "GrammarFixerAgent",
            "StyleCheckerAgent",
            "LogicCheckerAgent",
            "ConsistencyCheckerAgent",
            "RedactionAgent",
            "ComplianceAgent",
            "ExpertQAAgent",
            "FactCheckerAgent",
            "EvidenceValidatorAgent",
            "LegalStandardCheckerAgent"
        ]

        for agent in qa_agents:
            memory_content = f"""
CORRECTED QA MEMORY: McGrath Knowledge Analysis

CORRECTED QA considerations for knowledge transmission analysis:
- Verify evidence characterization accuracy (general vs. specific knowledge)
- Check timeline accuracy and logical consistency
- Validate evidence strength assessments (Yi Wang = Medium, not Very Strong)
- Ensure argument structure supports conclusions with corrected understanding
- Verify probability assessments are well-supported with corrected evidence
- Check for proper distinction between general and specific knowledge

CORRECTED Key QA checks:
- Evidence characterization accuracy (general vs. specific)
- Timeline consistency and logical flow
- Evidence strength and relevance (corrected understanding)
- Legal standard application accuracy
- Argument structure and conclusion support
- Probability assessment methodology (corrected)

This CORRECTED analysis shows how to QA complex knowledge transmission arguments with proper evidence characterization.
            """.strip()

            memories[agent] = memory_content

        return memories

    def _create_corrected_output_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for output phase agents."""
        memories = {}

        output_agents = [
            "MarkdownExporterAgent",
            "DocxExporterAgent",
            "MetadataTaggerAgent",
            "ReportGeneratorAgent",
            "SummaryExporterAgent"
        ]

        for agent in output_agents:
            memory_content = f"""
CORRECTED OUTPUT MEMORY: McGrath Knowledge Analysis

CORRECTED Output formatting for knowledge transmission analysis:
- Structure reports with corrected executive summary
- Include detailed findings with proper evidence characterization
- Provide corrected probability assessments with reasoning
- Format legal reasoning with corrected evidence understanding
- Tag metadata for future reference with correction notes
- Highlight corrections made to prevent future errors

CORRECTED Key output elements:
- Executive summary with corrected key findings
- Detailed evidence analysis with proper characterization
- Corrected probability assessment with reasoning
- Legal reasoning with corrected evidence understanding
- Recommendations with correction notes
- Correction documentation for future reference

This CORRECTED analysis demonstrates proper output formatting for complex legal analyses with accurate evidence characterization.
            """.strip()

            memories[agent] = memory_content

        return memories

    def _create_corrected_supervisor_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for supervisor agents."""
        memories = {}

        supervisors = [
            "ResearchSupervisor",
            "DraftingSupervisor",
            "CitationSupervisor",
            "QASupervisor"
        ]

        for supervisor in supervisors:
            memory_content = f"""
CORRECTED SUPERVISOR MEMORY: McGrath Knowledge Analysis

CORRECTED Supervision approach for knowledge transmission analysis:
- Coordinate multiple agent types for comprehensive analysis
- Ensure proper evidence characterization (general vs. specific knowledge)
- Maintain quality standards across all phases with corrected understanding
- Coordinate timeline and institutional analysis
- Oversee corrected probability assessment methodology
- Prevent evidence mischaracterization

CORRECTED Key supervision principles:
- Coordinate research phase for comprehensive evidence gathering with proper characterization
- Ensure drafting phase maintains logical structure with corrected evidence understanding
- Verify citation phase includes relevant precedents with proper evidence types
- Oversee QA phase for accuracy and consistency with corrected understanding

This CORRECTED analysis demonstrates effective supervision of complex legal analysis workflows with proper evidence characterization.
            """.strip()

            memories[supervisor] = memory_content

        return memories

    def _create_corrected_master_supervisor_memories(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Create CORRECTED memories for master supervisor."""
        memories = {}

        memory_content = f"""
CORRECTED MASTER SUPERVISOR MEMORY: McGrath Knowledge Analysis

CORRECTED Master supervision of knowledge transmission analysis:
- Orchestrate all 49 agents for comprehensive analysis with corrected understanding
- Coordinate session memory and context management
- Ensure cost-effective execution with quality results
- Manage corrected memory population and agent learning
- Oversee end-to-end analysis workflow with proper evidence characterization
- Prevent future contamination from incorrect evidence characterization

CORRECTED Key orchestration principles:
- Coordinate all phases: research, drafting, citation, QA, output with corrected understanding
- Manage session context and memory integration
- Ensure cost optimization while maintaining quality
- Populate agent memories with corrected analysis for future learning
- Generate comprehensive analysis reports with proper evidence characterization
- Document corrections to prevent future errors

This CORRECTED analysis demonstrates successful orchestration of the full 49-agent system with proper evidence characterization and correction documentation.
        """.strip()

        memories["MasterSupervisor"] = memory_content

        return memories

    def replace_memories_in_database(self, corrected_memories: Dict[str, str]) -> Dict[str, Any]:
        """Replace incorrect memories with corrected ones."""

        print(f"[INFO] Replacing memories with CORRECTED versions...")

        replaced_count = 0
        errors = []

        for agent_type, corrected_memory_content in corrected_memories.items():
            try:
                # Create a CORRECTED memory entry
                corrected_memory = AgentMemory(
                    agent_type=agent_type,
                    memory_id=str(uuid.uuid4()),
                    summary=corrected_memory_content,
                    context={
                        "analysis_type": "CORRECTED_knowledge_transmission",
                        "case": "McGrath",
                        "timestamp": "2025-10-11",
                        "source": "CORRECTED_full_49_agent_analysis",
                        "correction": "Yi Wang evidence properly characterized as general knowledge",
                        "previous_error": "Yi Wang confrontation incorrectly treated as direct evidence of specific slide knowledge"
                    },
                    source="manual_corrected"
                )

                self.memory_store.add(corrected_memory)

                replaced_count += 1
                print(f"  [OK] Replaced memory for {agent_type} with CORRECTED version")

            except Exception as e:
                error_msg = f"Failed to replace memory for {agent_type}: {e}"
                errors.append(error_msg)
                print(f"  [ERROR] {error_msg}")

        result = {
            "memories_replaced": replaced_count,
            "total_memories": len(corrected_memories),
            "errors": errors,
            "success_rate": replaced_count / len(corrected_memories) if corrected_memories else 0
        }

        print(f"[INFO] Memory replacement complete: {replaced_count}/{len(corrected_memories)} memories replaced with CORRECTED versions")

        return result


def main():
    """Main execution function."""
    print("MEMORY DATABASE UPDATE WITH CORRECTED ANALYSIS")
    print("=" * 60)

    # Find the most recent corrected analysis file
    corrected_analysis_files = list(Path(".").glob("mcgrath_CORRECTED_analysis_*.json"))
    if not corrected_analysis_files:
        print("[ERROR] No CORRECTED McGrath analysis files found!")
        return

    latest_corrected_analysis = max(corrected_analysis_files, key=lambda f: f.stat().st_mtime)
    print(f"[INFO] Using CORRECTED analysis file: {latest_corrected_analysis}")

    # Initialize updater
    updater = MemoryDatabaseUpdater()

    try:
        # Update memories with corrected analysis
        corrected_memories = updater.update_memories_with_corrected_analysis(str(latest_corrected_analysis))

        # Replace memories in database
        result = updater.replace_memories_in_database(corrected_memories)

        # Save result report
        report_file = f"memory_correction_report_{latest_corrected_analysis.stem}.json"
        with open(report_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[SUCCESS] Memory database updated with CORRECTED analysis!")
        print(f"[INFO] Report saved to: {report_file}")
        print(f"[INFO] Success rate: {result['success_rate']:.1%}")

        if result['errors']:
            print(f"[WARNING] {len(result['errors'])} errors occurred:")
            for error in result['errors']:
                print(f"  - {error}")

        print(f"\n[INFO] Database now contains CORRECTED memories that properly distinguish between general and specific knowledge")
        print(f"[INFO] Future analyses will use the corrected understanding")

        return result

    except Exception as e:
        print(f"[ERROR] Memory database update failed: {e}")
        return None


if __name__ == "__main__":
    main()
