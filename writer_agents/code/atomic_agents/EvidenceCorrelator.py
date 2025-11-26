"""Evidence correlation atomic agent.

Specialized agent for cross-referencing evidence, detecting contradictions, and building evidence maps.
Uses LangChain for multi-document queries and LLM for correlation analysis.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from ..atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
    from ..langchain_integration import LangChainSQLAgent
except ImportError:
    from atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
    from langchain_integration import LangChainSQLAgent


class EvidenceCorrelatorAgent(LLMAgent):
    """Correlate evidence across documents and detect contradictions.

    Single duty: Find documents mentioning same entities/events and detect contradictions.
    Method: LangChain multi-document queries + LLM correlation analysis.
    Output: Evidence correlation map with supporting/contradicting evidence.
    """

    duty = "Find documents mentioning same entities/events and detect contradictions"
    cost_tier = "mini"
    max_cost_per_run = 0.015

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def __init__(self):
        super().__init__()
        self.langchain_agent: Optional[LangChainSQLAgent] = None

    def _initialize_langchain(self) -> None:
        """Initialize LangChain SQL agent if not already done."""
        if self.langchain_agent is None:
            try:
                from pathlib import Path
                db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")
                self.langchain_agent = LangChainSQLAgent(
                    db_path=db_path,
                    verbose=False
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize LangChain agent: {e}")

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build evidence correlation prompt.

        Args:
            input_data: Should contain 'entities', 'claim', 'query', or 'documents'

        Returns:
            Formatted prompt for evidence correlation analysis
        """
        entities = input_data.get('entities', [])
        claim = input_data.get('claim', '')
        query = input_data.get('query', '')
        documents = input_data.get('documents', [])

        if claim:
            # Analyze a specific claim for supporting/contradicting evidence
            prompt = f"""Analyze the following claim for supporting and contradicting evidence:

CLAIM: {claim}

Please:
1. Identify what evidence would support this claim
2. Identify what evidence would contradict this claim
3. Look for inconsistencies or contradictions in the evidence
4. Assess the strength of supporting vs. contradicting evidence
5. Note any gaps in the evidence

For each piece of evidence, indicate:
- Whether it supports or contradicts the claim
- The strength of the evidence (strong, moderate, weak)
- The source or context
- Any limitations or uncertainties

Format your analysis as:
SUPPORTING EVIDENCE:
- [Evidence item with strength and source]

CONTRADICTING EVIDENCE:
- [Evidence item with strength and source]

CONTRADICTIONS DETECTED:
- [Description of contradictions]

OVERALL ASSESSMENT:
- [Summary of evidence balance]"""

        elif entities:
            # Correlate evidence across multiple entities
            entities_str = ", ".join(entities)
            prompt = f"""Find correlations and relationships between evidence involving these entities: {entities_str}

Please:
1. Identify documents or evidence that mention multiple entities together
2. Look for patterns in how entities interact
3. Detect any contradictions in how entities are described
4. Find supporting evidence that connects the entities
5. Note any temporal relationships between entity interactions

For each correlation found, include:
- The entities involved
- The type of relationship (supporting, contradicting, neutral)
- The evidence source
- The strength of the correlation
- Any contradictions or inconsistencies

Format as:
ENTITY CORRELATIONS:
- [Entity A] ↔ [Entity B]: [Relationship description]

SUPPORTING EVIDENCE:
- [Evidence that supports the correlation]

CONTRADICTING EVIDENCE:
- [Evidence that contradicts the correlation]

PATTERNS DETECTED:
- [Summary of patterns found]"""

        elif query:
            # General evidence correlation query
            prompt = f"""Analyze the following query for evidence correlations and contradictions:

QUERY: {query}

Please:
1. Find all relevant evidence related to this query
2. Identify correlations between different pieces of evidence
3. Detect any contradictions or inconsistencies
4. Assess the overall strength and coherence of the evidence
5. Note any gaps or missing evidence

Focus on:
- Cross-references between documents
- Temporal consistency
- Entity relationships
- Factual contradictions
- Supporting vs. contradicting evidence

Format your analysis with clear sections for correlations, contradictions, and overall assessment."""

        elif documents:
            # Analyze specific documents for correlations
            docs_text = "\n\n---DOCUMENT SEPARATOR---\n\n".join(documents)
            prompt = f"""Analyze the following documents for evidence correlations and contradictions:

DOCUMENTS:
{docs_text}

Please:
1. Identify correlations between information in different documents
2. Detect contradictions or inconsistencies
3. Find supporting evidence across documents
4. Note any gaps or missing information
5. Assess the overall coherence of the evidence

For each correlation or contradiction found, specify:
- Which documents are involved
- The specific information being correlated/contradicted
- The strength of the correlation/contradiction
- The implications for the overall case

Format as:
CROSS-DOCUMENT CORRELATIONS:
- [Description of correlation with document references]

CONTRADICTIONS DETECTED:
- [Description of contradiction with document references]

SUPPORTING EVIDENCE:
- [Evidence that supports claims across documents]

EVIDENCE GAPS:
- [Missing information or gaps in evidence]"""

        else:
            prompt = """Please provide evidence correlation input. I can analyze:
- Specific claims for supporting/contradicting evidence
- Multiple entities for correlations
- General queries for evidence analysis
- Specific documents for cross-references

Please specify what you'd like me to analyze."""

        return prompt

    def _extract_evidence_correlations(self, query: str) -> Dict[str, Any]:
        """Use LangChain to extract evidence correlations from database."""
        if not self.langchain_agent:
            return {"error": "LangChain agent not available"}

        try:
            # Query for evidence correlations
            correlation_query = f"""
            Find all documents and evidence related to: {query}

            Focus on:
            - Documents that mention multiple entities together
            - Communications between parties
            - Statements or claims made
            - Contradictory information
            - Supporting evidence

            Extract:
            - Document content
            - Entities mentioned
            - Claims or statements made
            - Dates or temporal information
            - Source information
            """

            result = self.langchain_agent.query_evidence(correlation_query)
            return {"langchain_results": result}

        except Exception as e:
            self.logger.error(f"LangChain evidence correlation query failed: {e}")
            return {"error": str(e)}

    def _find_entity_co_occurrences(self, entities: List[str]) -> Dict[str, Any]:
        """Find documents where multiple entities appear together."""
        if not self.langchain_agent:
            return {"error": "LangChain agent not available"}

        try:
            # Build query for entity co-occurrences
            entity_conditions = " OR ".join([f"content LIKE '%{entity}%'" for entity in entities])
            co_occurrence_query = f"""
            Find documents that mention multiple entities together.

            Look for documents containing: {", ".join(entities)}

            Extract:
            - Documents that mention multiple entities
            - The specific entities mentioned together
            - The context of their co-occurrence
            - Any relationships or interactions described
            - Temporal information if available
            """

            result = self.langchain_agent.query_evidence(co_occurrence_query)
            return {"co_occurrence_results": result}

        except Exception as e:
            self.logger.error(f"Entity co-occurrence query failed: {e}")
            return {"error": str(e)}

    def _analyze_contradictions(self, llm_response: str) -> Dict[str, Any]:
        """Extract contradiction analysis from LLM response."""
        contradictions = {
            "contradictions": [],
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "correlations": [],
            "confidence_score": 0.0
        }

        try:
            lines = llm_response.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Identify sections
                if "CONTRADICTING EVIDENCE" in line.upper():
                    current_section = "contradicting"
                elif "SUPPORTING EVIDENCE" in line.upper():
                    current_section = "supporting"
                elif "CORRELATIONS" in line.upper():
                    current_section = "correlations"
                elif "CONTRADICTIONS DETECTED" in line.upper():
                    current_section = "contradictions"
                elif line.startswith('-') or line.startswith('•'):
                    # Extract evidence item
                    evidence_item = line[1:].strip()
                    if current_section == "contradicting":
                        contradictions["contradicting_evidence"].append(evidence_item)
                    elif current_section == "supporting":
                        contradictions["supporting_evidence"].append(evidence_item)
                    elif current_section == "correlations":
                        contradictions["correlations"].append(evidence_item)
                    elif current_section == "contradictions":
                        contradictions["contradictions"].append(evidence_item)

            # Calculate confidence score based on evidence balance
            supporting_count = len(contradictions["supporting_evidence"])
            contradicting_count = len(contradictions["contradicting_evidence"])
            total_evidence = supporting_count + contradicting_count

            if total_evidence > 0:
                contradictions["confidence_score"] = abs(supporting_count - contradicting_count) / total_evidence

        except Exception as e:
            self.logger.error(f"Contradiction analysis failed: {e}")
            contradictions["error"] = str(e)

        return contradictions

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evidence correlation analysis.

        Args:
            input_data: Input containing entities, claim, query, or documents

        Returns:
            Evidence correlation analysis with contradictions and supporting evidence
        """
        try:
            # Initialize LangChain if needed
            self._initialize_langchain()

            # Build prompt
            prompt = self._build_prompt(input_data)

            # Get LLM response
            llm_response = self._call_llm(prompt)

            # Extract evidence correlations using LangChain
            langchain_data = {}
            if input_data.get('query'):
                langchain_data = self._extract_evidence_correlations(input_data['query'])
            elif input_data.get('entities'):
                langchain_data = self._find_entity_co_occurrences(input_data['entities'])

            # Analyze contradictions from LLM response
            contradiction_analysis = self._analyze_contradictions(llm_response)

            # Combine results
            correlation_result = {
                "success": True,
                "contradiction_analysis": contradiction_analysis,
                "langchain_data": langchain_data,
                "raw_llm_response": llm_response,
                "entities_analyzed": input_data.get('entities', []),
                "claim_analyzed": input_data.get('claim', ''),
                "query_analyzed": input_data.get('query', '')
            }

            return correlation_result

        except Exception as e:
            self.logger.error(f"Evidence correlation analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "contradiction_analysis": {"error": str(e)}
            }


