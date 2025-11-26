#!/usr/bin/env python3
"""
BN Query Mapper - Maps natural language questions to Bayesian Network queries.

This component:
- Extracts keywords/entities from questions
- Maps to BN nodes using ENTITY_TO_NODE_MAPPINGS + LLM
- Determines target nodes to query
- Constructs evidence dict for BN inference
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BNQuery:
    """A mapped BN query ready for inference."""
    evidence: Dict[str, str]  # node_id -> state
    target_nodes: List[str]   # Nodes to query for probabilities
    question_summary: str     # Summary of the original question
    confidence: float         # 0.0-1.0 confidence in mapping


class BNQueryMapper:
    """Maps natural language questions to BN queries."""

    def __init__(self, config_path: Optional[Path] = None, bn_model_path: Optional[Path] = None):
        """
        Initialize BN query mapper.

        Args:
            config_path: Optional path to BN node mappings JSON file
            bn_model_path: Optional path to BN model file (for extracting actual node names)
        """
        self.config_path = config_path or self._find_default_config()
        self.bn_model_path = bn_model_path or self._find_default_bn_model()
        self.mappings = self._load_mappings()
        self.bn_nodes = self._load_bn_nodes()
        logger.info(f"BNQueryMapper initialized with {len(self.mappings)} entity mappings")

    def _find_default_config(self) -> Path:
        """Find default configuration file path."""
        possible_paths = [
            Path(__file__).parent.parent / "config" / "bn_node_mappings.json",
            Path(__file__).parent.parent.parent / "writer_agents" / "config" / "bn_node_mappings.json",
        ]
        for path in possible_paths:
            if path.exists():
                return path
        return possible_paths[0]

    def _find_default_bn_model(self) -> Optional[Path]:
        """Find default BN model file path."""
        # Try multiple possible locations
        possible_paths = [
            # Relative to writer_agents/code
            Path(__file__).parent.parent.parent / "experiments" / "WizardWeb1.1.3.xdsl",
            Path(__file__).parent.parent.parent.parent / "experiments" / "WizardWeb1.1.3.xdsl",
            # From user rules
            Path(r"C:\Users\Owner\Desktop\WizardWeb\models\WizardWeb1.1.3.xdsl"),
            # Common locations
            Path(__file__).parent.parent.parent / "bayesian_network" / "code" / "experiments" / "WizardWeb1.1.3.xdsl",
            Path(__file__).parent.parent.parent / "Agents_1782_ML_Dataset" / "bayesian_network" / "code" / "experiments" / "WizardWeb1.1.3.xdsl",
            # Search for any .xdsl files
            Path(__file__).parent.parent.parent / "experiments" / "Witchweb113.xdsl",
            Path(__file__).parent.parent.parent / "bayesian_network" / "code" / "experiments" / "Witchweb113.xdsl",
        ]
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found BN model at: {path}")
                return path
        logger.warning("BN model file not found, will use mapping config only")
        return None

    def _load_mappings(self) -> Dict[str, str]:
        """Load entity-to-node mappings from config file."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Flatten nested structure
                    mappings = {}
                    entity_mappings = config.get("entity_to_node_mappings", {})
                    for category, entities in entity_mappings.items():
                        if isinstance(entities, dict):
                            mappings.update(entities)
                    return mappings
            except Exception as e:
                logger.warning(f"Could not load BN mappings: {e}")
        return self._get_default_mappings()

    def _get_default_mappings(self) -> Dict[str, str]:
        """Get default entity-to-node mappings."""
        return {
            "harvard": "Harvard_Involvement",
            "harvard university": "Harvard_Involvement",
            "harvard club": "Harvard_Club_Actions",
            "china": "PRC_Territory",
            "chinese": "PRC_Territory",
            "prc": "PRC_Territory",
            "ccp": "PRC_Government_Involvement",
            "national security": "National_Security_Risk",
            "national security matter": "National_Security_Risk",
            "sealing": "Sealing_Justification",
            "seal": "Sealing_Justification",
            "defamation": "Defamation_Claim",
            "lawsuit": "Litigation_Status",
            "statement 1": "Statement_1",
            "statement_1": "Statement_1",
        }

    def _load_bn_nodes(self) -> List[str]:
        """Load actual BN node names from model file if available."""
        if not self.bn_model_path or not self.bn_model_path.exists():
            # Return default nodes from mappings
            return list(set(self.mappings.values()))

        try:
            from .XdslParser import parse_xdsl
            xdsl_network = parse_xdsl(self.bn_model_path)
            nodes = list(xdsl_network.nodes.keys())
            logger.info(f"Loaded {len(nodes)} nodes from BN model: {self.bn_model_path}")
            return nodes
        except Exception as e:
            logger.warning(f"Could not parse BN model: {e}")
            return list(set(self.mappings.values()))

    def map_question_to_bn_query(self, question: str, context: Optional[str] = None) -> BNQuery:
        """
        Map a natural language question to a BN query.

        Args:
            question: User's question text
            context: Optional conversation context

        Returns:
            BNQuery with evidence dict, target nodes, and summary
        """
        question_lower = question.lower()

        # Extract entities from question
        entities = self._extract_entities(question_lower, context)

        # Map entities to BN nodes
        evidence = self._map_entities_to_evidence(entities, question_lower)

        # Determine target nodes (what the question is asking about)
        target_nodes = self._determine_target_nodes(question_lower, entities, evidence)

        # Extract additional evidence from question context
        evidence = self._extract_evidence_from_question(question_lower, context, evidence)

        # Generate summary
        question_summary = self._generate_summary(question, target_nodes, evidence)

        # Calculate confidence
        confidence = self._calculate_confidence(evidence, target_nodes, entities)

        return BNQuery(
            evidence=evidence,
            target_nodes=target_nodes,
            question_summary=question_summary,
            confidence=confidence
        )

    def _extract_entities(self, question_lower: str, context: Optional[str] = None) -> List[str]:
        """Extract entities/keywords from question."""
        entities = []

        # Check against mappings (case-insensitive)
        for entity, _ in self.mappings.items():
            if entity.lower() in question_lower:
                entities.append(entity)

        # Also check for common patterns
        patterns = [
            r"national security",
            r"statement\s+(\d+)",
            r"harvard",
            r"china|chinese|prc",
            r"sealing|seal",
            r"defamation",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, question_lower, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0).lower()
                if entity_text not in entities:
                    entities.append(entity_text)

        # Extract from context if available
        if context:
            context_lower = context.lower()
            for entity, _ in self.mappings.items():
                if entity.lower() in context_lower and entity not in entities:
                    entities.append(entity)

        return list(set(entities))  # Deduplicate

    def _map_entities_to_evidence(self, entities: List[str], question_lower: str) -> Dict[str, str]:
        """Map extracted entities to BN evidence dict."""
        evidence = {}

        for entity in entities:
            # Get mapped node
            mapped_node = None

            # Direct mapping
            if entity.lower() in self.mappings:
                mapped_node = self.mappings[entity.lower()]
            else:
                # Try case-insensitive partial match
                for key, value in self.mappings.items():
                    if key.lower() in entity.lower() or entity.lower() in key.lower():
                        mapped_node = value
                        break

            if mapped_node and mapped_node in self.bn_nodes:
                # Determine state based on context
                state = self._determine_entity_state(entity, question_lower, mapped_node)
                if state:
                    evidence[mapped_node] = state

        return evidence

    def _determine_entity_state(self, entity: str, question_lower: str, node: str) -> Optional[str]:
        """
        Determine the state value for an entity/node.

        Common patterns:
        - "present" / "involved" / "true" → "Present" or "Involved" or "True"
        - "not" / "absent" → "Absent" or "Not"
        - Default: "Present" for most nodes
        """
        # Check for negation
        entity_context = self._get_entity_context(entity, question_lower)

        if any(neg in entity_context for neg in ["not", "no", "absent", "lack of"]):
            # Try to find appropriate negative state
            if "Involved" in node or "Action" in node:
                return "Not_Involved"
            elif "Risk" in node:
                return "Low"
            else:
                return "Absent"

        # Default positive states
        if "Involvement" in node or "Action" in node:
            return "Involved"
        elif "Risk" in node:
            return "High"
        elif "Justification" in node:
            return "Justified"
        elif "Claim" in node:
            return "Present"
        elif "Status" in node:
            return "Active"
        elif "Evidence" in node or "Statement" in node:
            return "Present"
        else:
            return "Present"  # Safe default

    def _get_entity_context(self, entity: str, question_lower: str) -> str:
        """Get context around entity mention in question."""
        # Find position of entity
        entity_pos = question_lower.find(entity.lower())
        if entity_pos == -1:
            return ""

        # Extract 50 chars before and after
        start = max(0, entity_pos - 50)
        end = min(len(question_lower), entity_pos + len(entity) + 50)
        return question_lower[start:end]

    def _determine_target_nodes(self, question_lower: str, entities: List[str], evidence: Dict[str, str]) -> List[str]:
        """Determine which BN nodes to query for probabilities."""
        target_nodes = []

        # Check question for explicit target indicators
        if "national security" in question_lower or "national security matter" in question_lower:
            target_nodes.append("National_Security_Risk")

        if "sealing" in question_lower or "seal" in question_lower:
            target_nodes.append("Sealing_Justification")
            target_nodes.append("National_Security_Risk")  # Often related

        if "harvard" in question_lower:
            target_nodes.extend(["Harvard_Involvement", "Harvard_Club_Actions"])

        if "defamation" in question_lower:
            target_nodes.append("Defamation_Claim")

        if "probability" in question_lower or "percent" in question_lower or "chance" in question_lower:
            # If asking about probability but no specific target, use evidence nodes
            if not target_nodes and evidence:
                # Query the nodes we have evidence for
                target_nodes.extend(list(evidence.keys())[:3])  # Limit to first 3
            elif not target_nodes:
                # Default probability targets
                target_nodes.extend(["National_Security_Risk", "Sealing_Justification"])

        # Remove duplicates and ensure nodes exist in BN model
        target_nodes = [node for node in set(target_nodes) if node in self.bn_nodes]

        # If no targets found, use common default
        if not target_nodes:
            # Try to infer from evidence
            if evidence:
                target_nodes = list(evidence.keys())[:2]
            else:
                target_nodes = ["National_Security_Risk"]  # Safe default

        return target_nodes

    def _extract_evidence_from_question(self, question_lower: str, context: Optional[str],
                                       existing_evidence: Dict[str, str]) -> Dict[str, str]:
        """Extract additional evidence from question context."""
        evidence = existing_evidence.copy()

        # Check for "if factually plausible" or similar - indicates hypothetical evidence
        if "if" in question_lower or "assuming" in question_lower:
            # Extract condition after "if"
            if_match = re.search(r"if\s+([^,]+)", question_lower, re.IGNORECASE)
            if if_match:
                condition = if_match.group(1).strip()
                # Try to map condition to evidence
                for entity, node in self.mappings.items():
                    if entity.lower() in condition.lower():
                        if node in self.bn_nodes:
                            evidence[node] = "Present"  # Assume present for hypothetical

        # Check for statement numbers (e.g., "statement 1")
        statement_match = re.search(r"statement\s+(\d+)", question_lower, re.IGNORECASE)
        if statement_match:
            statement_num = statement_match.group(1)
            statement_node = f"Statement_{statement_num}"
            if statement_node in self.bn_nodes:
                evidence[statement_node] = "Present"

        return evidence

    def _generate_summary(self, question: str, target_nodes: List[str], evidence: Dict[str, str]) -> str:
        """Generate a summary of the mapped query."""
        parts = [
            f"Question: {question[:100]}",
            f"Target nodes: {', '.join(target_nodes)}",
            f"Evidence: {len(evidence)} nodes"
        ]
        return " | ".join(parts)

    def _calculate_confidence(self, evidence: Dict[str, str], target_nodes: List[str], entities: List[str]) -> float:
        """Calculate confidence in the mapping."""
        confidence = 0.5  # Base confidence

        # Boost for having evidence
        if evidence:
            confidence += 0.2

        # Boost for having target nodes
        if target_nodes:
            confidence += 0.2

        # Boost for entity matches
        if entities:
            confidence += min(len(entities) * 0.05, 0.1)

        return min(confidence, 1.0)

