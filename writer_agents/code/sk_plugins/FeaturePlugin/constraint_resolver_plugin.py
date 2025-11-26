#!/usr/bin/env python3
"""
Constraint Resolver Plugin - Resolves conflicts between hierarchical constraints.

This plugin coordinates constraint validation across multiple levels:
- Document level (word count, paragraph count, enumeration depth)
- Section level (section word count, section paragraph count, section enumeration depth)
- Paragraph level (sentences per paragraph)
- Sentence level (words per sentence)

It uses CatBoost feature importance scores and hierarchy weights to prioritize constraints
and automatically resolve conflicts when possible, flagging complex conflicts for manual review.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import (
    FunctionResult,
    EditRequest,
    DocumentLocation,
    PluginMetadata,
    kernel_function,
)
from .constraint_data_loader import get_loader
from .document_structure import parse_document_structure, DocumentStructure

logger = logging.getLogger(__name__)

class ConstraintLevel(Enum):
    """Hierarchy levels for constraints."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class Constraint:
    """Represents a single constraint."""
    level: ConstraintLevel
    constraint_type: str  # e.g., 'word_count', 'paragraph_count'
    section_name: Optional[str] = None  # None for document-level
    current_value: float = 0.0
    target_range: Tuple[float, float] = (0.0, 0.0)
    importance_score: float = 1.0  # From CatBoost feature importance
    plugin_name: str = ""
    gap: float = 0.0  # How far from target
    severity: str = "low"  # low, medium, high, critical


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""
    constraint: Constraint
    suggested_adjustment: Optional[Dict[str, Any]] = None


@dataclass
class ConflictResolution:
    """Result of conflict resolution."""
    resolved_constraints: List[Constraint] = field(default_factory=list)
    flagged_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    adjustments: List[EditRequest] = field(default_factory=list)


class ConstraintResolverPlugin(BaseFeaturePlugin):
    """Plugin to resolve conflicts between hierarchical constraints."""
    
    # Hierarchy weights: neutralize policy bias (all = 1.0) so CatBoost importances dominate
    HIERARCHY_WEIGHTS = {
        ConstraintLevel.DOCUMENT: 1.0,
        ConstraintLevel.SECTION: 1.0,
        ConstraintLevel.PARAGRAPH: 1.0,
        ConstraintLevel.SENTENCE: 1.0
    }
    
    # Priority threshold for auto-resolution (keep conservative)
    AUTO_RESOLVE_THRESHOLD = 0.3
    
    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "constraint_resolver", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("ConstraintResolverPlugin initialized")
        
        self.loader = get_loader()
        self.metadata = PluginMetadata(
            name="ConstraintResolverPlugin",
            description="Resolves hierarchical constraint conflicts across document levels",
            version="1.0.0",
            functions=["CollectConstraintViolations", "ResolveConstraintConflicts"]
        )

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ConstraintResolverPlugin",
            description="Resolves hierarchical constraint conflicts across document levels",
            version="1.0.0",
            functions=["CollectConstraintViolations", "ResolveConstraintConflicts"]
        )
    
    def _calculate_priority(self, constraint: Constraint) -> float:
        """
        Calculate priority score for a constraint.
        
        Priority = importance_score × hierarchy_weight
        """
        hierarchy_weight = self.HIERARCHY_WEIGHTS.get(constraint.level, 0.5)
        return constraint.importance_score * hierarchy_weight
    
    async def collect_all_constraints(
        self,
        document: "DocumentStructure",
        context: Dict[str, Any] = None
    ) -> List[ConstraintViolation]:
        """
        Collect all constraint violations from plugins.
        
        This method queries all relevant plugins to gather constraint violations.
        """
        violations = []
        
        # Get plugin registry
        try:
            from ...base_plugin import plugin_registry
            plugin_names = plugin_registry.list_plugins()
        except Exception as e:
            logger.warning(f"Could not access plugin registry: {e}")
            return violations
        
        # Collect constraints from different plugin types
        # Document-level constraints
        doc_plugins = [
            ("word_count", "word_count", ConstraintLevel.DOCUMENT),
            ("paragraph_count", "paragraph_count", ConstraintLevel.DOCUMENT),
            ("max_enumeration_depth", "enumeration_depth", ConstraintLevel.DOCUMENT),
        ]
        
        for plugin_name, constraint_type, level in doc_plugins:
            plugin = plugin_registry.get_plugin(plugin_name)
            if plugin and hasattr(plugin, f"validate_{plugin_name}"):
                try:
                    result = await getattr(plugin, f"validate_{plugin_name}")(document, context)
                    if result and result.data:
                        constraint = self._extract_constraint_from_result(
                            result.data, level, constraint_type, plugin_name
                        )
                        if constraint and constraint.gap > 0:
                            violations.append(ConstraintViolation(constraint=constraint))
                except Exception as e:
                    logger.debug(f"Error collecting constraint from {plugin_name}: {e}")
        
        # Section-level constraints
        sections = [
            'introduction', 'legal_standard', 'factual_background', 'privacy_harm',
            'danger_safety', 'public_interest', 'balancing_test', 'protective_measures', 'conclusion'
        ]
        
        section_constraint_types = [
            ('word_count', ConstraintLevel.SECTION),
            ('paragraph_structure', ConstraintLevel.PARAGRAPH),
            ('enumeration_depth', ConstraintLevel.SECTION),
            ('sentence_count', ConstraintLevel.PARAGRAPH),
            ('words_per_sentence', ConstraintLevel.SENTENCE),
        ]
        
        for section_name in sections:
            for constraint_type, level in section_constraint_types:
                plugin_name = f"{section_name}_{constraint_type}"
                plugin = plugin_registry.get_plugin(plugin_name)
                if plugin:
                    validate_method = f"validate_{section_name}_{constraint_type}"
                    if hasattr(plugin, validate_method):
                        try:
                            result = await getattr(plugin, validate_method)(document, context)
                            if result and result.data:
                                constraint = self._extract_constraint_from_result(
                                    result.data, level, constraint_type, plugin_name, section_name
                                )
                                if constraint and constraint.gap > 0:
                                    violations.append(ConstraintViolation(constraint=constraint))
                        except Exception as e:
                            logger.debug(f"Error collecting constraint from {plugin_name}: {e}")
        
        return violations
    
    def _extract_constraint_from_result(
        self,
        result_data: Dict[str, Any],
        level: ConstraintLevel,
        constraint_type: str,
        plugin_name: str,
        section_name: Optional[str] = None
    ) -> Optional[Constraint]:
        """Extract Constraint object from plugin validation result."""
        # Get importance score from CatBoost analysis (section-specific)
        importance = self.loader.get_feature_importance(section_name or "document", constraint_type)
        
        # Extract current value and target range
        current_value = result_data.get("word_count") or result_data.get("paragraph_count") or \
                       result_data.get("max_depth") or result_data.get("avg_sentences_per_paragraph") or \
                       result_data.get("avg_words_per_sentence") or result_data.get("enumeration_depth") or 0.0
        
        target_range = result_data.get("optimal_range") or result_data.get("target_range")
        
        # Guard: if target_range is missing or invalid, don't compute gaps (prevents mis-prioritization)
        if not target_range:
            logger.debug(f"No target_range found for {plugin_name}.{constraint_type} - skipping constraint extraction")
            return None
        
        # Normalize target_range to tuple
        if isinstance(target_range, list):
            if len(target_range) == 2:
                target_range = (float(target_range[0]), float(target_range[1]))
            else:
                logger.debug(f"Invalid target_range format for {plugin_name}.{constraint_type}: {target_range}")
                return None
        elif isinstance(target_range, tuple):
            if len(target_range) == 2:
                target_range = (float(target_range[0]), float(target_range[1]))
            else:
                logger.debug(f"Invalid target_range format for {plugin_name}.{constraint_type}: {target_range}")
                return None
        else:
            logger.debug(f"Invalid target_range type for {plugin_name}.{constraint_type}: {type(target_range)}")
            return None
        
        # Validate target_range values
        if target_range[0] < 0 or target_range[1] < 0 or target_range[0] > target_range[1]:
            logger.debug(f"Invalid target_range values for {plugin_name}.{constraint_type}: {target_range}")
            return None
        
        # Calculate gap only if target_range is valid
        gap = 0.0
        if target_range[0] > 0 or target_range[1] > 0:
            if current_value < target_range[0]:
                gap = target_range[0] - current_value
            elif current_value > target_range[1]:
                gap = current_value - target_range[1]
        
        # Get severity
        issues = result_data.get("issues", [])
        severity = "low"
        if issues:
            severity = issues[0].get("severity", "low")
        
        return Constraint(
            level=level,
            constraint_type=constraint_type,
            section_name=section_name,
            current_value=float(current_value),
            target_range=target_range,
            importance_score=importance,
            plugin_name=plugin_name,
            gap=gap,
            severity=severity
        )
    
    def resolve_constraints(
        self,
        violations: List[ConstraintViolation]
    ) -> ConflictResolution:
        """
        Resolve conflicts between constraints.
        
        Logic:
        1. Sort violations by priority (importance_score × hierarchy_weight)
        2. Group by constraint type
        3. Identify conflicts (e.g., section word count vs paragraph count)
        4. Auto-resolve: If one constraint has much higher priority, adjust lower one
        5. Flag complex: If priorities are close, report as needing manual review
        """
        if not violations:
            return ConflictResolution()
        
        # Sort by priority
        violations_with_priority = [
            (v, self._calculate_priority(v.constraint))
            for v in violations
        ]
        violations_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        # Group by constraint type and section
        constraint_groups: Dict[Tuple[str, Optional[str]], List[ConstraintViolation]] = {}
        for violation, priority in violations_with_priority:
            key = (violation.constraint.constraint_type, violation.constraint.section_name)
            if key not in constraint_groups:
                constraint_groups[key] = []
            constraint_groups[key].append(violation)
        
        resolved = []
        flagged = []
        recommendations = []
        
        # Process each group
        for (constraint_type, section_name), group_violations in constraint_groups.items():
            # Filter out violations with invalid target_ranges
            valid_violations = [
                v for v in group_violations
                if v.constraint.target_range and v.constraint.target_range[0] >= 0 and v.constraint.target_range[1] >= v.constraint.target_range[0]
            ]
            
            if not valid_violations:
                logger.debug(f"Skipping constraint group {constraint_type} in {section_name} - no valid target_ranges")
                continue
            
            if len(valid_violations) == 1:
                # Single violation - can auto-resolve
                violation = valid_violations[0]
                resolved.append(violation.constraint)
                recommendations.append({
                    "type": "constraint_adjustment",
                    "constraint_type": constraint_type,
                    "section": section_name,
                    "action": f"Adjust {constraint_type} to meet target range {violation.constraint.target_range}",
                    "priority": self._calculate_priority(violation.constraint)
                })
            else:
                # Multiple violations - check for conflicts
                priorities = [self._calculate_priority(v.constraint) for v in valid_violations]
                max_priority = max(priorities)
                min_priority = min(priorities)
                
                if max_priority - min_priority > self.AUTO_RESOLVE_THRESHOLD:
                    # Auto-resolve: prioritize highest priority constraint
                    highest = max(valid_violations, key=lambda v: self._calculate_priority(v.constraint))
                    resolved.append(highest.constraint)
                    recommendations.append({
                        "type": "constraint_resolution",
                        "constraint_type": constraint_type,
                        "section": section_name,
                        "action": f"Prioritizing highest priority constraint: {highest.constraint.plugin_name}",
                        "priority": self._calculate_priority(highest.constraint)
                    })
                else:
                    # Flag for manual review
                    flagged.append({
                        "constraint_type": constraint_type,
                        "section": section_name,
                        "violations": [
                            {
                                "plugin": v.constraint.plugin_name,
                                "priority": self._calculate_priority(v.constraint),
                                "gap": v.constraint.gap,
                                "target_range": v.constraint.target_range
                            }
                            for v in valid_violations
                        ],
                        "message": f"Multiple conflicting constraints for {constraint_type} in {section_name or 'document'}. Priorities are close, manual review needed."
                    })
        
        # Handle empty resolved list
        if not resolved:
            return ConflictResolution(
                resolved_constraints=[],
                flagged_conflicts=flagged,
                recommendations=recommendations
            )
        
        # Check if resolved contains ConstraintViolation objects or Constraint objects
        if resolved and isinstance(resolved[0], ConstraintViolation):
            resolved_constraints = [v.constraint for v in resolved]
        else:
            resolved_constraints = resolved
        
        return ConflictResolution(
            resolved_constraints=resolved_constraints,
            flagged_conflicts=flagged,
            recommendations=recommendations
        )
    
    async def resolve_all_constraints(
        self,
        document: "DocumentStructure",
        context: Dict[str, Any] = None
    ) -> FunctionResult:
        """
        Main entry point: collect and resolve all constraints.
        
        Returns FunctionResult with resolved constraints, flagged conflicts, and recommendations.
        """
        logger.info("Collecting all constraint violations...")
        violations = await self.collect_all_constraints(document, context)
        
        logger.info(f"Found {len(violations)} constraint violations")
        
        if not violations:
            return FunctionResult(
                success=True,
                data={
                    "violations_count": 0,
                    "resolved": [],
                    "flagged": [],
                    "recommendations": []
                },
                message="No constraint violations found"
            )
        
        # Resolve conflicts
        resolution = self.resolve_constraints(violations)
        
        logger.info(f"Resolved {len(resolution.resolved_constraints)} constraints, flagged {len(resolution.flagged_conflicts)} conflicts")
        
        return FunctionResult(
            success=len(resolution.flagged_conflicts) == 0,
            data={
                "violations_count": len(violations),
                "resolved_constraints": [
                    {
                        "level": c.level.value,
                        "type": c.constraint_type,
                        "section": c.section_name,
                        "current_value": c.current_value,
                        "target_range": c.target_range,
                        "gap": c.gap,
                        "priority": self._calculate_priority(c)
                    }
                    for c in resolution.resolved_constraints
                ],
                "flagged_conflicts": resolution.flagged_conflicts,
                "recommendations": resolution.recommendations,
                "summary": {
                    "total_violations": len(violations),
                    "auto_resolved": len(resolution.resolved_constraints),
                    "needs_manual_review": len(resolution.flagged_conflicts)
                }
            },
            message=f"Constraint resolution complete: {len(resolution.resolved_constraints)} auto-resolved, {len(resolution.flagged_conflicts)} flagged for review"
        )

    def _serialize_constraint(self, constraint: Constraint) -> Dict[str, Any]:
        return {
            "level": constraint.level.value,
            "type": constraint.constraint_type,
            "section": constraint.section_name,
            "current_value": constraint.current_value,
            "target_range": list(constraint.target_range),
            "gap": constraint.gap,
            "importance_score": constraint.importance_score,
            "plugin_name": constraint.plugin_name,
            "severity": constraint.severity,
        }

    def _serialize_violation(self, violation: ConstraintViolation) -> Dict[str, Any]:
        payload = {
            "constraint": self._serialize_constraint(violation.constraint),
        }
        if violation.suggested_adjustment:
            payload["suggested_adjustment"] = violation.suggested_adjustment
        return payload

    def _deserialize_context(self, context_json: Optional[str]) -> Dict[str, Any]:
        if not context_json:
            return {}
        if isinstance(context_json, dict):
            return context_json
        try:
            return json.loads(context_json)
        except Exception:
            logger.debug("Failed to parse context JSON for ConstraintResolverPlugin; using empty context")
            return {}

    def _parse_document(self, document_text: str) -> "DocumentStructure":
        return parse_document_structure(document_text or "")

    async def _register_functions(self) -> None:
        if not self.kernel:
            return

        @kernel_function(
            name="CollectConstraintViolations",
            description="Collect all constraint violations from the current draft"
        )
        async def collect_constraint_violations(
            document_text: str,
            context_json: str = ""
        ) -> str:
            document = self._parse_document(document_text)
            ctx = self._deserialize_context(context_json)
            violations = await self.collect_all_constraints(document, ctx)
            payload = {
                "violations_count": len(violations),
                "violations": [self._serialize_violation(v) for v in violations]
            }
            return json.dumps(payload)

        @kernel_function(
            name="ResolveConstraintConflicts",
            description="Resolve hierarchical constraint conflicts and return recommendations"
        )
        async def resolve_constraint_conflicts(
            document_text: str,
            context_json: str = ""
        ) -> str:
            document = self._parse_document(document_text)
            ctx = self._deserialize_context(context_json)
            result = await self.resolve_all_constraints(document, ctx)
            payload = {
                "success": result.success,
                "data": result.value or result.data,
                "message": result.metadata.get("message") if result.metadata else result.message
            }
            return json.dumps(payload)

        plugin_name = self.metadata.name if self.metadata else "ConstraintResolverPlugin"
        self.kernel.add_function(plugin_name=plugin_name, function=collect_constraint_violations)
        self.kernel.add_function(plugin_name=plugin_name, function=resolve_constraint_conflicts)
        self._functions["CollectConstraintViolations"] = collect_constraint_violations
        self._functions["ResolveConstraintConflicts"] = resolve_constraint_conflicts
