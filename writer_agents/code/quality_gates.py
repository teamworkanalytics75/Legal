"""Quality gate definitions and validation pipeline for hybrid orchestration."""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)

INTRO_SECTION_PATTERNS = [
    "introduction",
    "background",
    "factual background",
    "case background",
]

ANALYSIS_SECTION_PATTERNS = [
    "analysis",
    "legal standard",
    "argument",
    "legal argument",
    "discussion",
    "privacy harm",
    "balancing test",
]

CONCLUSION_SECTION_PATTERNS = [
    "conclusion",
    "requested relief",
    "prayer for relief",
    "wherefore",
]

DEFAULT_SECTION_ALIASES = {
    "introduction": INTRO_SECTION_PATTERNS,
    "background": INTRO_SECTION_PATTERNS,
    "factual background": INTRO_SECTION_PATTERNS,
    "case background": INTRO_SECTION_PATTERNS,
    "analysis": ANALYSIS_SECTION_PATTERNS,
    "argument": ANALYSIS_SECTION_PATTERNS,
    "legal standard": ANALYSIS_SECTION_PATTERNS,
    "discussion": ANALYSIS_SECTION_PATTERNS,
    "privacy harm": ANALYSIS_SECTION_PATTERNS,
    "balancing test": ANALYSIS_SECTION_PATTERNS,
    "conclusion": CONCLUSION_SECTION_PATTERNS,
    "requested relief": CONCLUSION_SECTION_PATTERNS,
    "prayer for relief": CONCLUSION_SECTION_PATTERNS,
    "wherefore": CONCLUSION_SECTION_PATTERNS,
}


class ValidationSeverity(Enum):
    """Severity levels for validation failures."""
    CRITICAL = "critical"    # Must fix - blocks commit
    HIGH = "high"           # Should fix - affects quality
    MEDIUM = "medium"       # Consider fixing - minor issues
    LOW = "low"             # Optional - style preferences


@dataclass
class QualityGate:
    """Configuration for a quality gate."""

    name: str
    description: str
    sk_function: str  # SK validation function name
    threshold: float  # Minimum score to pass (0.0-1.0)
    severity: ValidationSeverity = ValidationSeverity.HIGH
    required: bool = True
    weight: float = 1.0  # Weight in overall score calculation
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class ValidationResult:
    """Result from quality gate validation."""

    gate_name: str
    passed: bool
    score: float
    threshold: float
    details: str
    suggestions: List[str]
    severity: ValidationSeverity
    error: Optional[str] = None


@dataclass
class QualityGateResults:
    """Results from running all quality gates."""

    overall_score: float
    passed_gates: List[str]
    failed_gates: List[str]
    critical_failures: List[str]
    gate_results: Dict[str, ValidationResult]
    suggestions: List[str]
    can_proceed: bool  # Whether workflow can proceed to commit

    def get_summary(self) -> str:
        """Get human-readable summary of results."""
        total_gates = len(self.gate_results)
        passed_count = len(self.passed_gates)
        failed_count = len(self.failed_gates)
        critical_count = len(self.critical_failures)

        summary = f"""
Quality Gate Results Summary:
- Overall Score: {self.overall_score:.2f}
- Gates Passed: {passed_count}/{total_gates}
- Gates Failed: {failed_count}/{total_gates}
- Critical Failures: {critical_count}
- Can Proceed: {'Yes' if self.can_proceed else 'No'}

Failed Gates: {', '.join(self.failed_gates) if self.failed_gates else 'None'}
Critical Issues: {', '.join(self.critical_failures) if self.critical_failures else 'None'}
        """.strip()

        return summary


# Quality gate definitions
QUALITY_GATES = [
    QualityGate(
        name="citation_validity",
        description="Validate citation format and completeness",
        sk_function="ValidateCitationFormat",
        threshold=1.0,
        severity=ValidationSeverity.CRITICAL,
        required=True,
        weight=2.0,
        parameters={
            "required_format": "[Node:State]",
            "min_citations": 1
        }
    ),

    QualityGate(
        name="structure_complete",
        description="Verify all required sections are present",
        sk_function="ValidateStructure",
        threshold=1.0,
        severity=ValidationSeverity.CRITICAL,
        required=True,
        weight=2.0,
        parameters={
            "required_sections": [
                {
                    "label": "Introduction/Background",
                    "patterns": INTRO_SECTION_PATTERNS,
                },
                {
                    "label": "Legal Analysis/Argument",
                    "patterns": ANALYSIS_SECTION_PATTERNS,
                },
                {
                    "label": "Conclusion/Relief",
                    "patterns": CONCLUSION_SECTION_PATTERNS,
                },
            ],
            "min_word_count": 500
        }
    ),

    QualityGate(
        name="evidence_grounding",
        description="Ensure all claims have supporting evidence",
        sk_function="ValidateEvidenceGrounding",
        threshold=0.9,
        severity=ValidationSeverity.HIGH,
        required=True,
        weight=1.5,
        parameters={
            "min_evidence_coverage": 0.8,
            "require_citations": True
        }
    ),

    QualityGate(
        name="tone_consistency",
        description="Maintain professional legal tone throughout",
        sk_function="ValidateToneConsistency",
        threshold=0.85,
        severity=ValidationSeverity.MEDIUM,
        required=False,
        weight=1.0,
        parameters={
            "formal_language": True,
            "avoid_informal": True
        }
    ),

    QualityGate(
        name="argument_coherence",
        description="Ensure logical flow and argument coherence",
        sk_function="ValidateArgumentCoherence",
        threshold=0.8,
        severity=ValidationSeverity.HIGH,
        required=True,
        weight=1.5,
        parameters={
            "check_transitions": True,
            "verify_conclusions": True
        }
    ),

    QualityGate(
        name="legal_accuracy",
        description="Verify legal framework and precedent accuracy",
        sk_function="ValidateLegalAccuracy",
        threshold=0.85,
        severity=ValidationSeverity.HIGH,
        required=True,
        weight=1.5,
        parameters={
            "jurisdiction_specific": True,
            "check_precedents": True
        }
    ),

    QualityGate(
        name="privacy_harm_checklist",
        description="Domain-specific privacy harm validation",
        sk_function="ValidatePrivacyHarmChecklist",
        threshold=0.9,
        severity=ValidationSeverity.HIGH,
        required=True,
        weight=1.5,
        parameters={
            "harm_factors": ["intrusion", "disclosure", "misuse", "damage"],
            "legal_elements": ["conduct", "harm", "causation"]
        }
    ),

    QualityGate(
        name="grammar_spelling",
        description="Basic grammar and spelling check",
        sk_function="ValidateGrammarSpelling",
        threshold=0.95,
        severity=ValidationSeverity.MEDIUM,
        required=False,
        weight=0.5,
        parameters={
            "check_grammar": True,
            "check_spelling": True
        }
    )
]


class QualityGateRunner:
    """Runs quality gates on documents."""

    def __init__(self, sk_kernel=None):
        self.sk_kernel = sk_kernel
        self.gates = QUALITY_GATES

    async def run_all_gates(
        self,
        document: str,
        context: Dict[str, Any],
        gates: Optional[List[QualityGate]] = None
    ) -> QualityGateResults:
        """
        Run all quality gates on a document.

        Args:
            document: Document text to validate
            context: Context variables (evidence, posteriors, etc.)
            gates: Specific gates to run (default: all gates)

        Returns:
            QualityGateResults with comprehensive validation results
        """

        gates_to_run = gates or self.gates
        gate_results = {}
        passed_gates = []
        failed_gates = []
        critical_failures = []
        suggestions = []

        total_weight = 0.0
        weighted_score = 0.0

        logger.info(f"Running {len(gates_to_run)} quality gates")

        for gate in gates_to_run:
            try:
                result = await self._run_single_gate(gate, document, context)
                gate_results[gate.name] = result

                if result.passed:
                    passed_gates.append(gate.name)
                else:
                    failed_gates.append(gate.name)
                    suggestions.extend(result.suggestions)

                    if result.severity == ValidationSeverity.CRITICAL:
                        critical_failures.append(gate.name)

                # Calculate weighted score
                weighted_score += result.score * gate.weight
                total_weight += gate.weight

            except Exception as e:
                logger.error(f"Error running gate {gate.name}: {e}")

                # Create error result
                error_result = ValidationResult(
                    gate_name=gate.name,
                    passed=False,
                    score=0.0,
                    threshold=gate.threshold,
                    details=f"Gate execution failed: {e}",
                    suggestions=[f"Fix technical issue with {gate.name}"],
                    severity=gate.severity,
                    error=str(e)
                )

                gate_results[gate.name] = error_result
                failed_gates.append(gate.name)
                critical_failures.append(gate.name)
                total_weight += gate.weight

        # Calculate overall score
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

        # Determine if workflow can proceed
        can_proceed = len(critical_failures) == 0

        results = QualityGateResults(
            overall_score=overall_score,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            critical_failures=critical_failures,
            gate_results=gate_results,
            suggestions=suggestions,
            can_proceed=can_proceed
        )

        logger.info(f"Quality gates completed. Overall score: {overall_score:.2f}")
        return results

    async def _run_single_gate(
        self,
        gate: QualityGate,
        document: str,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Run a single quality gate."""

        try:
            if self.sk_kernel:
                # Use SK function if kernel is available
                variables = self._prepare_sk_variables(document, context, gate)
                result = await self.sk_kernel.invoke_function(
                    plugin_name="ValidationPlugin",
                    function_name=gate.sk_function,
                    variables=variables
                )

                payload = self._parse_sk_result(result)
                score = float(payload.get("score", 0.0))
                details = payload.get("details", "")
                suggestions = payload.get("suggestions", [])

            else:
                # Fallback to basic validation
                score, details, suggestions = await self._basic_validation(gate, document, context)

            passed = score >= gate.threshold

            return ValidationResult(
                gate_name=gate.name,
                passed=passed,
                score=score,
                threshold=gate.threshold,
                details=details,
                suggestions=suggestions,
                severity=gate.severity
            )

        except Exception as e:
            logger.error(f"Error in gate {gate.name}: {e}")
            return ValidationResult(
                gate_name=gate.name,
                passed=False,
                score=0.0,
                threshold=gate.threshold,
                details=f"Validation failed: {e}",
                suggestions=[f"Fix technical issue with {gate.name}"],
                severity=gate.severity,
                error=str(e)
            )

    async def _basic_validation(
        self,
        gate: QualityGate,
        document: str,
        context: Dict[str, Any]
    ) -> tuple[float, str, List[str]]:
        """Basic validation fallback when SK functions aren't available."""

        if gate.name == "citation_validity":
            return self._validate_citations_basic(document)
        elif gate.name == "structure_complete":
            return self._validate_structure_basic(document)
        elif gate.name == "evidence_grounding":
            return self._validate_evidence_basic(document, context)
        elif gate.name == "tone_consistency":
            return self._validate_tone_basic(document)
        else:
            # Default: assume passed
            return 1.0, "Basic validation passed", []

    def _validate_citations_basic(self, document: str) -> tuple[float, str, List[str]]:
        """Basic citation validation."""
        import re

        # Look for [Node:State] pattern
        citation_pattern = r'\[[^:]+:[^\]]+\]'
        citations = re.findall(citation_pattern, document)

        if not citations:
            return 0.0, "No citations found", ["Add evidence citations in [Node:State] format"]

        # Check citation quality
        valid_citations = [c for c in citations if ':' in c and len(c) > 3]
        score = len(valid_citations) / len(citations)

        details = f"Found {len(valid_citations)}/{len(citations)} valid citations"
        suggestions = []

        if score < 1.0:
            suggestions.append("Fix citation format to [Node:State]")

        return score, details, suggestions

    def _validate_structure_basic(self, document: str) -> tuple[float, str, List[str]]:
        """Basic structure validation."""
        required_sections = [
            ("Introduction/Background", INTRO_SECTION_PATTERNS),
            ("Legal Analysis/Argument", ANALYSIS_SECTION_PATTERNS),
            ("Conclusion/Relief", CONCLUSION_SECTION_PATTERNS),
        ]

        doc_lower = document.lower()
        found_sections = []
        missing_sections = []
        for label, patterns in required_sections:
            if self._doc_has_patterns(doc_lower, patterns):
                found_sections.append(label)
            else:
                missing_sections.append(label)

        score = len(found_sections) / len(required_sections)
        details = f"Found {len(found_sections)}/{len(required_sections)} required sections"

        suggestions = []
        if missing_sections:
            suggestions.append(f"Add missing sections: {', '.join(missing_sections)}")

        return score, details, suggestions

    def _validate_evidence_basic(self, document: str, context: Dict[str, Any]) -> tuple[float, str, List[str]]:
        """Basic evidence grounding validation."""
        evidence = context.get("evidence", {})

        if not evidence:
            return 1.0, "No evidence to validate", []

        # Count evidence references
        evidence_refs = 0
        doc_lower = document.lower()
        for node, description in evidence.items():
            node_lower = str(node).lower()
            tokens = {
                node_lower,
                node_lower.replace("fact_block_", ""),
                node_lower.replace("_", " "),
                f"[{node_lower}",
                f"[{node_lower.replace('fact_block_', '')}",
            }
            desc_lower = str(description or "").lower()
            if desc_lower:
                tokens.add(desc_lower)
            if any(token and token in doc_lower for token in tokens):
                evidence_refs += 1

        score = evidence_refs / len(evidence)
        details = f"Referenced {evidence_refs}/{len(evidence)} evidence items"

        suggestions = []
        if score < 0.9:
            suggestions.append("Include more evidence references in the analysis")

        return score, details, suggestions
    
    @staticmethod
    def _doc_has_patterns(document_lower: str, patterns: List[str]) -> bool:
        for pattern in patterns:
            token = str(pattern).strip().lower()
            if token and token in document_lower:
                return True
        return False

    def _validate_tone_basic(self, document: str) -> tuple[float, str, List[str]]:
        """Basic tone validation."""
        informal_words = ["gonna", "wanna", "yeah", "ok", "cool", "awesome", "stuff", "thing"]

        informal_count = sum(1 for word in informal_words if word in document.lower())

        score = max(0.0, 1.0 - (informal_count * 0.2))
        details = f"Found {informal_count} informal expressions"

        suggestions = []
        if informal_count > 0:
            suggestions.append("Use more formal legal language")

        return score, details, suggestions

    def _prepare_sk_variables(
        self,
        document: str,
        context: Dict[str, Any],
        gate: QualityGate
    ) -> Dict[str, Any]:
        """Prepare SK variables with JSON-safe payloads."""
        variables: Dict[str, Any] = {"document": document}

        if context:
            variables["context"] = self._safe_json_dumps(context)

        if gate.parameters:
            for key, value in gate.parameters.items():
                if isinstance(value, (dict, list, tuple)):
                    variables[key] = self._safe_json_dumps(value)
                else:
                    variables[key] = value

        return variables

    def _parse_sk_result(self, result: Any) -> Dict[str, Any]:
        """Normalize SK function output into a dictionary."""
        raw_value = self._extract_raw_value(result)

        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8", errors="ignore")

        if isinstance(raw_value, str):
            try:
                return json.loads(raw_value)
            except json.JSONDecodeError:
                return {"score": 0.0, "details": raw_value, "suggestions": []}

        if isinstance(raw_value, dict):
            return raw_value

        return {"score": 0.0, "details": str(raw_value), "suggestions": []}

    @staticmethod
    def _extract_raw_value(result: Any) -> Any:
        """Extract the underlying payload from SK invocation result."""
        for attr in ("value", "content", "result"):
            if hasattr(result, attr):
                candidate = getattr(result, attr)
                if candidate is not None:
                    return candidate
        return result

    @staticmethod
    def _safe_json_dumps(value: Any) -> str:
        """Serialize potentially complex objects to JSON."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_convert(item) for item in obj]
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            return str(obj)

        try:
            return json.dumps(value)
        except TypeError:
            return json.dumps(_convert(value))


# Convenience functions
def get_required_gates() -> List[QualityGate]:
    """Get only required quality gates."""
    return [gate for gate in QUALITY_GATES if gate.required]


def get_critical_gates() -> List[QualityGate]:
    """Get only critical quality gates."""
    return [gate for gate in QUALITY_GATES if gate.severity == ValidationSeverity.CRITICAL]


def get_gate_by_name(name: str) -> Optional[QualityGate]:
    """Get quality gate by name."""
    return next((gate for gate in QUALITY_GATES if gate.name == name), None)


# Export main classes and functions
__all__ = [
    "QualityGate",
    "ValidationResult",
    "QualityGateResults",
    "QualityGateRunner",
    "ValidationSeverity",
    "QUALITY_GATES",
    "get_required_gates",
    "get_critical_gates",
    "get_gate_by_name"
]
