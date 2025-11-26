"""Validation plugin for Semantic Kernel with citation, structure, and tone validation."""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction, KernelFunctionFromPrompt, KernelArguments
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from ..base_plugin import kernel_function

from ..base_plugin import BaseSKPlugin, PluginMetadata, ValidationFunction, FunctionResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from validation function."""
    passed: bool
    score: float
    details: str
    suggestions: List[str]
    errors: List[str]


def _serialize_validation_result(validation: ValidationResult) -> str:
    """Convert ValidationResult to JSON string."""
    return json.dumps({
        "score": float(validation.score),
        "passed": bool(validation.passed),
        "details": validation.details,
        "suggestions": list(validation.suggestions or []),
        "errors": list(validation.errors or [])
    })


def _ensure_sequence(value: Any, default: Optional[List[str]] = None) -> List[str]:
    """Ensure value is a list-like structure."""
    default_list = list(default) if default else []
    if value is None:
        return default_list

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            # Allow comma-separated strings
            return [item.strip() for item in value.split(",") if item.strip()]
        return default_list

    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    return default_list


def _ensure_mapping(value: Any) -> Dict[str, Any]:
    """Ensure value is a dictionary."""
    if value is None:
        return {}

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    return value if isinstance(value, dict) else {}


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


class CitationValidatorFunction(ValidationFunction):
    """Native function for citation format validation."""

    def __init__(self):
        super().__init__(
            name="CitationValidator",
            description="Validate citation format and completeness",
            validation_type="citation_format"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute citation validation."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        try:
            document = kwargs["document"]
            required_format = kwargs.get("required_format", "[Node:State]")

            # Validate citations
            validation_result = self._validate_citations(document, required_format)

            return FunctionResult(
                success=True,
                value=validation_result,
                metadata={"validation_type": "citation_format"}
            )

        except Exception as e:
            logger.error(f"Error in CitationValidator: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _validate_citations(self, document: str, required_format: str) -> ValidationResult:
        """Validate citation format in document."""

        # Extract citation pattern from required format
        if required_format == "[Node:State]":
            citation_pattern = r'\[[^:]+:[^\]]+\]'
        else:
            # Generic pattern for other formats
            citation_pattern = r'\[[^\]]+\]'

        citations = re.findall(citation_pattern, document)

        if not citations:
            return ValidationResult(
                passed=False,
                score=0.0,
                details="No citations found in document",
                suggestions=["Add evidence citations in the required format"],
                errors=["Missing citations"]
            )

        # Validate citation format
        valid_citations = []
        invalid_citations = []

        for citation in citations:
            if self._is_valid_citation(citation, required_format):
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)

        score = len(valid_citations) / len(citations) if citations else 0.0
        passed = score >= 0.9  # 90% of citations must be valid

        suggestions = []
        if invalid_citations:
            suggestions.append(f"Fix {len(invalid_citations)} invalid citations")
            suggestions.append(f"Ensure citations follow format: {required_format}")

        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Found {len(valid_citations)}/{len(citations)} valid citations",
            suggestions=suggestions,
            errors=[f"Invalid citation: {citation}" for citation in invalid_citations]
        )

    def _is_valid_citation(self, citation: str, required_format: str) -> bool:
        """Check if citation matches required format."""

        if required_format == "[Node:State]":
            # Must have format [Node:State]
            return ':' in citation and len(citation) > 3
        else:
            # Generic validation
            return len(citation) > 2 and citation.startswith('[') and citation.endswith(']')


class StructureValidatorFunction(ValidationFunction):
    """Native function for document structure validation."""

    def __init__(self):
        super().__init__(
            name="StructureValidator",
            description="Validate document structure and completeness",
            validation_type="structure_complete"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute structure validation."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        try:
            document = kwargs["document"]
            required_sections = kwargs.get("required_sections", ["Introduction", "Analysis", "Conclusion"])

            # Validate structure
            validation_result = self._validate_structure(document, required_sections)

            return FunctionResult(
                success=True,
                value=validation_result,
                metadata={"validation_type": "structure_complete"}
            )

        except Exception as e:
            logger.error(f"Error in StructureValidator: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _validate_structure(self, document: str, required_sections: List[str]) -> ValidationResult:
        """Validate document structure."""

        found_sections: List[str] = []
        missing_sections: List[str] = []
        doc_lower = document.lower()

        for section in required_sections:
            label, patterns = self._normalize_section_entry(section)
            if self._document_has_patterns(doc_lower, patterns):
                found_sections.append(label)
            else:
                missing_sections.append(label)

        total_required = len(required_sections) or 1
        score = len(found_sections) / total_required
        passed = score >= 1.0  # All required sections must be present

        suggestions = []
        if missing_sections:
            suggestions.append(f"Add missing sections: {', '.join(missing_sections)}")

        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Found {len(found_sections)}/{total_required} required sections",
            suggestions=suggestions,
            errors=[f"Missing section: {section}" for section in missing_sections]
        )

    def _normalize_section_entry(self, section: Any) -> (str, List[str]):
        """Return label and patterns for a required section definition."""
        if isinstance(section, dict):
            label = str(section.get("label") or section.get("name") or section.get("id") or "Section")
            patterns = _ensure_sequence(section.get("patterns") or section.get("aliases"))
        else:
            label = str(section)
            patterns = []

        # Gather default aliases based on label tokens (split by / or |)
        if not patterns:
            alias_patterns: List[str] = []
            for token in re.split(r"[\/|]", label):
                key = token.strip().lower()
                alias_patterns.extend(DEFAULT_SECTION_ALIASES.get(key, []))
            patterns = alias_patterns or [label]
        return label, patterns

    @staticmethod
    def _document_has_patterns(document_lower: str, patterns: List[str]) -> bool:
        """Return True if any pattern (case-insensitive substring) is present in the document."""
        for pattern in patterns:
            token = str(pattern).strip().lower()
            if not token:
                continue
            if token in document_lower:
                return True
        return False


class EvidenceGroundingValidatorFunction(ValidationFunction):
    """Native function for evidence grounding validation."""

    def __init__(self):
        super().__init__(
            name="EvidenceGroundingValidator",
            description="Validate that all claims have supporting evidence",
            validation_type="evidence_grounding"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute evidence grounding validation."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        try:
            document = kwargs["document"]
            context = _ensure_mapping(kwargs.get("context", {}))
            evidence = self._normalize_evidence_input(context.get("evidence"))

            # Validate evidence grounding
            validation_result = self._validate_evidence_grounding(document, evidence)

            return FunctionResult(
                success=True,
                value=validation_result,
                metadata={"validation_type": "evidence_grounding"}
            )

        except Exception as e:
            logger.error(f"Error in EvidenceGroundingValidator: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _normalize_evidence_input(self, evidence_input: Any) -> Dict[str, str]:
        """Normalize evidence payloads (dict or list) into a consistent mapping."""
        normalized: Dict[str, str] = {}
        if not evidence_input:
            return normalized

        if isinstance(evidence_input, dict):
            for key, value in evidence_input.items():
                if key:
                    normalized[str(key)] = str(value) if value is not None else ""
            return normalized

        if isinstance(evidence_input, list):
            for idx, item in enumerate(evidence_input, start=1):
                node_id = f"item_{idx}"
                state_text = ""
                if isinstance(item, dict):
                    node_id = str(item.get("node_id") or item.get("id") or node_id)
                    state_text = (
                        item.get("state")
                        or item.get("description")
                        or item.get("text")
                        or ""
                    )
                else:
                    state_text = str(item)
                normalized[node_id] = state_text
            return normalized

        if isinstance(evidence_input, str):
            normalized["evidence"] = evidence_input
            return normalized

        try:
            normalized["evidence"] = str(evidence_input)
        except Exception:
            normalized["evidence"] = ""
        return normalized

    def _validate_evidence_grounding(self, document: str, evidence_input: Any) -> ValidationResult:
        """Validate evidence grounding."""
        evidence = self._normalize_evidence_input(evidence_input)

        if not evidence:
            return ValidationResult(
                passed=True,
                score=1.0,
                details="No evidence to validate",
                suggestions=[],
                errors=[]
            )

        doc_lower = document.lower()
        evidence_refs = 0
        for node, description in evidence.items():
            node_lower = str(node).lower()
            normalized_tokens = {
                node_lower,
                node_lower.replace("fact_block_", ""),
                node_lower.replace("_", " "),
                f"[{node_lower}",
                f"[{node_lower.replace('fact_block_', '')}",
            }
            description_text = str(description or "").lower()
            if description_text:
                normalized_tokens.add(description_text)

            matched = any(token and token in doc_lower for token in normalized_tokens)
            if matched:
                evidence_refs += 1

        score = evidence_refs / len(evidence) if evidence else 1.0
        passed = score >= 0.8  # 80% of evidence should be referenced

        suggestions = []
        if score < 0.8:
            suggestions.append("Include more evidence references in the analysis")
            suggestions.append("Ensure all key evidence is cited")

        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Referenced {evidence_refs}/{len(evidence)} evidence items",
            suggestions=suggestions,
            errors=[]
        )


class ArgumentCoherenceValidatorFunction(ValidationFunction):
    """Heuristic validator for argument coherence."""

    def __init__(self):
        super().__init__(
            name="ArgumentCoherenceValidator",
            description="Evaluate transitions and conclusions for argument coherence",
            validation_type="argument_coherence"
        )
        self.transition_markers = [
            "therefore", "thus", "consequently", "as a result",
            "furthermore", "additionally", "however", "moreover"
        ]
        self.conclusion_markers = [
            "conclusion", "accordingly", "for these reasons",
            "we respectfully request", "in sum", "in summary"
        ]

    async def execute(self, **kwargs) -> FunctionResult:
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        document = kwargs["document"]
        check_transitions = bool(kwargs.get("check_transitions", True))
        verify_conclusions = bool(kwargs.get("verify_conclusions", True))

        validation_result = self._evaluate_coherence(
            document,
            check_transitions=check_transitions,
            verify_conclusions=verify_conclusions
        )

        return FunctionResult(
            success=True,
            value=validation_result,
            metadata={"validation_type": self.validation_type}
        )

    def _evaluate_coherence(
        self,
        document: str,
        check_transitions: bool,
        verify_conclusions: bool
    ) -> ValidationResult:
        paragraphs = [p.strip() for p in document.split("\n\n") if p.strip()]
        if not paragraphs:
            return ValidationResult(
                passed=True,
                score=1.0,
                details="Document too short for coherence analysis",
                suggestions=[],
                errors=[]
            )

        details = []
        suggestions: List[str] = []

        components: List[float] = []

        if check_transitions:
            transition_hits = sum(
                1 for paragraph in paragraphs
                if any(marker in paragraph.lower() for marker in self.transition_markers)
            )
            transition_score = transition_hits / len(paragraphs)
            components.append(transition_score)
            details.append(f"Transition markers in {transition_hits}/{len(paragraphs)} paragraphs")
            if transition_score < 0.75:
                suggestions.append("Add clearer transition sentences between sections to improve flow.")

        if verify_conclusions:
            conclusion_present = any(
                marker in document.lower()
                for marker in self.conclusion_markers
            )
            conclusion_score = 1.0 if conclusion_present else 0.0
            components.append(conclusion_score)
            details.append("Conclusion detected" if conclusion_present else "Missing explicit conclusion section")
            if not conclusion_present:
                suggestions.append("Include an explicit conclusion tying the legal theory to the requested relief.")

        score = sum(components) / len(components) if components else 1.0
        passed = score >= 0.8

        return ValidationResult(
            passed=passed,
            score=score,
            details="; ".join(details) if details else "Coherence heuristics satisfied",
            suggestions=suggestions,
            errors=[]
        )


class LegalAccuracyValidatorFunction(ValidationFunction):
    """Validator that checks for legal terminology and citations."""

    def __init__(self):
        super().__init__(
            name="LegalAccuracyValidator",
            description="Verify legal terminology and citation usage",
            validation_type="legal_accuracy"
        )
        self.legal_terms = [
            "statute", "precedent", "jurisdiction", "standard",
            "court", "authority", "rule", "burden", "evidence"
        ]
        self.citation_pattern = re.compile(r"\b[A-Z][a-zA-Z]+ v\. [A-Z][a-zA-Z]+")

    async def execute(self, **kwargs) -> FunctionResult:
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        document = kwargs["document"]
        jurisdiction_specific = bool(kwargs.get("jurisdiction_specific", True))
        check_precedents = bool(kwargs.get("check_precedents", True))

        validation_result = self._evaluate_accuracy(
            document,
            jurisdiction_specific=jurisdiction_specific,
            check_precedents=check_precedents
        )

        return FunctionResult(
            success=True,
            value=validation_result,
            metadata={"validation_type": self.validation_type}
        )

    def _evaluate_accuracy(
        self,
        document: str,
        jurisdiction_specific: bool,
        check_precedents: bool
    ) -> ValidationResult:
        lower_doc = document.lower()
        suggestions: List[str] = []

        term_hits = sum(1 for term in self.legal_terms if term in lower_doc)
        term_score = min(term_hits / max(len(self.legal_terms) / 2, 1), 1.0)
        if term_score < 0.6:
            suggestions.append("Reference applicable legal standards (statutes, burdens, or rules).")

        citations = self.citation_pattern.findall(document)
        citation_score = min(len(citations) / 2, 1.0) if check_precedents else 1.0
        if check_precedents and citation_score < 0.5:
            suggestions.append("Add controlling precedent citations (e.g., \"Case v. Case\").")

        jurisdiction_markers = ["district", "circuit", "united states", "state of"]
        jurisdiction_present = any(marker in lower_doc for marker in jurisdiction_markers)
        jurisdiction_score = 1.0 if not jurisdiction_specific else (1.0 if jurisdiction_present else 0.5)
        if jurisdiction_specific and not jurisdiction_present:
            suggestions.append("Identify the relevant court or jurisdiction to frame the legal standard.")

        score_components = [term_score, citation_score, jurisdiction_score]
        score = sum(score_components) / len(score_components)
        passed = score >= 0.8

        details = (
            f"Legal terms: {term_hits}, citations: {len(citations)}, "
            f"jurisdiction reference: {'yes' if jurisdiction_present else 'no'}"
        )

        return ValidationResult(
            passed=passed,
            score=score,
            details=details,
            suggestions=suggestions,
            errors=[]
        )


class PrivacyHarmChecklistValidatorFunction(ValidationFunction):
    """Validator to ensure privacy harm factors are covered."""

    def __init__(self):
        super().__init__(
            name="PrivacyHarmChecklistValidator",
            description="Validate privacy harm factors and legal elements",
            validation_type="privacy_harm_checklist"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        document = kwargs["document"]
        harm_factors = _ensure_sequence(kwargs.get("harm_factors"), [])
        legal_elements = _ensure_sequence(kwargs.get("legal_elements"), [])

        validation_result = self._evaluate_checklist(
            document,
            harm_factors=harm_factors,
            legal_elements=legal_elements
        )

        return FunctionResult(
            success=True,
            value=validation_result,
            metadata={"validation_type": self.validation_type}
        )

    def _evaluate_checklist(
        self,
        document: str,
        harm_factors: List[str],
        legal_elements: List[str]
    ) -> ValidationResult:
        lower_doc = document.lower()
        suggestions: List[str] = []

        harm_hits = [factor for factor in harm_factors if factor.lower() in lower_doc]
        legal_hits = [element for element in legal_elements if element.lower() in lower_doc]

        harm_score = len(harm_hits) / len(harm_factors) if harm_factors else 1.0
        legal_score = len(legal_hits) / len(legal_elements) if legal_elements else 1.0

        if harm_score < 1.0:
            missing = [factor for factor in harm_factors if factor not in harm_hits]
            suggestions.append(f"Address remaining privacy harm factors: {', '.join(missing)}.")

        if legal_score < 1.0:
            missing = [element for element in legal_elements if element not in legal_hits]
            suggestions.append(f"Connect facts to legal elements: {', '.join(missing)}.")

        score_components = [harm_score, legal_score]
        score = sum(score_components) / len(score_components)
        passed = score >= 0.85

        details = (
            f"Harm factors covered: {len(harm_hits)}/{len(harm_factors)}; "
            f"legal elements covered: {len(legal_hits)}/{len(legal_elements)}"
        )

        return ValidationResult(
            passed=passed,
            score=score,
            details=details,
            suggestions=suggestions,
            errors=[]
        )


class GrammarSpellingValidatorFunction(ValidationFunction):
    """Lightweight grammar and spelling heuristic validator."""

    def __init__(self):
        super().__init__(
            name="GrammarSpellingValidator",
            description="Check for capitalization, repeated punctuation, and spacing issues",
            validation_type="grammar_spelling"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        document = kwargs["document"]
        validation_result = self._evaluate_grammar(document)

        return FunctionResult(
            success=True,
            value=validation_result,
            metadata={"validation_type": self.validation_type}
        )

    def _evaluate_grammar(self, document: str) -> ValidationResult:
        sentences = [s.strip() for s in re.split(r"[.!?]", document) if s.strip()]
        total_sentences = len(sentences)

        if total_sentences == 0:
            return ValidationResult(
                passed=True,
                score=1.0,
                details="No complete sentences to evaluate",
                suggestions=[],
                errors=[]
            )

        capitalization_errors = sum(
            1 for sentence in sentences if sentence and not sentence[0].isupper()
        )

        repeated_spaces = document.count("  ")
        repeated_punctuation = len(re.findall(r"!!+|\?\?+|\.{3,}", document))

        penalty = (capitalization_errors + repeated_spaces + repeated_punctuation)
        normalized_penalty = min(penalty / max(total_sentences, 1), 1.0)
        score = max(0.0, 1.0 - normalized_penalty * 0.5)
        passed = score >= 0.9

        suggestions: List[str] = []
        if capitalization_errors:
            suggestions.append("Capitalize the first word of each sentence.")
        if repeated_spaces:
            suggestions.append("Remove repeated spaces between words.")
        if repeated_punctuation:
            suggestions.append("Replace repeated punctuation with standard legal prose.")

        details = (
            f"Capitalization issues: {capitalization_errors}, "
            f"double-spaces: {repeated_spaces}, repeated punctuation: {repeated_punctuation}"
        )

        return ValidationResult(
            passed=passed,
            score=score,
            details=details,
            suggestions=suggestions,
            errors=[]
        )


class ToneConsistencySemanticFunction(ValidationFunction):
    """Semantic function for tone consistency validation."""

    def __init__(self):
        super().__init__(
            name="ToneConsistencySemantic",
            description="Validate professional legal tone using LLM",
            validation_type="tone_consistency"
        )
        self.prompt_template = self._get_prompt_template()
        self._prompt_config = PromptTemplateConfig(
            name=self.name,
            description=self.description,
            template=self.prompt_template,
            template_format="semantic-kernel",
        )
        self._prompt_function: Optional[KernelFunctionFromPrompt] = None

    def _get_prompt_template(self) -> str:
        """Get the prompt template for tone validation."""
        return """
You are a legal writing expert. Analyze the tone and style of the following legal document.

DOCUMENT:
{{$document}}

REQUIREMENTS:
1. Check for professional legal tone
2. Identify informal language or colloquialisms
3. Verify formal legal writing style
4. Check for appropriate legal terminology

OUTPUT FORMAT:
{
  "score": 0.0-1.0,
  "passed": true/false,
  "details": "Brief analysis of tone",
  "suggestions": ["List of specific improvements"],
  "errors": ["List of tone issues"]
}

Focus on:
- Formal language vs informal
- Legal terminology usage
- Professional tone consistency
- Appropriate legal writing style
        """

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic tone validation."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        try:
            document = kwargs["document"]
            prompt_func = self._get_prompt_function()

            # Execute validation
            result = await prompt_func.invoke(kernel=kernel, arguments=KernelArguments(document=document))

            # Parse JSON result
            result_payload = result.value
            if isinstance(result_payload, list) and result_payload:
                first = result_payload[0]
                if hasattr(first, "items") and first.items:
                    first_item = first.items[0]
                    result_payload = getattr(first_item, "text", str(first_item))
                else:
                    result_payload = getattr(first, "text", str(first))
            elif hasattr(result_payload, "text"):
                result_payload = result_payload.text

            try:
                validation_data = json.loads(result_payload)
                validation_result = ValidationResult(
                    passed=validation_data.get("passed", False),
                    score=validation_data.get("score", 0.0),
                    details=validation_data.get("details", ""),
                    suggestions=validation_data.get("suggestions", []),
                    errors=validation_data.get("errors", [])
                )
            except (json.JSONDecodeError, TypeError):
                # Fallback to basic validation
                validation_result = self._basic_tone_validation(document)

            return FunctionResult(
                success=True,
                value=validation_result,
                metadata={
                    "validation_type": "tone_consistency",
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as e:
            logger.error(f"Error in ToneConsistencySemantic: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_prompt_function(self) -> KernelFunctionFromPrompt:
        if self._prompt_function is None:
            self._prompt_function = KernelFunctionFromPrompt(
                function_name=self.name,
                plugin_name="ValidationPlugin",
                description=self.description,
                prompt_template_config=self._prompt_config,
            )
        return self._prompt_function

    def _basic_tone_validation(self, document: str) -> ValidationResult:
        """Basic tone validation fallback."""

        informal_words = ["gonna", "wanna", "yeah", "ok", "cool", "awesome", "stuff", "thing"]

        informal_count = sum(1 for word in informal_words if word in document.lower())

        score = max(0.0, 1.0 - (informal_count * 0.2))
        passed = score >= 0.85

        suggestions = []
        if informal_count > 0:
            suggestions.append("Use more formal legal language")
            suggestions.append("Avoid colloquial expressions")

        return ValidationResult(
            passed=passed,
            score=score,
            details=f"Found {informal_count} informal expressions",
            suggestions=suggestions,
            errors=[]
        )


class ValidationPlugin(BaseSKPlugin):
    """Plugin for document validation functions."""

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.citation_validator = CitationValidatorFunction()
        self.structure_validator = StructureValidatorFunction()
        self.evidence_validator = EvidenceGroundingValidatorFunction()
        self.tone_validator = ToneConsistencySemanticFunction()
        self.argument_validator = ArgumentCoherenceValidatorFunction()
        self.legal_validator = LegalAccuracyValidatorFunction()
        self.privacy_validator = PrivacyHarmChecklistValidatorFunction()
        self.grammar_validator = GrammarSpellingValidatorFunction()

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="ValidationPlugin",
            description="Plugin for document validation and quality assurance",
            version="1.0.0",
            functions=[
                "ValidateCitationFormat",
                "ValidateStructure",
                "ValidateEvidenceGrounding",
                "ValidateToneConsistency",
                "ValidateArgumentCoherence",
                "ValidateLegalAccuracy",
                "ValidatePrivacyHarmChecklist",
                "ValidateGrammarSpelling"
            ]
        )

    async def _register_functions(self) -> None:
        """Register validation functions with the kernel."""

        # Register citation validator
        @kernel_function(
            name="ValidateCitationFormat",
            description="Validate citation format and completeness"
        )
        async def validate_citation_format(
            document: str,
            required_format: str = "[Node:State]",
            **_ignored: Any,
        ) -> str:
            """Validate citation format."""
            result = await self.citation_validator.execute(
                document=document,
                required_format=required_format
            )

            if result.success:
                validation = result.value
                return _serialize_validation_result(validation)
            else:
                raise RuntimeError(f"Citation validation failed: {result.error}")

        # Register structure validator
        @kernel_function(
            name="ValidateStructure",
            description="Validate document structure and completeness"
        )
        async def validate_structure(
            document: str,
            required_sections: str = '["Introduction", "Analysis", "Conclusion"]',
            **_ignored: Any,
        ) -> str:
            """Validate document structure."""
            sections = _ensure_sequence(
                required_sections,
                ["Introduction", "Analysis", "Conclusion"]
            )
            result = await self.structure_validator.execute(
                document=document,
                required_sections=sections
            )

            if result.success:
                validation = result.value
                return _serialize_validation_result(validation)
            else:
                raise RuntimeError(f"Structure validation failed: {result.error}")

        # Register evidence grounding validator
        @kernel_function(
            name="ValidateEvidenceGrounding",
            description="Validate that all claims have supporting evidence"
        )
        async def validate_evidence_grounding(
            document: str,
            context: Any = None,
            **_ignored: Any,
        ) -> str:
            """Validate evidence grounding."""
            context_data = _ensure_mapping(context)
            if not context_data:
                logger.debug("[ValidationPlugin] Evidence grounding called with empty context; skipping detailed validation")
            result = await self.evidence_validator.execute(
                document=document,
                context=context_data
            )

            if result.success:
                validation = result.value
                return _serialize_validation_result(validation)
            else:
                raise RuntimeError(f"Evidence grounding validation failed: {result.error}")

        # Register tone consistency validator
        @kernel_function(
            name="ValidateToneConsistency",
            description="Validate professional legal tone using LLM"
        )
        async def validate_tone_consistency(document: str, **_ignored: Any) -> str:
            """Validate tone consistency via LLM or fallback."""
            result = await self.tone_validator.execute(
                kernel=self.kernel,
                document=document
            )

            if result.success:
                validation = result.value
                return _serialize_validation_result(validation)
            else:
                raise RuntimeError(f"Tone validation failed: {result.error}")

        # Register argument coherence validator
        @kernel_function(
            name="ValidateArgumentCoherence",
            description="Validate transitions and overall argument coherence"
        )
        async def validate_argument_coherence(
            document: str,
            check_transitions: bool = True,
            verify_conclusions: bool = True,
            **_ignored: Any,
        ) -> str:
            result = await self.argument_validator.execute(
                document=document,
                check_transitions=check_transitions,
                verify_conclusions=verify_conclusions
            )

            if result.success:
                return _serialize_validation_result(result.value)
            raise RuntimeError(f"Argument coherence validation failed: {result.error}")

        # Register legal accuracy validator
        @kernel_function(
            name="ValidateLegalAccuracy",
            description="Validate legal terminology usage and citations"
        )
        async def validate_legal_accuracy(
            document: str,
            jurisdiction_specific: bool = True,
            check_precedents: bool = True,
            **_ignored: Any,
        ) -> str:
            result = await self.legal_validator.execute(
                document=document,
                jurisdiction_specific=jurisdiction_specific,
                check_precedents=check_precedents
            )

            if result.success:
                return _serialize_validation_result(result.value)
            raise RuntimeError(f"Legal accuracy validation failed: {result.error}")

        # Register privacy harm checklist validator
        @kernel_function(
            name="ValidatePrivacyHarmChecklist",
            description="Validate privacy harm factors and legal elements coverage"
        )
        async def validate_privacy_harm_checklist(
            document: str,
            harm_factors: str = '[]',
            legal_elements: str = '[]',
            **_ignored: Any,
        ) -> str:
            result = await self.privacy_validator.execute(
                document=document,
                harm_factors=_ensure_sequence(harm_factors, []),
                legal_elements=_ensure_sequence(legal_elements, [])
            )

            if result.success:
                return _serialize_validation_result(result.value)
            raise RuntimeError(f"Privacy harm checklist validation failed: {result.error}")

        # Register grammar/spelling validator
        @kernel_function(
            name="ValidateGrammarSpelling",
            description="Lightweight grammar and spelling heuristic validator"
        )
        async def validate_grammar_spelling(document: str, **_ignored: Any) -> str:
            result = await self.grammar_validator.execute(document=document)

            if result.success:
                return _serialize_validation_result(result.value)
            raise RuntimeError(f"Grammar/spelling validation failed: {result.error}")

        # Store function references
        self._functions["ValidateCitationFormat"] = validate_citation_format
        self._functions["ValidateStructure"] = validate_structure
        self._functions["ValidateEvidenceGrounding"] = validate_evidence_grounding
        self._functions["ValidateToneConsistency"] = validate_tone_consistency
        self._functions["ValidateArgumentCoherence"] = validate_argument_coherence
        self._functions["ValidateLegalAccuracy"] = validate_legal_accuracy
        self._functions["ValidatePrivacyHarmChecklist"] = validate_privacy_harm_checklist
        self._functions["ValidateGrammarSpelling"] = validate_grammar_spelling

        # Ensure functions are accessible via kernel.plugins even when
        # Semantic Kernel lacks create_plugin_from_functions (SK >= 1.37.1)
        self._mirror_functions_to_kernel()

    def _mirror_functions_to_kernel(self) -> None:
        """Mirror plugin functions into kernel.plugins for compatibility."""
        plugins_dict = getattr(self.kernel, "plugins", None)
        if isinstance(plugins_dict, dict):
            registry = plugins_dict.setdefault(self.metadata.name, {})
            for func_name, func in self._functions.items():
                registry[func_name] = func
            logger.debug(
                "[ValidationPlugin] Registered %d functions via kernel.plugins",
                len(self._functions),
            )
        else:
            logger.warning(
                "[ValidationPlugin] Kernel.plugins dict missing; functions only accessible via plugin registry"
            )


# Export classes
__all__ = [
    "ValidationPlugin",
    "CitationValidatorFunction",
    "StructureValidatorFunction",
    "EvidenceGroundingValidatorFunction",
    "ToneConsistencySemanticFunction",
    "ArgumentCoherenceValidatorFunction",
    "LegalAccuracyValidatorFunction",
    "PrivacyHarmChecklistValidatorFunction",
    "GrammarSpellingValidatorFunction",
    "ValidationResult"
]
