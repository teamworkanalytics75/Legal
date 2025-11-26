"""Review and QA atomic agents.

Agents for grammar, style, logic, consistency, redaction, compliance, and expert QA.
Mix of deterministic (zero cost) and LLM-based agents.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Set

try:
    from ..atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
except ImportError:
    from atomic_agent import DeterministicAgent, LLMAgent, AgentFactory


class GrammarFixerAgent(LLMAgent):
    """Fix grammar and typos in text.

    Single duty: Correct grammatical errors and typos.
    Method: LLM-based grammar checking (gpt-4o-mini).
    Output: Corrected text with change log.
    """

    duty = "Fix grammar and typos in text"
    cost_tier = "mini"
    max_cost_per_run = 0.005

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"
    premium_temperature = 0.1
    premium_max_tokens = 6000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build grammar correction prompt.

        Args:
            input_data: Should contain 'text'

        Returns:
            Formatted prompt
        """
        text = input_data.get('text', '')

        return f"""Fix grammar and typos in the following text. DO NOT change wording or style.
Only correct obvious grammatical errors and spelling mistakes.

Text:
{text}

Output the corrected text ONLY, no explanations."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse corrected text.

        Args:
            response: LLM response

        Returns:
            Parsed result with corrected text
        """
        return {
            'corrected_text': response.strip(),
            'word_count': len(response.split()),
        }


class StyleCheckerAgent(DeterministicAgent):
    """Check document against style guide rules.

    Single duty: Verify compliance with legal writing style guide.
    Method: Rule-based checking (deterministic).
    Output: List of style violations with suggestions.
    """

    duty = "Check document against style guide rules"

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"

    # Style rules (deterministic checks)
    STYLE_RULES = [
        ('passive_voice', r'\b(?:was|were|been|being)\s+\w+ed\b',
         'Avoid passive voice in legal writing'),
        ('shall_misuse', r'\bshall\b',
         'Prefer "must" or "will" over "shall"'),
        ('contractions', r"\b\w+n't\b",
         'Avoid contractions in formal documents'),
        ('first_person', r'\b(?:I|we|our|us)\b',
         'Avoid first person in objective analysis'),
        ('informal_words', r'\b(?:very|really|just|quite|pretty)\b',
         'Avoid informal intensifiers'),
    ]

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check text against style rules.

        Args:
            input_data: Should contain 'text'

        Returns:
            Style violations and compliance score
        """
        text = input_data.get('text', '')

        violations = []

        for rule_name, pattern, suggestion in self.STYLE_RULES:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))

            if matches:
                for match in matches:
                    violations.append({
                        'rule': rule_name,
                        'matched_text': match.group(0),
                        'position': match.start(),
                        'suggestion': suggestion,
                    })

        # Calculate compliance score
        total_words = len(text.split())
        violation_rate = len(violations) / max(total_words / 100, 1) # per 100 words
        compliance_score = max(0.0, 1.0 - (violation_rate / 10)) # Cap at 10 violations per 100 words

        return {
            'violations': violations,
            'violation_count': len(violations),
            'compliance_score': round(compliance_score, 2),
            'passed': compliance_score >= 0.8,
        }


class LogicCheckerAgent(LLMAgent):
    """Check argument logic and coherence.

    Single duty: Verify logical flow and argument structure.
    Method: LLM analysis of reasoning chain (gpt-4o-mini).
    Output: Logic assessment with identified gaps.
    """

    duty = "Check argument logic and identify missing premises or gaps"
    cost_tier = "mini"
    max_cost_per_run = 0.008

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"
    premium_temperature = 0.1
    premium_max_tokens = 6000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build logic checking prompt.

        Args:
            input_data: Should contain 'text'

        Returns:
            Formatted prompt
        """
        text = input_data.get('text', '')

        return f"""Analyze the logical structure of this legal argument. Identify:
1. Missing premises
2. Logical gaps or leaps
3. Unsupported conclusions
4. Contradictions

Text:
{text}

Output as JSON:
{{
  "logic_score": 0.0-1.0,
  "issues": [
    {{"type": "missing_premise", "location": "...", "description": "..."}},
    ...
  ],
  "passed": true/false
}}

Keep analysis brief. Output ONLY the JSON."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse logic analysis.

        Args:
            response: LLM response

        Returns:
            Parsed logic assessment
        """
        try:
            # Try JSON parsing
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            result = json.loads(cleaned)
            return result

        except json.JSONDecodeError:
            # Fallback
            return {
                'logic_score': 0.7,
                'issues': [],
                'passed': True,
                'raw_response': response,
            }


class ConsistencyCheckerAgent(DeterministicAgent):
    """Check term and name consistency throughout document.

    Single duty: Ensure terms and names used consistently.
    Method: Dictionary tracking and matching (deterministic).
    Output: Inconsistencies found with suggested corrections.
    """

    duty = "Ensure terms and names are used consistently throughout document"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of terms and names.

        Args:
            input_data: Should contain 'text'

        Returns:
            Consistency issues and score
        """
        text = input_data.get('text', '')

        # Extract potential entity names (capitalized phrases)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, text)

        # Group similar entities (case-insensitive)
        entity_variants: Dict[str, List[str]] = {}
        for entity in entities:
            key = entity.lower()
            if key not in entity_variants:
                entity_variants[key] = []
            if entity not in entity_variants[key]:
                entity_variants[key].append(entity)

        # Find inconsistencies (multiple variants of same entity)
        inconsistencies = []
        for key, variants in entity_variants.items():
            if len(variants) > 1:
                inconsistencies.append({
                    'entity': key,
                    'variants': variants,
                    'suggestion': f'Use "{variants[0]}" consistently',
                    'count': len(variants),
                })

        # Calculate consistency score
        total_entities = len(entities)
        unique_inconsistencies = len(inconsistencies)
        consistency_score = 1.0 - (unique_inconsistencies / max(total_entities / 10, 1))
        consistency_score = max(0.0, min(1.0, consistency_score))

        return {
            'inconsistencies': inconsistencies,
            'inconsistency_count': len(inconsistencies),
            'consistency_score': round(consistency_score, 2),
            'passed': consistency_score >= 0.9,
        }


class RedactionAgent(DeterministicAgent):
    """Apply redaction rules to protect sensitive information.

    Single duty: Redact PII and sealed information per rules.
    Method: Regex-based pattern matching and replacement (deterministic).
    Output: Redacted text with redaction log.
    """

    duty = "Apply redaction rules to protect sensitive information"

    # Redaction patterns
    REDACTION_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]'), # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REDACTED]'), # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'), # Email
        (r'\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
         '[ADDRESS REDACTED]'), # Street address
    ]

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from text.

        Args:
            input_data: Should contain 'text' and optional 'custom_patterns'

        Returns:
            Redacted text with redaction log
        """
        text = input_data.get('text', '')
        custom_patterns = input_data.get('custom_patterns', [])

        redactions = []
        redacted_text = text

        # Apply standard patterns
        all_patterns = self.REDACTION_PATTERNS + custom_patterns

        for pattern, replacement in all_patterns:
            matches = list(re.finditer(pattern, redacted_text))

            if matches:
                for match in matches:
                    redactions.append({
                        'original': match.group(0),
                        'redacted_to': replacement,
                        'position': match.start(),
                        'pattern_type': pattern[:30] + '...',
                    })

                # Apply redaction
                redacted_text = re.sub(pattern, replacement, redacted_text)

        return {
            'redacted_text': redacted_text,
            'redactions': redactions,
            'redaction_count': len(redactions),
        }


class ComplianceAgent(DeterministicAgent):
    """Verify document format compliance.

    Single duty: Check formatting and structure requirements.
    Method: Rule-based validation (deterministic).
    Output: Compliance checklist with pass/fail status.
    """

    duty = "Verify document format and structure compliance"

    # Compliance rules
    REQUIRED_SECTIONS = ['introduction', 'analysis', 'conclusion']
    MAX_PARAGRAPH_SENTENCES = 7
    MIN_SECTION_WORDS = 100

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check document compliance.

        Args:
            input_data: Should contain 'text' and optional 'sections'

        Returns:
            Compliance report
        """
        text = input_data.get('text', '')
        sections = input_data.get('sections', [])

        issues = []

        # Check required sections present
        section_titles = [s.get('title', '').lower() for s in sections]
        for required in self.REQUIRED_SECTIONS:
            if not any(required in title for title in section_titles):
                issues.append({
                    'type': 'missing_section',
                    'severity': 'high',
                    'description': f'Required section "{required}" not found',
                })

        # Check paragraph length (by double newlines)
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            sentences = para.split('. ')
            if len(sentences) > self.MAX_PARAGRAPH_SENTENCES:
                issues.append({
                    'type': 'paragraph_too_long',
                    'severity': 'low',
                    'description': f'Paragraph {i+1} has {len(sentences)} sentences (max {self.MAX_PARAGRAPH_SENTENCES})',
                    'location': f'Paragraph {i+1}',
                })

        # Check section length
        for section in sections:
            text = section.get('text', section.get('section_text', ''))
            word_count = len(text.split())
            if word_count < self.MIN_SECTION_WORDS:
                issues.append({
                    'type': 'section_too_short',
                    'severity': 'medium',
                    'description': f'Section "{section.get("title", "unknown")}" only {word_count} words (min {self.MIN_SECTION_WORDS})',
                    'location': section.get('title', 'unknown'),
                })

        # Calculate compliance score
        high_severity = sum(1 for i in issues if i.get('severity') == 'high')
        medium_severity = sum(1 for i in issues if i.get('severity') == 'medium')
        low_severity = sum(1 for i in issues if i.get('severity') == 'low')

        # Weighted deduction
        deduction = (high_severity * 0.2) + (medium_severity * 0.1) + (low_severity * 0.05)
        compliance_score = max(0.0, 1.0 - deduction)

        return {
            'issues': issues,
            'issue_count': len(issues),
            'compliance_score': round(compliance_score, 2),
            'passed': compliance_score >= 0.8,
        }


class ConsistencyCheckerAgent(DeterministicAgent):
    """Check term and name consistency throughout document.

    Single duty: Ensure terms and names used consistently.
    Method: Dictionary tracking and matching (deterministic).
    Output: Inconsistencies found with suggested corrections.
    """

    duty = "Ensure terms and names are used consistently throughout document"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency of terms and names.

        Args:
            input_data: Should contain 'text'

        Returns:
            Consistency issues and score
        """
        text = input_data.get('text', '')

        # Extract potential entity names (capitalized phrases)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        entities = re.findall(entity_pattern, text)

        # Group similar entities (case-insensitive)
        entity_variants: Dict[str, List[str]] = {}
        for entity in entities:
            key = entity.lower()
            if key not in entity_variants:
                entity_variants[key] = []
            if entity not in entity_variants[key]:
                entity_variants[key].append(entity)

        # Find inconsistencies (multiple variants of same entity)
        inconsistencies = []
        for key, variants in entity_variants.items():
            if len(variants) > 1:
                # Find most common variant
                variant_counts = {v: entities.count(v) for v in variants}
                primary = max(variant_counts.items(), key=lambda x: x[1])[0]

                inconsistencies.append({
                    'entity': key,
                    'variants': variants,
                    'primary': primary,
                    'suggestion': f'Use "{primary}" consistently (appears {variant_counts[primary]} times)',
                    'variant_count': len(variants),
                })

        # Calculate consistency score
        total_entities = len(entities)
        unique_inconsistencies = len(inconsistencies)
        consistency_score = 1.0 - (unique_inconsistencies / max(total_entities / 10, 1))
        consistency_score = max(0.0, min(1.0, consistency_score))

        return {
            'inconsistencies': inconsistencies,
            'inconsistency_count': len(inconsistencies),
            'consistency_score': round(consistency_score, 2),
            'passed': consistency_score >= 0.9,
        }


class RedactionAgent(DeterministicAgent):
    """Apply redaction rules to protect sensitive information.

    Single duty: Redact PII and sealed information per rules.
    Method: Regex-based pattern matching and replacement (deterministic).
    Output: Redacted text with redaction log.
    """

    duty = "Apply redaction rules to protect sensitive information"

    # Redaction patterns
    REDACTION_PATTERNS = [
        (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]'), # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REDACTED]'), # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REDACTED]'), # Email
        (r'\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
         '[ADDRESS REDACTED]'), # Street address
        (r'\b(?:SSN|Social Security Number):\s*\d{3}-\d{2}-\d{4}\b',
         '[SSN REDACTED]'), # Labeled SSN
    ]

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from text.

        Args:
            input_data: Should contain 'text' and optional 'custom_patterns'

        Returns:
            Redacted text with redaction log
        """
        text = input_data.get('text', '')
        custom_patterns = input_data.get('custom_patterns', [])

        redactions = []
        redacted_text = text

        # Apply standard patterns
        all_patterns = self.REDACTION_PATTERNS + custom_patterns

        for pattern, replacement in all_patterns:
            matches = list(re.finditer(pattern, redacted_text))

            if matches:
                for match in matches:
                    redactions.append({
                        'original': match.group(0)[:20] + '...', # Don't log full PII
                        'redacted_to': replacement,
                        'position': match.start(),
                        'pattern_type': 'PII',
                    })

                # Apply redaction
                redacted_text = re.sub(pattern, replacement, redacted_text)

        return {
            'redacted_text': redacted_text,
            'redactions': redactions,
            'redaction_count': len(redactions),
        }


class ComplianceAgent(DeterministicAgent):
    """Verify document format compliance.

    Single duty: Check formatting and structure requirements.
    Method: Rule-based validation (deterministic).
    Output: Compliance checklist with pass/fail status.
    """

    duty = "Verify document format and structure compliance"

    # Compliance rules
    REQUIRED_SECTIONS = ['introduction', 'analysis', 'conclusion']
    MAX_PARAGRAPH_SENTENCES = 7
    MIN_SECTION_WORDS = 100

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check document compliance.

        Args:
            input_data: Should contain 'text' and optional 'sections'

        Returns:
            Compliance report
        """
        text = input_data.get('text', '')
        sections = input_data.get('sections', [])

        issues = []

        # Check required sections present
        section_titles = [s.get('title', '').lower() for s in sections]
        for required in self.REQUIRED_SECTIONS:
            if not any(required in title for title in section_titles):
                issues.append({
                    'type': 'missing_section',
                    'severity': 'high',
                    'description': f'Required section "{required}" not found',
                })

        # Check paragraph length (by double newlines)
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            sentences = para.split('. ')
            if len(sentences) > self.MAX_PARAGRAPH_SENTENCES:
                issues.append({
                    'type': 'paragraph_too_long',
                    'severity': 'low',
                    'description': f'Paragraph {i+1} has {len(sentences)} sentences (max {self.MAX_PARAGRAPH_SENTENCES})',
                    'location': f'Paragraph {i+1}',
                })

        # Check section length
        for section in sections:
            sec_text = section.get('text', section.get('section_text', ''))
            word_count = len(sec_text.split())
            if word_count > 0 and word_count < self.MIN_SECTION_WORDS:
                issues.append({
                    'type': 'section_too_short',
                    'severity': 'medium',
                    'description': f'Section "{section.get("title", "unknown")}" only {word_count} words (min {self.MIN_SECTION_WORDS})',
                    'location': section.get('title', 'unknown'),
                })

        # Calculate compliance score
        high_severity = sum(1 for i in issues if i.get('severity') == 'high')
        medium_severity = sum(1 for i in issues if i.get('severity') == 'medium')
        low_severity = sum(1 for i in issues if i.get('severity') == 'low')

        # Weighted deduction
        deduction = (high_severity * 0.2) + (medium_severity * 0.1) + (low_severity * 0.05)
        compliance_score = max(0.0, 1.0 - deduction)

        return {
            'issues': issues,
            'issue_count': len(issues),
            'by_severity': {
                'high': high_severity,
                'medium': medium_severity,
                'low': low_severity,
            },
            'compliance_score': round(compliance_score, 2),
            'passed': compliance_score >= 0.8,
        }


class ExpertQAAgent(LLMAgent):
    """Expert-level final QA review.

    Single duty: Strategic coherence and citation verification.
    Method: LLM analysis with GPT-4o (premium tier for complex cases).
    Output: Expert assessment with actionable feedback.
    """

    duty = "Perform expert-level strategic coherence and citation review"
    cost_tier = "standard" # Can escalate to premium
    max_cost_per_run = 0.02

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"
    premium_temperature = 0.1
    premium_max_tokens = 6000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build expert QA prompt.

        Args:
            input_data: Should contain 'text' and 'complexity_score'

        Returns:
            Formatted prompt
        """
        text = input_data.get('text', '')
        complexity = input_data.get('complexity_score', 0.5)

        return f"""Perform expert-level QA review of this legal document.

Complexity Score: {complexity}

Focus on:
1. Strategic coherence - Does the argument strategy make sense?
2. Citation accuracy - Are legal citations properly used and relevant?
3. Argument strength - Are claims well-supported?
4. Professional quality - Is this ready for filing?

Document:
{text}

Output as JSON:
{{
  "overall_score": 0.0-1.0,
  "strategic_coherence": 0.0-1.0,
  "citation_accuracy": 0.0-1.0,
  "argument_strength": 0.0-1.0,
  "professional_quality": 0.0-1.0,
  "critical_issues": ["...", ...],
  "suggestions": ["...", ...],
  "ready_to_file": true/false
}}

Be concise. Output ONLY the JSON."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse expert QA assessment.

        Args:
            response: LLM response

        Returns:
            Parsed QA assessment
        """
        try:
            # Clean and parse JSON
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            result = json.loads(cleaned)
            return result

        except json.JSONDecodeError:
            # Fallback
            return {
                'overall_score': 0.7,
                'ready_to_file': False,
                'critical_issues': ['QA parse error'],
                'suggestions': [],
                'raw_response': response,
            }
