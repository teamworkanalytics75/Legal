"""Drafting atomic agents.

Agents for outline building, section writing, paragraph writing, and transitions.
All agents use LLM (gpt-4o-mini) with cost-optimized prompts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from ..atomic_agent import LLMAgent, AgentFactory
except ImportError:
    from atomic_agent import LLMAgent, AgentFactory


class OutlineBuilderAgent(LLMAgent):
    """Build structured section outline for legal document.

    Single duty: Create section headings and structure.
    Method: LLM analysis of case insights to determine sections.
    Output: Ordered list of sections with objectives.
    """

    duty = "Build structured section outline with headings and objectives"
    cost_tier = "mini"
    max_cost_per_run = 0.003

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build outline creation prompt.

        Args:
            input_data: Should contain 'insights' and 'summary'

        Returns:
            Formatted prompt
        """
        summary = input_data.get('summary', '')
        insights = input_data.get('insights', {})

        return f"""Create a structured outline for a legal memorandum.

Case Summary:
{summary}

Task: Generate section headings only. Output as JSON array with this structure:
[
  {{"section_id": "1", "title": "Introduction", "objective": "...", "estimated_words": 300}},
  {{"section_id": "2", "title": "...", "objective": "...", "estimated_words": 500}},
  ...
]

Keep it concise - 4-6 sections maximum. Focus on standard legal memo structure.
Output ONLY the JSON array, no other text."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse outline from LLM response.

        Args:
            response: LLM response

        Returns:
            Parsed outline dict
        """
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            sections = json.loads(cleaned)

            return {
                'sections': sections,
                'section_count': len(sections),
                'total_estimated_words': sum(s.get('estimated_words', 0) for s in sections),
            }

        except json.JSONDecodeError:
            # Fallback: extract section-like lines
            sections = []
            for i, line in enumerate(response.split("\n"), 1):
                if line.strip() and not line.strip().startswith("#"):
                    sections.append({
                        'section_id': str(i),
                        'title': line.strip(),
                        'objective': 'Generated from outline',
                        'estimated_words': 400,
                    })

            return {
                'sections': sections[:6], # Cap at 6
                'section_count': len(sections[:6]),
                'total_estimated_words': len(sections[:6]) * 400,
            }


class SectionWriterAgent(LLMAgent):
    """Write a single section of legal document.

    Single duty: Draft prose for one section based on outline.
    Method: LLM generation from section objective and context.
    Output: Written section text.
    """

    duty = "Write a single section of legal document based on outline and context"
    cost_tier = "mini"
    max_cost_per_run = 0.01

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build section writing prompt.

        Args:
            input_data: Should contain 'section', 'context', optional 'research'

        Returns:
            Formatted prompt
        """
        section = input_data.get('section', {})
        context = input_data.get('context', '')
        research = input_data.get('research', {})

        section_title = section.get('title', 'Section')
        section_objective = section.get('objective', 'Write section content')
        target_words = section.get('estimated_words', 400)

        prompt = f"""Write the following section for a legal memorandum:

Section: {section_title}
Objective: {section_objective}
Target Length: ~{target_words} words

Context:
{context}
"""

        if research:
            prompt += f"\n\nResearch Findings:\n{json.dumps(research, indent=2)}\n"

        prompt += """
Write professional legal prose. Be concise and precise. Use formal tone.
Output ONLY the section text, no headers or meta-commentary."""

        return prompt

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse section text from response.

        Args:
            response: LLM response

        Returns:
            Parsed section dict
        """
        # Clean response
        text = response.strip()

        return {
            'section_text': text,
            'word_count': len(text.split()),
        }


class ParagraphWriterAgent(LLMAgent):
    """Write a single paragraph within a section.

    Single duty: Draft one paragraph from a brief prompt.
    Method: LLM generation with strict length control.
    Output: Single paragraph text.
    """

    duty = "Write a single paragraph from a brief content prompt"
    cost_tier = "mini"
    max_cost_per_run = 0.005

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build paragraph writing prompt.

        Args:
            input_data: Should contain 'content_prompt' and optional 'max_words'

        Returns:
            Formatted prompt
        """
        content_prompt = input_data.get('content_prompt', '')
        max_words = input_data.get('max_words', 100)

        return f"""Write a single paragraph (max {max_words} words) about:

{content_prompt}

Use formal legal tone. Be concise. Output ONLY the paragraph text, nothing else."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse paragraph from response.

        Args:
            response: LLM response

        Returns:
            Parsed paragraph dict
        """
        text = response.strip()

        return {
            'paragraph_text': text,
            'word_count': len(text.split()),
        }


class TransitionAgent(LLMAgent):
    """Create smooth transitions between sections.

    Single duty: Write transition sentences connecting sections.
    Method: LLM analysis of section boundaries.
    Output: Transition sentences to insert.
    """

    duty = "Create smooth transitions between document sections"
    cost_tier = "mini"
    max_cost_per_run = 0.002

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build transition creation prompt.

        Args:
            input_data: Should contain 'section_a' and 'section_b'

        Returns:
            Formatted prompt
        """
        section_a = input_data.get('section_a', {})
        section_b = input_data.get('section_b', {})

        title_a = section_a.get('title', 'Previous section')
        title_b = section_b.get('title', 'Next section')
        text_a = section_a.get('text', '')[-200:] # Last 200 chars
        text_b = section_b.get('text', '')[:200] # First 200 chars

        return f"""Create a 1-2 sentence transition between these sections:

From section "{title_a}":
...{text_a}

To section "{title_b}":
{text_b}...

Write ONLY the transition sentence(s), nothing else. Keep it brief and professional."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse transition from response.

        Args:
            response: LLM response

        Returns:
            Parsed transition dict
        """
        text = response.strip()

        return {
            'transition_text': text,
            'word_count': len(text.split()),
        }
