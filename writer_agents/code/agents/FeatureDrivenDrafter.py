"""
Feature-Driven Drafting Agent for AutoGen.

Drafts motions/petitions following success formula rules extracted from trained model.
Uses quality rules to guide drafting and validation.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from ..agents import BaseAutoGenAgent, AgentFactory


def load_quality_rules() -> Dict[str, Any]:
    """Load petition quality rules from config."""
    workspace_root = Path(__file__).parent.parent.parent.parent.parent.parent
    config_path = workspace_root / "case_law_data" / "config" / "petition_quality_rules.json"

    if not config_path.exists():
        config_path = Path("case_law_data/config/petition_quality_rules.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find petition_quality_rules.json at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def build_system_message(rules: Dict[str, Any]) -> str:
    """Build system message for feature-driven drafter from quality rules."""

    # Extract key guidelines
    drafting_guidelines = rules.get('drafting_guidelines', {})
    section_structure = rules.get('section_structure', {})
    positive_signals = rules.get('positive_signals', {})
    negative_signals = rules.get('negative_signals', {})

    guidelines_text = []
    for guideline_name, guideline in drafting_guidelines.items():
        priority = guideline.get('priority', 'medium')
        guidance = guideline.get('guidance', '')
        guidelines_text.append(f"- **{guideline_name.replace('_', ' ').title()}** ({priority} priority): {guidance}")

    # Extract section word count targets
    section_targets = []
    section_word_counts = section_structure.get('section_word_counts', {})
    for section_name, config in section_word_counts.items():
        ideal = config.get('ideal', 0)
        min_val = config.get('min', 0)
        max_val = config.get('max', float('inf'))
        section_targets.append(f"- **{section_name.replace('_', ' ').title()}**: {ideal} words (target range: {min_val}-{max_val})")

    # Word count guidance
    total_word_count = section_structure.get('total_word_count', {})
    word_count_ideal = total_word_count.get('ideal', 1200)
    word_count_max = total_word_count.get('max', 2000)

    system_message = f"""You are a Feature-Driven Motion Drafter. Your task is to draft legal motions/petitions that align with proven success patterns extracted from a machine learning model trained on {rules['metadata'].get('n_samples', 107)} successful petitions.

## Core Principles

1. **Follow the Success Formula**: The quality rules below are derived from analyzing {rules['metadata'].get('model_accuracy', '87%')} accurate model that identified patterns in successful petitions.

2. **Structure is Critical**: Successful petitions follow a clear section structure with specific word count targets.

3. **Conciseness Wins**: Keep the total document length within {word_count_max} words. The ideal length is approximately {word_count_ideal} words.

## Required Section Structure

You MUST include the following sections in this order:
{chr(10).join([f"- {section}" for section in section_structure.get('required_sections', [])])}

### Section Word Count Targets

{chr(10).join(section_targets)}

**Total Document Length**: Target {word_count_ideal} words (maximum {word_count_max} words)

## Drafting Guidelines

{chr(10).join(guidelines_text)}

## Quality Checklist

Before finalizing your draft, ensure:

1. **Section Structure**:
   - All required sections are present
   - Section word counts are within target ranges
   - Sections are clearly labeled with headings

2. **Length & Conciseness**:
   - Total word count is between 650-{word_count_max} words
   - Character count is under 8000 characters
   - No unnecessary repetition or verbosity

3. **Citations**:
   - Use 2-5 citations strategically
   - Avoid citation overkill (more than 10 citations)
   - Citations should be relevant and well-integrated

4. **Language**:
   - Avoid excessive jurisdictional arguments (limit mentions of 'jurisdiction')
   - Minimize standing arguments unless essential
   - Focus on merits of the case

5. **Request Clarity**:
   - Clearly state the requested relief
   - Use phrases like "Petitioner respectfully requests" or "WHEREFORE"
   - Be specific about what you're asking the court to do

## Workflow

1. Analyze the provided case facts and legal requirements
2. Draft each section according to word count targets
3. Ensure overall structure follows the required format
4. Validate against the quality checklist above
5. Revise to meet word count and structure requirements

Remember: The goal is to create a petition that matches the mathematical formula for success, not just a legally correct document. Every word counts."""

    return system_message


class FeatureDrivenDrafter(BaseAutoGenAgent):
    """AutoGen agent that drafts motions/petitions following success formula rules."""

    def __init__(self, factory: AgentFactory, quality_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize feature-driven drafter.

        Args:
            factory: AgentFactory for creating the agent
            quality_rules: Optional quality rules dict (will load from file if not provided)
        """
        if quality_rules is None:
            quality_rules = load_quality_rules()

        system_message = build_system_message(quality_rules)
        super().__init__(factory, "FeatureDrivenDrafter", system_message)
        self.quality_rules = quality_rules

    async def draft_petition(
        self,
        case_facts: str,
        requested_relief: str,
        legal_basis: str,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Draft a petition following success formula rules.

        Args:
            case_facts: Factual background for the petition
            requested_relief: What the petitioner is requesting
            legal_basis: Legal authority/basis for the request
            additional_context: Any additional context or requirements

        Returns:
            Drafted petition text
        """
        task = f"""Draft a motion/petition with the following requirements:

**Case Facts:**
{case_facts}

**Requested Relief:**
{requested_relief}

**Legal Basis:**
{legal_basis}

{f"**Additional Context:**{chr(10)}{additional_context}" if additional_context else ""}

**Your Task:**
1. Draft a complete motion/petition following the section structure and word count targets specified in your system instructions
2. Ensure all required sections are included
3. Keep total length within the target range ({self.quality_rules.get('section_structure', {}).get('total_word_count', {}).get('ideal', 1200)} words ideal, {self.quality_rules.get('section_structure', {}).get('total_word_count', {}).get('max', 2000)} words maximum)
4. Use 2-5 strategic citations
5. Clearly state the requested relief in the conclusion section
6. Follow all drafting guidelines from your system instructions

**Output:** Return the complete drafted motion/petition with all sections clearly labeled."""

        return await self.run(task)

    async def revise_petition(
        self,
        current_draft: str,
        feedback: str,
        target_improvements: Optional[list] = None
    ) -> str:
        """
        Revise a petition draft based on feedback.

        Args:
            current_draft: Current draft text
            feedback: Feedback on what to improve
            target_improvements: Optional list of specific improvements to target

        Returns:
            Revised draft
        """
        improvements_text = ""
        if target_improvements:
            improvements_text = f"\n**Specific Improvements Needed:**\n" + "\n".join(f"- {item}" for item in target_improvements)

        task = f"""Revise the following petition draft based on the feedback provided:

**Current Draft:**
{current_draft}

**Feedback:**
{feedback}
{improvements_text}

**Your Task:**
1. Revise the draft to address the feedback
2. Maintain compliance with section structure and word count requirements
3. Ensure the revised draft still meets all quality guidelines
4. Preserve the clarity of the requested relief

**Output:** Return the complete revised draft."""

        return await self.run(task)

