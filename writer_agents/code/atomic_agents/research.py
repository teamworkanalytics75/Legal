"""Research atomic agents.

Agents for fact extraction, precedent finding/ranking, statute location, exhibit fetching.
Mix of deterministic (DB queries) and LLM-based agents.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from ..atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
except ImportError:
    from atomic_agent import DeterministicAgent, LLMAgent, AgentFactory


class FactExtractorAgent(LLMAgent):
    """Extract discrete facts from documents.

    Single duty: Extract facts as structured data.
    Method: LLM extraction with structured output (gpt-4o-mini).
    Output: List of fact objects.
    """

    duty = "Extract discrete facts from documents as structured data"
    cost_tier = "mini"
    max_cost_per_run = 0.005

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build fact extraction prompt.

        Args:
            input_data: Should contain 'text' or 'documents'

        Returns:
            Formatted prompt
        """
        text = input_data.get('text', '')
        documents = input_data.get('documents', [])

        if documents:
            # Extract from multiple documents
            doc_text = "\n\n---\n\n".join([
                f"Document {i+1}:\n{doc.get('text', '')[:500]}"
                for i, doc in enumerate(documents[:3]) # Limit to 3 docs for cost
            ])
        else:
            doc_text = text[:2000] # Limit length for cost

        return f"""Extract discrete facts from these documents.

Documents:
{doc_text}

Output as JSON array:
[
  {{"fact": "...", "source": "...", "confidence": 0.0-1.0}},
  ...
]

Extract 5-10 key facts only. Output ONLY the JSON array."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse extracted facts.

        Args:
            response: LLM response

        Returns:
            Parsed facts
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

            facts = json.loads(cleaned)

            return {
                'facts': facts,
                'fact_count': len(facts),
            }

        except json.JSONDecodeError:
            return {
                'facts': [],
                'fact_count': 0,
                'parse_error': True,
            }


class PrecedentFinderAgent(LLMAgent):
    """Find relevant precedent cases.

    Single duty: Identify candidate precedent cases.
    Method: DB query + mini LLM ranking.
    Output: List of relevant case citations.
    """

    duty = "Find relevant precedent cases from database"
    cost_tier = "mini"
    max_cost_per_run = 0.003

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build precedent search prompt.

        Args:
            input_data: Should contain 'query' or 'legal_issue'

        Returns:
            Formatted prompt
        """
        query = input_data.get('query', '')
        legal_issue = input_data.get('legal_issue', '')

        return f"""Given this legal issue, suggest 3-5 relevant precedent case names to search for:

Legal Issue: {legal_issue or query}

Output as JSON array of case names:
["Case Name 1", "Case Name 2", ...]

Output ONLY the JSON array."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse precedent suggestions.

        Args:
            response: LLM response

        Returns:
            Parsed precedent list
        """
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            precedents = json.loads(cleaned)

            return {
                'precedents': precedents,
                'precedent_count': len(precedents),
            }

        except json.JSONDecodeError:
            return {
                'precedents': [],
                'precedent_count': 0,
            }


class PrecedentRankerAgent(LLMAgent):
    """Rank precedent cases by relevance.

    Single duty: Score and rank found precedents.
    Method: LLM relevance scoring (gpt-4o-mini).
    Output: Ranked list of precedents with scores.
    """

    duty = "Rank precedent cases by relevance to legal issue"
    cost_tier = "mini"
    max_cost_per_run = 0.004

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"
    premium_temperature = 0.1
    premium_max_tokens = 6000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build ranking prompt.

        Args:
            input_data: Should contain 'precedents' and 'legal_issue'

        Returns:
            Formatted prompt
        """
        precedents = input_data.get('precedents', [])
        legal_issue = input_data.get('legal_issue', '')

        return f"""Rank these precedent cases by relevance to the legal issue.

Legal Issue: {legal_issue}

Precedents:
{json.dumps(precedents, indent=2)}

Output as JSON array (ordered by relevance, highest first):
[
  {{"case": "...", "relevance_score": 0.0-1.0, "reason": "brief reason"}},
  ...
]

Output ONLY the JSON array."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse ranked precedents.

        Args:
            response: LLM response

        Returns:
            Parsed ranking
        """
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)

            ranked = json.loads(cleaned)

            return {
                'ranked_precedents': ranked,
                'top_precedent': ranked[0] if ranked else None,
            }

        except json.JSONDecodeError:
            return {
                'ranked_precedents': [],
                'top_precedent': None,
            }


class PrecedentSummarizerAgent(LLMAgent):
    """Summarize precedent cases.

    Single duty: Create 1-2 sentence summaries of cases.
    Method: LLM summarization (gpt-4o-mini).
    Output: Brief case summaries.
    """

    duty = "Create 1-2 sentence summaries of precedent cases"
    cost_tier = "mini"
    max_cost_per_run = 0.003

    # Precision machine configuration
    meta_category = "precision"
    model_tier = "premium"
    output_strategy = "optimize"
    premium_temperature = 0.1
    premium_max_tokens = 6000

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build summarization prompt.

        Args:
            input_data: Should contain 'case_text' or 'precedent'

        Returns:
            Formatted prompt
        """
        case_text = input_data.get('case_text', '')
        precedent_name = input_data.get('precedent_name', 'Case')

        return f"""Summarize this legal precedent in 1-2 sentences:

Case: {precedent_name}
{case_text[:1000]}

Output ONLY the summary (1-2 sentences), nothing else."""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse case summary.

        Args:
            response: LLM response

        Returns:
            Parsed summary
        """
        summary = response.strip()

        return {
            'summary': summary,
            'sentence_count': len(summary.split('. ')),
        }


class StatuteLocatorAgent(DeterministicAgent):
    """Find statute text from database.

    Single duty: Locate full text of cited statutes.
    Method: Database queries (deterministic).
    Output: Statute text with metadata.
    """

    duty = "Locate full text of cited statutes from database"

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Locate statute in database.

        Args:
            input_data: Should contain 'statute_citation' and optional 'db_connection'

        Returns:
            Statute information
        """
        citation = input_data.get('statute_citation', '')
        db_connection = input_data.get('db_connection')

        # Placeholder for actual DB query
        # In production: SELECT * FROM statutes WHERE citation LIKE ?

        statute_info = {
            'citation': citation,
            'found': False,
            'text': None,
            'title': None,
        }

        if db_connection:
            # TODO: Actual database lookup
            pass

        return {
            'statute': statute_info,
            'located': statute_info['found'],
        }


class ExhibitFetcherAgent(DeterministicAgent):
    """Fetch exhibit files from storage.

    Single duty: Retrieve exhibit documents.
    Method: File system or database lookup (deterministic).
    Output: Exhibit file paths or content.
    """

    duty = "Retrieve exhibit documents from file system or database"

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"

    async def _deterministic_execution(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch exhibit files.

        Args:
            input_data: Should contain 'exhibit_ids' or 'exhibit_paths'

        Returns:
            Exhibit information
        """
        exhibit_ids = input_data.get('exhibit_ids', [])
        exhibit_paths = input_data.get('exhibit_paths', [])

        # Placeholder for actual file/DB fetching
        # In production: read files or query database

        exhibits = []
        for exhibit_id in exhibit_ids:
            exhibits.append({
                'id': exhibit_id,
                'path': f'exhibits/{exhibit_id}.pdf', # Placeholder
                'fetched': False, # Would be True after actual fetch
            })

        return {
            'exhibits': exhibits,
            'fetched_count': 0, # Would be actual count
        }
