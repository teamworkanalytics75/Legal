"""Enhanced research agents with LangChain integration.

Research agents that use LangChain SQLDatabaseToolkit for evidence retrieval
while maintaining compatibility with existing atomic agent patterns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..atomic_agent import LLMAgent, AgentFactory
    from ..langchain_integration import EvidenceRetrievalAgent
except ImportError:
    from atomic_agent import LLMAgent, AgentFactory
    from langchain_integration import EvidenceRetrievalAgent


class EnhancedPrecedentFinderAgent(LLMAgent):
    """Find relevant precedent cases using LangChain for evidence retrieval.

    Single duty: Identify candidate precedent cases using natural language database queries.
    Method: LangChain SQLDatabaseToolkit + LLM ranking.
    Output: List of relevant case citations with evidence.
    """

    duty = "Find relevant precedent cases using LangChain database queries"
    cost_tier = "mini"
    max_cost_per_run = 0.005  # Slightly higher due to LangChain usage

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def __init__(
        self,
        factory: AgentFactory,
        db_path: Optional[Path] = None,
        enable_langchain: bool = True,
        **kwargs
    ):
        """Initialize enhanced precedent finder.

        Args:
            factory: Agent factory for LLM configuration
            db_path: Path to SQLite database (defaults to lawsuit.db)
            enable_langchain: Whether to use LangChain for evidence retrieval
            **kwargs: Additional arguments for parent class
        """
        super().__init__(factory, **kwargs)

        # Set up database path
        if db_path is None:
            db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

        self.db_path = db_path
        self.enable_langchain = enable_langchain

        # Initialize evidence retrieval agent
        self.evidence_agent: Optional[EvidenceRetrievalAgent] = None
        if enable_langchain and db_path.exists():
            try:
                self.evidence_agent = EvidenceRetrievalAgent(
                    db_path=db_path,
                    factory=factory,
                    enable_langchain=True,
                    fallback_to_manual=True
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain evidence agent: {e}")
                self.evidence_agent = None

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build precedent search prompt.

        Args:
            input_data: Should contain 'query' or 'legal_issue'

        Returns:
            Formatted prompt
        """
        query = input_data.get('query', '')
        legal_issue = input_data.get('legal_issue', '')
        search_term = legal_issue or query

        return f"""Given this legal issue, suggest 3-5 relevant precedent case names to search for:

Legal Issue: {search_term}

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

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute precedent finding with LangChain evidence retrieval.

        Args:
            input_data: Input data containing legal issue/query

        Returns:
            Enhanced results with evidence from LangChain
        """
        # First, get precedent suggestions using LLM
        precedent_result = await super().execute(input_data)

        if not precedent_result.get('precedents'):
            return precedent_result

        # Enhance with LangChain evidence retrieval
        if self.evidence_agent:
            try:
                # Search for evidence about each precedent
                enhanced_precedents = []
                for precedent in precedent_result['precedents']:
                    # Use LangChain to find evidence about this precedent
                    evidence_result = self.evidence_agent.search_evidence(
                        query=f"Find evidence about precedent case: {precedent}",
                        limit=3,
                        context=f"Legal issue: {input_data.get('legal_issue', '')}"
                    )

                    enhanced_precedent = {
                        "case": precedent,
                        "evidence_found": evidence_result.get("success", False),
                        "evidence_count": len(evidence_result.get("results", [])),
                        "evidence_preview": evidence_result.get("answer", "")[:200] if evidence_result.get("success") else None
                    }

                    if evidence_result.get("success") and evidence_result.get("results"):
                        enhanced_precedent["evidence_results"] = evidence_result["results"]

                    enhanced_precedents.append(enhanced_precedent)

                # Update result with enhanced data
                precedent_result['enhanced_precedents'] = enhanced_precedents
                precedent_result['langchain_used'] = True
                precedent_result['evidence_retrieval_method'] = 'langchain_sql'

            except Exception as e:
                print(f"LangChain evidence retrieval failed: {e}")
                precedent_result['langchain_error'] = str(e)
                precedent_result['langchain_used'] = False
        else:
            precedent_result['langchain_used'] = False
            precedent_result['evidence_retrieval_method'] = 'none'

        return precedent_result

    # ------------------------------------------------------------------ #
    # LangChain metrics helpers
    # ------------------------------------------------------------------ #

    def reset_langchain_metrics(self) -> None:
        if self.evidence_agent and hasattr(self.evidence_agent, "reset_langchain_metrics"):
            self.evidence_agent.reset_langchain_metrics()  # type: ignore[attr-defined]

    def get_langchain_metrics(self) -> Dict[str, Any]:
        if self.evidence_agent and hasattr(self.evidence_agent, "get_langchain_metrics"):
            return self.evidence_agent.get_langchain_metrics()  # type: ignore[attr-defined]
        return {"queries_count": 0, "cost_estimate": 0.0}


class EnhancedFactExtractorAgent(LLMAgent):
    """Extract discrete facts using LangChain for document retrieval.

    Single duty: Extract facts as structured data using LangChain evidence retrieval.
    Method: LangChain SQLDatabaseToolkit + LLM extraction.
    Output: List of fact objects with source evidence.
    """

    duty = "Extract discrete facts using LangChain document retrieval"
    cost_tier = "mini"
    max_cost_per_run = 0.007  # Higher due to LangChain usage

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.3
    premium_max_tokens = 8000

    def __init__(
        self,
        factory: AgentFactory,
        db_path: Optional[Path] = None,
        enable_langchain: bool = True,
        **kwargs
    ):
        """Initialize enhanced fact extractor.

        Args:
            factory: Agent factory for LLM configuration
            db_path: Path to SQLite database (defaults to lawsuit.db)
            enable_langchain: Whether to use LangChain for evidence retrieval
            **kwargs: Additional arguments for parent class
        """
        super().__init__(factory, **kwargs)

        # Set up database path
        if db_path is None:
            db_path = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")

        self.db_path = db_path
        self.enable_langchain = enable_langchain

        # Initialize evidence retrieval agent
        self.evidence_agent: Optional[EvidenceRetrievalAgent] = None
        if enable_langchain and db_path.exists():
            try:
                self.evidence_agent = EvidenceRetrievalAgent(
                    db_path=db_path,
                    factory=factory,
                    enable_langchain=True,
                    fallback_to_manual=True
                )
            except Exception as e:
                print(f"Warning: Failed to initialize LangChain evidence agent: {e}")
                self.evidence_agent = None

    def _build_prompt(self, input_data: Dict[str, Any]) -> str:
        """Build fact extraction prompt.

        Args:
            input_data: Should contain 'text', 'documents', or 'query'

        Returns:
            Formatted prompt
        """
        text = input_data.get('text', '')
        documents = input_data.get('documents', [])
        query = input_data.get('query', '')

        if documents:
            # Extract from multiple documents
            doc_text = "\n\n---\n\n".join([
                f"Document {i+1}:\n{doc.get('text', '')[:500]}"
                for i, doc in enumerate(documents[:3]) # Limit to 3 docs for cost
            ])
        elif text:
            doc_text = text[:2000] # Limit length for cost
        else:
            doc_text = "No documents provided for fact extraction."

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

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact extraction with LangChain document retrieval.

        Args:
            input_data: Input data containing query or documents

        Returns:
            Enhanced results with facts extracted from LangChain-retrieved documents
        """
        # If we have a query but no documents, use LangChain to retrieve documents
        if self.evidence_agent and input_data.get('query') and not input_data.get('documents'):
            try:
                # Use LangChain to find relevant documents
                evidence_result = self.evidence_agent.search_evidence(
                    query=f"Find documents relevant to: {input_data['query']}",
                    limit=5,
                    context=input_data.get('context', '')
                )

                if evidence_result.get("success") and evidence_result.get("results"):
                    # Convert LangChain results to document format
                    documents = []
                    for result in evidence_result["results"]:
                        documents.append({
                            "text": result.get("preview", ""),
                            "id": result.get("id"),
                            "length": result.get("length", 0),
                            "source": "langchain_retrieval"
                        })

                    # Add documents to input data
                    input_data = input_data.copy()
                    input_data['documents'] = documents
                    input_data['langchain_documents'] = True

            except Exception as e:
                print(f"LangChain document retrieval failed: {e}")
                input_data['langchain_error'] = str(e)

        # Execute standard fact extraction
        result = await super().execute(input_data)

        # Add metadata about LangChain usage
        result['langchain_used'] = input_data.get('langchain_documents', False)
        result['evidence_retrieval_method'] = 'langchain_sql' if result['langchain_used'] else 'manual'

        return result


    # ------------------------------------------------------------------ #
    # LangChain metrics helpers
    # ------------------------------------------------------------------ #

    def reset_langchain_metrics(self) -> None:
        if self.evidence_agent and hasattr(self.evidence_agent, "reset_langchain_metrics"):
            self.evidence_agent.reset_langchain_metrics()  # type: ignore[attr-defined]

    def get_langchain_metrics(self) -> Dict[str, Any]:
        if self.evidence_agent and hasattr(self.evidence_agent, "get_langchain_metrics"):
            return self.evidence_agent.get_langchain_metrics()  # type: ignore[attr-defined]
        return {"queries_count": 0, "cost_estimate": 0.0}


# Factory functions for easy integration
def create_enhanced_precedent_finder(
    factory: AgentFactory,
    db_path: Optional[Path] = None,
    **kwargs
) -> EnhancedPrecedentFinderAgent:
    """Create enhanced precedent finder with LangChain integration.

    Args:
        factory: Agent factory for configuration
        db_path: Path to SQLite database
        **kwargs: Additional arguments

    Returns:
        Configured EnhancedPrecedentFinderAgent instance
    """
    return EnhancedPrecedentFinderAgent(factory, db_path, **kwargs)


def create_enhanced_fact_extractor(
    factory: AgentFactory,
    db_path: Optional[Path] = None,
    **kwargs
) -> EnhancedFactExtractorAgent:
    """Create enhanced fact extractor with LangChain integration.

    Args:
        factory: Agent factory for configuration
        db_path: Path to SQLite database
        **kwargs: Additional arguments

    Returns:
        Configured EnhancedFactExtractorAgent instance
    """
    return EnhancedFactExtractorAgent(factory, db_path, **kwargs)
