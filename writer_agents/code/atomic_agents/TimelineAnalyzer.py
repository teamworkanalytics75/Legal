"""Timeline analysis atomic agent.

Specialized agent for temporal reasoning, timeline construction, and date-based analysis.
Uses LangChain for database queries and LLM for timeline synthesis.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
    from ..langchain_integration import LangChainSQLAgent
except ImportError:
    from atomic_agent import DeterministicAgent, LLMAgent, AgentFactory
    from langchain_integration import LangChainSQLAgent


class TimelineAnalyzerAgent(LLMAgent):
    """Analyze temporal relationships and construct chronological timelines.

    Single duty: Extract events with dates and build chronological timelines.
    Method: LangChain database queries + LLM timeline synthesis.
    Output: Structured timeline with events, dates, and relationships.
    """

    duty = "Extract events with dates and build chronological timelines"
    cost_tier = "mini"
    max_cost_per_run = 0.01

    # Completeness machine configuration
    meta_category = "completeness"
    model_tier = "premium"
    output_strategy = "maximize"
    premium_temperature = 0.2  # Lower temperature for more consistent date parsing
    premium_max_tokens = 6000

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
        """Build timeline analysis prompt.

        Args:
            input_data: Should contain 'query', 'entity', 'date_range', or 'text'

        Returns:
            Formatted prompt for timeline analysis
        """
        query = input_data.get('query', '')
        entity = input_data.get('entity', '')
        date_range = input_data.get('date_range', ())
        text = input_data.get('text', '')

        if query:
            # Natural language query for timeline analysis
            prompt = f"""Analyze the following query for timeline information and extract all events with dates:

QUERY: {query}

Please:
1. Identify all events mentioned with specific dates
2. Extract any date ranges or temporal constraints
3. Identify entities involved in each event
4. Note any temporal relationships (before, after, during, etc.)
5. Flag any potential contradictions in timing

Format your response as a structured timeline with:
- Event descriptions
- Specific dates (YYYY-MM-DD format when possible)
- Entities involved
- Source references
- Confidence levels for each date/event

If the query involves a specific cutoff date (like "on or before April 19, 2019"), highlight this constraint and its implications."""

        elif entity and date_range:
            # Entity-specific timeline within date range
            start_date, end_date = date_range
            prompt = f"""Create a timeline for entity "{entity}" between {start_date} and {end_date}.

Extract all events involving {entity} within this date range and organize them chronologically.

For each event, include:
- Event description
- Specific date (YYYY-MM-DD format)
- Other entities involved
- Event significance
- Source information

Pay special attention to:
- Events that establish knowledge or awareness
- Communications or statements
- Actions taken or not taken
- Legal implications of timing"""

        elif text:
            # Extract timeline from provided text
            prompt = f"""Extract timeline information from the following text:

TEXT:
{text}

Please:
1. Identify all events with dates
2. Extract temporal relationships
3. Build chronological sequence
4. Note any temporal contradictions
5. Identify key entities and their involvement over time

Format as a structured timeline with dates, events, and relationships."""

        else:
            prompt = """Please provide timeline analysis input. I can analyze:
- Natural language queries for timeline information
- Entity-specific timelines within date ranges
- Text documents for timeline extraction

Please specify what you'd like me to analyze."""

        return prompt

    def _extract_timeline_from_langchain(self, query: str) -> Dict[str, Any]:
        """Use LangChain to extract timeline-related data from database."""
        if not self.langchain_agent:
            return {"error": "LangChain agent not available"}

        try:
            # Query for timeline-related information
            timeline_query = f"""
            Find all events, communications, or actions with dates related to: {query}

            Focus on:
            - Emails with dates
            - Statements or announcements with dates
            - Legal actions with dates
            - Communications between parties
            - Any temporal constraints or deadlines

            Extract the date, event description, and involved parties for each item.
            """

            result = self.langchain_agent.query_evidence(timeline_query)
            return {"langchain_results": result}

        except Exception as e:
            self.logger.error(f"LangChain timeline query failed: {e}")
            return {"error": str(e)}

    def _parse_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates and events from text using regex patterns."""
        events = []

        # Common date patterns
        date_patterns = [
            r'(\w+ \d{1,2}, \d{4})',  # "April 19, 2019"
            r'(\d{1,2}/\d{1,2}/\d{4})',  # "04/19/2019"
            r'(\d{4}-\d{2}-\d{2})',  # "2019-04-19"
            r'(on or before \w+ \d{1,2}, \d{4})',  # "on or before April 19, 2019"
            r'(before \w+ \d{1,2}, \d{4})',  # "before April 19, 2019"
            r'(after \w+ \d{1,2}, \d{4})',  # "after April 19, 2019"
        ]

        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(1)
                start_pos = max(0, match.start() - 100)
                end_pos = min(len(text), match.end() + 100)
                context = text[start_pos:end_pos]

                events.append({
                    "date_text": date_text,
                    "context": context.strip(),
                    "position": match.start()
                })

        return events

    def _synthesize_timeline(self, langchain_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """Combine LangChain results with LLM analysis into structured timeline."""
        timeline = {
            "events": [],
            "temporal_constraints": [],
            "entities": set(),
            "confidence_score": 0.0,
            "sources": []
        }

        try:
            # Parse LLM response for structured timeline
            if "timeline" in llm_response.lower() or "events" in llm_response.lower():
                # Try to extract structured data from LLM response
                lines = llm_response.split('\n')
                current_event = {}

                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_event:
                            timeline["events"].append(current_event)
                            current_event = {}
                        continue

                    # Look for date patterns
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2}|\w+ \d{1,2}, \d{4})', line)
                    if date_match:
                        if current_event:
                            timeline["events"].append(current_event)
                        current_event = {
                            "date": date_match.group(1),
                            "description": line,
                            "confidence": 0.8  # Default confidence
                        }
                    elif current_event and line.startswith('-'):
                        # Additional details for current event
                        current_event["description"] += " " + line[1:].strip()

                if current_event:
                    timeline["events"].append(current_event)

            # Add LangChain results if available
            if "langchain_results" in langchain_data:
                timeline["sources"].append("LangChain Database Query")
                timeline["confidence_score"] += 0.3

            # Sort events by date
            timeline["events"].sort(key=lambda x: self._parse_date_for_sorting(x.get("date", "")))

            # Extract entities
            for event in timeline["events"]:
                # Simple entity extraction (could be enhanced)
                text = event.get("description", "")
                if "Harvard" in text:
                    timeline["entities"].add("Harvard")
                if "OGC" in text:
                    timeline["entities"].add("OGC")
                if "Xi" in text:
                    timeline["entities"].add("Xi")

            timeline["entities"] = list(timeline["entities"])

        except Exception as e:
            self.logger.error(f"Timeline synthesis failed: {e}")
            timeline["error"] = str(e)

        return timeline

    def _parse_date_for_sorting(self, date_str: str) -> datetime:
        """Parse date string for chronological sorting."""
        try:
            # Try different date formats
            formats = [
                "%Y-%m-%d",
                "%B %d, %Y",
                "%m/%d/%Y",
                "%d/%m/%Y"
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # If no format works, return epoch
            return datetime(1970, 1, 1)

        except Exception:
            return datetime(1970, 1, 1)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute timeline analysis.

        Args:
            input_data: Input containing query, entity, date_range, or text

        Returns:
            Structured timeline analysis
        """
        try:
            # Initialize LangChain if needed
            self._initialize_langchain()

            # Build prompt
            prompt = self._build_prompt(input_data)

            # Get LLM response
            llm_response = self._call_llm(prompt)

            # Extract timeline data using LangChain if query provided
            langchain_data = {}
            if input_data.get('query'):
                langchain_data = self._extract_timeline_from_langchain(input_data['query'])

            # Synthesize final timeline
            timeline = self._synthesize_timeline(langchain_data, llm_response)

            # Add metadata
            timeline["agent"] = "TimelineAnalyzerAgent"
            timeline["timestamp"] = datetime.now().isoformat()
            timeline["input_query"] = input_data.get('query', '')

            return {
                "success": True,
                "timeline": timeline,
                "raw_llm_response": llm_response,
                "langchain_data": langchain_data
            }

        except Exception as e:
            self.logger.error(f"Timeline analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timeline": {"events": [], "error": str(e)}
            }


