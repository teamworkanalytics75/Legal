"""Legal research agent - analyzes case corpus and generates insights."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.agent import BackgroundAgent, AgentConfig


class LegalResearchAgent(BackgroundAgent):
    """Analyzes legal corpus and generates research insights."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.output_dir = Path("background_agents/outputs/research")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different output types
        (self.output_dir / "case_summaries").mkdir(exist_ok=True)
        (self.output_dir / "legal_principles").mkdir(exist_ok=True)
        (self.output_dir / "trend_analysis").mkdir(exist_ok=True)

    async def process(self, task: Any) -> Any:
        """
        Perform legal research analysis.

        Args:
            task: Dict with 'research_type' and parameters

        Returns:
            Dict with research findings
        """
        # Validate input
        if not isinstance(task, dict):
            self.logger.error(f"Invalid task data: {task}")
            return {'error': 'Task must be a dict'}

        research_type = task.get('research_type', 'summary')

        if research_type == 'summary':
            return await self._generate_case_summary(task)
        elif research_type == 'principles':
            return await self._extract_legal_principles(task)
        elif research_type == 'trends':
            return await self._analyze_trends(task)
        else:
            return {'error': f'Unknown research type: {research_type}'}

    async def _generate_case_summary(self, task: Dict) -> Dict:
        """Generate a summary of a case or set of cases."""
        cases = task.get('cases', [])

        if not cases:
            self.logger.warning("No cases provided for summary")
            return {'error': 'No cases provided', 'status': 'skipped'}

        prompt = f"""Analyze these legal cases and provide a comprehensive summary:

Cases: {json.dumps(cases, indent=2)}

Provide:
1. Overview of the cases
2. Common factual patterns
3. Legal issues presented
4. Outcomes and reasoning
5. Practical implications
6. Key takeaways for practitioners

Write in clear, professional language suitable for legal professionals."""

        summary = await self.llm_query(prompt, temperature=0.7, max_tokens=3000)

        result = {
            'research_type': 'summary',
            'timestamp': datetime.now().isoformat(),
            'cases_analyzed': len(cases),
            'summary': summary
        }

        # Save to file
        output_file = self.output_dir / "case_summaries" / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_file, 'w') as f:
            f.write(f"# Case Summary\n\n")
            f.write(f"**Generated:** {result['timestamp']}\n\n")
            f.write(f"**Cases Analyzed:** {result['cases_analyzed']}\n\n")
            f.write(f"## Summary\n\n{summary}\n")

        return result

    async def _extract_legal_principles(self, task: Dict) -> Dict:
        """Extract legal principles from cases."""
        cases = task.get('cases', [])

        prompt = f"""Extract the key legal principles from these cases:

Cases: {json.dumps(cases, indent=2)[:2000]}

For each principle, provide:
1. The principle statement
2. Supporting case(s)
3. Application context
4. Limitations or exceptions

Format as a structured list."""

        principles = await self.llm_query(prompt, temperature=0.5, max_tokens=2000)

        result = {
            'research_type': 'principles',
            'timestamp': datetime.now().isoformat(),
            'cases_analyzed': len(cases),
            'principles': principles
        }

        # Save to file
        output_file = self.output_dir / "legal_principles" / f"principles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_file, 'w') as f:
            f.write(f"# Legal Principles\n\n")
            f.write(f"**Generated:** {result['timestamp']}\n\n")
            f.write(f"{principles}\n")

        return result

    async def _analyze_trends(self, task: Dict) -> Dict:
        """Analyze trends across the corpus."""
        data = task.get('data', {})

        prompt = f"""Analyze trends in this legal data:

Data: {json.dumps(data, indent=2)[:2000]}

Identify:
1. Temporal trends (how things have changed over time)
2. Jurisdictional patterns
3. Outcome patterns
4. Emerging legal issues
5. Shifts in judicial reasoning

Provide specific examples and statistics where possible."""

        trends = await self.llm_query(prompt, temperature=0.6, max_tokens=2000)

        result = {
            'research_type': 'trends',
            'timestamp': datetime.now().isoformat(),
            'analysis': trends
        }

        # Save to file
        output_file = self.output_dir / "trend_analysis" / f"trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(output_file, 'w') as f:
            f.write(f"# Trend Analysis\n\n")
            f.write(f"**Generated:** {result['timestamp']}\n\n")
            f.write(f"{trends}\n")

        return result

