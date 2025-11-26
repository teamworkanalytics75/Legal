"""Pattern detection agent - identifies patterns in case outcomes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.agent import BackgroundAgent, AgentConfig


class PatternDetectionAgent(BackgroundAgent):
    """Detects patterns in legal case outcomes and reasoning."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.output_dir = Path("background_agents/outputs/patterns")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, task: Any) -> Any:
        """
        Detect patterns in case data.

        Args:
            task: Dict with 'cases' containing case data

        Returns:
            Dict with detected patterns
        """
        # Validate input
        if not isinstance(task, dict):
            self.logger.error(f"Invalid task data: {task}")
            return {'error': 'Task must be a dict'}

        cases = task.get('cases', [])

        if len(cases) < 5:
            self.logger.warning(f"Only {len(cases)} cases available, need at least 5")
            return {'error': 'Need at least 5 cases to detect patterns', 'status': 'skipped'}

        # Analyze patterns using LLM
        patterns = await self._detect_patterns(cases)

        result = {
            'timestamp': datetime.now().isoformat(),
            'cases_analyzed': len(cases),
            'patterns': patterns
        }

        # Save result
        output_file = self.output_dir / f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    async def _detect_patterns(self, cases: List[Dict]) -> Dict:
        """Use LLM to detect patterns."""
        # Prepare case summaries (limit to avoid token limits)
        case_summaries = []
        for i, case in enumerate(cases[:20]):  # Max 20 cases
            summary = {
                'id': i,
                'name': case.get('name', case.get('case_name', f'Case {i}')),
                'outcome': case.get('outcome', case.get('ruling', 'Unknown')),
                'facts': str(case.get('facts', case.get('summary', '')))[:500]
            }
            case_summaries.append(summary)

        prompt = f"""Analyze these legal cases and identify patterns:

Cases: {json.dumps(case_summaries, indent=2)}

Identify:
1. **Success Factors**: What factors correlate with positive outcomes?
2. **Failure Factors**: What factors correlate with negative outcomes?
3. **Jurisdictional Patterns**: Do outcomes vary by jurisdiction?
4. **Temporal Patterns**: Have outcomes changed over time?
5. **Factual Patterns**: What fact patterns appear repeatedly?
6. **Legal Reasoning Patterns**: What reasoning approaches do courts use?

For each pattern, provide:
- Description of the pattern
- Supporting cases (by ID)
- Statistical observation (e.g., "appears in 60% of successful cases")
- Practical implication

Return as JSON with these sections."""

        try:
            response = await self.llm_query(prompt, temperature=0.6, max_tokens=3000)

            # Try to parse JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                patterns = json.loads(json_match.group())
            else:
                patterns = {'raw_analysis': response}

            return patterns

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            return {'error': str(e)}

