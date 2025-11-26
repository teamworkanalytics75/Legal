"""Citation network builder - maps relationships between cases."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from ..core.agent import BackgroundAgent, AgentConfig


class CitationNetworkAgent(BackgroundAgent):
    """Builds and analyzes citation networks from legal cases."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.output_dir = Path("background_agents/outputs/networks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not NETWORKX_AVAILABLE:
            self.logger.warning("NetworkX not available - network analysis will be limited")

    async def process(self, task: Any) -> Any:
        """
        Build or analyze citation network.

        Args:
            task: Dict with 'cases' containing case data

        Returns:
            Dict with network analysis results
        """
        # Validate input
        if not isinstance(task, dict):
            self.logger.error(f"Invalid task data: {task}")
            return {'error': 'Task must be a dict'}

        cases = task.get('cases', [])

        if not cases:
            self.logger.warning("No cases provided for citation network")
            return {'error': 'No cases provided', 'status': 'skipped'}

        # Extract citations using LLM
        citations = await self._extract_citations(cases)

        # Build network if NetworkX available
        if NETWORKX_AVAILABLE:
            network_stats = self._build_network(citations)
        else:
            network_stats = {'note': 'NetworkX not available'}

        result = {
            'timestamp': datetime.now().isoformat(),
            'cases_processed': len(cases),
            'citations_found': len(citations),
            'network_stats': network_stats,
            'citations': citations
        }

        # Save result
        output_file = self.output_dir / f"citation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    async def _extract_citations(self, cases: List[Dict]) -> List[Dict]:
        """Extract citation relationships from cases."""
        all_citations = []

        for case in cases[:10]:  # Limit to first 10 to avoid token limits
            case_name = case.get('name', case.get('case_name', 'Unknown'))
            case_text = str(case)[:2000]  # Limit text

            prompt = f"""Extract all case citations from this legal document:

Document: {case_text}

List each citation in this format:
- Citing case: [current case]
- Cited case: [referenced case]
- Citation context: [brief description of why cited]

Return as a JSON array of objects with fields: citing_case, cited_case, context"""

            try:
                response = await self.llm_query(prompt, temperature=0.3)

                # Try to parse JSON
                import re
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    citations = json.loads(json_match.group())
                    all_citations.extend(citations)

            except Exception as e:
                self.logger.error(f"Citation extraction failed for {case_name}: {e}")

        return all_citations

    def _build_network(self, citations: List[Dict]) -> Dict:
        """Build network graph from citations."""
        if not NETWORKX_AVAILABLE:
            return {}

        G = nx.DiGraph()

        # Add edges from citations
        for citation in citations:
            citing = citation.get('citing_case', 'Unknown')
            cited = citation.get('cited_case', 'Unknown')
            context = citation.get('context', '')

            G.add_edge(citing, cited, context=context)

        # Calculate network metrics
        try:
            pagerank = nx.pagerank(G)
            most_influential = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]

            stats = {
                'total_cases': G.number_of_nodes(),
                'total_citations': G.number_of_edges(),
                'average_citations_per_case': G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
                'most_influential_cases': [
                    {'case': case, 'score': float(score)}
                    for case, score in most_influential
                ]
            }

            # Save graph
            output_file = self.output_dir / f"citation_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gexf"
            nx.write_gexf(G, output_file)
            self.logger.info(f"Saved network graph to {output_file}")

            return stats

        except Exception as e:
            self.logger.error(f"Network analysis failed: {e}")
            return {'error': str(e)}

