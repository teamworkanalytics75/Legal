#!/usr/bin/env python3
"""Generate LangChain capability-based seed queries from agent instructions.

This script converts extracted agent instructions into parametric LangChain
seed queries that teach universal skills and job-specific primitives.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


# Universal operator skills that every agent needs
UNIVERSAL_SKILLS = [
    {
        "id": "schema_discovery",
        "nl": "List all tables; for each, show 5 rows and column names",
        "priority": 1,
        "category": "schema"
    },
    {
        "id": "table_counts",
        "nl": "Show row counts per table, largest first",
        "priority": 1,
        "category": "schema"
    },
    {
        "id": "distinct_values",
        "nl": "Return distinct values of {column} and counts",
        "priority": 1,
        "category": "schema"
    },
    {
        "id": "text_search",
        "nl": "Find documents with {phrase} (case-insensitive) in extracted_text; return file_name, path, date",
        "priority": 1,
        "category": "search"
    },
    {
        "id": "phrase_cooccurrence",
        "nl": "Search for documents containing both {phrase1} and {phrase2} in the same record",
        "priority": 1,
        "category": "search"
    },
    {
        "id": "date_window",
        "nl": "List items between {start_date} and {end_date} ordered by date",
        "priority": 1,
        "category": "temporal"
    },
    {
        "id": "date_range",
        "nl": "Show the earliest and latest dates in date-like columns",
        "priority": 1,
        "category": "temporal"
    },
    {
        "id": "dedupe",
        "nl": "Return unique records keyed by content_hash, keeping the newest date_ingested",
        "priority": 2,
        "category": "data_quality"
    },
    {
        "id": "pagination",
        "nl": "Return top {n} results ordered by date with LIMIT and OFFSET",
        "priority": 2,
        "category": "performance"
    },
    {
        "id": "null_handling",
        "nl": "Count records where extracted_text is NULL or empty",
        "priority": 2,
        "category": "data_quality"
    },
    {
        "id": "audit_replay",
        "nl": "Replay the last 10 queries the agent ran and their token/cost from meta-memory",
        "priority": 2,
        "category": "audit"
    },
    {
        "id": "cost_report",
        "nl": "Report average token usage & cost per query from meta-memory log",
        "priority": 2,
        "category": "audit"
    },
    {
        "id": "parsing_resilience",
        "nl": "If the request is malformed or includes emojis 🧠, still return best effort results and a note",
        "priority": 3,
        "category": "error_handling"
    }
]


def generate_job_primitives(agent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate job-specific primitives for an agent based on its duty.

    Args:
        agent: Agent information dictionary

    Returns:
        List of primitive query templates
    """
    duty = agent.get('duty', '').lower()
    class_name = agent.get('class_name', '')
    primitives = []

    # Citation-related agents
    if 'citation' in duty or 'citation' in class_name.lower():
        primitives.extend([
            {
                "name": "citation_extract",
                "nl": "Find records matching citation pattern {regex}",
                "params": ["regex"],
                "category": "citation"
            },
            {
                "name": "citation_normalize",
                "nl": "Normalize citations to {format} using templates",
                "params": ["format"],
                "category": "citation"
            },
            {
                "name": "citation_verify",
                "nl": "Verify citations against case law database using SQL queries",
                "params": [],
                "category": "citation"
            }
        ])

    # Precedent-related agents
    if 'precedent' in duty or 'precedent' in class_name.lower():
        primitives.extend([
            {
                "name": "precedent_find",
                "nl": "Find relevant precedent cases using {criteria}",
                "params": ["criteria"],
                "category": "precedent"
            },
            {
                "name": "precedent_rank",
                "nl": "Rank precedent cases by relevance to {legal_issue}",
                "params": ["legal_issue"],
                "category": "precedent"
            },
            {
                "name": "precedent_summarize",
                "nl": "Create 1-2 sentence summaries of precedent cases",
                "params": [],
                "category": "precedent"
            }
        ])

    # Fact extraction agents
    if 'fact' in duty or 'fact' in class_name.lower():
        primitives.extend([
            {
                "name": "fact_extract",
                "nl": "Extract discrete facts from documents as structured data",
                "params": [],
                "category": "fact_extraction"
            },
            {
                "name": "entity_cooccurrence",
                "nl": "Find records where {entity1} and {entity2} co-occur",
                "params": ["entity1", "entity2"],
                "category": "fact_extraction"
            },
            {
                "name": "context_window",
                "nl": "Extract text surrounding {phrase} with {n} characters",
                "params": ["phrase", "n"],
                "category": "fact_extraction"
            }
        ])

    # Writing agents
    if 'write' in duty or 'write' in class_name.lower():
        primitives.extend([
            {
                "name": "outline_build",
                "nl": "Build structured section outline with headings and objectives",
                "params": [],
                "category": "writing"
            },
            {
                "name": "section_write",
                "nl": "Write a single section of legal document based on outline and context",
                "params": [],
                "category": "writing"
            },
            {
                "name": "paragraph_write",
                "nl": "Write a single paragraph from a brief content prompt",
                "params": [],
                "category": "writing"
            }
        ])

    # Quality assurance agents
    if any(word in duty for word in ['check', 'verify', 'consistency', 'grammar', 'style']):
        primitives.extend([
            {
                "name": "logic_check",
                "nl": "Check argument logic and identify missing premises or gaps",
                "params": [],
                "category": "quality"
            },
            {
                "name": "consistency_check",
                "nl": "Ensure terms and names are used consistently throughout document",
                "params": [],
                "category": "quality"
            },
            {
                "name": "grammar_fix",
                "nl": "Fix grammar and typos in text",
                "params": [],
                "category": "quality"
            }
        ])

    # Export agents
    if 'export' in duty or 'export' in class_name.lower():
        primitives.extend([
            {
                "name": "markdown_export",
                "nl": "Export document to Markdown format",
                "params": [],
                "category": "export"
            },
            {
                "name": "docx_export",
                "nl": "Export document to DOCX format",
                "params": [],
                "category": "export"
            }
        ])

    # Statute/exhibit agents
    if any(word in duty for word in ['statute', 'exhibit', 'locate', 'fetch']):
        primitives.extend([
            {
                "name": "statute_locate",
                "nl": "Locate full text of cited statutes from database",
                "params": [],
                "category": "reference"
            },
            {
                "name": "exhibit_fetch",
                "nl": "Retrieve exhibit documents from file system or database",
                "params": [],
                "category": "reference"
            }
        ])

    # If no specific primitives were generated, create generic ones
    if not primitives:
        primitives.extend([
            {
                "name": "generic_search",
                "nl": "Search for {query} in documents",
                "params": ["query"],
                "category": "generic"
            },
            {
                "name": "generic_extract",
                "nl": "Extract information related to {topic}",
                "params": ["topic"],
                "category": "generic"
            }
        ])

    return primitives


def generate_domain_anchors(agent: Dict[str, Any], count: int = 2) -> List[Dict[str, Any]]:
    """Generate thin domain anchors for an agent.

    Args:
        agent: Agent information dictionary
        count: Number of domain anchors to generate

    Returns:
        List of domain anchor queries
    """
    duty = agent.get('duty', '').lower()
    class_name = agent.get('class_name', '')
    anchors = []

    # Generate anchors based on agent type
    if 'citation' in duty or 'citation' in class_name.lower():
        anchors.extend([
            {"nl": "Find citations to Massachusetts cases", "limit": 50},
            {"nl": "List Bluebook-formatted citations", "limit": 100}
        ])
    elif 'precedent' in duty or 'precedent' in class_name.lower():
        anchors.extend([
            {"nl": "Find cases about discrimination in Massachusetts", "limit": 100},
            {"nl": "Show cases mentioning constructive knowledge", "limit": 50}
        ])
    elif 'fact' in duty or 'fact' in class_name.lower():
        anchors.extend([
            {"nl": "Extract facts about institutional actions", "limit": 100},
            {"nl": "Find procedural facts with dates", "limit": 50}
        ])
    elif 'write' in duty or 'write' in class_name.lower():
        anchors.extend([
            {"nl": "Build outline for legal memorandum", "limit": 10},
            {"nl": "Write section on legal standards", "limit": 5}
        ])
    else:
        # Generic anchors
        anchors.extend([
            {"nl": "Find documents related to legal proceedings", "limit": 100},
            {"nl": "Extract information about case timeline", "limit": 50}
        ])

    return anchors[:count]


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate LangChain capability-based seed queries')
    parser.add_argument('--agent-prompts', type=Path,
                       default=Path('config/agent_prompts.json'),
                       help='Input file with extracted agent information')
    parser.add_argument('--output', type=Path,
                       default=Path('config/langchain_capability_seeds.json'),
                       help='Output JSON file for seed queries')
    parser.add_argument('--add-universal-skills', action='store_true', default=True,
                       help='Include universal operator skills')
    parser.add_argument('--add-domain-anchors', type=int, default=2,
                       help='Number of domain anchors per agent')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if not args.agent_prompts.exists():
        print(f"Error: Input file {args.agent_prompts} does not exist")
        sys.exit(1)

    print(f"Generating capability seeds from {args.agent_prompts}")

    # Load agent information
    with open(args.agent_prompts, 'r', encoding='utf-8') as f:
        agent_data = json.load(f)

    agents = agent_data.get('agents', [])

    if args.verbose:
        print(f"Processing {len(agents)} agents")

    # Generate seed data
    seed_data = {
        'generation_info': {
            'source_file': str(args.agent_prompts),
            'total_agents': len(agents),
            'universal_skills_count': len(UNIVERSAL_SKILLS) if args.add_universal_skills else 0,
            'domain_anchors_per_agent': args.add_domain_anchors
        }
    }

    # Add universal skills
    if args.add_universal_skills:
        seed_data['universal_skills'] = UNIVERSAL_SKILLS

    # Generate agent-specific seeds
    seed_data['agents'] = {}
    total_primitives = 0
    total_anchors = 0

    for agent in agents:
        class_name = agent.get('class_name', 'UnknownAgent')

        # Generate job primitives
        primitives = generate_job_primitives(agent)

        # Generate domain anchors
        anchors = generate_domain_anchors(agent, args.add_domain_anchors)

        seed_data['agents'][class_name] = {
            'agent_info': {
                'duty': agent.get('duty'),
                'cost_tier': agent.get('cost_tier'),
                'meta_category': agent.get('meta_category'),
                'is_deterministic': agent.get('is_deterministic')
            },
            'job_primitives': primitives,
            'domain_anchors': anchors
        }

        total_primitives += len(primitives)
        total_anchors += len(anchors)

        if args.verbose:
            print(f"  - {class_name}: {len(primitives)} primitives, {len(anchors)} anchors")

    # Calculate totals
    total_queries = (len(UNIVERSAL_SKILLS) if args.add_universal_skills else 0) + total_primitives + total_anchors

    seed_data['summary'] = {
        'total_queries': total_queries,
        'universal_skills': len(UNIVERSAL_SKILLS) if args.add_universal_skills else 0,
        'job_primitives': total_primitives,
        'domain_anchors': total_anchors,
        'estimated_cost': total_queries * 0.0003  # Rough estimate
    }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(seed_data, f, indent=2, ensure_ascii=False)

    print(f"Generated {total_queries} seed queries to {args.output}")
    print(f"Breakdown: {seed_data['summary']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
