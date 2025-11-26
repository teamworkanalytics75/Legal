#!/usr/bin/env python3
"""Extract agent instructions from atomic agent code files.

This script harvests the `duty` strings and other instruction data from all
atomic agents to create a foundation for LangChain seed query generation.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional


def extract_agent_info_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Extract agent information from a Python file.

    Args:
        file_path: Path to the Python file containing agent definitions

    Returns:
        List of agent dictionaries with extracted information
    """
    agents = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Look for agent classes
                if 'Agent' in node.name and not node.name.startswith('_'):
                    agent_info = {
                        'class_name': node.name,
                        'file_path': str(file_path),
                        'duty': None,
                        'cost_tier': None,
                        'max_cost_per_run': None,
                        'meta_category': None,
                        'is_deterministic': None,
                        'docstring': ast.get_docstring(node)
                    }

                    # Extract class attributes
                    for item in node.body:
                        if isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    if target.id == 'duty' and isinstance(item.value, ast.Constant):
                                        agent_info['duty'] = item.value.value
                                    elif target.id == 'cost_tier' and isinstance(item.value, ast.Constant):
                                        agent_info['cost_tier'] = item.value.value
                                    elif target.id == 'max_cost_per_run' and isinstance(item.value, ast.Constant):
                                        agent_info['max_cost_per_run'] = item.value.value
                                    elif target.id == 'meta_category' and isinstance(item.value, ast.Constant):
                                        agent_info['meta_category'] = item.value.value
                                    elif target.id == 'is_deterministic' and isinstance(item.value, ast.Constant):
                                        agent_info['is_deterministic'] = item.value.value

                    # Only include if we found a duty (indicates it's a real agent)
                    if agent_info['duty']:
                        agents.append(agent_info)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    return agents


def extract_all_agents(source_dir: Path) -> List[Dict[str, Any]]:
    """Extract agent information from all Python files in source directory.

    Args:
        source_dir: Directory containing atomic agent files

    Returns:
        List of all extracted agent information
    """
    all_agents = []

    # Find all Python files in the atomic_agents directory
    for py_file in source_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            agents = extract_agent_info_from_file(py_file)
            all_agents.extend(agents)

    return all_agents


def categorize_agents(agents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize agents by their meta-category.

    Args:
        agents: List of agent information

    Returns:
        Dictionary mapping categories to agent lists
    """
    categories = {
        'completeness': [],
        'precision': [],
        'standard': [],
        'unknown': []
    }

    for agent in agents:
        category = agent.get('meta_category', 'unknown')
        if category in categories:
            categories[category].append(agent)
        else:
            categories['unknown'].append(agent)

    return categories


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract agent instructions from atomic agent code')
    parser.add_argument('--source', type=Path,
                       default=Path('writer_agents/code/atomic_agents'),
                       help='Source directory containing atomic agent files')
    parser.add_argument('--output', type=Path,
                       default=Path('config/agent_prompts.json'),
                       help='Output JSON file for extracted agent information')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: Source directory {args.source} does not exist")
        sys.exit(1)

    print(f"Extracting agent instructions from {args.source}")

    # Extract all agents
    agents = extract_all_agents(args.source)

    if args.verbose:
        print(f"Found {len(agents)} agents:")
        for agent in agents:
            print(f"  - {agent['class_name']}: {agent['duty']}")

    # Categorize agents
    categories = categorize_agents(agents)

    # Prepare output data
    output_data = {
        'extraction_info': {
            'source_directory': str(args.source),
            'total_agents': len(agents),
            'extraction_timestamp': str(Path().cwd()),
        },
        'agents': agents,
        'categories': categories,
        'summary': {
            'completeness_agents': len(categories['completeness']),
            'precision_agents': len(categories['precision']),
            'standard_agents': len(categories['standard']),
            'unknown_agents': len(categories['unknown'])
        }
    }

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(agents)} agents to {args.output}")
    print(f"Categories: {output_data['summary']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
