"""
Petition Validator Native Function for Semantic Kernel.

Validates petition drafts against success formula quality rules.
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..base_plugin import kernel_function


def load_quality_rules() -> Dict[str, Any]:
    """Load petition quality rules from config."""
    # Try to find the config file relative to this file
    workspace_root = Path(__file__).parent.parent.parent.parent.parent.parent
    config_path = workspace_root / "case_law_data" / "config" / "petition_quality_rules.json"

    if not config_path.exists():
        # Fallback path
        config_path = Path("case_law_data/config/petition_quality_rules.json")

    if not config_path.exists():
        raise FileNotFoundError(f"Could not find petition_quality_rules.json at {config_path}")

    with open(config_path, 'r') as f:
        return json.load(f)


def extract_sections(text: str) -> Dict[str, str]:
    """Extract sections from petition text."""
    sections = {
        'introduction': '',
        'legal_standard': '',
        'factual_background': '',
        'argument': '',
        'conclusion': ''
    }

    # Section detection patterns (from BuildPetitionFeatures.py)
    patterns = {
        'introduction': [
            r'\bI\.\s+INTRODUCTION\b',
            r'\bINTRODUCTION\b',
            r'^INTRODUCTION',
            r'\b1\.\s+INTRODUCTION\b'
        ],
        'legal_standard': [
            r'\b(?:II\.|2\.)\s+LEGAL\s+STANDARD\b',
            r'\bLEGAL\s+STANDARD\b',
            r'\bSTANDARD\s+OF\s+REVIEW\b',
            r'\bLEGAL\s+AUTHORITY\b'
        ],
        'factual_background': [
            r'\b(?:III\.|3\.)\s+FACTUAL\s+BACKGROUND\b',
            r'\bFACTUAL\s+BACKGROUND\b',
            r'\bBACKGROUND\b',
            r'\bFACTS\b'
        ],
        'argument': [
            r'\b(?:IV\.|4\.)\s+ARGUMENT\b',
            r'\bARGUMENT\b',
            r'\bDISCUSSION\b',
            r'\bANALYSIS\b'
        ],
        'conclusion': [
            r'\b(?:V\.|5\.|VIII\.|VIII\.)\s+CONCLUSION\b',
            r'\bCONCLUSION\b',
            r'\bREQUESTED\s+RELIEF\b',
            r'\bWHEREFORE\b'
        ]
    }

    for section_name, section_patterns in patterns.items():
        combined_pattern = '|'.join(section_patterns)
        match = re.search(combined_pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Extract next ~2000 chars as section content
            start = match.end()
            end = min(len(text), start + 2000)
            sections[section_name] = text[start:end]

    return sections


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def count_citations(text: str) -> int:
    """Count citations in text (simple heuristic)."""
    # Look for citation patterns like "v.", "U.S.", numbers, etc.
    citation_patterns = [
        r'\d+\s+[A-Z]\.\s*\d+',  # "542 U.S."
        r'\d+\s+F\.\s*(?:2d|3d|Supp\.)',  # "123 F.2d"
        r'v\.\s+[A-Z]',  # "v. Name"
    ]

    count = 0
    for pattern in citation_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))

    return count


def validate_feature_values(text: str, rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate petition features against quality rules.

    Returns:
        Dictionary with validation results for each feature
    """
    results = {
        'passed': True,
        'warnings': [],
        'errors': [],
        'feature_scores': {},
        'overall_score': 0.0
    }

    word_count = count_words(text)
    char_count = len(text)
    citation_count = count_citations(text)
    citation_rate = (citation_count / word_count * 1000) if word_count > 0 else 0

    sections = extract_sections(text)

    # Check positive signals
    if 'positive_signals' in rules:
        for feature, config in rules['positive_signals'].items():
            if 'word_count' in feature or 'char_count' in feature:
                continue  # Handle separately

            if feature == 'citation_rate':
                value = citation_rate
                min_val = config.get('min', 0)
                ideal = config.get('ideal', 2)
                max_val = config.get('max', 5)

                if value < min_val:
                    results['warnings'].append(f"{feature}: {value:.2f} below minimum {min_val}")
                    results['passed'] = False
                elif value > max_val:
                    results['warnings'].append(f"{feature}: {value:.2f} above maximum {max_val}")
                else:
                    # Score based on distance from ideal
                    if value <= ideal:
                        score = (value - min_val) / (ideal - min_val) if ideal > min_val else 1.0
                    else:
                        score = 1.0 - ((value - ideal) / (max_val - ideal)) if max_val > ideal else 1.0
                    results['feature_scores'][feature] = max(0, min(1, score))

    # Check negative signals (things to minimize)
    if 'negative_signals' in rules:
        for feature, config in rules['negative_signals'].items():
            if feature == 'word_count':
                value = word_count
                min_val = config.get('min', 0)
                ideal = config.get('ideal', 1200)
                max_val = config.get('max', 2000)

                if value > max_val:
                    results['errors'].append(f"Word count {value} exceeds maximum {max_val}. Keep concise!")
                    results['passed'] = False
                elif value < min_val:
                    results['warnings'].append(f"Word count {value} below minimum {min_val}")
                else:
                    # Score: closer to ideal is better
                    if ideal > 0:
                        distance = abs(value - ideal) / ideal
                        results['feature_scores'][feature] = max(0, 1.0 - distance)

            elif feature == 'char_count':
                value = char_count
                max_val = config.get('max', 8000)

                if value > max_val:
                    results['errors'].append(f"Character count {value} exceeds maximum {max_val}")
                    results['passed'] = False
            elif feature == 'citation_count':
                value = citation_count
                max_val = config.get('max', 10)

                if value > max_val:
                    results['warnings'].append(f"Citation count {value} exceeds recommended maximum {max_val}")

    # Check section structure
    if 'section_structure' in rules:
        required_sections = rules['section_structure'].get('required_sections', [])
        section_word_counts = rules['section_structure'].get('section_word_counts', {})

        for section in required_sections:
            section_key = section.lower().replace(' ', '_')
            if section_key in sections:
                section_text = sections[section_key]
                if not section_text.strip():
                    results['errors'].append(f"Section '{section}' is empty or not found")
                    results['passed'] = False
                else:
                    word_count_section = count_words(section_text)
                    if section_key in section_word_counts:
                        expected = section_word_counts[section_key]
                        ideal = expected.get('ideal', 0)
                        min_val = expected.get('min', 0)
                        max_val = expected.get('max', float('inf'))

                        if word_count_section < min_val:
                            results['warnings'].append(f"{section} section: {word_count_section} words (minimum: {min_val})")
                        elif word_count_section > max_val:
                            results['warnings'].append(f"{section} section: {word_count_section} words (maximum: {max_val})")
                        else:
                            # Score based on proximity to ideal
                            if ideal > 0:
                                distance = abs(word_count_section - ideal) / ideal
                                results['feature_scores'][f"{section_key}_word_count"] = max(0, 1.0 - distance * 0.5)
            else:
                results['errors'].append(f"Required section '{section}' not found")
                results['passed'] = False

    # Calculate overall score
    if results['feature_scores']:
        results['overall_score'] = sum(results['feature_scores'].values()) / len(results['feature_scores'])
    else:
        results['overall_score'] = 0.5 if results['passed'] else 0.0

    return results


@kernel_function(
    description="Validates a petition draft against quality rules extracted from successful petitions. Returns validation results with scores, warnings, and errors.",
    name="ValidatePetitionQuality"
)
def petition_validator_native_function(
    petition_text: str,
    **_ignored: Any,
) -> str:
    """
    Validate a petition draft against success formula quality rules.

    Args:
        petition_text: The petition text to validate

    Returns:
        JSON string with validation results
    """
    try:
        rules = load_quality_rules()
        validation_results = validate_feature_values(petition_text, rules)

        # Format results
        result = {
            'passed': validation_results['passed'],
            'overall_score': round(validation_results['overall_score'], 2),
            'errors': validation_results['errors'],
            'warnings': validation_results['warnings'],
            'feature_scores': {k: round(v, 2) for k, v in validation_results['feature_scores'].items()},
            'recommendations': []
        }

        # Add recommendations based on failures
        if not validation_results['passed']:
            result['recommendations'].append("Review errors and warnings above")

        if validation_results['overall_score'] < 0.7:
            result['recommendations'].append("Overall score is below 0.7 - consider revising to improve alignment with successful petitions")

        if 'word_count' in validation_results['feature_scores']:
            if validation_results['feature_scores']['word_count'] < 0.6:
                result['recommendations'].append("Consider adjusting document length to be more concise")

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            'passed': False,
            'error': f"Validation failed: {str(e)}",
            'overall_score': 0.0
        }, indent=2)
