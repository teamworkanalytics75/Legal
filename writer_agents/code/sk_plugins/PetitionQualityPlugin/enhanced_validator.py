"""
Enhanced validator with position-aware validation and hierarchical constraint checking.

Extends the base validator with:
- Position-aware validation (checks where sections/arguments appear)
- Sequential constraint checking (Intel factors order)
- Hierarchical scoring (document → section → feature → granular)
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from ..base_plugin import kernel_function

# Import base validator functions
from .petition_validator_function import (
    load_quality_rules,
    extract_sections,
    count_words,
    count_citations
)


def load_constraint_system(version: str = "1.0") -> Dict[str, Any]:
    """Load constraint system from versioned JSON."""
    workspace_root = Path(__file__).parent.parent.parent.parent.parent.parent
    constraint_path = workspace_root / "case_law_data" / "config" / "constraint_system_versions" / f"v{version}_base.json"

    # Try generated version if base doesn't exist
    if not constraint_path.exists():
        constraint_path = workspace_root / "case_law_data" / "config" / "constraint_system_versions" / f"v{version}_generated.json"

    if not constraint_path.exists():
        raise FileNotFoundError(f"Could not find constraint system v{version} at {constraint_path}")

    with open(constraint_path, 'r') as f:
        return json.load(f)


def check_position_constraint(
    position: float,
    constraint: Dict[str, Any],
    doc_length: int,
    section_start: int = None,
    section_end: int = None
) -> Tuple[bool, str]:
    """
    Check if a position satisfies positional constraints.

    Args:
        position: Character position in document
        constraint: Position constraint from constraint system
        doc_length: Total document length
        section_start: Optional section start position
        section_end: Optional section end position

    Returns:
        Tuple of (is_valid, message)
    """
    if 'position' not in constraint:
        return True, ""

    pos_config = constraint['position']

    # Check relative position
    if pos_config.get('relative', False):
        relative_pos = position / doc_length if doc_length > 0 else 0

        char_start = pos_config.get('char_start', 0)
        char_end = pos_config.get('char_end', 1.0)

        if relative_pos < char_start or relative_pos > char_end:
            return False, f"Position {relative_pos:.2%} outside optimal range ({char_start:.2%}-{char_end:.2%})"

    # Check absolute position
    else:
        char_start = pos_config.get('char_start', 0)
        char_end = pos_config.get('char_end', doc_length)

        if position < char_start or position > char_end:
            return False, f"Position {position} outside optimal range ({char_start}-{char_end})"

    # Special case: last N chars
    if 'last_500_chars' in str(pos_config.get('char_start', '')):
        required_pos = doc_length - 500
        if position < required_pos:
            return False, f"Must appear in last 500 characters (currently at position {position})"

    return True, ""


def check_sequential_constraints(
    text: str,
    sections: Dict[str, str],
    sequential_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Check sequential constraints (e.g., Intel factors order).

    Returns:
        List of validation results for sequential constraints
    """
    results = []

    if 'intel_factors' in sequential_config:
        factors = sequential_config['intel_factors']

        # Find positions of each factor
        factor_positions = {}
        for factor_name, factor_config in factors.items():
            # Search for factor mentions
            factor_patterns = {
                'f1_participation': r'factor\s+1|first\s+factor|participation',
                'f2_receptivity': r'factor\s+2|second\s+factor|receptivity',
                'f3_circumvention': r'factor\s+3|third\s+factor|circumvent',
                'f4_burden': r'factor\s+4|fourth\s+factor|(?:undue|burden|intrusive)'
            }

            pattern = factor_patterns.get(factor_name, '')
            if pattern:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    factor_positions[factor_name] = match.start()

        # Check order
        if 'f1_participation' in factor_positions and 'f2_receptivity' in factor_positions:
            if factor_positions['f1_participation'] > factor_positions['f2_receptivity']:
                results.append({
                    'type': 'sequential',
                    'constraint': 'intel_factors',
                    'status': 'warning',
                    'message': 'Factor 1 (participation) should precede Factor 2 (receptivity)'
                })

    return results


def validate_hierarchical(
    text: str,
    constraint_system: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate petition against hierarchical constraint system.

    Returns:
        Dictionary with hierarchical validation results
    """
    results = {
        'passed': True,
        'overall_score': 0.0,
        'scores': {
            'document_level': 0.0,
            'section_level': 0.0,
            'feature_level': 0.0
        },
        'details': {
            'document_level': [],
            'section_level': [],
            'feature_level': []
        },
        'warnings': [],
        'errors': []
    }

    doc_length = len(text)
    word_count = count_words(text)
    char_count = len(text)

    # Document-level validation
    doc_level = constraint_system.get('document_level', {})

    # Word count
    if 'word_count' in doc_level:
        wc_constraint = doc_level['word_count']
        min_wc = wc_constraint.get('min', 0)
        ideal_wc = wc_constraint.get('ideal', 1200)
        max_wc = wc_constraint.get('max', 2000)

        if word_count < min_wc:
            results['errors'].append(f"Word count {word_count} below minimum {min_wc}")
            results['passed'] = False
        elif word_count > max_wc:
            results['errors'].append(f"Word count {word_count} exceeds maximum {max_wc}")
            results['passed'] = False
        else:
            # Score based on distance from ideal
            distance = abs(word_count - ideal_wc) / ideal_wc if ideal_wc > 0 else 1.0
            score = max(0, 1.0 - distance)
            results['scores']['document_level'] += score * wc_constraint.get('weight', 1.0)
            results['details']['document_level'].append({
                'constraint': 'word_count',
                'value': word_count,
                'score': score,
                'status': 'pass'
            })

    # Character count
    if 'char_count' in doc_level:
        cc_constraint = doc_level['char_count']
        max_cc = cc_constraint.get('max', 8000)

        if char_count > max_cc:
            results['errors'].append(f"Character count {char_count} exceeds maximum {max_cc}")
            results['passed'] = False
        else:
            score = 1.0 if char_count <= max_cc else 0.0
            results['scores']['document_level'] += score * cc_constraint.get('weight', 1.0)
            results['details']['document_level'].append({
                'constraint': 'char_count',
                'value': char_count,
                'score': score,
                'status': 'pass'
            })

    # Section-level validation
    sections = extract_sections(text)
    section_constraints = constraint_system.get('sections', {})

    for section_name, section_config in section_constraints.items():
        section_text = sections.get(section_name, '')

        if not section_text:
            # Check if section is required
            results['errors'].append(f"Required section '{section_name}' not found")
            results['passed'] = False
            continue

        # Check word count
        if 'word_count' in section_config:
            wc_config = section_config['word_count']
            section_wc = count_words(section_text)

            min_wc = wc_config.get('min', 0)
            ideal_wc = wc_config.get('ideal', 0)
            max_wc = wc_config.get('max', float('inf'))

            if section_wc < min_wc:
                results['warnings'].append(
                    f"{section_name} section: {section_wc} words (minimum: {min_wc})"
                )
            elif section_wc > max_wc:
                results['warnings'].append(
                    f"{section_name} section: {section_wc} words (maximum: {max_wc})"
                )

            # Score based on proximity to ideal
            if ideal_wc > 0:
                distance = abs(section_wc - ideal_wc) / ideal_wc
                score = max(0, 1.0 - distance * 0.5)
                weight = wc_config.get('weight', 1.0)
                results['scores']['section_level'] += score * weight

                results['details']['section_level'].append({
                    'section': section_name,
                    'constraint': 'word_count',
                    'value': section_wc,
                    'ideal': ideal_wc,
                    'score': score,
                    'status': 'pass' if min_wc <= section_wc <= max_wc else 'warning'
                })

        # Check position
        if section_text:
            # Find section position in document
            section_start = text.find(section_text)
            if section_start >= 0:
                is_valid, message = check_position_constraint(
                    section_start,
                    section_config,
                    doc_length
                )

                if not is_valid:
                    results['warnings'].append(f"{section_name} section: {message}")

                results['details']['section_level'].append({
                    'section': section_name,
                    'constraint': 'position',
                    'position': section_start,
                    'relative_position': section_start / doc_length if doc_length > 0 else 0,
                    'status': 'pass' if is_valid else 'warning',
                    'message': message
                })

    # Feature-level validation (simplified - would need feature extraction)
    feature_constraints = constraint_system.get('feature_constraints', {})

    # For now, just log that feature-level validation would happen here
    results['details']['feature_level'].append({
        'note': 'Feature-level validation requires full feature extraction pipeline'
    })

    # Calculate overall score using hierarchical weights
    scoring = constraint_system.get('scoring', {})
    hierarchical_weights = scoring.get('hierarchical', {
        'document_level': 0.3,
        'section_level': 0.4,
        'feature_level': 0.3
    })

    # Normalize scores
    doc_score = results['scores']['document_level'] / 10.0 if results['scores']['document_level'] > 0 else 0.0
    section_score = results['scores']['section_level'] / 20.0 if results['scores']['section_level'] > 0 else 0.0
    feature_score = 0.5  # Placeholder

    results['overall_score'] = (
        doc_score * hierarchical_weights['document_level'] +
        section_score * hierarchical_weights['section_level'] +
        feature_score * hierarchical_weights['feature_level']
    )

    # Check sequential constraints
    if 'argument' in section_constraints:
        arg_config = section_constraints['argument']
        if 'sequential_constraints' in arg_config:
            seq_results = check_sequential_constraints(
                text,
                sections,
                arg_config['sequential_constraints']
            )
            results['details']['section_level'].extend(seq_results)
            results['warnings'].extend([r['message'] for r in seq_results if r['status'] == 'warning'])

    return results


@kernel_function(
    description="Enhanced validation with position-aware and hierarchical constraint checking. Validates petition drafts against hierarchical constraint system.",
    name="ValidatePetitionConstraints"
)
def enhanced_validator_native_function(
    petition_text: str,
    constraint_version: str = "1.0",
    **_ignored: Any,
) -> str:
    """
    Validate petition against hierarchical constraint system.

    Args:
        petition_text: The petition text to validate
        constraint_version: Version of constraint system to use (default: "1.0")

    Returns:
        JSON string with hierarchical validation results
    """
    try:
        constraint_system = load_constraint_system(version=constraint_version)
        validation_results = validate_hierarchical(petition_text, constraint_system)

        # Format results
        result = {
            'passed': validation_results['passed'],
            'overall_score': round(validation_results['overall_score'], 3),
            'scores': {
                'document_level': round(validation_results['scores']['document_level'], 2),
                'section_level': round(validation_results['scores']['section_level'], 2),
                'feature_level': round(validation_results['scores']['feature_level'], 2)
            },
            'errors': validation_results['errors'],
            'warnings': validation_results['warnings'],
            'details': validation_results['details'],
            'constraint_version': constraint_version
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            'passed': False,
            'error': f"Validation failed: {str(e)}",
            'overall_score': 0.0
        }, indent=2)

