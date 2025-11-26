#!/usr/bin/env python3
"""
Section Utilities - Helper functions for extracting and analyzing sections.

Shared utilities for section-specific plugins.
"""

import re
import logging
from typing import Dict, Optional, Tuple, List, Any

logger = logging.getLogger(__name__)

# Section detection patterns (from extract_perfect_outline_features.py)
SECTION_PATTERNS = {
    'introduction': r'\b(introduction|intro)\b',
    'legal_standard': r'\b(legal standard|legal framework|governing law|applicable law)\b',
    'factual_background': r'\b(factual background|facts|background|factual basis)\b',
    'privacy_harm': r'\b(privacy|good cause|privacy harm|privacy interest)\b',
    'danger_safety': r'\b(danger|safety|threat|risk|harassment|retaliation)\b',
    'public_interest': r'\b(public interest|public access|public right|transparency)\b',
    'balancing_test': r'\b(balancing|balance|weigh|outweigh)\b',
    'protective_measures': r'\b(protective measures|proposed relief|requested relief|protective order)\b',
    'conclusion': r'\b(conclusion|concluding|for the foregoing reasons)\b',
}


def extract_section_text(text: str, section_name: str) -> Optional[str]:
    """
    Extract text for a specific section from document.
    
    Args:
        text: Full document text
        section_name: Name of section to extract
        
    Returns:
        Section text or None if section not found
    """
    if section_name not in SECTION_PATTERNS:
        return None
    
    text_lower = text.lower()
    pattern = SECTION_PATTERNS[section_name]
    
    # Find section start
    matches = list(re.finditer(pattern, text_lower))
    if not matches:
        return None
    
    start_pos = matches[0].start()
    
    # Find section end (start of next section or end of document)
    end_pos = len(text)
    
    # Find all section positions
    all_section_positions = []
    for name, pat in SECTION_PATTERNS.items():
        for match in re.finditer(pat, text_lower):
            all_section_positions.append((match.start(), name))
    
    # Sort by position
    all_section_positions.sort()
    
    # Find next section after this one
    for pos, name in all_section_positions:
        if pos > start_pos and name != section_name:
            end_pos = pos
            break
    
    section_text = text[start_pos:end_pos].strip()
    return section_text if section_text else None


def find_section_positions(text: str) -> List[Tuple[int, str]]:
    """
    Find all section positions in document.
    
    Returns:
        List of (position, section_name) tuples, sorted by position
    """
    text_lower = text.lower()
    positions = []
    
    for section_name, pattern in SECTION_PATTERNS.items():
        matches = list(re.finditer(pattern, text_lower))
        if matches:
            positions.append((matches[0].start(), section_name))
    
    positions.sort()
    return positions


def extract_paragraphs_from_section(section_text: str) -> List[str]:
    """
    Extract paragraphs from a section.
    
    Args:
        section_text: Text of the section
        
    Returns:
        List of paragraph strings
    """
    if not section_text:
        return []
    
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'(?:\r?\n\s*){2,}', section_text.strip())
    
    # If no double newlines found, try single newlines
    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in section_text.splitlines() if p.strip()]
    else:
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def extract_sentences_from_paragraph(paragraph_text: str) -> List[str]:
    """
    Extract sentences from a paragraph.
    
    Args:
        paragraph_text: Text of the paragraph
        
    Returns:
        List of sentence strings
    """
    if not paragraph_text:
        return []
    
    # Simple sentence splitting: split on period, exclamation, question mark
    # followed by space and capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.!?])$', paragraph_text)
    
    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # If no sentences found, treat entire paragraph as one sentence
    if not sentences:
        sentences = [paragraph_text.strip()]
    
    return sentences


def count_words_in_sentence(sentence_text: str) -> int:
    """
    Count words in a sentence.
    
    Args:
        sentence_text: Text of the sentence
        
    Returns:
        Word count (alphanumeric tokens)
    """
    if not sentence_text:
        return 0
    
    words = re.findall(r"[A-Za-z0-9']+", sentence_text)
    return len(words)


async def find_section_examples_from_corpus(
    plugin_instance: Any,
    section_name: str,
    feature_type: str,
    optimal_range: Optional[Tuple[float, float]] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Find examples of successful motions with optimal section structure from corpus.
    
    Args:
        plugin_instance: Plugin instance with corpus search methods (must have find_winning_cases, search_case_law)
        section_name: Name of section (e.g., 'legal_standard', 'introduction')
        feature_type: Type of feature (e.g., 'word_count', 'paragraph_structure', 'enumeration_depth')
        optimal_range: Optional tuple of (min, max) for optimal range
        limit: Maximum number of examples to return
        
    Returns:
        List of corpus examples with case_name, citation, section_text, similarity_score, etc.
    """
    if not plugin_instance:
        logger.debug("No plugin instance provided for corpus search")
        return []
    
    # Check if plugin has corpus search methods
    if not (hasattr(plugin_instance, 'find_winning_cases') and hasattr(plugin_instance, 'search_case_law')):
        logger.debug(f"Plugin {plugin_instance.__class__.__name__} does not have corpus search methods")
        return []
    
    examples = []
    
    try:
        # Build search keywords based on section and feature
        section_keywords = {
            'introduction': ['introduction', 'intro'],
            'legal_standard': ['legal standard', 'governing law', 'applicable law'],
            'factual_background': ['factual background', 'facts', 'background'],
            'privacy_harm': ['privacy', 'privacy harm', 'privacy interest'],
            'danger_safety': ['danger', 'safety', 'threat', 'risk'],
            'public_interest': ['public interest', 'public access'],
            'balancing_test': ['balancing', 'balance', 'weigh'],
            'protective_measures': ['protective measures', 'proposed relief', 'protective order'],
            'conclusion': ['conclusion', 'concluding'],
        }
        
        feature_keywords = {
            'word_count': ['word count', 'length', 'words'],
            'paragraph_structure': ['paragraph', 'structure', 'paragraphs'],
            'enumeration_depth': ['enumeration', 'numbered', 'bulleted', 'list'],
            'sentence_count': ['sentence', 'sentences'],
            'words_per_sentence': ['words per sentence', 'sentence length'],
        }
        
        # Combine keywords
        keywords = section_keywords.get(section_name, [section_name])
        keywords.extend(feature_keywords.get(feature_type, []))
        
        # Search for winning cases
        winning_cases = []
        if hasattr(plugin_instance, 'find_winning_cases'):
            try:
                winning_cases = await plugin_instance.find_winning_cases(
                    keywords=keywords,
                    min_keywords=1,
                    limit=limit * 2  # Get more to filter
                )
            except Exception as e:
                logger.debug(f"find_winning_cases failed: {e}")
        
        # Search by semantic similarity
        similar_cases = []
        if hasattr(plugin_instance, 'search_case_law'):
            try:
                query = f"{section_name} section {feature_type} successful motion optimal structure"
                similar_cases = await plugin_instance.search_case_law(
                    query=query,
                    top_k=limit * 2,
                    min_similarity=0.3
                )
            except Exception as e:
                logger.debug(f"search_case_law failed: {e}")
        
        # Combine and deduplicate
        all_cases = {}
        for case in winning_cases + similar_cases:
            case_name = case.get('case_name', '')
            if case_name and case_name not in all_cases:
                all_cases[case_name] = case
        
        # Extract section text from each case and check if it meets optimal range
        for case_name, case in list(all_cases.items())[:limit * 2]:
            case_text = case.get('cleaned_text') or case.get('text_snippet') or ''
            if not case_text:
                continue
            
            # Extract section text
            section_text = extract_section_text(case_text, section_name)
            if not section_text:
                continue
            
            # Calculate feature value based on feature_type
            feature_value = None
            if feature_type == 'word_count':
                words = re.findall(r"[A-Za-z0-9']+", section_text)
                feature_value = len(words)
            elif feature_type == 'paragraph_structure':
                paragraphs = extract_paragraphs_from_section(section_text)
                feature_value = len(paragraphs)
            elif feature_type == 'enumeration_depth':
                # Simple enumeration depth calculation
                lines = section_text.split('\n')
                max_depth = 0
                for line in lines:
                    # Count indentation or enumeration markers
                    depth = 0
                    if re.match(r'^\s*[•\-\*]', line) or re.match(r'^\s*\d+\.', line):
                        depth = len(re.match(r'^\s*', line).group()) // 2
                    max_depth = max(max_depth, depth)
                feature_value = max_depth
            elif feature_type == 'sentence_count':
                paragraphs = extract_paragraphs_from_section(section_text)
                total_sentences = sum(len(extract_sentences_from_paragraph(p)) for p in paragraphs)
                feature_value = total_sentences
            elif feature_type == 'words_per_sentence':
                paragraphs = extract_paragraphs_from_section(section_text)
                all_sentences = []
                for p in paragraphs:
                    all_sentences.extend(extract_sentences_from_paragraph(p))
                if all_sentences:
                    total_words = sum(count_words_in_sentence(s) for s in all_sentences)
                    feature_value = total_words / len(all_sentences) if all_sentences else 0
                else:
                    feature_value = 0
            
            # Check if meets optimal range
            meets_optimal = True
            if optimal_range and feature_value is not None:
                min_val, max_val = optimal_range
                meets_optimal = min_val <= feature_value <= max_val
            
            # Build example dict
            example = {
                'case_name': case.get('case_name', 'Unknown'),
                'citation': case.get('citation', case.get('court', 'Unknown')),
                'court': case.get('court', 'Unknown'),
                'section_text': section_text[:500],  # Limit snippet length
                'section_name': section_name,
                'feature_type': feature_type,
                'feature_value': feature_value,
                'similarity_score': case.get('similarity_score', case.get('keyword_count', 0) / len(keywords) if keywords else 0),
                'meets_optimal_range': meets_optimal,
                'source_db': case.get('source_db', 'unknown'),
                'date_filed': case.get('date_filed', '')
            }
            
            examples.append(example)
        
        # Sort by similarity score and meets_optimal_range (prioritize optimal examples)
        examples.sort(key=lambda x: (x['meets_optimal_range'], x['similarity_score']), reverse=True)
        
        # Return top examples
        return examples[:limit]
        
    except Exception as e:
        logger.warning(f"Corpus search failed for {section_name}.{feature_type}: {e}")
        return []

