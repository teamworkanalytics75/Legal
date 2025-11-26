#!/usr/bin/env python3
"""
Corpus Validation Plugin - Orchestrates corpus-based validation of generated drafts.

Compares generated draft to successful motions in corpus to ensure content matches
patterns from successful motions.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from semantic_kernel import Kernel
from .base_feature_plugin import BaseFeaturePlugin
from ..base_plugin import FunctionResult
from .section_utils import find_section_examples_from_corpus, extract_section_text

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class CorpusValidationPlugin(BaseFeaturePlugin):
    """Plugin to validate drafts against corpus patterns."""

    def __init__(self, kernel: Kernel, chroma_store, rules_dir: Path, memory_store=None, **kwargs):
        super().__init__(kernel, "corpus_validation", chroma_store, rules_dir, memory_store=memory_store, **kwargs)
        logger.info("CorpusValidationPlugin initialized")

    async def validate_draft_against_corpus(
        self,
        document: "DocumentStructure",
        context: Dict[str, Any] = None
    ) -> FunctionResult:
        """
        Validate that generated draft matches patterns from successful motions in corpus.
        
        Args:
            document: DocumentStructure to validate
            context: Optional context with section-specific validation results
            
        Returns:
            FunctionResult with similarity scores, pattern matches, and recommendations
        """
        text = document.get_full_text()
        
        # Get section-specific validation results from context if available
        section_results = context.get("section_validation_results", {}) if context else {}
        
        all_sections = [
            'introduction', 'legal_standard', 'factual_background',
            'privacy_harm', 'danger_safety', 'public_interest',
            'balancing_test', 'protective_measures', 'conclusion'
        ]
        
        corpus_matches = {}
        pattern_matches = {}
        recommendations = []
        overall_similarity = 0.0
        
        for section_name in all_sections:
            section_text = extract_section_text(text, section_name)
            if not section_text:
                continue
            
            # Search corpus for similar successful sections
            try:
                examples = await find_section_examples_from_corpus(
                    plugin_instance=self,
                    section_name=section_name,
                    feature_type='word_count',  # Use word_count as primary feature
                    optimal_range=None,  # Don't filter by range, just find similar
                    limit=3
                )
                
                if examples:
                    # Calculate average similarity
                    similarities = [ex.get('similarity_score', 0) for ex in examples]
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                    
                    corpus_matches[section_name] = {
                        'examples': examples,
                        'avg_similarity': avg_similarity,
                        'top_match': examples[0] if examples else None
                    }
                    
                    # Check if section matches patterns from successful motions
                    section_validation = section_results.get(section_name, {})
                    meets_targets = section_validation.get('meets_target', False)
                    
                    pattern_matches[section_name] = {
                        'meets_targets': meets_targets,
                        'similarity': avg_similarity,
                        'has_similar_examples': len(examples) > 0
                    }
                    
                    # Generate recommendations if section doesn't match patterns
                    if not meets_targets and examples:
                        recommendations.append({
                            'section': section_name,
                            'type': 'corpus_pattern_mismatch',
                            'priority': 'medium',
                            'message': f"{section_name.replace('_', ' ').title()} section doesn't match patterns from successful motions",
                            'action': f"Review examples from successful motions: {', '.join([ex.get('case_name', 'Unknown') for ex in examples[:2]])}",
                            'examples': [
                                {
                                    'case_name': ex.get('case_name', 'Unknown'),
                                    'citation': ex.get('citation', ''),
                                    'similarity': ex.get('similarity_score', 0),
                                    'snippet': ex.get('section_text', '')[:200]
                                }
                                for ex in examples[:2]
                            ]
                        })
            
            except Exception as e:
                logger.debug(f"Corpus validation failed for {section_name}: {e}")
        
        # Calculate overall similarity score
        if corpus_matches:
            similarities = [match['avg_similarity'] for match in corpus_matches.values()]
            overall_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Determine if draft matches corpus patterns
        matches_patterns = overall_similarity > 0.5 and len([m for m in pattern_matches.values() if m['meets_targets']]) >= len(all_sections) * 0.7
        
        return FunctionResult(
            success=matches_patterns,
            data={
                'overall_similarity': overall_similarity,
                'matches_patterns': matches_patterns,
                'corpus_matches': corpus_matches,
                'pattern_matches': pattern_matches,
                'recommendations': recommendations,
                'sections_validated': list(corpus_matches.keys())
            },
            message=f"Draft corpus validation complete. Overall similarity: {overall_similarity:.2f}, Matches patterns: {matches_patterns}"
        )

    async def find_similar_successful_motions(
        self,
        document: "DocumentStructure",
        limit: int = 5
    ) -> FunctionResult:
        """
        Find successful motions in corpus that are similar to the current draft.
        
        Args:
            document: DocumentStructure to find similar motions for
            limit: Maximum number of similar motions to return
            
        Returns:
            FunctionResult with list of similar successful motions
        """
        text = document.get_full_text()
        
        # Search for similar motions using semantic similarity
        similar_motions = []
        
        try:
            # Use semantic search to find similar motions
            if hasattr(self, 'search_case_law'):
                results = await self.search_case_law(
                    query=text[:1000],  # Use first 1000 chars as query
                    top_k=limit,
                    min_similarity=0.4
                )
                
                for result in results:
                    similar_motions.append({
                        'case_name': result.get('case_name', 'Unknown'),
                        'citation': result.get('citation', 'Unknown'),
                        'court': result.get('court', 'Unknown'),
                        'similarity_score': result.get('similarity_score', 0),
                        'snippet': result.get('text_snippet', '')[:300],
                        'source': result.get('source_db', 'unknown')
                    })
        except Exception as e:
            logger.debug(f"Failed to find similar motions: {e}")
        
        return FunctionResult(
            success=len(similar_motions) > 0,
            data={
                'similar_motions': similar_motions,
                'count': len(similar_motions)
            },
            message=f"Found {len(similar_motions)} similar successful motions"
        )

