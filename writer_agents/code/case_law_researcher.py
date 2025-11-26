"""
Case Law Researcher - Queries case law databases for relevant precedents.

This component searches case databases for supporting case law on user's themes,
extracts relevant passages, generates explanations, and returns structured results.
"""

import logging
import sys
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add case_law_data/scripts to path for imports
project_root = Path(__file__).parent.parent.parent
case_law_scripts = project_root / "case_law_data" / "scripts"
if str(case_law_scripts) not in sys.path:
    sys.path.insert(0, str(case_law_scripts))

logger = logging.getLogger(__name__)

# Try to import research components
try:
    from research_similar_cases import (
        SimilarCasesResearcher,
        HKStatement,
        WinningCase,
        KeywordWinningCasesFinder
    )
    from citation_searchers.sqlite_searcher import SQLiteSearcher
    RESEARCH_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Research components not available: {e}")
    SimilarCasesResearcher = None
    SQLiteSearcher = None
    HKStatement = None
    RESEARCH_COMPONENTS_AVAILABLE = False


class CaseLawResearcher:
    """
    Main research orchestrator for case law queries.

    Queries case databases for relevant precedents based on case insights,
    extracts relevant passages, and generates explanations of findings.
    """

    def __init__(
        self,
        db_paths: Optional[List[Path]] = None,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        memory_store=None  # EpisodicMemoryBank for learning
    ):
        """
        Initialize case law researcher.

        Args:
            db_paths: List of database paths to search. If None, uses default paths.
            model_name: BERT model name for embeddings
            memory_store: EpisodicMemoryBank for querying past research and learning
        """
        if not RESEARCH_COMPONENTS_AVAILABLE:
            logger.error("Research components not available. Cannot initialize CaseLawResearcher.")
            self.enabled = False
            return

        # Default database paths
        if db_paths is None:
            project_root = Path(__file__).parent.parent.parent
            db_paths = [
                project_root / "case_law_data" / "1782_corpus.db",
                project_root / "case_law_data" / "harvard_corpus.db",
                project_root / "case_law_data" / "ma_federal_motions.db",
                project_root / "case_law_data" / "china_corpus.db",
                project_root / "case_law_data" / "appellate_corpus.db"
            ]

        # Filter to existing databases
        self.db_paths = [p for p in db_paths if p.exists()]

        if not self.db_paths:
            logger.warning("No case law databases found. CaseLawResearcher will be limited.")
            self.enabled = False
            return

        self.enabled = True
        self.model_name = model_name

        # Initialize SQLite searcher for semantic similarity
        try:
            self.sqlite_searcher = SQLiteSearcher(
                db_paths=self.db_paths,
                model_name=model_name
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SQLiteSearcher: {e}")
            self.sqlite_searcher = None

        # Initialize SimilarCasesResearcher for full research pipeline
        try:
            self.researcher = SimilarCasesResearcher(
                db_paths=self.db_paths,
                sqlite_searcher=self.sqlite_searcher
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SimilarCasesResearcher: {e}")
            self.researcher = None

        # Initialize keyword-only fallback researcher
        try:
            self.keyword_finder = KeywordWinningCasesFinder(self.db_paths)
        except Exception as e:
            logger.warning(f"Failed to initialize KeywordWinningCasesFinder: {e}")
            self.keyword_finder = None

        self.memory_store = memory_store
        logger.info(f"CaseLawResearcher initialized with {len(self.db_paths)} database(s)")
        if memory_store:
            logger.info("Memory store enabled - will query past research before running")

    def extract_themes_from_case_insights(self, insights: Any) -> Dict[str, Any]:
        """
        Extract research themes from CaseInsights.

        Args:
            insights: CaseInsights object with case summary, evidence, etc.

        Returns:
            Dictionary with keywords, tags, and query text for research
        """
        # Extract keywords from case summary
        summary_text = getattr(insights, 'summary', '') or ''
        evidence_list = getattr(insights, 'evidence', []) or []

        # Build query text from summary and evidence
        query_parts = [summary_text]
        for ev in evidence_list[:10]:  # Limit to first 10 evidence items
            if hasattr(ev, 'description'):
                query_parts.append(ev.description)
            elif isinstance(ev, dict):
                query_parts.append(ev.get('description', ''))

        query_text = " ".join(query_parts)

        # Extract keywords (common legal/research terms)
        keywords = []
        query_lower = query_text.lower()

        # National security related
        if any(term in query_lower for term in ['national security', 'national security', 'reputation', 'reputational']):
            keywords.extend(['national security', 'reputation'])

        # Foreign government
        if any(term in query_lower for term in ['foreign government', 'china', 'chinese communist party', 'ccp']):
            keywords.extend(['foreign government', 'china', 'ccp'])

        # Academic institutions
        if any(term in query_lower for term in ['harvard', 'academic', 'university', 'institution']):
            keywords.extend(['harvard', 'academic institution'])

        # Other common terms
        for term in ['privacy', 'harassment', 'retaliation', 'defamation', 'torture', 'collective punishment']:
            if term in query_lower:
                keywords.append(term)

        # Build tags dictionary (matching database schema)
        tags = {
            'tag_national_security': 1 if any(kw in keywords for kw in ['national security', 'reputation']) else 0,
            'tag_foreign_government': 1 if any(kw in keywords for kw in ['foreign government', 'china', 'ccp']) else 0,
            'tag_academic_institution': 1 if any(kw in keywords for kw in ['harvard', 'academic institution']) else 0,
            'tag_human_rights': 1 if any(kw in keywords for kw in ['torture', 'harassment', 'retaliation']) else 0,
            'tag_defamation': 1 if 'defamation' in keywords else 0,
            'tag_online_platform': 1 if 'esuwiki' in query_lower else 0
        }

        return {
            'keywords': list(set(keywords)),  # Deduplicate
            'tags': tags,
            'query_text': query_text[:2000],  # Limit length
            'summary': summary_text
        }

    def research_case_law(
        self,
        insights: Any,
        top_k: int = 50,
        min_similarity: float = 0.3
    ) -> Dict[str, Any]:
        """
        Research case law relevant to the case insights.

        Args:
            insights: CaseInsights object
            top_k: Number of cases to return
            min_similarity: Minimum similarity threshold

        Returns:
            Dictionary with research results including cases, explanations, and summary
        """
        if not self.enabled:
            logger.warning("CaseLawResearcher not enabled, returning empty results")
            return {
                'success': False,
                'error': 'CaseLawResearcher not enabled',
                'cases': [],
                'explanations': {},
                'summary': {}
            }

        logger.info("Starting case law research...")

        # Extract themes from insights first
        themes = self.extract_themes_from_case_insights(insights)
        logger.info(f"Extracted themes: {', '.join(themes['keywords'])}")

        # Query past research before doing new research
        past_research = None
        if self.memory_store:
            try:
                query_text = themes.get('query_text', '') or (insights.summary[:200] if hasattr(insights, 'summary') else '')

                if query_text:
                    past_memories = self.memory_store.retrieve(
                        agent_type="CaseLawResearcher",
                        query=query_text,
                        k=3,
                        memory_types=["query", "execution"]
                    )
                    if past_memories:
                        logger.info(f"Found {len(past_memories)} similar past research queries - using insights to inform research")
                        # Could use past research to refine query or skip redundant research
                        past_research = past_memories
            except Exception as e:
                logger.debug(f"Could not query past research: {e}")

        # Use SimilarCasesResearcher if available
        if self.researcher:
            try:
                # Create HKStatement-like object for research
                hk_text = themes['query_text']
                research_results = self.researcher.research(hk_text=hk_text)

                # Generate explanations
                explanations = self._generate_explanations(research_results, themes)

                # Build summary
                summary = self._build_research_summary(research_results, themes)

                result = {
                    'success': True,
                    'themes': themes,
                    'research_results': research_results,
                    'cases': research_results.get('combined_results', [])[:top_k],
                    'semantic_matches': research_results.get('semantic_results', [])[:top_k],
                    'winning_cases': research_results.get('winning_cases', [])[:top_k],
                    'explanations': explanations,
                    'summary': summary
                }
                return self._augment_with_keyword_fallback(result, themes, top_k)
            except Exception as e:
                logger.error(f"Research failed: {e}")
                result = {
                    'success': False,
                    'error': str(e),
                    'cases': [],
                    'explanations': {},
                    'summary': {}
                }
                return self._augment_with_keyword_fallback(result, themes, top_k)
        else:
            # Fallback: use SQLiteSearcher directly
            logger.info("Using SQLiteSearcher directly (SimilarCasesResearcher not available)")
            if self.sqlite_searcher:
                try:
                    candidates = self.sqlite_searcher.search(
                        query_text=themes['query_text'],
                        top_k=top_k,
                        min_similarity=min_similarity
                    )

                    # Convert candidates to case dicts
                    cases = []
                    for cand in candidates:
                        cases.append({
                            'cluster_id': getattr(cand, 'cluster_id', None),
                            'case_name': getattr(cand, 'case_name', 'Unknown'),
                            'court': getattr(cand, 'court', 'Unknown'),
                            'similarity_score': getattr(cand, 'similarity_score', 0.0),
                            'citation': getattr(cand, 'citation', ''),
                            'text_snippet': getattr(cand, 'text_snippet', '')[:500] if hasattr(cand, 'text_snippet') else ''
                        })

                    explanations = self._generate_simple_explanations(cases, themes)
                    summary = self._build_simple_summary(cases, themes)

                    result = {
                        'success': True,
                        'themes': themes,
                        'cases': cases,
                        'explanations': explanations,
                        'summary': summary
                    }
                    return self._augment_with_keyword_fallback(result, themes, top_k)
                except Exception as e:
                    logger.error(f"SQLite search failed: {e}")
                    result = {
                        'success': False,
                        'error': str(e),
                        'cases': [],
                        'explanations': {},
                        'summary': {}
                    }
                    return self._augment_with_keyword_fallback(result, themes, top_k)
            else:
                result = {
                    'success': False,
                    'error': 'No research components available',
                    'cases': [],
                    'explanations': {},
                    'summary': {}
                }
                return self._augment_with_keyword_fallback(result, themes, top_k)

    def _generate_explanations(
        self,
        research_results: Dict[str, Any],
        themes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate human-readable explanations of research findings.

        Args:
            research_results: Results from SimilarCasesResearcher
            themes: Extracted themes and keywords

        Returns:
            Dictionary with explanations organized by theme
        """
        explanations = {}

        # Get cases
        combined_cases = research_results.get('combined_results', [])
        winning_cases = research_results.get('winning_cases', [])

        # Group cases by theme
        theme_explanations = {}
        for theme_keyword in themes['keywords']:
            theme_cases = []
            for case in combined_cases[:20]:  # Top 20 cases
                case_text = str(case.get('text', '') or case.get('text_snippet', '')).lower()
                if theme_keyword.lower() in case_text:
                    theme_cases.append(case)

            if theme_cases:
                theme_explanations[theme_keyword] = {
                    'count': len(theme_cases),
                    'cases': theme_cases[:5],  # Top 5 for this theme
                    'explanation': f"Found {len(theme_cases)} cases related to '{theme_keyword}'. "
                                 f"Top case: {theme_cases[0].get('case_name', 'Unknown')} "
                                 f"({theme_cases[0].get('court', 'Unknown')})"
                }

        # Overall explanation
        total_cases = len(combined_cases)
        total_winning = len(winning_cases)

        explanations['overall'] = {
            'total_cases_found': total_cases,
            'winning_cases': total_winning,
            'themes_with_cases': len(theme_explanations),
            'summary': f"Found {total_cases} relevant cases, including {total_winning} winning cases. "
                      f"Research identified support for {len(theme_explanations)} key themes."
        }

        explanations['by_theme'] = theme_explanations

        # Top cases explanation
        top_cases = combined_cases[:10]
        explanations['top_cases'] = []
        for case in top_cases:
            similarity = case.get('similarity_score')
            if similarity is None:
                similarity = 0.0
            elif not isinstance(similarity, (int, float)):
                similarity = 0.0
            
            explanations['top_cases'].append({
                'case_name': case.get('case_name', 'Unknown'),
                'court': case.get('court', 'Unknown'),
                'relevance': similarity,
                'why_relevant': f"High similarity ({similarity:.2f}) to case themes"
            })

        return explanations

    def _generate_simple_explanations(
        self,
        cases: List[Dict[str, Any]],
        themes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate simple explanations when using SQLiteSearcher directly."""
        top_cases = []
        for case in cases[:10]:
            similarity = case.get('similarity_score')
            if similarity is None:
                similarity = 0.0
            elif not isinstance(similarity, (int, float)):
                similarity = 0.0
            
            top_cases.append({
                'case_name': case.get('case_name', 'Unknown'),
                'court': case.get('court', 'Unknown'),
                'relevance': similarity,
                'why_relevant': f"Semantic similarity score: {similarity:.2f}"
            })
        
        explanations = {
            'overall': {
                'total_cases_found': len(cases),
                'summary': f"Found {len(cases)} relevant cases based on semantic similarity."
            },
            'top_cases': top_cases
        }
        return explanations

    def _build_research_summary(
        self,
        research_results: Dict[str, Any],
        themes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build summary of research findings."""
        summary = research_results.get('summary', {})

        return {
            'themes_searched': themes['keywords'],
            'semantic_matches': summary.get('semantic_matches', 0),
            'winning_cases': summary.get('winning_cases', 0),
            'patterns_found': summary.get('patterns_found', 0),
            'research_status': 'complete'
        }

    def _build_simple_summary(
        self,
        cases: List[Dict[str, Any]],
        themes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build simple summary when using SQLiteSearcher directly."""
        return {
            'themes_searched': themes['keywords'],
            'cases_found': len(cases),
            'research_status': 'complete'
        }

    def _augment_with_keyword_fallback(
        self,
        base_results: Dict[str, Any],
        themes: Dict[str, Any],
        limit: int
    ) -> Dict[str, Any]:
        """
        Ensure we return cases by falling back to keyword-winning search when needed.
        """
        needs_fallback = not base_results.get('cases')
        if not needs_fallback:
            return base_results

        fallback = self._keyword_fallback_search(themes, limit=limit)
        if not fallback:
            return base_results

        logger.info("Keyword fallback search produced %d cases", len(fallback['cases']))
        base_results['cases'] = fallback['cases']
        base_results['winning_cases'] = fallback.get('winning_cases', [])
        
        # Replace explanations with fallback explanations (they have the correct format)
        base_results['explanations'] = fallback['explanations']
        base_results['summary'] = fallback['summary']

        base_results['success'] = True
        base_results.setdefault('warnings', []).append("keyword_fallback_used")
        base_results['fallback_method'] = fallback['fallback_method']
        return base_results

    def _keyword_fallback_search(
        self,
        themes: Dict[str, Any],
        limit: int = 50
    ) -> Optional[Dict[str, Any]]:
        """Fallback keyword search using available corpora when semantic search fails."""
        if not self.keyword_finder:
            return None

        keywords = themes.get('keywords') or []
        if not keywords:
            keywords = self._derive_keywords_from_text(
                themes.get('query_text') or themes.get('summary') or ""
            )

        if not keywords:
            return None

        try:
            winning_cases = self.keyword_finder.find_winning_cases(
                keywords=keywords,
                tags=themes.get('tags', {}),
                min_keywords=1,
                limit=limit,
                prioritize_1782=True
            )
        except Exception as e:
            logger.error(f"Keyword fallback search failed: {e}")
            return None

        if not winning_cases:
            return None

        case_dicts = [self._winning_case_to_dict(case) for case in winning_cases[:limit]]
        
        # Generate explanations that match the expected format
        explanations = {
            'overall': {
                'total_cases_found': len(case_dicts),
                'winning_cases': len([c for c in case_dicts if c.get('favorable_to_plaintiff', 0) == 1]),
                'themes_with_cases': len(themes.get('keywords', [])),
                'summary': f"Found {len(case_dicts)} relevant cases using keyword search. "
                          f"Research identified support for {len(themes.get('keywords', []))} key themes."
            },
            'by_theme': {},
            'top_cases': [
                {
                    'case_name': case.get('case_name', 'Unknown'),
                    'court': case.get('court', 'Unknown'),
                    'relevance': case.get('similarity_score', 0.5),
                    'why_relevant': f"Keyword match: {', '.join(case.get('keywords_found', [])[:3])}"
                }
                for case in case_dicts[:10]
            ]
        }
        
        # Group cases by theme keywords
        for keyword in themes.get('keywords', []):
            theme_cases = [c for c in case_dicts if keyword.lower() in str(c.get('text_snippet', '')).lower()]
            if theme_cases:
                explanations['by_theme'][keyword] = {
                    'count': len(theme_cases),
                    'cases': theme_cases[:5],
                    'explanation': f"Found {len(theme_cases)} cases related to '{keyword}'"
                }
        
        summary = {
            'themes_searched': themes.get('keywords', []),
            'cases_found': len(case_dicts),
            'winning_cases': len([c for c in case_dicts if c.get('favorable_to_plaintiff', 0) == 1]),
            'research_status': 'complete'
        }

        return {
            'cases': case_dicts,
            'winning_cases': [c for c in case_dicts if c.get('favorable_to_plaintiff', 0) == 1],
            'explanations': explanations,
            'summary': summary,
            'fallback_method': 'keyword_search'
        }

    @staticmethod
    def _derive_keywords_from_text(text: str, max_terms: int = 10) -> List[str]:
        """Derive fallback keywords from plain text when explicit tags are missing."""
        if not text:
            return []

        tokens = re.findall(r"[a-zA-Z]{5,}", text.lower())
        if not tokens:
            return []

        stopwords = {
            "therefore", "however", "whereas", "hereby",
            "which", "their", "there", "about", "other",
            "these", "those", "would", "could", "should",
            "plaintiff", "defendant", "motion", "court", "legal"
        }

        keywords = []
        for token in tokens:
            if token in stopwords:
                continue
            if token not in keywords:
                keywords.append(token)
            if len(keywords) >= max_terms:
                break

        return keywords

    @staticmethod
    def _winning_case_to_dict(case: WinningCase) -> Dict[str, Any]:
        """Convert WinningCase dataclass to dictionary for downstream usage."""
        text_snippet = getattr(case, 'text_snippet', None) or ''
        text_snippet = text_snippet[:500] if text_snippet else ''
        
        return {
            'cluster_id': getattr(case, 'cluster_id', None),
            'case_name': getattr(case, 'case_name', 'Unknown'),
            'court': getattr(case, 'court', 'Unknown'),
            'date_filed': getattr(case, 'date_filed', None),
            'citation': getattr(case, 'citation', ''),
            'keywords_found': getattr(case, 'keywords_found', []) or [],
            'keyword_count': getattr(case, 'keyword_count', 0) or 0,
            'tags': getattr(case, 'tags', {}) or {},
            'favorable_to_plaintiff': getattr(case, 'favorable_to_plaintiff', 0) or 0,
            'text_snippet': text_snippet,
            'similarity_score': getattr(case, 'similarity_score', None) or 0.5,
            'source': 'keyword_fallback'
        }
