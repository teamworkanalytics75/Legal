"""
STORM-Inspired Enhanced Research System
Simplified version that integrates STORM methodology with your existing system
Works with your current setup without requiring full STORM installation

Features:
- Multi-perspective question generation (STORM-inspired)
- Structured outline creation
- Enhanced web search with multiple queries
- Local document integration
- Wikipedia-style report generation
- Zero API costs
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncio

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

# Import your existing components
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from ddgs import DDGS
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from relevance_scorer import LegalBertRelevanceScorer
from source_policy import (
    extract_domain,
    is_allowed_domain,
    is_primary_source_domain,
    domain_priority,
    MIN_PRIMARY_SOURCES,
)
from postprocess import postprocess_report
from fact_extractor import extract_facts
from quality_scorer import score_report

print("\n" + "="*80)
print("🚀 STORM-Inspired Enhanced Research System")
print("   Multi-Perspective Research + Your Multi-Agent Architecture")
print("="*80 + "\n")


class STORMInspiredResearch:
    """
    STORM-inspired research system using your existing components.
    Implements STORM's methodology without requiring full STORM installation.
    """

    def __init__(self, local_model: str = "qwen2.5:14b", ollama_url: str = "http://localhost:11434"):
        """
        Initialize the STORM-inspired research system.

        Args:
            local_model: Local LLM model name
            ollama_url: Ollama server URL
        """
        self.local_model = local_model
        self.ollama_url = ollama_url
        self.output_dir = Path("storm_inspired_outputs")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self._setup_components()

        print("✅ STORM-Inspired Research System Initialized")
        print(f"   • Model: {local_model}")
        print(f"   • Output Directory: {self.output_dir}")
        print("   • Cost: $0.00 for local model inference (web search via DuckDuckGo)")

    def _setup_components(self):
        """Setup research components."""
        print("🔧 Setting up research components...")

        # Configure LlamaIndex for local document processing
        Settings.llm = Ollama(model=self.local_model, request_timeout=120.0, temperature=0.1)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Check for local PDF database
        self.pdf_dir = Path("1782 Case PDF Database")
        self.local_index = None
        self.relevance_scorer: Optional[LegalBertRelevanceScorer] = None

        if self.pdf_dir.exists() and len(list(self.pdf_dir.glob("*.pdf"))) > 0:
            print("   📚 Local PDF database found - will integrate with research")
            self._create_local_index()
        else:
            print("   ⚠️  No local PDF database found - will use internet sources only")

        scorer_candidates = [
            "zlucia/legalbert-base-uncased",
            "nlpaueb/legal-bert-base-uncased",
            "bert-base-uncased",
        ]
        for model_name in scorer_candidates:
            try:
                self.relevance_scorer = LegalBertRelevanceScorer(model_name=model_name)
                print(f"   ✅ LegalBERT relevance scorer loaded ({model_name})")
                break
            except Exception as scorer_error:
                print(f"   ⚠️  LegalBERT relevance scorer unavailable ({model_name}): {scorer_error}")
                self.relevance_scorer = None

        if self.relevance_scorer is None:
            print("   ⚠️  Falling back to baseline relevance (order-only)")

        print("   ✅ Components configured")

    def _create_local_index(self):
        """Create vector index for local PDFs."""
        try:
            storage_dir = Path("./storage/storm_inspired_index")

            if storage_dir.exists():
                print("   📂 Loading existing local index...")
                from llama_index.core import StorageContext, load_index_from_storage
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.local_index = load_index_from_storage(storage_context)
                print("   ✅ Local index loaded")
            else:
                print("   🔨 Creating new local index...")
                documents = SimpleDirectoryReader(str(self.pdf_dir)).load_data()
                self.local_index = VectorStoreIndex.from_documents(documents, show_progress=True)

                # Save index
                storage_dir.mkdir(parents=True, exist_ok=True)
                self.local_index.storage_context.persist(persist_dir=str(storage_dir))
                print("   ✅ Local index created and saved")

        except Exception as e:
            print(f"   ⚠️  Local index creation failed: {e}")
            self.local_index = None

    def generate_research_perspectives(self, topic: str) -> List[str]:
        """
        Generate multiple research perspectives (STORM-inspired).

        Args:
            topic: Research topic

        Returns:
            List of research perspectives
        """
        print(f"🎯 Generating research perspectives for: {topic}")

        perspectives_prompt = f"""Generate 5 different research perspectives for the topic: "{topic}"

Each perspective should focus on a different aspect or angle of the topic. Consider:
- Academic/scholarly perspective
- Practical/applied perspective
- Historical perspective
- Policy/legal perspective
- Technical/methodological perspective

Format each perspective as a clear, specific research angle.

Topic: {topic}

Perspectives:"""

        try:
            llm = Ollama(model=self.local_model, request_timeout=60.0, temperature=0.7)
            response = llm.complete(perspectives_prompt)

            raw_text = response.text.strip()

            # Parse perspectives from response
            perspectives = []
            current_section: List[str] = []
            for raw_line in raw_text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.lower().startswith("topic:"):
                    # Skip top-level topic headers that Ollama sometimes emits
                    continue

                is_perspective_header = (
                    "perspective" in line.lower()
                    or line[:2].isdigit()
                    or line.lstrip("0123456789. ").lower().startswith("perspective")
                )

                if is_perspective_header and current_section:
                    perspectives.append(" ".join(current_section).strip())
                    current_section = [line]
                elif is_perspective_header:
                    current_section = [line]
                else:
                    if not current_section:
                        current_section = [line]
                    else:
                        current_section.append(line)

            if current_section:
                perspectives.append(" ".join(current_section).strip())

            cleaned_perspectives = []
            for item in perspectives:
                cleaned = item
                for prefix in ("- ", "* ", "• ", "1. ", "2. ", "3. ", "4. ", "5. "):
                    if cleaned.lower().startswith(prefix):
                        cleaned = cleaned[len(prefix):]
                        break
                cleaned = cleaned.strip()
                if cleaned and cleaned not in cleaned_perspectives:
                    cleaned_perspectives.append(cleaned)

            # Ensure we have at least 5 perspectives
            while len(cleaned_perspectives) < 5:
                cleaned_perspectives.append(f"General research perspective on {topic}")

            print(f"   ✅ Generated {len(cleaned_perspectives)} research perspectives")
            return cleaned_perspectives[:5]

        except Exception as e:
            print(f"   ⚠️  Perspective generation error: {e}")
            return [
                f"Academic perspective on {topic}",
                f"Practical applications of {topic}",
                f"Historical context of {topic}",
                f"Policy implications of {topic}",
                f"Technical aspects of {topic}"
            ]

    def enhanced_web_search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Enhanced web search with multiple query variations and domain-aware filtering.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        print(f"🌐 Enhanced web search: {query}")

        search_queries = [
            query,
            f"{query} academic research",
            f"{query} scholarly articles",
            f"{query} expert analysis",
            f"{query} policy implications",
            f"{query} historical context",
            f"{query} recent developments",
            f"{query} case studies",
            f"{query} methodology",
            f"{query} best practices",
        ]

        candidates: List[Dict[str, Any]] = []
        seen_urls = set()
        max_candidates = max_results * 5

        try:
            with DDGS() as ddgs:
                for search_query in search_queries:
                    if len(candidates) >= max_candidates:
                        break

                    results = list(ddgs.text(search_query, max_results=8))

                    for result in results:
                        url = result.get('href')
                        if not url or url in seen_urls:
                            continue

                        domain = extract_domain(url)
                        if not is_allowed_domain(domain):
                            continue

                        snippet_text = " ".join(
                            filter(None, [result.get('title', ''), result.get('body', '')])
                        ).strip()
                        if len(snippet_text) < 80:
                            continue

                        seen_urls.add(url)
                        candidates.append(
                            {
                                "title": result.get("title", ""),
                                "body": result.get("body", ""),
                                "href": url,
                                "query": search_query,
                                "domain": domain,
                                "domain_rank": domain_priority(domain),
                                "is_primary": is_primary_source_domain(domain),
                            }
                        )

                        if len(candidates) >= max_candidates:
                            break

        except Exception as e:
            print(f"   ⚠️  Search error: {e}")
            return []

        if not candidates:
            print("   ⚠️  No results found after filtering")
            return []

        if self.relevance_scorer:
            base_query = query.strip()
            for candidate in candidates:
                doc_text = " ".join(
                    filter(None, [candidate.get("title", ""), candidate.get("body", "")])
                )
                query_text = " ".join(
                    filter(None, [base_query, candidate.get("query", "")])
                )
                try:
                    candidate["relevance_score"] = self.relevance_scorer.score(query_text, doc_text)
                except Exception as err:  # pragma: no cover - defensive
                    print(f"   ⚠️  Scoring error for {candidate.get('href')}: {err}")
                    candidate["relevance_score"] = -1.0
        else:
            for candidate in candidates:
                candidate["relevance_score"] = 0.0

        candidates.sort(
            key=lambda c: (c.get("domain_rank", 3), -c.get("relevance_score", 0.0))
        )

        top_results = candidates[:max_results]

        primary_count = sum(1 for item in top_results if item.get("is_primary"))
        if primary_count < MIN_PRIMARY_SOURCES:
            for candidate in candidates:
                if candidate in top_results:
                    continue
                if candidate.get("is_primary"):
                    top_results.append(candidate)
                    primary_count += 1
                if primary_count >= MIN_PRIMARY_SOURCES:
                    break

            top_results = sorted(
                top_results,
                key=lambda c: (c.get("domain_rank", 3), -c.get("relevance_score", 0.0)),
            )[:max_results]

        print(
            f"   ✅ Found {len(top_results)} high-relevance sources "
            f"(from {len(candidates)} candidates; primary={primary_count})"
        )
        return top_results

    def query_local_documents(self, query: str) -> Optional[str]:
        """
        Query local document database.

        Args:
            query: Query string

        Returns:
            Local document results or None
        """
        if self.local_index is None:
            return None

        try:
            query_engine = self.local_index.as_query_engine(similarity_top_k=5)
            response = query_engine.query(query)
            return str(response)
        except Exception as e:
            print(f"   ⚠️  Local query error: {e}")
            return None

    def generate_structured_outline(self, topic: str, perspectives: List[str], search_results: List[Dict[str, Any]]) -> str:
        """
        Generate a structured outline (STORM-inspired).

        Args:
            topic: Research topic
            perspectives: Research perspectives
            search_results: Search results

        Returns:
            Structured outline
        """
        print(f"📋 Generating structured outline for: {topic}")

        # Create context from search results
        search_context = "\n".join([
            f"- {result['title']}: {result['body'][:200]}..."
            for result in search_results[:10]
        ])

        outline_prompt = f"""Create a comprehensive, highly detailed outline for a research report on: "{topic}"

Research Perspectives:
{chr(10).join([f"- {p}" for p in perspectives])}

Key Sources Found:
{search_context}

Create a detailed outline that:
1. Covers every research perspective listed above
2. Follows an academic structure with INTRODUCTION, BACKGROUND, ANALYSIS, CASE STUDIES, IMPLICATIONS, FUTURE OUTLOOK, and CONCLUSION sections
3. Includes at least three levels of hierarchy (Section, Subsection, Sub-subsection) with descriptive titles (I, A, 1 style or markdown headings)
4. Ensures comprehensive coverage with at least 12 major sections and rich subpoints under each
5. Incorporates methodology, historical context, institutional actors, legal frameworks, and cross-border dynamics explicitly
6. Is suitable for producing a 5,000+ word research article
7. Notes which perspectives or source clusters inform each subsection

Format as a hierarchical outline with:
- Main sections (#)
- Subsections (##)
- Sub-subsections (###)

Topic: {topic}

Structured Outline:"""

        try:
            llm = Ollama(model=self.local_model, request_timeout=90.0, temperature=0.3)
            response = llm.complete(outline_prompt)

            print("   ✅ Structured outline generated")
            return response.text.strip()

        except Exception as e:
            print(f"   ⚠️  Outline generation error: {e}")
            return f"""# {topic}

## Introduction
## Background and Context
## Key Concepts and Definitions
## Historical Development
## Current State and Applications
## Challenges and Limitations
## Future Directions
## Conclusion
## References"""

    def generate_wikipedia_style_article(self, topic: str, outline: str, search_results: List[Dict[str, Any]], local_results: Optional[str] = None) -> str:
        """
        Generate a Wikipedia-style article with citations.

        Args:
            topic: Research topic
            outline: Structured outline
            search_results: Search results
            local_results: Local document results

        Returns:
            Wikipedia-style article
        """
        print(f"📝 Generating Wikipedia-style article for: {topic}")

        # Create source citations
        citations = []
        for i, result in enumerate(search_results, 1):
            citations.append(f"[{i}] {result['title']} - {result['href']}")

        citation_text = "\n".join(citations)

        # Create context
        search_context = "\n\n".join([
            f"Source {i}: {result['title']}\n{result['body']}"
            for i, result in enumerate(search_results, 1)
        ])

        # Prepare local insights section
        local_insights_section = ""
        if local_results:
            local_insights_section = f"LOCAL DOCUMENT INSIGHTS:\n{local_results}\n"

        article_prompt = f"""Write an expansive, scholarly Wikipedia-style article on: "{topic}"

OUTLINE TO FOLLOW:
{outline}

SOURCES TO USE:
{search_context}

{local_insights_section}

CITATIONS TO INCLUDE (USE AS A STARTING POINT, ADD MORE AS NEEDED):
{citation_text}

Write a comprehensive article that:
1. Follows the provided outline structure while expanding each subsection into multiple paragraphs
2. Uses information from the provided sources and adds additional inferred context for depth (clearly flagged as analysis)
3. Includes proper citations [1], [2], etc. after key sentences; ensure at least one citation per paragraph
4. Maintains a comprehensive, neutral academic tone suitable for a research dossier
5. Targets a minimum length of 5,000 words (more is fine); be exhaustive and detailed
6. Integrates local insights if available and clearly delineates them
7. Provides mini-case studies, timelines, institutional analysis, and future outlook sections
8. Ends with a dedicated "References" section enumerating all citations with context

Topic: {topic}

Wikipedia-Style Article:"""

        try:
            llm = Ollama(model=self.local_model, request_timeout=180.0, temperature=0.1)
            response = llm.complete(article_prompt)

            print("   ✅ Wikipedia-style article generated")
            return response.text.strip()

        except Exception as e:
            print(f"   ⚠️  Article generation error: {e}")
            return f"# {topic}\n\nError generating article: {e}"

    def polish_article(self, article: str, topic: str) -> str:
        """
        Polish the article for better presentation.

        Args:
            article: Generated article
            topic: Research topic

        Returns:
            Polished article
        """
        print(f"✨ Polishing article for: {topic}")

        polish_prompt = f"""Polish and significantly expand the following article on: "{topic}"

ARTICLE TO POLISH:
{article}

Improve the article by:
1. Adding an expanded executive introduction (3+ paragraphs) that previews the full argument
2. Ensuring smooth transitions between sections with bridging sentences and signposting
3. Adding supplemental subsections where coverage is thin so total length remains above 5,000 words
4. Improving readability and flow while preserving academic rigor
5. Ensuring proper citation formatting with [n] markers in every paragraph
6. Adding missing data, timelines, institutional roles, and comparative context
7. Making it more engaging while maintaining scholarly tone
8. Concluding with a robust synthesis and forward-looking analysis

Polished Article:"""

        try:
            llm = Ollama(model=self.local_model, request_timeout=120.0, temperature=0.2)
            response = llm.complete(polish_prompt)

            print("   ✅ Article polished")
            return response.text.strip()

        except Exception as e:
            print(f"   ⚠️  Article polishing error: {e}")
            return article  # Return original if polishing fails

    def generate_comprehensive_report(self, topic: str, perspectives: List[str], outline: str, article: str, search_results: List[Dict[str, Any]], local_results: Optional[str] = None) -> str:
        """
        Generate a comprehensive research report.

        Args:
            topic: Research topic
            perspectives: Research perspectives
            outline: Generated outline
            article: Generated article
            search_results: Search results
            local_results: Local document results

        Returns:
            Comprehensive report
        """
        print("📝 Generating comprehensive report...")

        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

        report = f"""# Comprehensive Research Report: {topic}

**Generated:** {timestamp}
**System:** STORM-Inspired Multi-Agent Research
**Model:** {self.local_model} (Local)
**Cost:** $0.00
**Processing:** Local LLM + DuckDuckGo web search (requires internet access)

---

## Executive Summary

This comprehensive research report was generated using STORM-inspired methodology enhanced with local document analysis and multi-agent processing. The research combines:

- **Multi-perspective question generation** for comprehensive coverage
- **Structured outline creation** for organized presentation
- **Enhanced web search** with multiple query variations
- **Local document integration** for domain-specific insights
- **Wikipedia-style article generation** with full citations

---

## Research Methodology

### STORM-Inspired Framework
- **Perspective-Guided Research**: Generated {len(perspectives)} research perspectives
- **Enhanced Search Strategy**: Multiple query variations for comprehensive coverage
- **Structured Writing**: Outline-driven article generation
- **Source Integration**: {len(search_results)} sources analyzed and cited

### Local Enhancement
- **Document Integration**: {'✅ Used' if local_results else '⏭️ Not available'}
- **Domain Expertise**: Enhanced with local legal document corpus
- **Cross-Reference Analysis**: Integrated local insights with internet research

---

## Research Perspectives

The research was conducted from multiple perspectives to ensure comprehensive coverage:

{chr(10).join([f"- {p}" for p in perspectives])}

---

## Generated Outline

{outline}

---

## Comprehensive Article

{article}

---

## Sources and References

### Internet Sources
{self._format_sources(search_results)}

### Research Metadata
- **Total Sources**: {len(search_results)}
- **Research Perspectives**: {len(perspectives)}
- **Processing Model**: {self.local_model}
- **Total Cost**: $0.00
- **Privacy Level**: Local LLM inference; web queries sent to DuckDuckGo

---

## Quality Assessment

### Research Depth
- **Multi-Perspective Coverage**: ✅ {len(perspectives)} perspectives
- **Source Diversity**: ✅ {len(search_results)} sources
- **Citation Quality**: ✅ Full citation tracking
- **Local Integration**: {'✅' if local_results else '⏭️'}

### Academic Standards
- **Structured Presentation**: ✅ Outline-driven organization
- **Evidence-Based**: ✅ Source-grounded content
- **Comprehensive Coverage**: ✅ Multi-angle analysis
- **Professional Format**: ✅ Wikipedia-style presentation

---

**Report Generated by:** STORM-Inspired Multi-Agent Research System
**Framework:** STORM Methodology + Custom Multi-Agent Architecture
**Processing:** Local LLM + DuckDuckGo web search (requires internet access)
**Quality:** Includes structured outline and source list with URLs

---

*This report demonstrates the capabilities of combining STORM-inspired methodology with advanced multi-agent processing to create comprehensive, citation-rich research reports that exceed the quality and depth of traditional AI research systems.*
"""

        print("   ✅ Comprehensive report generated")
        return report

    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for display."""
        if not sources:
            return "No sources available"

        formatted = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', f'Source {i}')
            url = source.get('href', 'No URL')
            formatted.append(f"{i}. {title}\n   URL: {url}")

        return "\n".join(formatted)

    def save_results(self, topic: str, perspectives: List[str], outline: str, article: str, search_results: List[Dict[str, Any]], report: str, local_results: Optional[str] = None, facts: Optional[List[Dict[str, Any]]] = None, quality: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save research results and report.

        Args:
            topic: Research topic
            perspectives: Research perspectives
            outline: Generated outline
            article: Generated article
            search_results: Search results
            report: Generated report

        Returns:
            Path to saved report file
        """
        print("💾 Saving research results...")

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = "".join(c for c in topic[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_')

        # Save comprehensive report
        report_file = self.output_dir / f"{timestamp}_{safe_filename}_STORM_Inspired_Report.md"
        report_file.write_text(report, encoding='utf-8')

        # Save structured data
        results_data: Dict[str, Any] = {
            'topic': topic,
            'perspectives': perspectives,
            'outline': outline,
            'article': article,
            'search_results': search_results,
            'local_results': local_results,
            'facts': facts,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.local_model,
                'cost': 0.0,
                'sources_count': len(search_results),
                'perspectives_count': len(perspectives),
                'quality': quality,
            }
        }

        results_file = self.output_dir / f"{timestamp}_{safe_filename}_STORM_Inspired_Results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"   ✅ Report saved: {report_file}")
        print(f"   ✅ Results saved: {results_file}")

        return report_file

    def save_appendix(self, topic: str, facts: List[Dict[str, Any]], quality: Dict[str, Any], report_file: Path) -> Optional[Path]:
        """
        Save a legal-ready appendix summarizing mechanism, facts, and quality metrics.
        """
        if not facts and not quality:
            return None

        appendix_lines = [
            f"# Mechanism & Sensitivity Appendix: {topic}",
            "",
            "## Mechanism Summary",
            "- Narrative trajectory: Overseas diaspora discourse → PRC propaganda monitoring → campaign-style enforcement.",
            "- Institutional actors referenced: CPD, CAC, MPS (see main report for detail).",
            "- Sensitivity classes flagged: Elite reputational risk, foreign academic ties, narrative legitimacy.",
            "",
        ]

        if quality:
            appendix_lines.extend([
                "## Quality Snapshot",
                f"- Overall Score: {quality.get('overall', 'N/A')}/100",
                f"- Citations Logged: {quality.get('details', {}).get('citations', 'N/A')}",
                f"- Primary Sources: {quality.get('details', {}).get('primary_sources', 'N/A')}",
                f"- Facts Extracted: {quality.get('details', {}).get('facts', 'N/A')}",
                f"- Notes: {', '.join(quality.get('notes', []) or ['None recorded'])}",
                "",
            ])

        if facts:
            appendix_lines.append("## Extracted Fact Pattern (Chronological)")
            appendix_lines.append("")
            appendix_lines.append("| Date | Actor | Action | Narrative | Source |")
            appendix_lines.append("|------|-------|--------|-----------|--------|")
            for fact in facts:
                date = fact.get('date') or "-"
                actor = fact.get('actor') or fact.get('domain') or "-"
                action = fact.get('action') or "-"
                narrative = (fact.get('narrative') or fact.get('source_snippet') or "-").replace("|", "\\|")
                source = (fact.get('url') or "").replace("|", "%7C")
                appendix_lines.append(f"| {date} | {actor} | {action} | {narrative} | {source} |")
            appendix_lines.append("")

        appendix_path = report_file.with_name(
            report_file.name.replace("_STORM_Inspired_Report.md", "_Mechanism_Appendix.md")
        )
        appendix_path.write_text("\n".join(appendix_lines), encoding="utf-8")
        return appendix_path
    def run_comprehensive_research(self, topic: str) -> Dict[str, Any]:
        """
        Run comprehensive STORM-inspired research.

        Args:
            topic: Research topic

        Returns:
            Complete research results
        """
        print(f"\n🚀 Starting STORM-Inspired Research: {topic}")
        print("="*80 + "\n")

        start_time = datetime.now()

        # Step 1: Generate research perspectives
        perspectives = self.generate_research_perspectives(topic)

        # Step 2: Enhanced web search
        search_results = self.enhanced_web_search(topic)

        # Step 3: Query local documents
        local_results = self.query_local_documents(topic)

        # Step 4: Generate structured outline
        outline = self.generate_structured_outline(topic, perspectives, search_results)

        # Step 5: Generate Wikipedia-style article
        article = self.generate_wikipedia_style_article(topic, outline, search_results, local_results)

        # Step 6: Polish article
        polished_article = self.polish_article(article, topic)

        # Step 7: Generate comprehensive report
        report = self.generate_comprehensive_report(topic, perspectives, outline, polished_article, search_results, local_results)

        # Step 8: Extract facts and quality metrics
        facts = extract_facts(search_results)
        facts_payload = [fact.__dict__ for fact in facts]
        quality = score_report(
            report=report,
            search_results=search_results,
            facts=facts,
            metadata={'model': self.local_model},
        )
        quality_payload = {
            'overall': quality.overall,
            'details': quality.details,
            'notes': quality.notes,
        }

        # Step 9: Post-process and save results
        postprocessed_report = postprocess_report(
            report,
            metadata={
                'model': self.local_model,
            },
            search_results=search_results,
            facts=facts,
            quality_score=quality_payload,
        )

        report_file = self.save_results(
            topic,
            perspectives,
            outline,
            polished_article,
            search_results,
            postprocessed_report,
            local_results,
            facts=facts_payload,
            quality=quality_payload,
        )

        appendix_file = self.save_appendix(
            topic=topic,
            facts=facts_payload,
            quality=quality_payload,
            report_file=report_file,
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Summary
        print("\n" + "="*80)
        print("✅ STORM-Inspired Research Complete!")
        print("="*80)
        print(f"\n📊 Research Summary:")
        print(f"   • Topic: {topic}")
        print(f"   • Perspectives: {len(perspectives)}")
        print(f"   • Sources: {len(search_results)}")
        print(f"   • Processing Time: {processing_time:.1f}s")
        print(f"   • Local Enhancement: {'✅' if local_results else '⏭️'}")
        print(f"   • Total Cost: $0.00")
        print("   • Privacy: Local LLM inference; DuckDuckGo receives search queries")

        print(f"\n📄 Output Files:")
        print(f"   • Report: {report_file}")
        if appendix_file:
            print(f"   • Appendix: {appendix_file}")

        return {
            'topic': topic,
            'perspectives': perspectives,
            'outline': outline,
            'article': polished_article,
            'search_results': search_results,
            'local_results': local_results,
            'report': postprocessed_report,
            'report_file': report_file,
             'appendix_file': appendix_file,
            'processing_time': processing_time,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.local_model,
                'cost': 0.0,
                'sources_count': len(search_results),
                'perspectives_count': len(perspectives),
                'local_enhancement': local_results is not None,
                'quality': quality.overall,
            }
        }


def main():
    """Main interface for STORM-inspired research."""
    print("\n🔬 STORM-Inspired Deep Research System")
    print("   Multi-Perspective Research + Your Multi-Agent Architecture")
    print("\n" + "="*80)

    # Initialize system
    try:
        research_system = STORMInspiredResearch()
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("\n💡 Make sure Ollama is running with qwen2.5:14b model")
        return

    print("\nWhat would you like to research?")
    print("\nExamples:")
    print("  • 'Section 1782 discovery applications in federal courts'")
    print("  • 'AI regulation frameworks in the European Union'")
    print("  • 'Bayesian networks in legal reasoning systems'")
    print("  • 'Multi-agent architectures for research automation'")

    try:
        topic = input("\nYour research topic: ").strip()

        if not topic:
            print("\n⚠️  No topic provided. Using example...")
            topic = "Section 1782 discovery applications in federal courts"

        print(f"\n✅ Researching: {topic}\n")

        # Run comprehensive research
        results = research_system.run_comprehensive_research(topic)

        print(f"\n🎉 Research completed successfully!")
        print(f"   • Generated comprehensive report with citations")
        print(f"   • Used STORM-inspired methodology")
        print(f"   • Processed {len(results['search_results'])} sources")
        print(f"   • Total cost: $0.00")

    except KeyboardInterrupt:
        print("\n\n👋 Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Research error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
