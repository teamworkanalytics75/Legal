#!/usr/bin/env python3
"""
Base Feature Plugin with Rule Loading.

Base class for atomic feature plugins backed by ML-derived rules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

from ..base_plugin import (
    BaseSKPlugin,
    PluginMetadata,
    FunctionResult,
    EditRequest,
    DocumentLocation,
    kernel_function,
)
from ...sk_compat import register_functions_with_kernel

logger = logging.getLogger(__name__)

# Forward reference for DocumentStructure (imported in method to avoid circular dependency)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .document_structure import DocumentStructure


class BaseFeaturePlugin(BaseSKPlugin):
    """Base class for atomic feature plugins backed by ML-derived rules."""

    def __init__(
        self,
        kernel: Kernel,
        feature_name: str,
        chroma_store,
        rules_dir: Path,
        memory_store=None,  # EpisodicMemoryBank for long-term learning
        db_paths=None,  # List[Path] - case law database paths
        enable_langchain: bool = True,  # Enable LangChain SQL agents (Phase 2)
        enable_courtlistener: bool = False,  # Enable CourtListener API (Phase 4, optional)
        enable_storm: bool = False  # Enable STORM research (Phase 5, optional)
    ):
        # Set attributes needed during BaseSKPlugin initialization
        self.feature_name = feature_name
        self.chroma_store = chroma_store
        self.rules_dir = rules_dir
        self.memory_store = memory_store  # EpisodicMemoryBank integration

        # Initialize base class (will call _get_metadata, which now can use feature_name)
        super().__init__(kernel)
        self.rules = self._load_rules()

        # Initialize with metadata from rules
        self.metadata = PluginMetadata(
            name=f"{feature_name.title()}Plugin",
            description=f"Atomic plugin for {feature_name} feature analysis",
            version="1.0.0",
            functions=[f"query_{feature_name}", f"generate_{feature_name}_argument", f"validate_{feature_name}"]
        )

        # Initialize native and semantic functions
        self._initialize_functions()

        # Initialize database/research access (Phase 1+)
        self._initialize_database_access(db_paths, enable_langchain, enable_courtlistener, enable_storm)

    def _load_rules(self) -> Dict[str, Any]:
        """Load rule configuration for this feature."""
        # Handle case where rules_dir might be None
        if not self.rules_dir:
            logger.debug(f"Rules directory not provided for {self.feature_name}, using defaults")
            return self._default_rules()

        # Check if rules_dir exists and is a directory
        if isinstance(self.rules_dir, Path) and not self.rules_dir.exists():
            logger.debug(f"Rules directory does not exist: {self.rules_dir}, using defaults")
            return self._default_rules()

        rules_file = self.rules_dir / f"{self.feature_name}_rules.json"

        if not rules_file.exists():
            logger.debug(f"Rules file not found: {rules_file}, using defaults")
            return self._default_rules()

        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            logger.debug(f"Loaded rules for {self.feature_name}")
            return rules
        except Exception as e:
            logger.warning(f"Failed to load rules for {self.feature_name}: {e}")
            return self._default_rules()

    def _default_rules(self) -> Dict[str, Any]:
        """Default rules if file loading fails."""
        return {
            "feature_name": self.feature_name,
            "shap_importance": 0.0,
            "minimum_threshold": 1,
            "successful_case_average": 2.0,
            "chroma_query_template": f"{self.feature_name} {self.feature_name}",
            "validation_criteria": {
                "min_mentions": 1,
                "required_context": [self.feature_name]
            }
        }

    def _initialize_functions(self) -> None:
        """Initialize native/semantic/validation wrappers with kernel_function decorators."""
        feature_slug = self.feature_name
        feature_label = feature_slug.replace("_", " ")

        @kernel_function(
            name=f"query_{feature_slug}",
            description=f"Query Chroma/rule-based guidance for {feature_label}."
        )
        async def query_feature(case_context: str = "", **kwargs) -> str:
            """Expose native rule/query pipeline to Semantic Kernel."""
            result = await self._execute_native(case_context=case_context, **kwargs)
            payload = result.value if result.success else {"error": result.error or "native execution failed"}
            return json.dumps(payload, default=str)

        @kernel_function(
            name=f"generate_{feature_slug}_argument",
            description=f"Generate AI-enhanced argument for {feature_label}."
        )
        async def generate_argument(case_context: str = "", **kwargs) -> str:
            """Expose semantic generation helper to Semantic Kernel."""
            result = await self._execute_semantic(case_context=case_context, **kwargs)
            payload = result.value if result.success else {"error": result.error or "semantic execution failed"}
            return json.dumps(payload, default=str)

        @kernel_function(
            name=f"validate_{feature_slug}",
            description=f"Validate draft compliance for {feature_label} signals."
        )
        async def validate_feature(draft_text: str) -> str:
            """Expose rule validation to Semantic Kernel."""
            result = await self.validate_draft(draft_text)
            payload = result.value if result.success else {"error": result.error or "validation failed"}
            return json.dumps(payload, default=str)

        self.native_function = query_feature
        self.semantic_function = generate_argument
        self.validation_function = validate_feature

    async def query_chroma(
        self,
        case_context: str,
        collections: Optional[List[str]] = None,
        n_results: int = 10,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Query Chroma using rule-defined template.

        Phase 3: Enhanced Chroma integration with multi-collection support.

        Args:
            case_context: Context to search for
            collections: Optional list of collection names to search (default: ["case_law_legal"])
            n_results: Number of results per collection
            min_score: Minimum similarity score threshold

        Returns:
            List of results, ranked by relevance and source priority
        """
        try:
            query_template = self.rules.get('chroma_query_template', f"{self.feature_name} {case_context}")
            query = query_template.format(case_context=case_context[:200])  # Limit context length

            # Default collections if not specified
            if collections is None:
                collections = self.rules.get('chroma_collections', ["case_law_legal"])

            # Collection priority/weighting (from rules or defaults)
            collection_weights = self.rules.get('collection_weights', {})
            default_weight = 1.0

            all_results = []

            # Query each collection
            if hasattr(self.chroma_store, 'query'):
                for collection in collections:
                    try:
                        collection_results = await self.chroma_store.query(
                            collection_name=collection,
                            query_text=query,
                            n_results=n_results
                        )

                        # Add collection metadata and weight
                        weight = collection_weights.get(collection, default_weight)
                        for result in collection_results:
                            if isinstance(result, dict):
                                result['collection'] = collection
                                result['weight'] = weight
                                result['score'] = result.get('score', 0.0) * weight
                                if result.get('score', 0.0) >= min_score:
                                    all_results.append(result)
                            else:
                                # Handle non-dict results
                                all_results.append({
                                    'text': str(result),
                                    'collection': collection,
                                    'weight': weight,
                                    'score': min_score,
                                    'metadata': {}
                                })
                    except Exception as e:
                        logger.warning(f"Chroma query failed for collection {collection}: {e}")
                        continue
            else:
                # Fallback for testing
                all_results = [{"text": f"Sample result for {query}", "metadata": {"case_id": "test"}, "collection": "case_law_legal", "score": 0.5}]

            # Rank results by weighted score
            all_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)

            # Remove duplicates (same text from different collections)
            seen_texts = set()
            deduplicated_results = []
            for result in all_results:
                text_key = result.get('text', '')[:100]  # First 100 chars as key
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    deduplicated_results.append(result)

            logger.info(f"Chroma query for {self.feature_name}: {len(deduplicated_results)} results from {len(collections)} collection(s)")
            return deduplicated_results[:n_results * len(collections)]  # Return top results

        except Exception as e:
            logger.error(f"Chroma query failed for {self.feature_name}: {e}")
            return []

    async def extract_patterns(self, chroma_results: List[Dict]) -> Dict[str, Any]:
        """Extract argument patterns from successful cases."""
        patterns = {
            "common_phrases": [],
            "citation_contexts": [],
            "argument_structures": [],
            "success_indicators": []
        }

        # Extract patterns from results
        for result in chroma_results:
            text = result.get("text", "")
            metadata = result.get("metadata", {})

            # Extract common phrases (simplified)
            recommended_phrases = self.rules.get("recommended_phrases", [])
            for phrase in recommended_phrases:
                if phrase.lower() in text.lower():
                    patterns["common_phrases"].append(phrase)

            # Extract citation contexts
            if "citation" in metadata:
                patterns["citation_contexts"].append({
                    "citation": metadata["citation"],
                    "context": text[:200]
                })

        # Remove duplicates
        patterns["common_phrases"] = list(set(patterns["common_phrases"]))

        logger.info(f"Extracted patterns for {self.feature_name}: {len(patterns['common_phrases'])} phrases")
        return patterns

    async def generate_argument(self, patterns: Dict, case_context: str) -> str:
        """Generate motion text using patterns and rules."""
        try:
            # Get recommended phrases from rules
            recommended_phrases = self.rules.get("recommended_phrases", [])
            common_phrases = patterns.get("common_phrases", [])

            # Combine and prioritize phrases
            all_phrases = list(set(recommended_phrases + common_phrases))

            # Generate argument based on feature type
            if self.feature_name == "mentions_privacy":
                argument = self._generate_privacy_argument(all_phrases, case_context)
            elif self.feature_name == "mentions_harassment":
                argument = self._generate_harassment_argument(all_phrases, case_context)
            elif self.feature_name == "mentions_safety":
                argument = self._generate_safety_argument(all_phrases, case_context)
            elif self.feature_name == "mentions_retaliation":
                argument = self._generate_retaliation_argument(all_phrases, case_context)
            else:
                argument = self._generate_generic_argument(all_phrases, case_context)

            logger.info(f"Generated {self.feature_name} argument: {len(argument)} characters")
            return argument

        except Exception as e:
            logger.error(f"Argument generation failed for {self.feature_name}: {e}")
            return f"Error generating {self.feature_name} argument: {str(e)}"

    def _generate_privacy_argument(self, phrases: List[str], context: str) -> str:
        """Generate privacy-specific argument."""
        return f"""
The disclosure of personal information in this case would cause significant privacy harm.
{phrases[0] if phrases else 'Privacy interests'} are at stake, and the {phrases[1] if len(phrases) > 1 else 'personal information'}
involved in this matter should be protected from public disclosure. The {phrases[2] if len(phrases) > 2 else 'expectation of privacy'}
in this context outweighs any public interest in disclosure.
""".strip()

    def _generate_harassment_argument(self, phrases: List[str], context: str) -> str:
        """Generate harassment-specific argument."""
        return f"""
The disclosure of identifying information would likely lead to {phrases[0] if phrases else 'harassment'}
and {phrases[1] if len(phrases) > 1 else 'retaliation'}. The risk of {phrases[2] if len(phrases) > 2 else 'adverse action'}
is substantial and well-documented in similar cases. Protecting against such {phrases[3] if len(phrases) > 3 else 'reprisal'}
is essential to ensure fair access to the courts.
""".strip()

    def _generate_safety_argument(self, phrases: List[str], context: str) -> str:
        """Generate safety-specific argument."""
        return f"""
Public disclosure would create {phrases[0] if phrases else 'safety concerns'} for the parties involved.
The potential for {phrases[1] if len(phrases) > 1 else 'danger'} or {phrases[2] if len(phrases) > 2 else 'threats'}
outweighs any public interest in transparency. The court should prioritize {phrases[3] if len(phrases) > 3 else 'security'}
over public access in this matter.
""".strip()

    def _generate_retaliation_argument(self, phrases: List[str], context: str) -> str:
        """Generate retaliation-specific argument."""
        return f"""
Disclosure would create a substantial risk of {phrases[0] if phrases else 'retaliation'}
and {phrases[1] if len(phrases) > 1 else 'adverse action'}. The causal connection between
public disclosure and {phrases[2] if len(phrases) > 2 else 'reprisal'} is well-established.
Protecting against such {phrases[3] if len(phrases) > 3 else 'retaliatory conduct'}
is essential for fair access to justice.
""".strip()

    def _generate_generic_argument(self, phrases: List[str], context: str) -> str:
        """Generate generic argument for other features."""
        return f"""
This case involves significant {self.feature_name.replace('_', ' ')} considerations.
The {phrases[0] if phrases else 'relevant factors'} support the requested relief.
{phrases[1] if len(phrases) > 1 else 'Additional considerations'} further support
the motion to seal/pseudonym.
""".strip()

    async def generate_edit_requests(
        self,
        text: str,
        structure: 'DocumentStructure',
        context: Optional[Dict[str, Any]] = None
    ) -> List[EditRequest]:
        """
        Generate location-specific edit requests for improving the document.

        This is the new method for plugins to request edits at specific locations.
        Plugins should override this method to generate EditRequests based on their
        specific validation logic.

        Args:
            text: Full document text
            structure: Parsed document structure with paragraph/sentence tracking
            context: Optional context dict with:
                - weak_features: Dict of weak features from CatBoost analysis
                - research_results: Pre-loaded research results
                - validation_state: Current validation state

        Returns:
            List of EditRequest objects specifying where and what to edit
        """
        # Default implementation: return empty list
        # Plugins can override to accept context parameter
        return []

    def _get_weak_feature_priority(self, context: Optional[Dict[str, Any]] = None) -> float:
        """Get priority score for this plugin's feature based on weak_features context.

        Args:
            context: Optional context dict with weak_features

        Returns:
            Priority score (0.0 to 1.0), higher = more important to fix
        """
        if not context or not context.get("weak_features"):
            return 0.5  # Default priority

        weak_features = context["weak_features"]

        # Check if this plugin's feature is in weak_features
        if self.feature_name in weak_features:
            feature_analysis = weak_features[self.feature_name]
            gap = feature_analysis.get("gap", 0)
            target = feature_analysis.get("target", 1)

            # Priority based on gap size (larger gap = higher priority)
            if target > 0:
                gap_ratio = abs(gap) / target
                return min(gap_ratio, 1.0)  # Clamp to 1.0

        return 0.0  # Feature not weak, low priority

    def _get_research_results(self, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get pre-loaded research results from context.

        Args:
            context: Optional context dict with research_results

        Returns:
            Research results dict or None
        """
        if not context:
            return None
        return context.get("research_results")

    def _should_prioritize_feature(self, context: Optional[Dict[str, Any]] = None, threshold: float = 0.3) -> bool:
        """Check if this plugin's feature should be prioritized based on context.

        Args:
            context: Optional context dict with weak_features
            threshold: Minimum priority score to return True (default: 0.3)

        Returns:
            True if feature should be prioritized, False otherwise
        """
        priority = self._get_weak_feature_priority(context)
        return priority >= threshold

    async def validate_draft(self, draft_text: str) -> FunctionResult:
        """Validate draft against rule criteria."""
        try:
            validation = self.rules.get("validation_criteria", {})
            text_lower = draft_text.lower()

            # Check minimum mentions
            min_mentions = validation.get("min_mentions", 1)
            feature_count = text_lower.count(self.feature_name.replace("_", " "))

            # Check required context
            required_context = validation.get("required_context", [])
            context_present = all(context in text_lower for context in required_context)

            # Calculate score
            score = 0.0
            if feature_count >= min_mentions:
                score += 0.5
            if context_present:
                score += 0.5

            success = score >= 0.7  # 70% threshold

            return FunctionResult(
                success=success,
                value={
                    "score": score,
                    "feature_count": feature_count,
                    "min_required": min_mentions,
                    "context_present": context_present,
                    "validation_passed": success
                },
                error=None if success else f"Validation failed: score {score:.2f} < 0.7"
            )

        except Exception as e:
            logger.error(f"Validation failed for {self.feature_name}: {e}")
            return FunctionResult(
                success=False,
                value=None,
                error=str(e)
            )

    async def _execute_native(self, **kwargs) -> FunctionResult:
        """Execute native function for rule-based operations."""
        try:
            case_context = kwargs.get("case_context", "")

            # Query Chroma
            results = await self.query_chroma(case_context)

            # Extract patterns
            patterns = await self.extract_patterns(results)

            # Generate argument
            argument = await self.generate_argument(patterns, case_context)

            return FunctionResult(
                success=True,
                value={
                    "argument": argument,
                    "patterns": patterns,
                    "results_count": len(results)
                },
                metadata={"feature": self.feature_name, "method": "native"}
            )

        except Exception as e:
            logger.error(f"Native execution failed for {self.feature_name}: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _get_metadata(self) -> PluginMetadata:
        """Provide metadata during BaseSKPlugin initialization.

        During BaseSKPlugin.__init__, self.metadata may not yet be set. Use
        feature_name (if available) to construct minimal metadata so plugin
        registration can proceed, then BaseFeaturePlugin.__init__ will assign
        the full metadata.
        """
        try:
            fname = getattr(self, "feature_name", "feature")
            return PluginMetadata(
                name=f"{fname.title()}Plugin",
                description=f"Atomic plugin for {fname} feature analysis",
                version="1.0.0",
                functions=[f"query_{fname}", f"generate_{fname}_argument", f"validate_{fname}"]
            )
        except Exception:
            # Fallback minimal metadata
            return PluginMetadata(
                name="FeaturePlugin",
                description="Atomic feature plugin",
                version="1.0.0",
                functions=["query", "generate_argument", "validate"]
            )

    async def _register_functions(self) -> None:
        """Register plugin functions with the kernel."""
        if not self.kernel:
            logger.warning("Kernel unavailable; skipping feature plugin registration")
            return

        try:
            plugin_name = self.metadata.name if self.metadata else f"{self.feature_name.title()}Plugin"
            functions_to_register = [
                self.native_function,
                self.semantic_function,
                getattr(self, "validation_function", None),
            ]
            functions_to_register = [func for func in functions_to_register if func is not None]

            plugin = register_functions_with_kernel(
                self.kernel,
                plugin_name,
                functions_to_register,
            )

            registered = getattr(plugin, "functions", {})
            if registered:
                self._functions.update(registered)
            else:
                # Fallback: register helper in add_plugin mode returns dict assignment
                self._functions.update(self.kernel.plugins.get(plugin_name, {}))

            logger.info("Registered %s functions: %s", plugin_name, list(self._functions.keys()))
        except Exception as e:
            logger.error(f"Failed to register functions for {self.feature_name}: {e}")

    async def _execute_semantic(self, **kwargs) -> FunctionResult:
        """Execute semantic function for AI-powered generation."""
        try:
            case_context = kwargs.get("case_context", "")

            # Use semantic kernel for AI generation
            if self.kernel:
                # Create a prompt based on rules
                prompt = f"""
                Generate a legal argument for {self.feature_name} in a motion to seal/pseudonym.

                Case context: {case_context}

                Required elements from successful cases:
                - Minimum mentions: {self.rules.get('minimum_threshold', 1)}
                - Recommended phrases: {', '.join(self.rules.get('recommended_phrases', []))}

                Generate a compelling argument that addresses these requirements.
                """

                # Use semantic kernel to generate response
                response = await self.kernel.invoke(
                    self.kernel.create_function_from_prompt(
                        prompt,
                        function_name=f"generate_{self.feature_name}",
                        plugin_name=self.metadata.name
                    )
                )

                return FunctionResult(
                    success=True,
                    value={
                        "argument": str(response),
                        "method": "semantic",
                        "prompt_used": prompt
                    },
                    metadata={"feature": self.feature_name, "method": "semantic"}
                )
            else:
                # Fallback to native method
                return await self._execute_native(**kwargs)

        except Exception as e:
            logger.error(f"Semantic execution failed for {self.feature_name}: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _query_memory(
        self,
        query: str,
        agent_type: Optional[str] = None,
        k: int = 5,
        memory_types: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Query episodic memory for past similar operations.

        Args:
            query: Query string for semantic search
            agent_type: Optional agent type filter (defaults to plugin class name)
            k: Number of results to return
            memory_types: Optional list of memory types to filter by

        Returns:
            List of relevant memory entries
        """
        if not self.memory_store:
            return []

        if not query or not query.strip():
            logger.debug("Empty query string provided, skipping memory query")
            return []

        try:
            agent_type = agent_type or self.__class__.__name__
            results = self.memory_store.retrieve(
                agent_type=agent_type,
                query=query[:500],  # Limit query length for performance
                k=k,
                memory_types=memory_types
            )
            if results:
                logger.debug(f"Memory query returned {len(results)} results for {agent_type}")
            return results
        except Exception as e:
            logger.debug(f"Memory query failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    def _store_memory(
        self,
        summary: str,
        context: Dict[str, Any],
        agent_type: Optional[str] = None,
        memory_type: str = "execution",
        source: str = "plugin"
    ) -> None:
        """
        Store operation result in episodic memory.

        Args:
            summary: Summary of the operation
            context: Additional context data
            agent_type: Optional agent type (defaults to plugin class name)
            memory_type: Type of memory (execution, query, edit, etc.)
            source: Source identifier
        """
        if not self.memory_store:
            return

        try:
            from datetime import datetime
            import uuid
            import sys
            from pathlib import Path

            # Try relative import first, then absolute
            try:
                from ...EpisodicMemoryBank import EpisodicMemoryEntry
            except (ImportError, ValueError):
                try:
                    # Try adding parent directory to path
                    code_dir = Path(__file__).parent.parent.parent
                    if str(code_dir) not in sys.path:
                        sys.path.insert(0, str(code_dir))
                    from EpisodicMemoryBank import EpisodicMemoryEntry
                except ImportError:
                    logger.warning("EpisodicMemoryEntry not available, memory storage skipped")
                    return

            agent_type = agent_type or self.__class__.__name__
            memory = EpisodicMemoryEntry(
                agent_type=agent_type,
                memory_id=str(uuid.uuid4()) if hasattr(uuid, 'uuid4') else f"{datetime.now().isoformat()}",
                summary=summary,
                context=context,
                source=source,
                timestamp=datetime.now(),
                memory_type=memory_type
            )
            self.memory_store.add(memory)
            logger.debug(f"Stored memory for {agent_type}: {summary[:50]}...")
        except Exception as e:
            logger.debug(f"Memory storage failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _initialize_database_access(
        self,
        db_paths: Optional[List[Path]],
        enable_langchain: bool,
        enable_courtlistener: bool,
        enable_storm: bool
    ) -> None:
        """
        Initialize database and research access for plugins.

        Phase 1: Core database access (SQLiteSearcher)
        Phase 2: LangChain SQL agents (optional)
        Phase 4: CourtListener API (optional, fallback only)
        Phase 5: STORM research (optional, deep research)
        """
        # Initialize database paths
        if db_paths is None:
            # Auto-detect default database paths
            project_root = Path(__file__).parent.parent.parent.parent.parent
            db_paths = [
                project_root / "case_law_data" / "1782_corpus.db",
                project_root / "case_law_data" / "harvard_corpus.db",
                project_root / "case_law_data" / "ma_federal_motions.db",
                project_root / "case_law_data" / "china_corpus.db",
                project_root / "case_law_data" / "appellate_corpus.db",
            ]

        # Filter to existing databases
        self.db_paths = [Path(p) for p in db_paths if p and Path(p).exists()]

        # Phase 1: Initialize SQLite searcher for case law databases
        self.sqlite_searcher = None
        if self.db_paths:
            try:
                # Add case_law_data scripts to path
                import sys
                scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "case_law_data" / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))

                from citation_searchers.sqlite_searcher import SQLiteSearcher

                self.sqlite_searcher = SQLiteSearcher(
                    db_paths=self.db_paths,
                    source_name=f"{self.feature_name}_sqlite",
                    model_name="nlpaueb/legal-bert-base-uncased"
                )
                logger.info(f"Initialized SQLite searcher for {self.feature_name} with {len(self.db_paths)} database(s)")
            except Exception as e:
                logger.warning(f"SQLite searcher not available for {self.feature_name}: {e}")
                self.sqlite_searcher = None

        # Phase 2: Initialize LangChain SQL agent (optional)
        self.langchain_agent = None
        if enable_langchain and self.db_paths:
            try:
                from writer_agents.code.LangchainIntegration import LangChainSQLAgent
                from writer_agents.code.agents import ModelConfig

                # Use first database for LangChain (unified_corpus preferred)
                langchain_db = self.db_paths[0]
                self.langchain_agent = LangChainSQLAgent(
                    db_path=str(langchain_db),
                    model_config=ModelConfig(model="gpt-4o-mini"),
                    verbose=False
                )
                logger.info(f"Initialized LangChain SQL agent for {self.feature_name}")
            except Exception as e:
                logger.debug(f"LangChain agent not available for {self.feature_name}: {e}")
                self.langchain_agent = None

        # Phase 4: Initialize CourtListener searcher (optional, fallback only)
        self.courtlistener_searcher = None
        if enable_courtlistener:
            try:
                import sys
                scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "case_law_data" / "scripts" / "citation_searchers"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))

                from courtlistener_searcher import CourtListenerSearcher

                self.courtlistener_searcher = CourtListenerSearcher(
                    enabled=True,
                    source_name=f"{self.feature_name}_courtlistener"
                )
                logger.info(f"Initialized CourtListener searcher for {self.feature_name}")
            except Exception as e:
                logger.debug(f"CourtListener searcher not available for {self.feature_name}: {e}")
                self.courtlistener_searcher = None

        # Phase 5: Initialize STORM research (optional, deep research)
        self.storm_researcher = None
        if enable_storm:
            try:
                import sys
                scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "scripts"
                if str(scripts_dir) not in sys.path:
                    sys.path.insert(0, str(scripts_dir))

                from STORMInspiredResearch import STORMInspiredResearch

                self.storm_researcher = STORMInspiredResearch()
                logger.info(f"Initialized STORM researcher for {self.feature_name}")
            except Exception as e:
                logger.debug(f"STORM researcher not available for {self.feature_name}: {e}")
                self.storm_researcher = None

    async def search_case_law(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        keywords: Optional[List[str]] = None,
        tags: Optional[Dict[str, int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search case law databases using semantic similarity.

        Phase 1: Core database access method.

        Args:
            query: Legal text to search for
            top_k: Maximum number of results
            min_similarity: Minimum similarity threshold
            keywords: Optional keywords for keyword-based search
            tags: Optional tags for filtering (e.g., {'tag_national_security': 1})

        Returns:
            List of case dictionaries with case_name, court, citation, similarity_score, etc.
        """
        if not self.sqlite_searcher:
            logger.debug(f"SQLite searcher not available for {self.feature_name}")
            return []

        try:
            # Semantic similarity search
            candidates = self.sqlite_searcher.search(
                query_text=query[:2000],  # Limit for embedding
                top_k=top_k,
                min_similarity=min_similarity
            )

            # Convert CitationCandidate objects to dictionaries
            results = []
            for candidate in candidates:
                # Get text_snippet if available, otherwise use empty string
                text_snippet = getattr(candidate, 'text_snippet', None) or getattr(candidate, 'text', None) or ''
                results.append({
                    'case_name': candidate.case_name or 'Unknown',
                    'court': candidate.court or 'Unknown',
                    'citation': candidate.citation,
                    'similarity_score': candidate.similarity_score,
                    'cluster_id': getattr(candidate, 'cluster_id', None),
                    'source': candidate.source,
                    'text_snippet': text_snippet,
                    'metadata': getattr(candidate, 'metadata', None) or {}
                })

            logger.info(f"Found {len(results)} similar cases for {self.feature_name}")
            return results

        except Exception as e:
            logger.error(f"Case law search failed for {self.feature_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    async def find_winning_cases(
        self,
        keywords: List[str],
        tags: Optional[Dict[str, int]] = None,
        min_keywords: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Find winning cases containing specified keywords and matching tags.

        Phase 1: Core database access method for keyword-based case finding.

        Args:
            keywords: List of keywords to search for
            tags: Optional tags for filtering (e.g., {'tag_national_security': 1})
            min_keywords: Minimum number of keywords that must match
            limit: Maximum number of cases to return

        Returns:
            List of winning case dictionaries
        """
        if not self.db_paths:
            logger.debug(f"No databases available for winning case search")
            return []

        winning_cases = []

        try:
            import sqlite3

            for db_path in self.db_paths:
                if not db_path.exists():
                    continue

                try:
                    with sqlite3.connect(str(db_path)) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()

                        # Check if database has any cases with favorable_to_plaintiff = 1
                        # If not, we'll make the filter optional and prioritize wins in ordering
                        cursor.execute("PRAGMA table_info(cases)")
                        columns = [row[1] for row in cursor.fetchall()]
                        has_winning_cases = False
                        if 'favorable_to_plaintiff' in columns:
                            cursor.execute("SELECT COUNT(*) FROM cases WHERE favorable_to_plaintiff = 1")
                            has_winning_cases = cursor.fetchone()[0] > 0

                        # Build keyword search conditions
                        keyword_conditions = []
                        keyword_params = []
                        for keyword in keywords:
                            keyword_conditions.append("LOWER(cleaned_text) LIKE ?")
                            keyword_params.append(f"%{keyword.lower()}%")

                        # Build tag conditions
                        tag_conditions = []
                        if tags:
                            for tag, value in tags.items():
                                if value == 1:
                                    tag_conditions.append(f"{tag} = 1")

                        # Build WHERE clause
                        where_parts = [
                            f"({' OR '.join(keyword_conditions)})",  # At least one keyword
                            "cleaned_text IS NOT NULL",
                            "cleaned_text != ''"
                        ]
                        
                        # Only require favorable_to_plaintiff = 1 if database has winning cases
                        # Otherwise, we'll prioritize wins in ORDER BY instead
                        if has_winning_cases and 'favorable_to_plaintiff' in columns:
                            where_parts.append("favorable_to_plaintiff = 1")

                        if tag_conditions:
                            where_parts.append(f"({' OR '.join(tag_conditions)})")

                        # Build ORDER BY - prioritize wins if database has no winning cases filtered
                        order_by = "date_filed DESC"
                        if not has_winning_cases and 'favorable_to_plaintiff' in columns:
                            order_by = """
                                CASE
                                    WHEN favorable_to_plaintiff = 1 THEN 0
                                    ELSE 1
                                END,
                                date_filed DESC
                            """

                        query = f"""
                            SELECT
                                cluster_id,
                                case_name,
                                court,
                                date_filed,
                                cleaned_text,
                                favorable_to_plaintiff
                            FROM cases
                            WHERE {' AND '.join(where_parts)}
                            ORDER BY {order_by}
                            LIMIT ?
                        """

                        params = keyword_params + [limit]
                        cursor.execute(query, params)
                        rows = cursor.fetchall()

                        for row in rows:
                            # Count matching keywords
                            text_lower = (row['cleaned_text'] or '').lower()
                            keyword_count = sum(1 for kw in keywords if kw.lower() in text_lower)

                            if keyword_count >= min_keywords:
                                winning_cases.append({
                                    'cluster_id': row['cluster_id'],
                                    'case_name': row['case_name'] or 'Unknown',
                                    'court': row['court'] or 'Unknown',
                                    'date_filed': row['date_filed'] or '',
                                    'keywords_found': [kw for kw in keywords if kw.lower() in text_lower],
                                    'keyword_count': keyword_count,
                                    'favorable_to_plaintiff': row['favorable_to_plaintiff'],
                                    'text_snippet': (row['cleaned_text'] or '')[:200],
                                    'source_db': db_path.name
                                })

                except Exception as e:
                    logger.warning(f"Error querying {db_path.name}: {e}")
                    continue

            # Sort by keyword count (most matches first)
            winning_cases.sort(key=lambda x: x['keyword_count'], reverse=True)

            logger.info(f"Found {len(winning_cases)} winning cases with keywords for {self.feature_name}")
            return winning_cases[:limit]

        except Exception as e:
            logger.error(f"Winning case search failed for {self.feature_name}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []

    async def query_langchain(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query database using LangChain natural language SQL.

        Phase 2: LangChain SQL agent integration.

        Args:
            query: Natural language query (e.g., "Find cases where privacy harm was successfully argued")
            context: Optional context for the query

        Returns:
            Dictionary with query results
        """
        if not self.langchain_agent:
            return {"error": "LangChain agent not available"}

        try:
            result = self.langchain_agent.query_evidence(query, context)

            # Store query in memory for learning
            if self.memory_store:
                self._store_memory(
                    summary=f"LangChain query: {query[:100]}",
                    context={
                        "query": query,
                        "context": context,
                        "result_count": len(result.get("results", [])) if isinstance(result, dict) else 0,
                        "query_type": "langchain_sql"
                    },
                    memory_type="query",
                    source=f"{self.feature_name}_langchain"
                )

            return result

        except Exception as e:
            logger.error(f"LangChain query failed for {self.feature_name}: {e}")
            return {"error": str(e)}

    async def search_courtlistener(
        self,
        query: str,
        top_k: int = 10,
        courts: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search CourtListener API for cases (fallback when local DBs don't have cases).

        Phase 4: CourtListener API integration (optional, costs money).

        Args:
            query: Search query
            top_k: Maximum number of results
            courts: Optional list of court identifiers to filter by

        Returns:
            List of case dictionaries from CourtListener
        """
        if not self.courtlistener_searcher:
            return []

        try:
            candidates = self.courtlistener_searcher.search(
                query_text=query,
                top_k=top_k,
                courts=courts
            )

            # Convert to dictionaries
            results = []
            for candidate in candidates:
                results.append({
                    'case_name': candidate.case_name or 'Unknown',
                    'court': candidate.court or 'Unknown',
                    'citation': candidate.citation,
                    'similarity_score': candidate.similarity_score,
                    'source': candidate.source,
                    'metadata': candidate.metadata or {}
                })

            logger.info(f"CourtListener search for {self.feature_name}: {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"CourtListener search failed for {self.feature_name}: {e}")
            return []

    async def research_deep(
        self,
        topic: str,
        perspectives: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform deep STORM-style multi-perspective research.

        Phase 5: STORM research system integration (optional, slow but thorough).

        Args:
            topic: Research topic (e.g., "privacy harm in academic institutions")
            perspectives: Optional number of research perspectives (default: STORM default)

        Returns:
            Dictionary with research results, sources, and analysis
        """
        if not self.storm_researcher:
            return {"error": "STORM researcher not available"}

        try:
            # STORM research is typically synchronous, wrap if needed
            # This is a placeholder - actual implementation depends on STORMInspiredResearch API
            result = self.storm_researcher.research(topic)

            # Store research in memory
            if self.memory_store:
                self._store_memory(
                    summary=f"STORM research: {topic[:100]}",
                    context={
                        "topic": topic,
                        "source_count": len(result.get("sources", [])) if isinstance(result, dict) else 0,
                        "research_type": "storm_deep"
                    },
                    memory_type="query",
                    source=f"{self.feature_name}_storm"
                )

            return result

        except Exception as e:
            logger.error(f"STORM research failed for {self.feature_name}: {e}")
            return {"error": str(e)}

    async def unified_query(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        top_k: int = 10,
        min_similarity: float = 0.3,
        keywords: Optional[List[str]] = None,
        tags: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """
        Unified query interface across all research sources.

        Phase 6: Unified query interface - queries all available sources and aggregates results.

        Args:
            query: Query text
            sources: Optional list of sources to query (default: all available)
                Options: ["database", "chroma", "langchain", "courtlistener", "storm"]
            top_k: Maximum results per source
            min_similarity: Minimum similarity threshold for database/Chroma
            keywords: Optional keywords for keyword-based search
            tags: Optional tags for filtering

        Returns:
            Dictionary with aggregated results from all sources, ranked and deduplicated
        """
        if sources is None:
            # Default: query all available sources
            sources = ["database", "chroma"]
            if self.langchain_agent:
                sources.append("langchain")
            if self.courtlistener_searcher:
                sources.append("courtlistener")
            if self.storm_researcher:
                sources.append("storm")

        all_results = []
        source_results = {}

        # Query each source
        if "database" in sources and self.sqlite_searcher:
            try:
                db_results = await self.search_case_law(query, top_k=top_k, min_similarity=min_similarity, keywords=keywords, tags=tags)
                source_results["database"] = db_results
                for result in db_results:
                    result['source'] = 'database'
                    result['source_priority'] = 1.0  # High priority for database
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Database query failed: {e}")

        if "chroma" in sources and self.chroma_store:
            try:
                chroma_results = await self.query_chroma(query, n_results=top_k, min_score=min_similarity)
                source_results["chroma"] = chroma_results
                for result in chroma_results:
                    result['source'] = 'chroma'
                    result['source_priority'] = 0.8  # Medium-high priority for Chroma
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Chroma query failed: {e}")

        if "langchain" in sources and self.langchain_agent:
            try:
                langchain_results = await self.query_langchain(query, context=query)
                source_results["langchain"] = langchain_results
                if isinstance(langchain_results, dict) and "results" in langchain_results:
                    for result in langchain_results["results"][:top_k]:
                        result['source'] = 'langchain'
                        result['source_priority'] = 0.9  # High priority for LangChain
                        all_results.append(result)
            except Exception as e:
                logger.warning(f"LangChain query failed: {e}")

        if "courtlistener" in sources and self.courtlistener_searcher:
            try:
                courtlistener_results = await self.search_courtlistener(query, top_k=top_k)
                source_results["courtlistener"] = courtlistener_results
                for result in courtlistener_results:
                    result['source'] = 'courtlistener'
                    result['source_priority'] = 0.7  # Medium priority (external API)
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"CourtListener query failed: {e}")

        if "storm" in sources and self.storm_researcher:
            try:
                storm_results = await self.research_deep(query)
                source_results["storm"] = storm_results
                if isinstance(storm_results, dict) and "sources" in storm_results:
                    for result in storm_results["sources"][:top_k]:
                        result['source'] = 'storm'
                        result['source_priority'] = 0.6  # Lower priority (slow, thorough)
                        all_results.append(result)
            except Exception as e:
                logger.warning(f"STORM research failed: {e}")

        # Rank and deduplicate results
        # Calculate combined score: (similarity_score * source_priority) for ranking
        for result in all_results:
            similarity = result.get('similarity_score', result.get('score', 0.0))
            priority = result.get('source_priority', 0.5)
            result['combined_score'] = similarity * priority

        # Sort by combined score
        all_results.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)

        # Deduplicate by case name or citation
        seen_cases = set()
        deduplicated_results = []
        for result in all_results:
            case_key = (
                result.get('case_name', '') or
                result.get('citation', '') or
                result.get('text', '')[:50]
            )
            if case_key and case_key not in seen_cases:
                seen_cases.add(case_key)
                deduplicated_results.append(result)

        logger.info(f"Unified query for {self.feature_name}: {len(deduplicated_results)} unique results from {len(sources)} source(s)")

        return {
            "results": deduplicated_results[:top_k * len(sources)],
            "source_results": source_results,
            "total_results": len(deduplicated_results),
            "sources_queried": sources,
            "query": query
        }
