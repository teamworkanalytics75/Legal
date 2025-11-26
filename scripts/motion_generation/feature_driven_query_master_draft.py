#!/usr/bin/env python3
"""
Feature-Driven Query System for Master Draft

Queries databases based on top 20 predictive features, saves results for ML,
and feeds everything into the master draft with continuous refinement.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Load OpenAI API key from config
def load_openai_api_key():
    """Load OpenAI API key from config file."""
    config_path = Path(__file__).parent / "config" / "OpenaiConfig.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                api_key = config.get("openai_api_key") or config.get("OPENAI_API_KEY")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    return True
        except Exception as e:
            print(f"Warning: Could not load API key from config: {e}")
    return False

# Auto-load API key on import
load_openai_api_key()

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "writer_agents" / "code"))
sys.path.insert(0, str(Path(__file__).parent / "document_ingestion"))

try:
    from LangchainIntegration import LangChainSQLAgent
    from agents import ModelConfig
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from SimpleQueryAgent import SimpleLegalQueryAgent
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

from WorkflowStrategyExecutor import (
    WorkflowStrategyExecutor,
    WorkflowStrategyConfig,
    WorkflowState,
    WorkflowPhase
)
from tasks import WriterDeliverable, DraftSection, PlanDirective
from insights import CaseInsights


class FeatureDrivenQuerySystem:
    """Query system driven by top 20 predictive features."""

    def __init__(self):
        """Initialize feature-driven query system."""
        self.top_features = self._load_top_20_features()
        self.query_log = []
        self.query_results = {}

        # Initialize query agents
        self.agents = {}
        self._initialize_agents()

    def _load_top_20_features(self) -> List[Dict]:
        """Load top 20 features from petition model."""
        fi_path = Path("case_law_data/models/petition_model_feature_importance.csv")
        if not fi_path.exists():
            return []

        df = pd.read_csv(fi_path)
        top_20 = df.head(20)

        features = []
        for idx, row in top_20.iterrows():
            features.append({
                'rank': idx + 1,
                'feature': row['feature'],
                'importance': row['importance']
            })

        return features

    def _initialize_agents(self):
        """Initialize query agents for available databases."""
        # Try to load API key from config if not set
        if not os.getenv("OPENAI_API_KEY"):
            load_openai_api_key()

        api_key = os.getenv("OPENAI_API_KEY")
        if LANGCHAIN_AVAILABLE and api_key:
            model_config = ModelConfig(model="gpt-4o-mini")

            # Unified corpus (case law)
            db_path = Path("case_law_data/unified_corpus.db")
            if db_path.exists():
                try:
                    self.agents['unified_corpus'] = LangChainSQLAgent(db_path, model_config, verbose=False)
                except Exception as e:
                    print(f"Warning: Failed to initialize unified_corpus agent: {e}")

            # Exported lawsuit docs
            lawsuit_db = Path("case_law_data/lawsuit_docs_langchain.db")
            if lawsuit_db.exists():
                try:
                    self.agents['lawsuit_docs'] = LangChainSQLAgent(lawsuit_db, model_config, verbose=False)
                except Exception as e:
                    print(f"Warning: Failed to initialize lawsuit_docs agent: {e}")

        # MySQL fallback
        if MYSQL_AVAILABLE:
            try:
                self.agents['lawsuit_docs_mysql'] = SimpleLegalQueryAgent()
            except Exception as e:
                print(f"Warning: MySQL agent not available: {e}")

    def generate_feature_queries(self) -> Dict[str, List[str]]:
        """Generate queries based on top 20 features."""
        queries_by_feature = {}

        for feat in self.top_features:
            feature_name = feat['feature']
            queries = []

            # Generate queries based on feature type
            if 'word_count' in feature_name:
                section = feature_name.replace('_word_count', '').replace('_', ' ')
                queries.extend([
                    f"Find examples of successful {section} sections in case law",
                    f"Show me how {section} sections are structured in successful motions",
                    f"What makes a strong {section} section in legal documents?",
                ])
            elif 'keyword_' in feature_name:
                term = feature_name.replace('keyword_', '').replace('_', ' ')
                queries.extend([
                    f"Find documents that effectively use {term} terminology",
                    f"Show examples of how {term} is discussed in successful cases",
                    f"What are the best practices for using {term} in legal documents?",
                ])
            elif 'citation' in feature_name:
                queries.extend([
                    f"Find examples of effective citation usage in successful motions",
                    f"Show me how citations are structured in granted petitions",
                    f"What citation patterns appear in successful legal documents?",
                ])
            elif 'has_' in feature_name:
                section = feature_name.replace('has_', '').replace('_', ' ')
                queries.extend([
                    f"Find documents that include {section} sections",
                    f"Show examples of effective {section} sections",
                ])
            elif feature_name in ['char_count', 'word_count', 'log_char_count']:
                queries.extend([
                    f"Find examples of successful document length and structure",
                    f"What is the optimal document length for successful motions?",
                ])

            if queries:
                queries_by_feature[feature_name] = queries[:2]  # Top 2 queries per feature

        return queries_by_feature

    def query_feature(self, feature_name: str, query: str, database: str = 'unified_corpus') -> Dict:
        """Query database about a specific feature."""
        agent = self.agents.get(database)
        if not agent:
            return {"success": False, "error": f"Agent for {database} not available"}

        try:
            # Use LangChain for SQLite
            if isinstance(agent, LangChainSQLAgent):
                result = agent.query_evidence(query)
            # Use SimpleQueryAgent for MySQL
            elif hasattr(agent, 'search'):
                results = agent.search(query)
                result = {
                    "success": len(results) > 0,
                    "answer": f"Found {len(results)} relevant documents",
                    "documents": results
                }
            else:
                return {"success": False, "error": "Unknown agent type"}

            # Log query
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "feature": feature_name,
                "query": query,
                "database": database,
                "success": result.get("success", False),
                "has_answer": bool(result.get("answer")),
                "sql": result.get("executed_sql") if isinstance(result, dict) else None
            }
            self.query_log.append(log_entry)

            # Store result
            if feature_name not in self.query_results:
                self.query_results[feature_name] = []
            self.query_results[feature_name].append({
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })

            return result

        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_query_log(self, path: Optional[Path] = None):
        """Save query log for ML training."""
        if path is None:
            path = Path("case_law_data/query_logs") / f"feature_queries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "features": self.top_features,
            "query_log": self.query_log,
            "results_summary": {
                feat['feature']: len(self.query_results.get(feat['feature'], []))
                for feat in self.top_features
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"Saved query log: {path}")
        return path

    def compile_query_results(self) -> str:
        """Compile all query results into text for master draft."""
        compiled = []
        compiled.append("# Research Results Based on Top 20 Predictive Features\n")
        compiled.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for feat in self.top_features:
            feature_name = feat['feature']
            importance = feat['importance']

            if feature_name in self.query_results:
                compiled.append(f"## {feature_name} (Importance: {importance:.2f})\n\n")

                for query_result in self.query_results[feature_name]:
                    result = query_result['result']
                    if result.get("success") and result.get("answer"):
                        compiled.append(f"**Query**: {query_result['query']}\n")
                        compiled.append(f"**Result**: {result['answer'][:500]}...\n\n")

                compiled.append("\n")

        return "\n".join(compiled)


async def run_feature_driven_master_draft_workflow():
    """Run feature-driven query and master draft update workflow."""
    print("="*70)
    print("FEATURE-DRIVEN MASTER DRAFT WORKFLOW")
    print("="*70)

    # Initialize query system
    print("\n[1/5] Initializing feature-driven query system...")
    query_system = FeatureDrivenQuerySystem()
    print(f"   Loaded {len(query_system.top_features)} features")
    print(f"   Initialized {len(query_system.agents)} query agent(s)")

    # Generate queries
    print("\n[2/5] Generating feature-based queries...")
    feature_queries = query_system.generate_feature_queries()
    total_queries = sum(len(qs) for qs in feature_queries.values())
    print(f"   Generated {total_queries} queries across {len(feature_queries)} features")

    # Execute queries
    print("\n[3/5] Executing queries...")
    print("   This may take a few minutes...\n")

    query_count = 0
    for feature_name, queries in feature_queries.items():
        print(f"   Querying: {feature_name}...")
        for query in queries:
            # Try unified_corpus first, then lawsuit_docs
            result = query_system.query_feature(feature_name, query, 'unified_corpus')
            if not result.get("success"):
                result = query_system.query_feature(feature_name, query, 'lawsuit_docs')

            query_count += 1
            if query_count % 5 == 0:
                print(f"      Progress: {query_count}/{total_queries} queries completed")

    print(f"\n   Completed {query_count} queries")

    # Save query log for ML
    print("\n[4/5] Saving query log for ML training...")
    log_path = query_system.save_query_log()

    # Compile results
    print("\n[5/5] Compiling results and updating master draft...")
    research_results = query_system.compile_query_results()

    # Update master draft
    config = WorkflowStrategyConfig(
        master_draft_mode=True,
        master_draft_title="Motion for Seal and Pseudonym - Master Draft",
        google_docs_enabled=True,
        google_drive_folder_id="1MZwep4pb9M52lSLLGQAd3quslA8A5iBu",
        google_docs_capture_version_history=True,
        google_docs_learning_enabled=True,
        markdown_export_enabled=True,
        markdown_export_path="outputs/master_drafts"
    )

    executor = WorkflowStrategyExecutor(config)

    # Load existing master draft or create new
    from DocumentMetadataRecorder import DocumentMetadataRecorder
    doc_recorder = DocumentMetadataRecorder()
    existing_doc = doc_recorder.get_doc_by_title(
        config.master_draft_title,
        folder_id=config.google_drive_folder_id
    )

    # Create deliverable with research results
    sections = [
        DraftSection(
            section_id="feature_research",
            title="Feature-Based Research Results",
            body=research_results
        ),
        DraftSection(
            section_id="outline",
            title="Outline: Top 20 Predictive Features",
            body="[Existing outline section - will be preserved]"
        )
    ]

    plan = PlanDirective(
        objective="Update master draft with feature-driven research results",
        deliverable_format="Legal memorandum with research-backed content",
        tone="Professional legal",
        style_constraints=["Based on top 20 predictive features", "ML-driven insights"]
    )

    # Create case insights
    insights = CaseInsights(
        reference_id="MASTER_DRAFT_001",
        case_style="Master Draft - Motion for Seal and Pseudonym",
        summary="Master draft continuously refined with feature-driven research",
        jurisdiction="US",
        evidence={},
        posteriors={}
    )

    # Create deliverable
    full_content = f"{research_results}\n\n---\n\n## Master Draft Content\n\n[Existing content will be preserved and enhanced]"

    deliverable = WriterDeliverable(
        plan=plan,
        sections=sections,
        edited_document=full_content,
        metadata={
            "workflow_type": "feature_driven_update",
            "query_count": query_count,
            "features_queried": len(feature_queries),
            "log_path": str(log_path),
            "update_timestamp": datetime.now().isoformat()
        }
    )

    state = WorkflowState(
        phase=WorkflowPhase.COMMIT,
        iteration=1
    )

    # Commit to master draft
    try:
        await executor._commit_to_google_docs(deliverable, insights, state)

        if state.google_doc_url:
            print("\n" + "="*70)
            print("[SUCCESS] MASTER DRAFT UPDATED!")
            print("="*70)
            print(f"\n[DOC] Google Doc URL:")
            print(f"{state.google_doc_url}\n")
            print(f"[RESEARCH] Query log saved: {log_path}")
            print(f"[RESULTS] {query_count} queries executed")
            print(f"[FEATURES] {len(feature_queries)} features researched")

            print("\nThe master draft now includes:")
            print("  - Feature-based research results")
            print("  - Evidence from case law databases")
            print("  - ML-ready query logs")
            print("\nThe writing system will validate and refine based on constraints.")
        else:
            print("[WARNING] Document updated but URL not available")

    except Exception as e:
        print(f"\n[ERROR] Failed to update master draft: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_feature_driven_master_draft_workflow())

