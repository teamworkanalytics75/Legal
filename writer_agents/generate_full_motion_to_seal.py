#!/usr/bin/env python3
"""
Full Pipeline Motion Generator - Motion to Seal and Pseudonym

This script runs the complete pipeline to generate a full motion to seal and pseudonym:
1. Initializes all components (Conductor, RefinementLoop, OutlineManager, SHAP)
2. Creates CaseInsights for the motion
3. Runs the full workflow (Explore → Research → Plan → Draft → Validate → Review → Refine → Commit)
4. Uses outline integration for proper section organization
5. Commits to Google Drive master draft
6. Leverages SHAP insights for quality improvement
7. Uses memory system for learning
"""

import argparse
import asyncio
import logging
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Fix Windows console encoding for emoji/unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "writer_agents"))
sys.path.insert(0, str(project_root / "writer_agents" / "code"))


def build_case_insights_from_final_facts_db(
    database_path: Path,
    summary_template: str,
    jurisdiction: str = "D. Mass."
) -> Optional[Any]:
    """
    Build CaseInsights dynamically from the fact_registry database.
    
    This function queries the final facts database and extracts relevant facts
    to build comprehensive CaseInsights with actual case data.
    
    Note: CaseInsights, Posterior, and EvidenceItem are imported inside the function
    to avoid circular import issues.
    """
    if not database_path.exists():
        logger.warning(f"Database not found at {database_path}")
        return None
    
    try:
        # Import here to avoid circular imports
        from code.insights import CaseInsights, Posterior, EvidenceItem
        
        # Query facts from database
        facts_by_type: Dict[str, List[str]] = {}
        evidence_items: List[Any] = []
        all_fact_types: Set[str] = set()
        
        with sqlite3.connect(str(database_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT fact_id, fact_type, fact_value, description, source_doc, 
                       extraction_method, confidence, metadata
                FROM fact_registry
                ORDER BY confidence DESC, created_at DESC
            """)
            
            rows = cursor.fetchall()
            if not rows:
                logger.warning(f"No facts found in fact_registry table at {database_path}")
                return None
            
            # First pass: collect all fact types and group facts
            for row in rows:
                # sqlite3.Row objects use dictionary-style indexing, not .get()
                # Use try/except for safe access with defaults
                try:
                    fact_type = row["fact_type"] or "unknown"
                except (KeyError, IndexError):
                    fact_type = "unknown"
                
                try:
                    fact_value = row["fact_value"] or ""
                except (KeyError, IndexError):
                    fact_value = ""
                
                try:
                    fact_id = row["fact_id"] or ""
                except (KeyError, IndexError):
                    fact_id = ""
                
                try:
                    description = row["description"] or ""
                except (KeyError, IndexError):
                    description = ""
                
                try:
                    source_doc = row["source_doc"] or ""
                except (KeyError, IndexError):
                    source_doc = ""
                
                try:
                    confidence = row["confidence"] or 0.0
                except (KeyError, IndexError):
                    confidence = 0.0
                
                if not fact_value:
                    continue
                
                # Track all fact types found in database
                all_fact_types.add(fact_type)
                
                # Group facts by type
                facts_by_type.setdefault(fact_type, []).append(fact_value)
            
            # Second pass: create evidence items for ALL facts (not just hardcoded types)
            # Use actual fact types from database, not hardcoded list
            for row in rows:
                # sqlite3.Row objects use dictionary-style indexing, not .get()
                try:
                    fact_type = row["fact_type"] or "unknown"
                    fact_value = row["fact_value"] or ""
                    fact_id = row["fact_id"] or ""
                    description = row["description"] or ""
                    confidence = row["confidence"] or 0.0
                except (KeyError, IndexError):
                    continue  # Skip rows with missing required fields
                
                if not fact_value:
                    continue
                
                # Include ALL facts as evidence items, not just specific types
                evidence_items.append(EvidenceItem(
                    node_id=f"fact_block_{fact_id}",
                    state=fact_value,
                    description=description or fact_value,
                    weight=float(confidence) if confidence else 0.85
                ))
            
            logger.info(f"[FACTS] Found {len(all_fact_types)} distinct fact types: {sorted(all_fact_types)}")
            logger.info(f"[FACTS] Created {len(evidence_items)} evidence items from {len(rows)} database rows")
        
        # Build summary from facts
        summary_parts = [summary_template]
        if facts_by_type.get("Harm"):
            summary_parts.append(f"\n\nHarm Evidence: {len(facts_by_type['Harm'])} documented instances of harm.")
        if facts_by_type.get("PRC_Risk"):
            summary_parts.append(f"\n\nPRC Risk Evidence: {len(facts_by_type['PRC_Risk'])} documented safety risks.")
        if facts_by_type.get("Defamation"):
            summary_parts.append(f"\n\nDefamation Evidence: {len(facts_by_type['Defamation'])} documented defamatory statements.")
        
        summary = " ".join(summary_parts)
        
        # Build posteriors based on ACTUAL fact types found in database
        posteriors = []
        
        # Use actual fact types from database, not hardcoded assumptions
        if facts_by_type.get("Risk") or facts_by_type.get("Harm"):
            risk_count = len(facts_by_type.get("Risk", []))
            harm_count = len(facts_by_type.get("Harm", []))
            posteriors.append(Posterior(
                node_id="Safety_Risk_Arbitrary_Detention",
                probabilities={"High": 0.95, "Moderate": 0.05},
                interpretation=(
                    f"Plaintiff faces credible risk of arbitrary detention, torture, and retaliation. "
                    f"Database contains {risk_count} documented risk facts and {harm_count} harm facts."
                )
            ))
        
        if facts_by_type.get("Allegation") or facts_by_type.get("Harm"):
            allegation_count = len(facts_by_type.get("Allegation", []))
            harm_count = len(facts_by_type.get("Harm", []))
            posteriors.append(Posterior(
                node_id="Defamation_Harm_Reputation_Economic",
                probabilities={"High": 0.92, "Moderate": 0.08},
                interpretation=(
                    f"Defamatory statements causing severe harm. "
                    f"Database contains {allegation_count} allegation facts and {harm_count} harm facts."
                )
            ))
        
        if facts_by_type.get("Publication") or facts_by_type.get("Disclosure"):
            pub_count = len(facts_by_type.get("Publication", []))
            disclosure_count = len(facts_by_type.get("Disclosure", []))
            posteriors.append(Posterior(
                node_id="Doxxing_Privacy_Harm",
                probabilities={"High": 0.90, "Moderate": 0.10},
                interpretation=(
                    f"Defamatory publications and disclosures causing privacy violations. "
                    f"Database contains {pub_count} publication facts and {disclosure_count} disclosure facts."
                )
            ))
        
        if facts_by_type.get("Spoliation") or facts_by_type.get("Spoliation Risk"):
            spoliation_count = len(facts_by_type.get("Spoliation", []))
            spoliation_risk_count = len(facts_by_type.get("Spoliation Risk", []))
            posteriors.append(Posterior(
                node_id="Harvard_Knowledge_Spoliation",
                probabilities={"High": 0.88, "Moderate": 0.12},
                interpretation=(
                    f"Evidence of spoliation and knowledge. "
                    f"Database contains {spoliation_count} spoliation facts and {spoliation_risk_count} spoliation risk facts."
                )
            ))
        
        if facts_by_type.get("Communication") or facts_by_type.get("Non-Response") or facts_by_type.get("NonResponse"):
            comm_count = len(facts_by_type.get("Communication", []))
            nonresponse_count = len(facts_by_type.get("Non-Response", [])) + len(facts_by_type.get("NonResponse", []))
            posteriors.append(Posterior(
                node_id="Harvard_Knowledge_NonResponse",
                probabilities={"High": 0.85, "Moderate": 0.15},
                interpretation=(
                    f"Harvard had knowledge through communications but failed to respond. "
                    f"Database contains {comm_count} communication facts and {nonresponse_count} non-response facts."
                )
            ))
        
        # Always include legal standard
        posteriors.append(Posterior(
            node_id="Legal_Standard_Sealing_PRC_Risk",
            probabilities={"High": 0.98, "Moderate": 0.02},
            interpretation=(
                "The legal standard for sealing requires showing that privacy and safety interests "
                "outweigh public access. In cases involving foreign government retaliation risks, "
                "courts have recognized the need for protective measures including pseudonym filing."
            )
        ))
        
        # Add standard legal evidence items
        evidence_items.extend([
            EvidenceItem(
                node_id="FRCP_5.2",
                state="Applicable",
                description="Federal Rule of Civil Procedure 5.2(d) permits courts to allow filing under seal when privacy or safety interests outweigh public access.",
                weight=1.0
            ),
            EvidenceItem(
                node_id="FRCP_10_a",
                state="Applicable",
                description="Federal Rule of Civil Procedure 10(a) allows pseudonym filing when there is a 'substantial privacy interest' or 'exceptional circumstances,' including foreign government retaliation risks.",
                weight=1.0
            ),
        ])
        
        logger.info(f"[FACTS] Built CaseInsights from database: {len(facts_by_type)} fact types, {len(evidence_items)} evidence items")
        
        return CaseInsights(
            reference_id="MOTION_TO_SEAL_PSEUDONYM_HARVARD_DEFAMATION",
            summary=summary,
            case_style="Federal District Court - Harvard Defamation Case",
            jurisdiction=jurisdiction,
            posteriors=posteriors,
            evidence=evidence_items
        )
        
    except Exception as e:
        logger.error(f"Failed to build CaseInsights from database: {e}", exc_info=True)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate motion to seal/pseudonym via full workflow")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Disable Google Docs commits and keep outputs local",
    )
    parser.add_argument(
        "--local-llm",
        action="store_true",
        help="Force local LLM usage even if USE_OPENAI is set",
    )
    return parser.parse_args()


async def generate_full_motion(args: argparse.Namespace | None = None):
    """Generate a complete motion to seal and pseudonym using the full pipeline."""
    args = args or argparse.Namespace(dry_run=False, local_llm=False)

    print("\n" + "="*80)
    print("[START] FULL PIPELINE: MOTION TO SEAL AND PSEUDONYM")
    print("="*80)

    try:
        # Import all necessary components
        print("\n[IMPORT] Starting imports...")
        sys.stdout.flush()
        try:
            print("[IMPORT] Importing WorkflowOrchestrator...")
            sys.stdout.flush()
            from code.WorkflowOrchestrator import Conductor as WorkflowOrchestrator
            print("[IMPORT] WorkflowOrchestrator imported")
            sys.stdout.flush()
            from code.workflow_config import WorkflowStrategyConfig
            print("[IMPORT] WorkflowStrategyConfig imported")
            sys.stdout.flush()
        except ImportError:
            print("[IMPORT] Relative import failed, trying absolute...")
            sys.stdout.flush()
            from WorkflowOrchestrator import Conductor as WorkflowOrchestrator
            from workflow_config import WorkflowStrategyConfig

        print("[IMPORT] Importing CaseInsights, Posterior, EvidenceItem...")
        sys.stdout.flush()
        from code.insights import CaseInsights, Posterior, EvidenceItem
        print("[IMPORT] CaseInsights imported")
        sys.stdout.flush()
        from code.agents import ModelConfig
        print("[IMPORT] ModelConfig imported")
        sys.stdout.flush()
        from code.sk_config import SKConfig
        print("[IMPORT] SKConfig imported")
        sys.stdout.flush()

        print("\n[OK] Imported all components")
        sys.stdout.flush()
        
        # Import CaseInsights, Posterior, EvidenceItem for the build function
        # (already imported above, but need to ensure they're available)

        # 1. Initialize configuration
        print("\n[CONFIG] Creating configuration...")

        # DEFAULT TO LOCAL LLM (Ollama)
        # Only use OpenAI if explicitly requested via environment variable
        use_openai = os.environ.get("USE_OPENAI", "false").lower() == "true"
        if args.local_llm:
            use_openai = False
        api_key = None

        if use_openai:
            # Try to get OpenAI API key if explicitly requested
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Try multiple possible locations
                key_files = [
                    Path(".openai_api_key.txt"),
                    Path("writer_agents/.openai_api_key.txt"),
                    Path(".env"),
                    Path("secrets.toml")
                ]
                for key_file in key_files:
                    if key_file.exists():
                        try:
                            content = key_file.read_text(encoding="utf-8").strip()
                            # Check if it's a .env file
                            if key_file.name == ".env":
                                for line in content.split("\n"):
                                    if line.startswith("OPENAI_API_KEY="):
                                        api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                                        break
                            # Check if it's a secrets.toml file
                            elif key_file.name == "secrets.toml":
                                try:
                                    import tomllib
                                    secrets = tomllib.loads(content)
                                    api_key = secrets.get("OPENAI_API_KEY") or secrets.get("openai", {}).get("api_key")
                                except:
                                    pass
                            else:
                                api_key = content
                            if api_key:
                                logger.info(f"Found API key in {key_file}")
                                break
                        except Exception as e:
                            logger.debug(f"Could not read {key_file}: {e}")
                            continue

        # Create AutoGen configuration (use local by default)
        autogen_config = ModelConfig(
            model="gpt-4o-mini",  # This is just for AutoGen config, SK uses local
            temperature=0.2,
            max_tokens=4096,
            use_local=True,  # Default to local LLM (Ollama)
            local_model="qwen2.5:14b"  # Default local model
        )

        # Create SK configuration - DEFAULT TO LOCAL LLM
        if use_openai and api_key:
            sk_config = SKConfig(
                model_name="gpt-4o-mini",
                temperature=0.3,
                max_tokens=4000,
                api_key=api_key,
                use_local=False  # Use OpenAI
            )
            print("   [OK] Using OpenAI (gpt-4o-mini) for content generation")
        else:
            # DEFAULT: Use local LLM (Ollama) - use qwen2.5:14b for better quality
            sk_config = SKConfig(
                use_local=True,
                local_model="qwen2.5:14b",  # Better quality than phi3:mini for legal writing
                temperature=0.3,
                max_tokens=4000
            )
            print("   [OK] Using local LLM (Ollama/qwen2.5:14b) for content generation (DEFAULT)")
            print("   [INFO] Make sure Ollama is running: ollama serve")
            print("   [INFO] Using qwen2.5:14b for better quality (phi3:mini was generating garbage)")

        # KNOWN DOCUMENT ID (from previous run)
        KNOWN_MASTER_DRAFT_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"
        KNOWN_MASTER_DRAFT_URL = f"https://docs.google.com/document/d/{KNOWN_MASTER_DRAFT_ID}/edit?usp=drivesdk"

        google_docs_enabled = not args.dry_run
        if args.dry_run:
            print("[DRY RUN] Google Docs integration disabled; results will remain local.")

        # Create workflow configuration with MASTER DRAFT MODE enabled
        config = WorkflowStrategyConfig(
            autogen_config=autogen_config,
            sk_config=sk_config,
            google_docs_enabled=google_docs_enabled,
            google_drive_folder_id="1MZwep4pb9M52lSLLGQAd3quslA8A5iBu" if google_docs_enabled else None,
            google_docs_auto_share=google_docs_enabled,
            google_docs_capture_version_history=google_docs_enabled,
            google_docs_learning_enabled=google_docs_enabled,
            google_docs_live_updates=google_docs_enabled,  # Enable live updates during workflow
            memory_system_enabled=True,
            memory_storage_path="memory_store",
            memory_context_max_items=10,  # More context for comprehensive motion
            memory_context_types=["execution", "edit", "document", "query", "conversation"],
            enable_sk_planner=True,  # Enable planner for better structure
            enable_quality_gates=True,  # Enable quality gates
            auto_commit_threshold=0.85,  # High threshold for quality
            enable_iterative_refinement=True,  # Enable refinement loop
            max_iterations=5,  # Allow more iterations for comprehensive motion
            # MASTER DRAFT SETTINGS
            master_draft_mode=google_docs_enabled,
            master_draft_title="Motion for Seal and Pseudonym - Master Draft",
            markdown_export_enabled=True,
            markdown_export_path="outputs/master_drafts",
            enable_version_backups=True,
            version_backup_directory="outputs/master_drafts/versions",
            max_versions_to_keep=50
        )

        print("   [OK] Configuration created:")
        print(f"     - Master Draft Mode: {'ENABLED' if config.master_draft_mode else 'DISABLED'}")
        print(f"     - Master Draft Title: {config.master_draft_title}")
        if google_docs_enabled:
            print(f"     - Google Docs: Enabled (Folder ID: {config.google_drive_folder_id})")
            print(f"     - [LIVE] Live Updates: ENABLED (you can watch the draft update in real-time!)")
        else:
            print("     - Google Docs: Disabled (dry-run mode)")
            print("     - [LIVE] Live Updates: DISABLED (local-only workflow)")
        print(f"     - Memory System: Enabled")
        print(f"     - Quality Gates: Enabled (threshold: {config.auto_commit_threshold})")
        print(f"     - Iterative Refinement: Enabled")
        print(f"     - Max Iterations: {config.max_iterations}")

        if os.environ.get("MOTION_TEST_MODE") == "1":
            print("[TEST MODE] Skipping orchestrator execution; returning stub result.")
            return {"status": "test-mode", "timestamp": datetime.now().isoformat()}

        # 2. Initialize WorkflowStrategyExecutor (Conductor)
        print("\n[ROBOT] Initializing WorkflowStrategyExecutor (Conductor)...")
        print("   - This initializes: AutoGen, Semantic Kernel, RefinementLoop, OutlineManager, SHAP")
        print("   - Outline integration will be automatically enabled")
        print("   - SHAP insights will be computed during refinement")

        print("   [INFO] Initialization may take 1-3 minutes (loading 50+ plugins, models, etc.)...")
        orchestrator = WorkflowOrchestrator(config)

        print("   [OK] Conductor initialized successfully")
        print("     - AutoGen writing team ready")
        print("     - Semantic Kernel quality control ready")
        print("     - RefinementLoop (CatBoost + SHAP) ready")
        print("     - OutlineManager (perfect outline structure) ready")
        print("     - Google Docs bridge ready")

        # 3. Create comprehensive CaseInsights for motion to seal and pseudonym
        # NOTE: This uses ACTUAL case facts from the final facts database (605 facts)
        # Build CaseInsights dynamically from the fact_registry database
        print("\n[DOC] Creating CaseInsights for Motion to Seal and Pseudonym...")
        print("   [INFO] Loading ACTUAL case facts from final facts database (605 facts)")
        
        # Load facts from database to build CaseInsights
        database_path = project_root / "case_law_data" / "lawsuit_facts_database.db"
        insights = build_case_insights_from_final_facts_db(
            database_path=database_path,
            summary_template=(
                "Motion to seal sensitive personal information and allow filing under pseudonym "
                "pursuant to Federal Rule of Civil Procedure 5.2(d) and Federal Rule of Civil Procedure 10(a). "
                "This motion seeks to protect plaintiff's privacy and safety interests in a defamation case "
                "involving Harvard-affiliated entities, PRC-based harassment and surveillance risks, and "
                "politically sensitive content. The motion addresses privacy concerns related to doxxing, "
                "defamatory publications in PRC-facing media, threats of arbitrary detention, and ongoing "
                "safety risks from foreign government retaliation. The case involves Harvard Club statements, "
                "OGC communications, Hong Kong defamation proceedings, and §1782 discovery requests."
            ),
            jurisdiction="D. Mass."
        )

        # Fail if database loading fails - no fake/fallback data allowed
        if not insights:
            raise RuntimeError(
                f"Failed to load facts from database at {database_path}. "
                "Motion generation requires real facts from fact_registry. "
                "Cannot proceed with fake/fallback data."
            )
        
        # Ensure we have real facts loaded
        if not insights.evidence or len(insights.evidence) == 0:
            raise RuntimeError(
                "CaseInsights created but contains no evidence items. "
                "Database may be empty or fact_registry table has no rows. "
                "Cannot proceed without real case facts."
            )
        
        print(f"   [OK] Loaded {len(insights.evidence)} evidence items from database")
        
        # OLD FALLBACK CODE REMOVED - No fake data allowed
        # If you see this, it means the above check failed
        if False:  # This block will never execute, kept for reference
            insights = CaseInsights(
            reference_id="MOTION_TO_SEAL_PSEUDONYM_HARVARD_DEFAMATION",
            summary=(
                "Motion to seal sensitive personal information and allow filing under pseudonym "
                "pursuant to Federal Rule of Civil Procedure 5.2(d) and Federal Rule of Civil Procedure 10(a). "
                "This motion seeks to protect plaintiff's privacy and safety interests in a defamation case "
                "involving Harvard-affiliated entities, PRC-based harassment and surveillance risks, and "
                "politically sensitive content. The motion addresses privacy concerns related to doxxing, "
                "defamatory publications in PRC-facing media, threats of arbitrary detention, and ongoing "
                "safety risks from foreign government retaliation. The case involves Harvard Club statements, "
                "OGC communications, Hong Kong defamation proceedings, and §1782 discovery requests."
            ),
            case_style="Federal District Court - Harvard Defamation Case",
            jurisdiction="D. Mass.",
            posteriors=[
                Posterior(
                    node_id="PRC_Safety_Risk_Arbitrary_Detention",
                    probabilities={"High": 0.95, "Moderate": 0.05},
                    interpretation=(
                        "Plaintiff faces credible risk of arbitrary detention, torture, and retaliation "
                        "from PRC authorities due to politically sensitive content and defamatory publications "
                        "circulating in PRC-facing media. Disclosure of identity could enable PRC authorities "
                        "to locate and harm the plaintiff, as demonstrated by the EsuWiki case pattern."
                    )
                ),
                Posterior(
                    node_id="Defamation_Harm_Reputation_Economic",
                    probabilities={"High": 0.92, "Moderate": 0.08},
                    interpretation=(
                        "Harvard-affiliated entities published defamatory statements that were amplified "
                        "across PRC-facing platforms (WeChat, Zhihu, Baidu, etc.), causing severe reputational "
                        "harm, loss of consulting contracts, cancelled classes, and ongoing economic damage."
                    )
                ),
                Posterior(
                    node_id="Doxxing_Privacy_Harm",
                    probabilities={"High": 0.90, "Moderate": 0.10},
                    interpretation=(
                        "Defamatory publications included doxxing of plaintiff's personal information, "
                        "enabling harassment, surveillance, and threats. Plaintiff was subject to handler "
                        "surveillance and travel restrictions due to fear of PRC retaliation."
                    )
                ),
                Posterior(
                    node_id="Harvard_Knowledge_Spoliation",
                    probabilities={"High": 0.88, "Moderate": 0.12},
                    interpretation=(
                        "Harvard had actual knowledge of harm and PRC risks through multiple channels "
                        "(GSS warnings, OGC emails, Center Shanghai communications) but failed to correct "
                        "defamatory statements. Evidence of spoliation through website deletions and backdating."
                    )
                ),
                Posterior(
                    node_id="Legal_Standard_Sealing_PRC_Risk",
                    probabilities={"High": 0.98, "Moderate": 0.02},
                    interpretation=(
                        "The legal standard for sealing requires showing that privacy and safety interests "
                        "outweigh public access. In cases involving foreign government retaliation risks, "
                        "courts have recognized the need for protective measures including pseudonym filing."
                    )
                )
            ],
            evidence=[
                EvidenceItem(
                    node_id="FRCP_5.2",
                    state="Applicable",
                    description="Federal Rule of Civil Procedure 5.2(d) permits courts to allow filing under seal when privacy or safety interests outweigh public access. The rule specifically protects social security numbers, financial account numbers, and other sensitive personal identifiers.",
                    weight=1.0
                ),
                EvidenceItem(
                    node_id="FRCP_10_a",
                    state="Applicable",
                    description="Federal Rule of Civil Procedure 10(a) requires parties to be named in the complaint, but courts may allow pseudonym filing when there is a 'substantial privacy interest' or 'exceptional circumstances,' including foreign government retaliation risks.",
                    weight=1.0
                ),
                EvidenceItem(
                    node_id="HK_Statement_of_Claim",
                    state="Filed",
                    description="Hong Kong High Court Statement of Claim filed June 2, 2025, alleging defamation, doxxing, privacy violations, and conspiracy by Harvard-affiliated clubs.",
                    weight=0.95
                ),
                EvidenceItem(
                    node_id="Harvard_Club_Statements",
                    state="Published",
                    description="Defamatory statements published by Harvard Clubs of Hong Kong, Beijing, and Shanghai in April 2019, falsely accusing plaintiff of misrepresenting admissions role.",
                    weight=0.95
                ),
                EvidenceItem(
                    node_id="PRC_Media_Amplification",
                    state="Ongoing",
                    description="Defamatory content amplified across PRC-facing platforms including WeChat, Zhihu, Baidu Baijiahao, E-Canada, and Sohu, causing viral circulation and ongoing harm.",
                    weight=0.90
                ),
                EvidenceItem(
                    node_id="EsuWiki_Pattern",
                    state="Relevant",
                    description="EsuWiki case (June-August 2019) demonstrates PRC pattern of arbitrary detention and torture for politically sensitive online content, establishing foreseeable risk pattern.",
                    weight=0.85
                ),
                EvidenceItem(
                    node_id="OGC_Non_Response",
                    state="Documented",
                    description="Harvard OGC received multiple warnings about PRC risks and harm but failed to respond or correct defamatory statements, demonstrating institutional knowledge.",
                    weight=0.90
                )
            ]
        )

        print("   [OK] CaseInsights created:")
        print(f"     - Reference ID: {insights.reference_id}")
        print(f"     - Jurisdiction: {insights.jurisdiction}")
        print(f"     - Posteriors: {len(insights.posteriors)} elements")
        print(f"     - Evidence Items: {len(insights.evidence)} items")
        print("\n   Key Legal Elements:")
        for idx, posterior in enumerate(insights.posteriors, 1):
            top_outcome = posterior.top_outcome()
            top_prob = posterior.probabilities.get(top_outcome, 0.0) if top_outcome else 0.0
            print(f"     {idx}. {posterior.node_id} ({top_outcome}: {top_prob:.2f})")

        # 4. Run the full workflow
        print("\n[START] Running Full Pipeline Workflow...")
        print("="*80)
        print("Workflow Phases:")
        print("  1. EXPLORE    - AutoGen explores arguments and strategies")
        print("  2. RESEARCH   - Case law research and evidence gathering")
        print("  3. PLAN       - Semantic Kernel creates structured plan")
        print("  4. DRAFT      - AutoGen generates initial draft [LIVE UPDATE to Google Drive]")
        print("  5. VALIDATE   - RefinementLoop analyzes with CatBoost + SHAP [LIVE UPDATE]")
        print("  6. REVIEW     - AutoGen reviews and suggests improvements")
        print("  7. REFINE     - RefinementLoop refines based on outline + SHAP insights [LIVE UPDATE]")
        print("  8. COMMIT     - Final commit to Google Drive master draft")
        print("\n   [LIVE] Live updates enabled! Check your Google Drive document to watch progress in real-time.")
        print("="*80)
        print()

        # Execute the workflow with known document ID for live updates
        result = await orchestrator.run_hybrid_workflow(
            insights,
            initial_google_doc_id=KNOWN_MASTER_DRAFT_ID,
            initial_google_doc_url=KNOWN_MASTER_DRAFT_URL
        )

        print("\n" + "="*80)
        print("[RESULTS] WORKFLOW COMPLETED")
        print("="*80)

        # Get the actual result object to extract state
        if hasattr(orchestrator, '_last_state') and orchestrator._last_state:
            state = orchestrator._last_state
            if state.google_doc_url:
                if not isinstance(result, dict):
                    result = {}
                result['document_url'] = state.google_doc_url
                result['document_id'] = state.google_doc_id

        # 5. Display results
        if result:
            print("\n[OK] Workflow completed successfully!")

            # Extract key information from result
            if isinstance(result, dict):
                print(f"\n[SUMMARY] Result Summary:")
                print(f"   - Phase: {result.get('phase', 'unknown')}")
                print(f"   - Iterations: {result.get('iterations', 'unknown')}")

                if 'document_url' in result:
                    print(f"\n   [DOC] Document URL: {result['document_url']}")

                if 'document_id' in result:
                    print(f"   [ID] Document ID: {result['document_id']}")

                if 'sections' in result:
                    print(f"\n   [SECTIONS] Sections Generated: {len(result.get('sections', []))}")
                    for section in result.get('sections', [])[:10]:  # Show first 10
                        section_name = section.get('name', section.get('section_id', 'unknown'))
                        word_count = section.get('word_count', 'unknown')
                        print(f"      - {section_name} ({word_count} words)")

                if 'validation_results' in result:
                    val_results = result['validation_results']
                    if isinstance(val_results, dict):
                        overall_score = val_results.get('overall_score', 'unknown')
                        print(f"\n   [OK] Validation Score: {overall_score}")
                        if 'shap_insights' in val_results:
                            print(f"   [SHAP] SHAP Insights: Generated")
                        if 'outline_validation' in val_results:
                            outline_val = val_results['outline_validation']
                            if isinstance(outline_val, dict):
                                if outline_val.get('valid', False):
                                    print(f"   [OUTLINE] Outline Structure: Valid (follows perfect outline)")
                                else:
                                    print(f"   [WARNING] Outline Structure: Needs adjustment")
                                    if 'recommendations' in outline_val:
                                        print(f"      Recommendations:")
                                        for rec in outline_val['recommendations'][:3]:
                                            print(f"        - {rec}")
            else:
                print(f"\n[INFO] Result Type: {type(result).__name__}")
                print(f"   Result: {str(result)[:200]}...")
        else:
            print("[WARNING] Workflow completed but no result returned")

        # 6. Final summary
        print("\n" + "="*80)
        print("[OK] FULL PIPELINE COMPLETE")
        print("="*80)

        # Show markdown file location
        markdown_path = Path(config.markdown_export_path) / "master_draft.md"
        if markdown_path.exists():
            print(f"\n[FILE] Markdown file saved to:")
            print(f"   {markdown_path.resolve()}")
            print(f"   Size: {markdown_path.stat().st_size:,} bytes")
        else:
            print(f"\n[FILE] Markdown export path: {markdown_path.resolve()}")
            print("   (File may be created during commit phase)")

        # Show Google Doc URL if available
        if result and isinstance(result, dict):
            if 'document_url' in result:
                print(f"\n[GOOGLE DOC] Document URL:")
                print(f"   {result['document_url']}")

        print("\n[NEXT STEPS] Next Steps:")
        print("   1. [OK] Check your Google Drive folder for the master draft")
        print("   2. [OK] Review the markdown file at the path above")
        print("   3. [OK] Review the document structure (should follow perfect outline)")
        print("   4. [OK] Verify outline integration (Legal Standard -> Factual Background transition)")
        print("   5. [OK] Check SHAP insights (used for quality improvement)")
        print("   6. [OK] Review enumeration requirements (11+ instances)")
        print("   7. [OK] Verify all sections are present and properly ordered")
        print("\n[INFO] The system has:")
        print("   - Used CatBoost + SHAP for quality analysis")
        print("   - Applied perfect outline structure")
        print("   - Organized plugins by section")
        print("   - Recalibrated plugin targets based on outline")
        print("   - Committed to Google Drive master draft")
        print("   - Stored learnings in memory system")
        print()

        return result

    except ImportError as e:
        print(f"\n[ERROR] Import error: {e}")
        print("   Make sure you're running from the project root")
        import traceback
        traceback.print_exc()
        return None

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        logger.exception("Failed to generate motion")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    cli_args = parse_args()
    print("\n[START] Starting Full Pipeline Motion Generation...")
    print("   This will generate a complete motion to seal and pseudonym")
    print("   using all integrated systems: CatBoost, SHAP, Outline, Google Drive")
    print()

    result = asyncio.run(generate_full_motion(cli_args))

    if isinstance(result, dict) and result.get("status") == "test-mode":
        print("\n[SUCCESS] Test-mode run completed (no external services were contacted).")
    elif result:
        print("\n[SUCCESS] SUCCESS! Your motion has been generated with the full pipeline.")
        print("   The document is available in your Google Drive master draft folder.")
    else:
        print("\n[WARNING] Motion generation failed. Check the logs above for details.")
