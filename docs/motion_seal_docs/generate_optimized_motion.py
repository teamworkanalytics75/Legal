#!/usr/bin/env python3
"""
Master Workflow Script: Generate Optimized Motion to Seal/Pseudonym

End-to-end pipeline: CatBoost Analysis → Semantic Kernel Enforcement → Autogen Writing → Validation

This script generates motion drafts optimized for success based on CatBoost-identified features.
"""

import asyncio
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add writer_agents to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code.WorkflowOrchestrator import Conductor, WorkflowCommitBlocked
from code.workflow_config import WorkflowStrategyConfig
from code.sk_config import SKConfig
from code.insights import CaseInsights, EvidenceItem
from code.agents import ModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = REPO_ROOT / "writer_agents" / "outputs" / "motion_generation_results.json"
DEFAULT_DIAGNOSTICS_OUTPUT = REPO_ROOT / "writer_agents" / "outputs" / "fact_usage_analysis.json"
DIAGNOSTICS_SCRIPT = REPO_ROOT / "writer_agents" / "scripts" / "diagnose_facts_issue.py"

DEFAULT_MASTER_DRAFT_ID = "1rM_J3hIkG28PTY66TgvscGzmgqJiNiMSgiCO1va1tsE"
DEFAULT_MASTER_DRAFT_TITLE = "Motion for Seal and Pseudonym - Master Draft"

STREAMING_ARTIFACT_PATTERNS = [
    re.compile(r"\[StreamingChatMessageContent[^\]]*\]", re.DOTALL),
    re.compile(r"StreamingChatMessageContent\([^\)]*\)", re.DOTALL),
    re.compile(r"\[StreamingTextContent[^\]]*\]", re.DOTALL),
    re.compile(r"StreamingTextContent\([^\)]*\)", re.DOTALL),
    # More aggressive patterns for nested structures
    re.compile(r"\[Streaming\w+[^\]]*?inner_content=[^\]]*?\]", re.DOTALL),
    re.compile(r"Streaming\w+\([^\)]*?inner_content=[^\)]*?\)", re.DOTALL),
]


def _sanitize_final_motion_text(text: Any) -> str:
    """Remove streaming object representations and normalize whitespace."""
    if text is None:
        return ""
    cleaned = str(text)
    
    # First, try to extract any remaining text from object representations
    # Handle nested StreamingChatMessageContent with inner_content
    # Pattern for [StreamingChatMessageContent(...)] with nested content
    nested_pattern = r'\[StreamingChatMessageContent[^\]]*?inner_content=[^\)]*?content=[\'"]([^\'"]+)[\'"][^\]]*?\]'
    matches = list(re.finditer(nested_pattern, cleaned, re.DOTALL))
    for match in reversed(matches):  # Reverse to preserve indices
        extracted_text = match.group(1)
        cleaned = cleaned[:match.start()] + extracted_text + cleaned[match.end():]
    
    # Also try to extract from StreamingTextContent items
    text_content_pattern = r'StreamingTextContent[^\)]*?text=[\'"]([^\'"]+)[\'"][^\)]*?\)'
    text_matches = list(re.finditer(text_content_pattern, cleaned, re.DOTALL))
    for match in reversed(text_matches):
        extracted_text = match.group(1)
        cleaned = cleaned[:match.start()] + extracted_text + cleaned[match.end():]
    
    # Remove all remaining object representations (more aggressive)
    for pattern in STREAMING_ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    
    # Remove any remaining object-like patterns (fallback)
    cleaned = re.sub(r'\[?Streaming\w+\([^\)]*\)\]?', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'Streaming\w+\([^\)]*\)', '', cleaned, flags=re.DOTALL)
    
    # Clean up artifacts
    cleaned = cleaned.replace("][", "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _load_case_corpus_payload(preferred_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load lawsuit facts JSON (fact_blocks + extracted_facts) if available."""
    candidates: List[Path] = []
    seen: Set[Path] = set()

    def _add_candidate(candidate: Optional[Path]) -> None:
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    _add_candidate(preferred_path)
    default_lawsuit_facts = REPO_ROOT / "writer_agents" / "outputs" / "lawsuit_facts_extracted.json"
    default_case_insights = REPO_ROOT / "writer_agents" / "outputs" / "case_insights.json"
    _add_candidate(default_lawsuit_facts)
    _add_candidate(default_case_insights)

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning(f"Could not load lawsuit facts from {candidate}: {exc}")
            continue

        fact_blocks = payload.get("fact_blocks") or {}
        extracted = payload.get("extracted_facts") or {}
        if fact_blocks or extracted:
            logger.info(f"Loaded personal corpus facts from {candidate}")
            return {"fact_blocks": fact_blocks, "extracted_facts": extracted}

    return None


def run_preflight_checks(case_insights_file: Optional[Path] = None) -> bool:
    """Validate that core artifacts exist before launching the workflow."""
    warnings: List[str] = []

    # Check CatBoost model coverage
    models_dir = REPO_ROOT / "case_law_data" / "models"
    required_models = [
        "catboost_outline_unified.cbm",
        "catboost_motion_seal_pseudonym.cbm",
        "section_1782_discovery_model.cbm"
    ]
    missing_models = [name for name in required_models if not (models_dir / name).exists()]
    if missing_models:
        warnings.append(
            f"Missing CatBoost models in {models_dir}: {', '.join(missing_models)}"
        )
    else:
        logger.info(
            "Pre-flight OK - CatBoost models detected (%s)",
            ", ".join(required_models)
        )

    # Check personal corpus readiness (lawsuit_facts_extracted.json or lawsuit_source_documents artifacts)
    case_insights_candidates: List[Path] = []
    if case_insights_file:
        case_insights_candidates.append(case_insights_file)
    default_lawsuit_facts = REPO_ROOT / "writer_agents" / "outputs" / "lawsuit_facts_extracted.json"
    default_case_insights = REPO_ROOT / "writer_agents" / "outputs" / "case_insights.json"  # Backward compatibility
    if all(candidate != default_lawsuit_facts for candidate in case_insights_candidates):
        case_insights_candidates.append(default_lawsuit_facts)
    if all(candidate != default_case_insights for candidate in case_insights_candidates):
        case_insights_candidates.append(default_case_insights)  # Backward compatibility

    case_insights_ready = False
    case_insights_errors: List[str] = []
    for candidate in case_insights_candidates:
        try:
            if candidate.exists():
                with candidate.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                fact_blocks = data.get("fact_blocks") or {}
                extracted = data.get("extracted_facts") or {}
                if fact_blocks or extracted:
                    case_insights_ready = True
                    logger.info(
                        "Pre-flight OK - Loaded %d fact blocks from %s",
                        len(fact_blocks),
                        candidate
                    )
                    break
                case_insights_errors.append(
                    f"{candidate} is readable but contains no fact blocks."
                )
        except Exception as exc:  # noqa: BLE001
            case_insights_errors.append(f"Unable to read {candidate}: {exc}")

    personal_corpus_ready = case_insights_ready
    corpus_dir = REPO_ROOT / "case_law_data" / "lawsuit_source_documents"
    # Support old name for backward compatibility
    if not corpus_dir.exists():
        old_corpus_dir = REPO_ROOT / "case_law_data" / "tmp_corpus"
        if old_corpus_dir.exists():
            corpus_dir = old_corpus_dir
    if not personal_corpus_ready and corpus_dir.exists():
        try:
            if any(corpus_dir.iterdir()):
                personal_corpus_ready = True
                logger.info(
                    "Pre-flight OK - Personal corpus directory populated at %s",
                    corpus_dir
                )
        except Exception as exc:  # noqa: BLE001
            case_insights_errors.append(f"Unable to inspect {corpus_dir}: {exc}")

    metadata_candidates = [
        REPO_ROOT / "case_law_data" / "results" / "personal_corpus_features.csv",
        REPO_ROOT / "case_law_data" / "results" / "personal_corpus_metadata.json"
    ]
    if not personal_corpus_ready and any(path.exists() for path in metadata_candidates):
        personal_corpus_ready = True
        logger.info(
            "Pre-flight OK - Personal corpus metadata detected in case_law_data/results"
        )

    if not personal_corpus_ready:
        warnings.extend(case_insights_errors or [
            "Personal corpus artifacts missing. "
            "Add writer_agents/outputs/lawsuit_facts_extracted.json or populate case_law_data/lawsuit_source_documents."
        ])

    # Check outline feature extraction availability
    outline_script = REPO_ROOT / "case_law_data" / "scripts" / "extract_perfect_outline_features.py"
    outline_ready = False
    if outline_script.exists():
        try:
            spec = importlib.util.spec_from_file_location("preflight_outline", outline_script)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                if hasattr(module, "extract_perfect_outline_features"):
                    outline_ready = True
                    logger.info(
                        "Pre-flight OK - extract_perfect_outline_features available"
                    )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"Outline feature extraction module raised {exc}. "
                "Run case_law_data/scripts/extract_perfect_outline_features.py to regenerate."
            )
    else:
        warnings.append(
            f"Outline feature extraction script missing at {outline_script}. "
            "Ensure outline utilities are in place."
        )

    if not outline_ready and outline_script.exists():
        warnings.append(
            "Outline feature extraction script found but could not load extract_perfect_outline_features."
        )

    if warnings:
        logger.warning("Pre-flight checks reported %d issue(s):", len(warnings))
        for issue in warnings:
            logger.warning(" - %s", issue)
        logger.warning("Continuing despite pre-flight warnings.")
        return False

    logger.info("Pre-flight checks complete - all critical artifacts detected.")
    return True


def _write_results_file(results: Dict[str, Any], destination: Path) -> None:
    """Persist motion generation results to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    
    # Sanitize final_motion
    final_motion = payload.get("final_motion", "")
    sanitized = _sanitize_final_motion_text(final_motion)
    
    # Validate no artifacts remain
    if "StreamingChatMessageContent" in sanitized or "StreamingTextContent" in sanitized:
        logger.warning("Streaming artifacts detected in final_motion after sanitization, applying additional cleanup")
        # More aggressive cleanup
        sanitized = re.sub(r'\[?Streaming\w+[^\]]*?\]?', '', sanitized, flags=re.DOTALL)
        sanitized = re.sub(r'Streaming\w+\([^\)]*\)', '', sanitized, flags=re.DOTALL)
        # Final check
        if "StreamingChatMessageContent" in sanitized or "StreamingTextContent" in sanitized:
            logger.error("Streaming artifacts still present after aggressive cleanup - motion text may be corrupted")
    
    payload["final_motion"] = sanitized
    
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Results saved to %s", destination)


def _run_diagnostics_cli(motion_results: Path, output_path: Path) -> None:
    """Invoke the diagnostics script using the current interpreter."""
    cmd = [
        sys.executable,
        str(DIAGNOSTICS_SCRIPT),
        "--motion-results",
        str(motion_results),
        "--output-json",
        str(output_path),
    ]
    logger.info("Running diagnostics: %s", " ".join(str(part) for part in cmd))
    try:
        subprocess.run(cmd, check=True)
        logger.info("Diagnostics completed successfully")
    except subprocess.CalledProcessError as exc:
        logger.error("Diagnostics failed (exit code %s)", exc.returncode)
        raise


def create_case_insights_from_input(
    case_summary: str,
    jurisdiction: str = "D. Massachusetts",
    evidence: Optional[Dict[str, Any]] = None,
    posteriors: Optional[Dict[str, Any]] = None
) -> CaseInsights:
    """Create CaseInsights from user input (dict-friendly).

    Evidence and posteriors may be dicts; we pass them through as lists of dicts
    for downstream compatibility.
    """
    ev_list = []
    if isinstance(evidence, dict):
        # Flatten fact blocks/extracted facts into evidence items
        # Align node_id format with CaseFactsProvider expectation: 'fact_block_<key>'
        fact_blocks = evidence.get("fact_blocks") or {}
        for k, v in fact_blocks.items():
            ev_list.append(EvidenceItem(node_id=f"fact_block_{k}", state=str(v)).to_dict())
        # Do not inject extracted_facts as evidence to avoid polluting targeted fact blocks
        # extracted = evidence.get("extracted_facts") or {}
    elif isinstance(evidence, list):
        # Normalize list items to dict form compatible with CaseInsights/EvidenceItem
        normalized: list[dict] = []
        for item in evidence:
            if isinstance(item, dict):
                node_id = item.get("node_id")
                state = item.get("state")
                if node_id and state is not None:
                    normalized.append(EvidenceItem(node_id=node_id, state=str(state)).to_dict())
            else:
                node_id = getattr(item, "node_id", None)
                state = getattr(item, "state", None)
                if node_id and state is not None:
                    normalized.append(EvidenceItem(node_id=node_id, state=str(state)).to_dict())
        ev_list = normalized

    post_list = []
    if isinstance(posteriors, dict):
        # Expect {node_id: {state: prob}}
        for node, probs in posteriors.items():
            post_list.append({"node_id": node, "probabilities": probs})
    elif isinstance(posteriors, list):
        post_list = posteriors

    return CaseInsights(
        reference_id="local_case",
        summary=case_summary,
        jurisdiction=jurisdiction,
        evidence=ev_list,
        posteriors=post_list,
        case_style="Motion for Leave to File Under Seal and to Proceed Under Pseudonym"
    )


async def generate_optimized_motion(
    case_summary: str,
    jurisdiction: str = "D. Massachusetts",
    evidence: Optional[Dict[str, Any]] = None,
    posteriors: Optional[Dict[str, Any]] = None,
    autogen_model: str = "gpt-4o",
    sk_use_local: bool = True,  # Default to local LLM (Phi-3 Mini)
    target_confidence: float = 0.70,
    max_iterations: int = 10,
    refinement_mode: str = "auto",  # auto|on|off
    workflow_overrides: Optional[Dict[str, Any]] = None,
    google_docs_enabled: bool = True,
    master_draft_id: Optional[str] = DEFAULT_MASTER_DRAFT_ID,
    master_draft_title: Optional[str] = DEFAULT_MASTER_DRAFT_TITLE,
    google_drive_folder_id: Optional[str] = None,
    hk_statement_path: Optional[Path] = None,
    ogc_emails_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate optimized motion using CatBoost → SK → Autogen pipeline.

    Args:
        case_summary: Brief summary of the case
        jurisdiction: Court jurisdiction
        evidence: Evidence dictionary (optional)
        posteriors: Bayesian network posteriors (optional)
        autogen_model: Model to use for Autogen agents
        sk_use_local: Whether to use local model for Semantic Kernel
        target_confidence: Target CatBoost confidence threshold (default 0.70 = 70%)
        max_iterations: Maximum workflow iterations

    Returns:
        Dictionary with final motion, validation results, and metrics
    """
    logger.info("="*70)
    logger.info("Generating Optimized Motion to Seal/Pseudonym")
    logger.info("="*70)

    evidence_payload = evidence or _load_case_corpus_payload()

    # Create case insights
    insights = create_case_insights_from_input(
        case_summary=case_summary,
        jurisdiction=jurisdiction,
        evidence=evidence_payload,
        posteriors=posteriors
    )

    # Configure workflow
    # Respect environment variable for local LLM usage
    use_local_autogen = os.environ.get("MATRIX_USE_LOCAL", "true").strip().lower() in {"1", "true", "yes", "on"}
    autogen_config = ModelConfig(
        model=autogen_model,
        use_local=use_local_autogen,
        local_model=os.environ.get("MATRIX_LOCAL_MODEL", "qwen2.5:14b"),
        local_base_url=os.environ.get("MATRIX_LOCAL_URL", "http://localhost:11434")
    )
    sk_config = SKConfig(use_local=sk_use_local)  # Default to local LLM (qwen2.5:14b)

    # Determine iterative refinement default
    enable_iterative_refinement = (
        True if refinement_mode == "on" else
        False if refinement_mode == "off" else
        (max_iterations > 1)
    )

    if refinement_mode == "auto" and enable_iterative_refinement:
        logger.info(
            "Iterative refinement auto-enabled (max_iterations=%s, refinement_mode=auto)",
            max_iterations,
        )

    workflow_config = WorkflowStrategyConfig(
        autogen_config=autogen_config,
        sk_config=sk_config,
        max_iterations=max_iterations,
        enable_sk_planner=True,
        enable_quality_gates=True,
        auto_commit_threshold=target_confidence,
        enable_autogen_review=True,
        validation_strict=True,
        enable_iterative_refinement=enable_iterative_refinement,
        enable_chroma_integration=True,
        chroma_persist_directory="./chroma_collections"
    )
    
    effective_overrides: Dict[str, Any] = dict(workflow_overrides or {})
    if google_docs_enabled:
        effective_overrides.setdefault("google_docs_enabled", True)
        effective_overrides.setdefault("master_draft_mode", True)
        effective_overrides.setdefault(
            "master_draft_title",
            master_draft_title or DEFAULT_MASTER_DRAFT_TITLE,
        )
        if google_drive_folder_id:
            effective_overrides.setdefault(
                "google_drive_folder_id",
                google_drive_folder_id,
            )
        effective_overrides.setdefault("google_docs_live_updates", True)
        effective_overrides.setdefault("markdown_export_enabled", True)
    else:
        effective_overrides.setdefault("google_docs_enabled", False)
        effective_overrides.setdefault("master_draft_mode", False)

    if effective_overrides:
        for key, value in effective_overrides.items():
            if hasattr(workflow_config, key):
                setattr(workflow_config, key, value)
                logger.info(f"Applied workflow override: {key} = {value}")
            else:
                logger.warning(f"Ignoring unknown workflow override: {key}")

    # Initialize conductor
    executor = Conductor(config=workflow_config)

    # Run workflow
    logger.info("Starting hybrid workflow...")
    try:
        initial_doc_id = (
            master_draft_id
            if workflow_config.google_docs_enabled and master_draft_id
            else None
        )
        initial_doc_url = (
            f"https://docs.google.com/document/d/{initial_doc_id}/edit"
            if initial_doc_id
            else None
        )

        try:
            deliverable = await executor.run_hybrid_workflow(
                insights,
                initial_google_doc_id=initial_doc_id,
                initial_google_doc_url=initial_doc_url,
            )
        except TypeError as exc:
            # Backwards compatibility for stub executors used in tests
            if "unexpected keyword argument" in str(exc):
                deliverable = await executor.run_hybrid_workflow(insights)
            else:
                raise

        # Extract final document and validation results
        final_document = deliverable.edited_document or deliverable.sections[0].body if deliverable.sections else ""
        final_document = _sanitize_final_motion_text(final_document)
        google_doc_id = None

        # Get validation results from executor state if available
        validation_results = {}
        if hasattr(executor, '_last_state') and executor._last_state:
            google_doc_id = getattr(executor._last_state, "google_doc_id", None)
            validation_results = executor._last_state.validation_results or {}
        # Also try to get from deliverable metadata if available
        if not validation_results and hasattr(deliverable, 'metadata') and deliverable.metadata:
            validation_results = deliverable.metadata.get('validation_results', {})
        if not google_doc_id and hasattr(deliverable, 'metadata') and deliverable.metadata:
            google_doc_id = deliverable.metadata.get('google_doc_id')

        meets_threshold_flag = validation_results.get("meets_threshold")
        validation_passed = True if meets_threshold_flag is None else bool(meets_threshold_flag)
        blocking_reasons: List[str] = []
        if not validation_passed:
            failed_gates = validation_results.get("failed_gates") or []
            if failed_gates:
                blocking_reasons.append(f"Failed gates: {', '.join(failed_gates)}")
            personal_summary = (validation_results.get("personal_facts_verification") or {})
            if personal_summary.get("critical_facts_missing"):
                missing = personal_summary.get("facts_missing", [])
                if missing:
                    blocking_reasons.append(f"Missing facts: {', '.join(missing[:5])}")
            catboost_validation = validation_results.get("catboost_validation") or {}
            if not catboost_validation.get("meets_threshold", True):
                blocking_reasons.append(
                    f"CatBoost confidence {catboost_validation.get('predicted_success_probability', 0.0):.2f} below target"
                )

        # Compile results
        results = {
            "success": validation_passed,
            "final_motion": final_document,
            "google_doc_id": google_doc_id,
            "validation_results": validation_results,
            "deliverable": {
                "plan": deliverable.plan.dict() if hasattr(deliverable.plan, 'dict') else str(deliverable.plan),
                "sections": [s.dict() if hasattr(s, 'dict') else {"title": s.title, "body": s.body} for s in deliverable.sections],
                "reviews": [r.dict() if hasattr(r, 'dict') else str(r) for r in deliverable.reviews]
            },
            "metrics": {
                "target_confidence": target_confidence,
                "catboost_confidence": validation_results.get("catboost_validation", {}).get("predicted_success_probability", 0.0),
                "overall_score": validation_results.get("overall_score", 0.0),
                "meets_target": validation_results.get("catboost_validation", {}).get("meets_threshold", False)
            }
        }
        if not validation_passed:
            reasons_text = "; ".join(blocking_reasons) if blocking_reasons else "Validation thresholds not met."
            results["error"] = reasons_text
            results["blocking_reasons"] = blocking_reasons

        logger.info("="*70)
        logger.info("Motion Generation Complete")
        logger.info("="*70)
        logger.info(f"CatBoost Confidence: {results['metrics']['catboost_confidence']:.1%}")
        logger.info(f"Overall Score: {results['metrics']['overall_score']:.2f}")
        logger.info(f"Meets Target: {results['metrics']['meets_target']}")

        return results

    except WorkflowCommitBlocked as blocked:
        logger.error(f"Workflow blocked: {blocked}")
        state = getattr(executor, "_last_state", None)
        google_doc_id = getattr(state, "google_doc_id", None) if state else None
        validation_results = state.validation_results if state and state.validation_results else {}
        draft_result = getattr(state, "draft_result", {}) if state else {}
        final_document = (
            draft_result.get("complete_motion")
            or draft_result.get("privacy_harm_section")
            or draft_result.get("edited_document")
            or ""
        )
        final_document = _sanitize_final_motion_text(final_document)
        catboost_block = validation_results.get("catboost_validation", {}) if validation_results else {}
        results = {
            "success": False,
            "final_motion": final_document,
            "google_doc_id": google_doc_id,
            "validation_results": validation_results,
            "error": str(blocked),
            "blocking_reasons": [str(blocked)],
            "metrics": {
                "target_confidence": target_confidence,
                "catboost_confidence": catboost_block.get("predicted_success_probability", 0.0),
                "overall_score": validation_results.get("overall_score", 0.0),
                "meets_target": catboost_block.get("meets_threshold", False),
            },
        }
        return results

    except Exception:
        logger.exception("Motion generation failed")
        return {
            "success": False,
            "error": "Motion generation failed",
            "final_motion": "",
            "validation_results": {}
        }


def main():
    """Command-line interface for motion generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate optimized motion to seal/pseudonym using CatBoost → SK → Autogen pipeline"
    )
    parser.add_argument(
        "--case-summary",
        required=True,
        help="Brief summary of the case"
    )
    parser.add_argument(
        "--jurisdiction",
        default="D. Massachusetts",
        help="Court jurisdiction (default: D. Massachusetts)"
    )
    parser.add_argument(
        "--evidence",
        type=str,
        help="Path to JSON file with evidence dictionary"
    )
    parser.add_argument(
        "--case-insights-file",
        type=str,
        help="Path to writer_agents/outputs/lawsuit_facts_extracted.json (defaults to that path if present, also supports case_insights.json for backward compatibility)"
    )
    parser.add_argument(
        "--posteriors",
        type=str,
        help="Path to JSON file with Bayesian network posteriors"
    )
    parser.add_argument(
        "--autogen-model",
        default="gpt-4o",
        help="Model for Autogen agents (default: gpt-4o)"
    )
    parser.add_argument(
        "--target-confidence",
        type=float,
        default=0.70,
        help="Target CatBoost confidence threshold (default: 0.70)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum workflow iterations (default: 10)"
    )
    parser.add_argument(
        "--refinement-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Iterative refinement: auto (on if max_iterations>1), on, or off (default: auto)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (JSON format)"
    )
    parser.add_argument(
        "--run-diagnostics",
        action="store_true",
        help="Run fact-flow diagnostics after motion generation"
    )
    parser.add_argument(
        "--diagnostics-output",
        type=str,
        default=str(DEFAULT_DIAGNOSTICS_OUTPUT),
        help="Path for the fact usage analysis JSON (default: writer_agents/outputs/fact_usage_analysis.json)"
    )
    parser.add_argument(
        "--master-draft-id",
        default=DEFAULT_MASTER_DRAFT_ID,
        help="Google Doc ID of the master motion draft to update (default: %(default)s)"
    )
    parser.add_argument(
        "--master-draft-title",
        default=DEFAULT_MASTER_DRAFT_TITLE,
        help="Title used to locate the master draft in Google Drive"
    )
    parser.add_argument(
        "--google-drive-folder-id",
        type=str,
        help="Optional Google Drive folder ID that contains the master draft"
    )
    parser.add_argument(
        "--no-google-docs",
        dest="google_docs_enabled",
        action="store_false",
        help="Disable Google Docs live updates and write local drafts only"
    )
    parser.set_defaults(google_docs_enabled=True)

    args = parser.parse_args()
    case_insights_override: Optional[Path] = None
    if args.case_insights_file:
        case_insights_override = Path(args.case_insights_file).expanduser().resolve()

    run_preflight_checks(case_insights_override)

    # Load evidence and posteriors if provided
    evidence = None
    if args.evidence:
        with open(args.evidence, 'r') as f:
            evidence = json.load(f)

    if evidence is None:
        evidence = _load_case_corpus_payload(case_insights_override)

    posteriors = None
    if args.posteriors:
        with open(args.posteriors, 'r') as f:
            posteriors = json.load(f)

    # Run generation
    results = asyncio.run(generate_optimized_motion(
        case_summary=args.case_summary,
        jurisdiction=args.jurisdiction,
        evidence=evidence,
        posteriors=posteriors,
        autogen_model=args.autogen_model,
        target_confidence=args.target_confidence,
        max_iterations=args.max_iterations,
        refinement_mode=args.refinement_mode,
        google_docs_enabled=args.google_docs_enabled,
        master_draft_id=args.master_draft_id,
        master_draft_title=args.master_draft_title,
        google_drive_folder_id=args.google_drive_folder_id,
    ))

    # Always persist canonical results for downstream diagnostics
    _write_results_file(results, DEFAULT_RESULTS_PATH)

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        _write_results_file(results, output_path)

        # Also save motion text as separate file
        motion_path = output_path.with_suffix('.txt')
        with open(motion_path, 'w') as f:
            f.write(_sanitize_final_motion_text(results.get("final_motion", "")))
        logger.info(f"Motion text saved to {motion_path}")

    if args.run_diagnostics:
        diagnostics_output = Path(args.diagnostics_output).expanduser().resolve()
        diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            _run_diagnostics_cli(DEFAULT_RESULTS_PATH, diagnostics_output)
        except subprocess.CalledProcessError:
            logger.error("Diagnostics run failed; see log above for details.")

    # Print summary
    if results.get("success"):
        print("\n" + "="*70)
        print("SUCCESS - Motion Generated")
        print("="*70)
        print(f"CatBoost Confidence: {results['metrics']['catboost_confidence']:.1%}")
        print(f"Overall Score: {results['metrics']['overall_score']:.2f}")
        print(f"Target Met: {'✓' if results['metrics']['meets_target'] else '✗'}")
        print("\nFinal motion:")
        print("-"*70)
        sanitized_motion = _sanitize_final_motion_text(results.get("final_motion", ""))
        print(sanitized_motion[:500] + "..." if len(sanitized_motion) > 500 else sanitized_motion)
    else:
        print("\n" + "="*70)
        print("ERROR - Motion Generation Failed")
        print("="*70)
        print(f"Error: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
