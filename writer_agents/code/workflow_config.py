"""Shared workflow configuration for orchestrators.

Defines WorkflowStrategyConfig once to avoid drift between orchestrator modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Union

from .agents import ModelConfig
from .sk_config import SKConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStrategyConfig:
    """Configuration for workflow strategy executor (shared).

    Auto-enables iterative refinement when max_iterations > 1 unless explicitly set.
    """

    # Model configurations
    autogen_config: ModelConfig = field(default_factory=ModelConfig)
    sk_config: Optional[SKConfig] = field(default=None)

    # Local LLM settings (override for specific tasks)
    use_local_for_drafting: bool = True  # Use local for drafting (high volume)
    use_local_for_review: bool = False   # Use OpenAI for final review (quality critical)

    # Workflow settings
    max_iterations: int = 3
    enable_sk_planner: bool = True
    enable_quality_gates: bool = True
    auto_commit_threshold: float = 0.85
    constraint_system_version: str = "1.0"

    # Phase settings
    exploration_rounds: int = 2
    validation_strict: bool = True
    enable_autogen_review: bool = True
    enable_iterative_refinement: Optional[bool] = None  # Auto: True when max_iterations > 1 unless explicitly set

    # Chroma integration settings
    chroma_persist_directory: str = "./chroma_collections"
    enable_chroma_integration: bool = True

    # Google Docs integration settings
    google_docs_enabled: bool = False
    google_drive_folder_id: Optional[str] = None
    google_docs_auto_share: bool = True
    google_docs_capture_version_history: bool = True
    google_docs_learning_enabled: bool = True
    google_docs_live_updates: bool = True  # Enable live updates during workflow (write to G Drive as draft progresses)
    streaming_max_chars: int = 80000  # Safety cap for live LLM streaming chunks
    final_motion_max_chars: int = 120000  # Upper bound for finalized motion text

    # Master draft settings
    master_draft_title: Optional[str] = None  # e.g., "Motion for Seal and Pseudonym - Master Draft"
    master_draft_mode: bool = False  # If True, always use master draft title and update existing doc
    markdown_export_enabled: bool = True  # Export master draft to markdown
    markdown_export_path: str = "outputs/master_drafts"  # Path for markdown exports

    # Version management settings
    enable_version_backups: bool = True  # Create backups before updating master draft
    save_backups_for_ml: bool = True  # Save backups to ML training directory
    version_backup_directory: str = "outputs/master_drafts/versions"  # Where to store version backups
    max_versions_to_keep: int = 50  # Maximum number of versions to keep

    # Episodic memory system settings
    memory_system_enabled: bool = True
    memory_storage_path: Union[str, Path] = "memory_store"
    memory_context_max_items: int = 5
    memory_context_types: List[str] = field(
        default_factory=lambda: ["execution", "edit", "document", "query", "conversation"]
    )

    # Multi-model ensemble settings
    multi_model_config: Optional[Any] = field(default=None)  # MultiModelConfig instance

    # Plugin resource access configuration
    plugin_db_paths: Optional[List[Path]] = None  # Auto-detect if None
    plugin_enable_langchain: bool = True  # LangChain SQL agents
    plugin_enable_courtlistener: bool = False  # CourtListener API (costs money)
    plugin_enable_storm: bool = False  # STORM research (slow but thorough)
    plugin_enable_unified_query: bool = True  # Unified query interface

    # Section plugin toggles (allow disabling groups for faster iterations)
    enable_section_word_count_plugins: bool = True
    enable_section_paragraph_structure_plugins: bool = True
    enable_section_enumeration_depth_plugins: bool = True
    enable_section_sentence_count_plugins: bool = True
    enable_section_words_per_sentence_plugins: bool = True
    enable_constraint_resolver_plugin: bool = True

    # Complete motion generation settings
    enable_complete_motion_generation: bool = True  # Generate all sections instead of just privacy harm

    def __post_init__(self) -> None:
        # Auto-enable iterative refinement when max_iterations > 1 if not explicitly set
        if self.enable_iterative_refinement is None:
            auto_enabled = bool(self.max_iterations and self.max_iterations > 1)
            self.enable_iterative_refinement = auto_enabled
            if auto_enabled:
                logger.info(
                    "WorkflowStrategyConfig auto-enabled iterative refinement "
                    "(max_iterations=%s, refinement_mode=auto)",
                    self.max_iterations,
                )
