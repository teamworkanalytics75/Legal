#!/usr/bin/env python3
"""
Complete Word Monitor Plugin - CatBoost Success Signal

Monitors usage of the word "complete" which is a CatBoost success signal.
Category: Thoroughness
Success Rate: Word count #1 predictor
"""

import logging
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel
from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin

logger = logging.getLogger(__name__)


class CompleteWordMonitorPlugin(CatBoostWordMonitorPlugin):
    """
    Plugin for monitoring the word "complete" in motion documents.
    
    This word is identified as a CatBoost success signal:
    - Category: Thoroughness
    - Success Rate: Word count #1 predictor
    
    The plugin tracks frequency, context, and usage patterns to help
    calibrate with other plugins (sentences, paragraphs, arguments, etc.).
    """

    def __init__(
        self,
        kernel: Kernel,
        chroma_store=None,
        rules_dir: Optional[Path] = None,
        memory_store=None,
        **kwargs
    ):
        """Initialize Complete word monitor plugin."""
        super().__init__(
            kernel=kernel,
            word="complete",
            word_category="Thoroughness",
            success_rate="Word count #1 predictor",
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            memory_store=memory_store,
            **kwargs
        )
        logger.info("CompleteWordMonitorPlugin initialized for monitoring word: 'complete'")
