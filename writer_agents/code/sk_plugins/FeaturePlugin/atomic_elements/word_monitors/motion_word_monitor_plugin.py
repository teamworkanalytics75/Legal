#!/usr/bin/env python3
"""
Motion Word Monitor Plugin - CatBoost Success Signal

Monitors usage of the word "motion" which is a CatBoost success signal.
Category: Motion Language
Success Rate: 50% success
"""

import logging
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel
from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin

logger = logging.getLogger(__name__)


class MotionWordMonitorPlugin(CatBoostWordMonitorPlugin):
    """
    Plugin for monitoring the word "motion" in motion documents.
    
    This word is identified as a CatBoost success signal:
    - Category: Motion Language
    - Success Rate: 50% success
    
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
        """Initialize Motion word monitor plugin."""
        super().__init__(
            kernel=kernel,
            word="motion",
            word_category="Motion Language",
            success_rate="50% success",
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            memory_store=memory_store,
            **kwargs
        )
        logger.info("MotionWordMonitorPlugin initialized for monitoring word: 'motion'")
