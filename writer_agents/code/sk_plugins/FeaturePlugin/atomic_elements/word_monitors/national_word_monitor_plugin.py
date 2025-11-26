#!/usr/bin/env python3
"""
National Word Monitor Plugin - CatBoost Success Signal

Monitors usage of the word "national" which is a CatBoost success signal.
Category: National Security
Success Rate: 45.3% success
"""

import logging
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel
from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin

logger = logging.getLogger(__name__)


class NationalWordMonitorPlugin(CatBoostWordMonitorPlugin):
    """
    Plugin for monitoring the word "national" in motion documents.
    
    This word is identified as a CatBoost success signal:
    - Category: National Security
    - Success Rate: 45.3% success
    
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
        """Initialize National word monitor plugin."""
        super().__init__(
            kernel=kernel,
            word="national",
            word_category="National Security",
            success_rate="45.3% success",
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            memory_store=memory_store,
            **kwargs
        )
        logger.info("NationalWordMonitorPlugin initialized for monitoring word: 'national'")
