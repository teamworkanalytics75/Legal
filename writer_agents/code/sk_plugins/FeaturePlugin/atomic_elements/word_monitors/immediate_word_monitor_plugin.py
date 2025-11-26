#!/usr/bin/env python3
"""
Immediate Word Monitor Plugin - CatBoost Success Signal

Monitors usage of the word "immediate" which is a CatBoost success signal.
Category: Endangerment Implicit
Success Rate: Good
"""

import logging
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel
from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin

logger = logging.getLogger(__name__)


class ImmediateWordMonitorPlugin(CatBoostWordMonitorPlugin):
    """
    Plugin for monitoring the word "immediate" in motion documents.
    
    This word is identified as a CatBoost success signal:
    - Category: Endangerment Implicit
    - Success Rate: Good
    
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
        """Initialize Immediate word monitor plugin."""
        super().__init__(
            kernel=kernel,
            word="immediate",
            word_category="Endangerment Implicit",
            success_rate="Good",
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            memory_store=memory_store,
            **kwargs
        )
        logger.info("ImmediateWordMonitorPlugin initialized for monitoring word: 'immediate'")
