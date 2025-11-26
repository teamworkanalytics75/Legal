"""
Drafting plugin for Semantic Kernel.

Provides native and semantic functions for structured legal writing tasks.
"""

from __future__ import annotations

import logging

from semantic_kernel import Kernel

from .privacy_harm_native import privacy_harm_native_function
from .factual_timeline import factual_timeline_native_function
from ..shared.semantic_function_loader import load_semantic_function
from ...sk_compat import register_functions_with_kernel

PLUGIN_NAME = "DraftingPlugin"
logger = logging.getLogger(__name__)


def register(kernel: Kernel) -> None:
    """
    Attach drafting functions to the kernel.
    """
    plugin = register_functions_with_kernel(
        kernel,
        PLUGIN_NAME,
        [
            privacy_harm_native_function,
            factual_timeline_native_function,
        ],
    )

    # Semantic function (prompt) for advanced privacy harm drafting
    kernel.plugins[PLUGIN_NAME]["PrivacyHarmSemantic"] = load_semantic_function(
        kernel,
        package_path=__package__,
        function_directory="PrivacyHarmFunction",
    )
