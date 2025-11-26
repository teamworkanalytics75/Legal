"""
Validation plugin housing deterministic and semantic quality gates.
"""

from __future__ import annotations

import logging

from semantic_kernel import Kernel

from .citation_validator import validate_citation_format
from .structure_validator import validate_structure
from ..shared.semantic_function_loader import load_semantic_function
from ...sk_compat import register_functions_with_kernel

PLUGIN_NAME = "ValidationPlugin"
logger = logging.getLogger(__name__)


def register(kernel: Kernel) -> None:
    plugin = register_functions_with_kernel(
        kernel,
        PLUGIN_NAME,
        [
            validate_citation_format,
            validate_structure,
        ],
    )

    kernel.plugins[PLUGIN_NAME]["ValidateToneConsistency"] = load_semantic_function(
        kernel,
        package_path=__package__,
        function_directory="ToneConsistencyFunction",
    )
