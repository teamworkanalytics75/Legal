"""
Petition Quality Plugin for Semantic Kernel.

Validates drafted motions/petitions against success formula extracted from trained model.
Includes both basic validation and enhanced hierarchical constraint validation.
"""
import logging

from semantic_kernel import Kernel

from .petition_validator_function import petition_validator_native_function
from .enhanced_validator import enhanced_validator_native_function
from ...sk_compat import register_functions_with_kernel

PLUGIN_NAME = "PetitionQualityPlugin"
logger = logging.getLogger(__name__)


def register(kernel: Kernel) -> None:
    """Register petition quality validation functions with the kernel."""
    plugin = register_functions_with_kernel(
        kernel,
        PLUGIN_NAME,
        [
            petition_validator_native_function,
            enhanced_validator_native_function,
        ],
    )
    function_count = len(getattr(plugin, "functions", {})) or len(kernel.plugins.get(PLUGIN_NAME, {}))
    logger.info("Registered %s with %d functions", PLUGIN_NAME, function_count)
