"""
Assembly plugin for constructing final documents.
"""

from __future__ import annotations

import logging

from semantic_kernel import Kernel

from .document_assembler import assemble_document
from .exhibit_linker import link_exhibits
from ...sk_compat import register_functions_with_kernel


PLUGIN_NAME = "AssemblyPlugin"
logger = logging.getLogger(__name__)


def register(kernel: Kernel) -> None:
    register_functions_with_kernel(
        kernel,
        PLUGIN_NAME,
        [
            assemble_document,
            link_exhibits,
        ],
    )
