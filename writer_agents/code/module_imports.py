"""Safe import helper for package-safe imports.

Usage:
    mod = safe_import(
        'writer_agents.code.sk_plugins.FeaturePlugin.transition_legal_to_factual_plugin',
        'sk_plugins.FeaturePlugin.transition_legal_to_factual_plugin',
        'transition_legal_to_factual_plugin'
    )
    TransitionLegalToFactualPlugin = getattr(mod, 'TransitionLegalToFactualPlugin')
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional


def safe_import(*module_names: str) -> ModuleType:
    last_err: Optional[Exception] = None
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise ImportError("No module names provided to safe_import")

