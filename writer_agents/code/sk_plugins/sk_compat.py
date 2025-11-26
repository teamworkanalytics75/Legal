"""Compatibility shim for legacy ``code.sk_plugins.sk_compat`` imports.

Older modules imported Semantic Kernel helpers from ``code.sk_plugins.sk_compat``
before the helpers were consolidated under ``code.sk_compat``. Recent refactors
updated call sites, but some runtime tooling (e.g., dynamic import strings)
still probes the legacy module path during initialization, producing noisy
``ModuleNotFoundError`` warnings even though fallbacks succeed.

To keep the runtime quiet while remaining backwards compatible, we re-export
all helpers from the canonical ``code.sk_compat`` module.
"""

from ..sk_compat import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
