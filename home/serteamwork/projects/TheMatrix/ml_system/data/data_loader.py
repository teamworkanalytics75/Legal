"""Compatibility shim: snake_case import for DataLoader module.

Also exposes a `mysql` attribute (with a `connector.connect` member), so tests
can patch `ml_system.data.data_loader.mysql.connector.connect` regardless of
environment. This mirrors the symbols exposed by the canonical module.
"""
from .DataLoader import *  # noqa: F401,F403

import types as _types
try:  # Best-effort import to expose mysql for test patching
    import mysql as _mysql  # type: ignore
    mysql = _mysql  # noqa: F401
except Exception:
    mysql = _types.SimpleNamespace(connector=_types.SimpleNamespace(connect=None))  # noqa: F401
