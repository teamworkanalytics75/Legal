# Writer Agents Code Package
"""
Initialize shared aliases so absolute imports like ``sk_plugins`` resolve to the
package-local implementation under ``code.sk_plugins``. This prevents Python
from loading two distinct module instances when files alternate between
absolute and relative imports, which previously caused plugin registries to
diverge and break Semantic Kernel plugin lookups.
"""

from importlib import import_module
import sys


def _ensure_sk_plugins_alias() -> None:
    """Alias ``sk_plugins`` to ``code.sk_plugins`` once for the entire runtime."""
    if "sk_plugins" in sys.modules:
        return
    try:
        module = import_module(".sk_plugins", __name__)
    except ImportError:
        return
    sys.modules["sk_plugins"] = module


_ensure_sk_plugins_alias()
