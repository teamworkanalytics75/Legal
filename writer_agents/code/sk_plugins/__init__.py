"""Plugin registry and base classes for Semantic Kernel plugins."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Awaitable
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

try:
    from ..sk_compat import register_functions_with_kernel
except ImportError:  # pragma: no cover - fallback for legacy import paths
    from sk_compat import register_functions_with_kernel  # type: ignore[misc]

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a SK plugin."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "TheMatrix Writer Agents"
    functions: List[str] = None

    def __post_init__(self):
        if self.functions is None:
            self.functions = []


class BaseSKPlugin(ABC):
    """Abstract base class for Semantic Kernel plugins."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.metadata = self._get_metadata()
        self._functions: Dict[str, KernelFunction] = {}

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    async def _register_functions(self) -> None:
        """Register all plugin functions with the kernel."""
        pass

    async def initialize(self) -> None:
        """Initialize the plugin by registering its functions."""
        logger.info(f"Initializing plugin: {self.metadata.name}")
        await self._register_functions()
        logger.info(f"Plugin {self.metadata.name} initialized with {len(self._functions)} functions")

    def get_function(self, name: str) -> Optional[KernelFunction]:
        """Get a function by name."""
        return self._functions.get(name)

    def list_functions(self) -> List[str]:
        """List all function names in this plugin."""
        return list(self._functions.keys())

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata


class PluginRegistry:
    """Registry for managing SK plugins."""

    def __init__(self):
        self._plugins: Dict[str, BaseSKPlugin] = {}
        self._kernel: Optional[Kernel] = None

    def set_kernel(self, kernel: Kernel) -> None:
        """Set the kernel for plugin registration."""
        self._kernel = kernel

    async def register_plugin(self, plugin: BaseSKPlugin) -> None:
        """Register a plugin with the kernel."""
        if not self._kernel:
            raise RuntimeError("Kernel not set. Call set_kernel() first.")

        plugin_name = plugin.metadata.name
        if plugin_name in self._plugins:
            logger.warning(f"Plugin {plugin_name} already registered. Overwriting.")

        self._plugins[plugin_name] = plugin
        await plugin.initialize()

        # Ensure functions are available via Semantic Kernel's plugin system
        if plugin._functions:
            try:
                register_functions_with_kernel(self._kernel, plugin_name, plugin._functions.values())
            except Exception as exc:
                logger.warning(f"Failed to register {plugin_name} functions with kernel: {exc}")

        logger.info(f"Plugin {plugin_name} registered successfully")

    def get_plugin(self, name: str) -> Optional[BaseSKPlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        """List all registered plugin names."""
        return list(self._plugins.keys())

    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        plugin = self.get_plugin(name)
        return plugin.get_metadata() if plugin else None


# Global plugin registry instance
plugin_registry = PluginRegistry()


def register_plugin(plugin: BaseSKPlugin) -> Awaitable[None]:
    """Convenience helper that returns the registration coroutine."""
    return plugin_registry.register_plugin(plugin)


# Export commonly used classes and functions
__all__ = [
    "BaseSKPlugin",
    "PluginMetadata",
    "PluginRegistry",
    "plugin_registry",
    "register_plugin"
]
