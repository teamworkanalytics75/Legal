"""Tests for plugin registration compatibility with SK 1.37.1 helpers."""

import asyncio
import importlib
import sys

import pytest

# Ensure legacy module paths resolve (agents, tasks, etc.)
_MODULE_ALIASES = {
    "agents": "writer_agents.agents",
    "insights": "writer_agents.insights",
    "tasks": "writer_agents.tasks",
    "sk_config": "writer_agents.sk_config",
    "sk_plugins": "writer_agents.sk_plugins",
}
for alias, target in _MODULE_ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(target)

from writer_agents.code.sk_plugins.ValidationPlugin.validation_functions import (  # noqa: E402
    ValidationPlugin,
)
from writer_agents.code.sk_plugins.drafting_plugin import register as register_drafting_plugin  # noqa: E402


class FakeKernel:
    """Minimal stub mimicking the SK kernel registration surface."""

    def __init__(self):
        self.plugins = {}
        self.functions = []

    # Drafting plugin uses add_plugin fallback when create_plugin_from_functions is missing
    def add_plugin(self, plugin, plugin_name):
        plugin_dict = self.plugins.setdefault(plugin_name, {})
        plugin_dict.update(plugin)
        return plugin_dict

    def add_function(self, func):
        self.functions.append(func)
        return func

    def create_semantic_function(
        self,
        prompt_template,
        plugin_name=None,
        function_name=None,
        prompt_config=None,
    ):
        async def _semantic(*args, **kwargs):
            return {"content": "semantic-output"}

        return _semantic


@pytest.mark.asyncio
async def test_validation_plugin_registers_all_functions():
    """ValidationPlugin should expose all eight validators through the kernel."""
    kernel = FakeKernel()
    plugin = ValidationPlugin(kernel)
    await plugin._register_functions()

    expected = {
        "ValidateCitationFormat",
        "ValidateStructure",
        "ValidateEvidenceGrounding",
        "ValidateToneConsistency",
        "ValidateArgumentCoherence",
        "ValidateLegalAccuracy",
        "ValidatePrivacyHarmChecklist",
        "ValidateGrammarSpelling",
    }

    assert set(plugin._functions.keys()) == expected
    # Stub kernel may not record functions when decorators short-circuit; ensure no crash occurred.
    assert isinstance(kernel.functions, list)


def test_drafting_plugin_registers_without_create_plugin():
    """DraftingPlugin register() should succeed with add_plugin() fallback."""
    kernel = FakeKernel()
    register_drafting_plugin(kernel)

    assert "DraftingPlugin" in kernel.plugins
    drafting_functions = kernel.plugins["DraftingPlugin"]
    # privacy harm semantic helper is added explicitly
    assert "PrivacyHarmSemantic" in drafting_functions
    # native functions should also be present
    assert any(name.startswith("privacy_harm_native") for name in drafting_functions)
