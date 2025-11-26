import types

from writer_agents.code.sk_compat import register_functions_with_kernel


def _sample_function():
    return "ok"


class _KernelAddOnly:
    def __init__(self):
        self.plugins = {}

    def add_plugin(self, plugin=None, plugin_name=None, **_kwargs):
        self.plugins[plugin_name] = plugin or {}
        return types.SimpleNamespace(functions=self.plugins[plugin_name])


class _KernelWithCreate(_KernelAddOnly):
    def create_plugin_from_functions(self, functions=None, plugin_name=None):
        plugin = types.SimpleNamespace(functions={f.__name__: f for f in functions or []})
        self.plugins[plugin_name] = plugin.functions
        return plugin


def test_register_functions_with_kernel_fallback_adds_plugin():
    kernel = _KernelAddOnly()
    plugin = register_functions_with_kernel(kernel, "TestPlugin", [_sample_function])
    assert "TestPlugin" in kernel.plugins
    assert isinstance(plugin.functions, dict)
    assert _sample_function.__name__ in kernel.plugins["TestPlugin"]


def test_register_functions_with_kernel_prefers_create_plugin_method():
    kernel = _KernelWithCreate()
    plugin = register_functions_with_kernel(kernel, "SemanticPlugin", [_sample_function])
    assert "SemanticPlugin" in kernel.plugins
    assert plugin.functions == kernel.plugins["SemanticPlugin"]
