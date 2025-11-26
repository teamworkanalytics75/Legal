"""Compatibility helpers for Semantic Kernel imports and plugin registration."""

from enum import Enum
from typing import Tuple, Type, Callable, Iterable, Optional, Dict, Any
from dataclasses import dataclass

import logging

logger = logging.getLogger(__name__)


def get_chat_components() -> Tuple[Type, Type, Type]:
    """
    Return ChatHistory, ChatMessageContent, and AuthorRole classes across SK versions.
    """
    try:
        from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole

        logger.debug("Loaded chat components from semantic_kernel.contents")
        return ChatHistory, ChatMessageContent, AuthorRole
    except (ImportError, AttributeError):
        try:
            from semantic_kernel.contents.chat_history import ChatHistory  # type: ignore
            from semantic_kernel.contents.chat_message_content import ChatMessageContent  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Semantic Kernel chat components unavailable; please upgrade semantic-kernel."
            ) from exc

        try:
            from semantic_kernel.contents.chat_role import ChatRole  # type: ignore

            logger.debug("Loaded chat components from legacy semantic_kernel.contents.* modules")
            return ChatHistory, ChatMessageContent, ChatRole
        except ImportError:
            logger.warning(
                "semantic_kernel.contents.chat_role not found; using basic AuthorRole shim"
            )

            class AuthorRoleShim(str, Enum):
                USER = "user"
                ASSISTANT = "assistant"
                SYSTEM = "system"

            return ChatHistory, ChatMessageContent, AuthorRoleShim


def get_memory_record() -> Type:
    """
    Return MemoryRecord class across SK versions.
    """
    try:
        from semantic_kernel.memory.memory_record import MemoryRecord

        logger.debug("Loaded MemoryRecord from semantic_kernel.memory.memory_record")
        return MemoryRecord
    except ImportError:
        try:
            from semantic_kernel.memory import MemoryRecord

            logger.debug("Loaded MemoryRecord from semantic_kernel.memory")
            return MemoryRecord
        except ImportError:
            logger.warning("MemoryRecord not found; using shim dataclass")

            @dataclass
            class MemoryRecordShim:
                id: str
                text: str
                embedding: Optional[Any] = None
                metadata: Optional[Dict[str, Any]] = None
                timestamp: Optional[str] = None
                is_reference: bool = False

            return MemoryRecordShim


def get_kernel_function_decorator() -> Callable:
    """
    Return kernel_function decorator across SK versions.
    """
    try:
        from semantic_kernel.functions.kernel_function_decorator import kernel_function

        logger.debug("Loaded kernel_function from kernel_function_decorator")
        return kernel_function
    except ImportError:
        try:
            from semantic_kernel.functions import kernel_function

            logger.debug("Loaded kernel_function from semantic_kernel.functions")
            return kernel_function
        except ImportError:
            logger.warning("kernel_function decorator not found; using no-op shim")

            def kernel_function_shim(*decorator_args, **decorator_kwargs):
                def decorator(func):
                    return func

                if decorator_args and callable(decorator_args[0]) and not decorator_kwargs:
                    return decorator(decorator_args[0])
                return decorator

            return kernel_function_shim
def register_functions_with_kernel(kernel: Any, plugin_name: str, functions: Iterable[Callable]) -> Any:
    """
    Register kernel_function-decorated callables regardless of SK version.

    Returns the plugin object provided by Semantic Kernel.
    """
    function_list = list(functions)

    if not function_list:
        raise ValueError(f"No functions provided for plugin {plugin_name}")

    if hasattr(kernel, "create_plugin_from_functions"):
        plugin = kernel.create_plugin_from_functions(
            functions=function_list,
            plugin_name=plugin_name,
        )
        logger.info("Registered %s via create_plugin_from_functions()", plugin_name)
        return plugin

    if not hasattr(kernel, "add_plugin"):
        raise AttributeError("Kernel instance lacks both create_plugin_from_functions and add_plugin")

    plugin_dict: Dict[str, Callable] = {}
    for idx, func in enumerate(function_list, start=1):
        func_name = getattr(func, "__name__", None) or getattr(func, "name", None) or f"Function{idx}"
        if func_name in plugin_dict:
            func_name = f"{func_name}_{idx}"
        plugin_dict[func_name] = func

    plugin = kernel.add_plugin(plugin=plugin_dict, plugin_name=plugin_name)
    logger.info("Registered %s via add_plugin() fallback (SK>=1.37.1)", plugin_name)
    return plugin


__all__ = [
    "get_chat_components",
    "get_memory_record",
    "get_kernel_function_decorator",
    "register_functions_with_kernel",
]
