"""Base plugin classes for Semantic Kernel plugins."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

try:
    from ..sk_compat import get_kernel_function_decorator
except (ImportError, ValueError):
    try:
        from writer_agents.code.sk_compat import get_kernel_function_decorator
    except ImportError:
        try:
            from sk_compat import get_kernel_function_decorator  # type: ignore
        except ImportError:
            import sys
            from pathlib import Path

            parent_dir = Path(__file__).resolve().parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            from sk_compat import get_kernel_function_decorator  # type: ignore

kernel_function = get_kernel_function_decorator()

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    description: str
    version: str
    functions: List[str]


class BaseSKPlugin(ABC):
    """Base class for Semantic Kernel plugins."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.metadata = self._get_metadata()  # ✅ Set metadata in __init__ (required by PluginRegistry)
        self._functions: Dict[str, KernelFunction] = {}

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abstractmethod
    async def _register_functions(self) -> None:
        """Register functions with the kernel."""
        pass

    async def initialize(self) -> None:
        """Initialize the plugin."""
        await self._register_functions()
        logger.info(f"Plugin {self.metadata.name} initialized with {len(self._functions)} functions")

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self.metadata

    def get_functions(self) -> Dict[str, KernelFunction]:
        """Get registered functions."""
        return self._functions.copy()


@dataclass
class FunctionResult:
    """Result from a plugin function execution."""
    success: bool
    value: Any = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    data: Optional[Any] = None  # Alternative to value for structured data
    message: Optional[str] = None  # Alternative to error for human-readable messages
    
    def __post_init__(self):
        """Map data/message to value/metadata for backwards compatibility."""
        # If data is provided but value is not, use data as value
        if self.data is not None and self.value is None:
            self.value = self.data
        # If message is provided, store it in metadata for informational purposes
        if self.message is not None:
            if self.metadata is None:
                self.metadata = {}
            if "message" not in self.metadata:
                self.metadata["message"] = self.message


@dataclass
class DocumentLocation:
    """Reference to a specific location in a document."""
    paragraph_index: Optional[int] = None  # 0-based paragraph index
    sentence_index: Optional[int] = None   # 0-based sentence index within paragraph
    character_offset: Optional[int] = None  # Absolute character position (fallback)
    position_type: str = "after"  # "before", "after", "replace"
    section_name: Optional[str] = None  # Optional section identifier

    def __post_init__(self):
        """Validate location data."""
        if self.paragraph_index is None and self.sentence_index is None and self.character_offset is None:
            raise ValueError("At least one location identifier must be provided")
        if self.position_type not in ["before", "after", "replace"]:
            raise ValueError(f"Invalid position_type: {self.position_type}. Must be 'before', 'after', or 'replace'")


@dataclass
class EditRequest:
    """Request to edit a document at a specific location."""
    plugin_name: str
    location: DocumentLocation
    edit_type: str  # "insert", "replace", "delete"
    content: str
    priority: int  # Higher = more important (0-100)
    affected_plugins: List[str]  # Plugins that need re-validation
    metadata: Dict[str, Any]  # Additional context
    reason: str  # Why this edit is needed

    def __post_init__(self):
        """Validate edit request."""
        if self.edit_type not in ["insert", "replace", "delete"]:
            raise ValueError(f"Invalid edit_type: {self.edit_type}. Must be 'insert', 'replace', or 'delete'")
        if self.priority < 0 or self.priority > 100:
            raise ValueError(f"Priority must be between 0 and 100, got {self.priority}")
        if self.edit_type == "delete" and not self.content:
            # For delete, content might be empty or contain what to delete
            pass
        if not isinstance(self.affected_plugins, list):
            self.affected_plugins = list(self.affected_plugins) if self.affected_plugins else []


@dataclass
class EditResult:
    """Result of applying an edit."""
    success: bool
    original_text: str
    modified_text: str
    applied_edits: List[EditRequest]
    failed_edits: List[EditRequest]
    metrics_impact: Dict[str, Any]  # Changes to word count, sentence count, etc.

    def __post_init__(self):
        """Validate edit result."""
        if not isinstance(self.applied_edits, list):
            self.applied_edits = list(self.applied_edits) if self.applied_edits else []
        if not isinstance(self.failed_edits, list):
            self.failed_edits = list(self.failed_edits) if self.failed_edits else []
        if not isinstance(self.metrics_impact, dict):
            self.metrics_impact = self.metrics_impact or {}


class BasePluginFunction(ABC):
    """Base class for plugin functions."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> FunctionResult:
        """Execute the function with given parameters."""
        pass


class NativeFunction(BasePluginFunction):
    """Base class for native (deterministic) functions."""

    def __init__(self, name: str, description: str, func: Callable):
        super().__init__(name, description)
        self.func = func

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute the native function."""
        try:
            result = self.func(**kwargs)
            return FunctionResult(success=True, value=result)
        except Exception as e:
            logger.error(f"Error in native function {self.name}: {e}")
            return FunctionResult(success=False, value=None, error=str(e))


class SemanticFunction(BasePluginFunction):
    """Base class for semantic (LLM-based) functions."""

    def __init__(self, name: str, description: str, prompt_template: str):
        super().__init__(name, description)
        self.prompt_template = prompt_template

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute the semantic function using the kernel."""
        try:
            # Create semantic function from template
            semantic_func = kernel.create_function_from_prompt(
                prompt_template=self.prompt_template,
                function_name=self.name,
                description=self.description
            )

            # Execute with provided variables
            result = await kernel.invoke_function(semantic_func, variables=kwargs)

            return FunctionResult(
                success=True,
                value=result.value,
                metadata={"tokens_used": getattr(result, 'usage_metadata', None)}
            )
        except Exception as e:
            logger.error(f"Error in semantic function {self.name}: {e}")
            return FunctionResult(success=False, value=None, error=str(e))


class DraftingFunction(BasePluginFunction):
    """Base class for document drafting functions."""

    def __init__(self, name: str, description: str, section_type: str):
        super().__init__(name, description)
        self.section_type = section_type

    def _validate_inputs(self, **kwargs) -> bool:
        """Validate required inputs for drafting functions."""
        required = ["evidence", "posteriors", "case_summary"]
        return all(key in kwargs for key in required)

    def _format_citations(self, evidence: Dict[str, Any]) -> str:
        """Format evidence as citations."""
        citations = []
        for node, state in evidence.items():
            citations.append(f"[{node}:{state}]")
        return " ".join(citations)

    def _extract_key_posteriors(self, posteriors: Dict[str, Any], threshold: float = 0.7) -> Dict[str, Any]:
        """Extract high-confidence posteriors."""
        return {
            node: prob for node, prob in posteriors.items()
            if isinstance(prob, (int, float)) and prob >= threshold
        }


class ValidationFunction(BasePluginFunction):
    """Base class for validation functions."""

    def __init__(self, name: str, description: str, validation_type: str):
        super().__init__(name, description)
        self.validation_type = validation_type

    def _validate_inputs(self, **kwargs) -> bool:
        """Validate required inputs for validation functions."""
        return "document" in kwargs or "text" in kwargs

    def _calculate_score(self, passed_checks: int, total_checks: int) -> float:
        """Calculate validation score."""
        return passed_checks / total_checks if total_checks > 0 else 0.0


class AssemblyFunction(BasePluginFunction):
    """Base class for document assembly functions."""

    def __init__(self, name: str, description: str, assembly_type: str):
        super().__init__(name, description)
        self.assembly_type = assembly_type

    def _validate_inputs(self, **kwargs) -> bool:
        """Validate required inputs for assembly functions."""
        return "sections" in kwargs or "documents" in kwargs


# Export base classes
__all__ = [
    "PluginMetadata",
    "BaseSKPlugin",
    "BasePluginFunction",
    "NativeFunction",
    "SemanticFunction",
    "DraftingFunction",
    "ValidationFunction",
    "AssemblyFunction",
    "FunctionResult",
    "DocumentLocation",
    "EditRequest",
    "EditResult"
]
