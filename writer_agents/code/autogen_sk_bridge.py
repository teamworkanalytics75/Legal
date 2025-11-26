"""Bridge between AutoGen and Semantic Kernel for hybrid orchestration."""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from semantic_kernel import Kernel

logger = logging.getLogger(__name__)


@dataclass
class SKFunctionDescriptor:
    """Descriptor for SK function to be used as AutoGen tool."""

    name: str
    description: str
    plugin_name: str
    function_name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]


class AutoGenSKBridge:
    """Bridge for integrating SK functions as AutoGen tools."""

    def __init__(self, sk_kernel: Kernel):
        self.sk_kernel = sk_kernel
        self.registered_functions: Dict[str, SKFunctionDescriptor] = {}

    def register_sk_function_as_tool(self, descriptor: SKFunctionDescriptor) -> None:
        """Register an SK function as an AutoGen tool."""
        self.registered_functions[descriptor.name] = descriptor
        logger.info(f"Registered SK function {descriptor.name} as AutoGen tool")

    def get_autogen_tool_function(self, function_name: str) -> Callable:
        """Get AutoGen tool function for SK function."""

        if function_name not in self.registered_functions:
            raise ValueError(f"Function {function_name} not registered")

        descriptor = self.registered_functions[function_name]

        async def autogen_tool_function(**kwargs) -> str:
            """AutoGen tool function that calls SK function."""
            try:
                result = await self.sk_kernel.invoke_function(
                    plugin_name=descriptor.plugin_name,
                    function_name=descriptor.function_name,
                    variables=kwargs
                )

                return str(result.value)

            except Exception as e:
                logger.error(f"Error calling SK function {function_name}: {e}")
                return f"Error: {e}"

        return autogen_tool_function

    def get_all_tool_descriptors(self) -> List[Dict[str, Any]]:
        """Get all registered tool descriptors for AutoGen."""

        descriptors = []
        for name, descriptor in self.registered_functions.items():
            descriptors.append({
                "name": name,
                "description": descriptor.description,
                "parameters": descriptor.input_schema
            })

        return descriptors


class SKToolRegistry:
    """Registry for SK functions as AutoGen tools."""

    def __init__(self, sk_kernel: Kernel):
        self.bridge = AutoGenSKBridge(sk_kernel)
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register default SK tools for AutoGen."""

        # Privacy Harm Drafting Tool
        privacy_harm_tool = SKFunctionDescriptor(
            name="draft_privacy_harm",
            description="Draft privacy harm analysis section using SK semantic function",
            plugin_name="PrivacyHarmPlugin",
            function_name="PrivacyHarmSemantic",
            input_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Evidence dictionary in JSON format"
                    },
                    "posteriors": {
                        "type": "string",
                        "description": "Bayesian network posteriors in JSON format"
                    },
                    "case_summary": {
                        "type": "string",
                        "description": "Case summary text"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Legal jurisdiction (US, EU, CA, etc.)",
                        "default": "US"
                    }
                },
                "required": ["evidence", "posteriors", "case_summary"]
            },
            output_schema={
                "type": "string",
                "description": "Formatted privacy harm analysis section"
            }
        )

        # Privacy Harm Native Tool
        privacy_harm_native_tool = SKFunctionDescriptor(
            name="draft_privacy_harm_native",
            description="Draft privacy harm analysis using SK native function with templates",
            plugin_name="PrivacyHarmPlugin",
            function_name="PrivacyHarmNative",
            input_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Evidence dictionary in JSON format"
                    },
                    "posteriors": {
                        "type": "string",
                        "description": "Bayesian network posteriors in JSON format"
                    },
                    "case_summary": {
                        "type": "string",
                        "description": "Case summary text"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Legal jurisdiction (US, EU, CA, etc.)",
                        "default": "US"
                    }
                },
                "required": ["evidence", "posteriors", "case_summary"]
            },
            output_schema={
                "type": "string",
                "description": "Formatted privacy harm analysis section with structured output"
            }
        )

        # Factual Timeline Tool
        factual_timeline_tool = SKFunctionDescriptor(
            name="draft_factual_timeline",
            description="Draft factual timeline section using SK functions",
            plugin_name="FactualTimelinePlugin",
            function_name="FactualTimelineSemantic",
            input_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Evidence dictionary in JSON format"
                    },
                    "posteriors": {
                        "type": "string",
                        "description": "Bayesian network posteriors in JSON format"
                    },
                    "case_summary": {
                        "type": "string",
                        "description": "Case summary text"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Legal jurisdiction (US, EU, CA, etc.)",
                        "default": "US"
                    }
                },
                "required": ["evidence", "posteriors", "case_summary"]
            },
            output_schema={
                "type": "string",
                "description": "Formatted factual timeline section"
            }
        )

        # Causation Analysis Tool
        causation_analysis_tool = SKFunctionDescriptor(
            name="draft_causation_analysis",
            description="Draft causation analysis section using SK functions",
            plugin_name="CausationAnalysisPlugin",
            function_name="CausationAnalysisSemantic",
            input_schema={
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "string",
                        "description": "Evidence dictionary in JSON format"
                    },
                    "posteriors": {
                        "type": "string",
                        "description": "Bayesian network posteriors in JSON format"
                    },
                    "case_summary": {
                        "type": "string",
                        "description": "Case summary text"
                    },
                    "jurisdiction": {
                        "type": "string",
                        "description": "Legal jurisdiction (US, EU, CA, etc.)",
                        "default": "US"
                    }
                },
                "required": ["evidence", "posteriors", "case_summary"]
            },
            output_schema={
                "type": "string",
                "description": "Formatted causation analysis section"
            }
        )

        # Citation Validation Tool
        citation_validation_tool = SKFunctionDescriptor(
            name="validate_citations",
            description="Validate citation format and completeness",
            plugin_name="ValidationPlugin",
            function_name="ValidateCitationFormat",
            input_schema={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Document text to validate"
                    },
                    "required_format": {
                        "type": "string",
                        "description": "Required citation format",
                        "default": "[Node:State]"
                    }
                },
                "required": ["document"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "details": {"type": "string"},
                    "suggestions": {"type": "array", "items": {"type": "string"}}
                }
            }
        )

        # Structure Validation Tool
        structure_validation_tool = SKFunctionDescriptor(
            name="validate_structure",
            description="Validate document structure and completeness",
            plugin_name="ValidationPlugin",
            function_name="ValidateStructure",
            input_schema={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Document text to validate"
                    },
                    "required_sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Required section names",
                        "default": ["Introduction", "Analysis", "Conclusion"]
                    }
                },
                "required": ["document"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "details": {"type": "string"},
                    "suggestions": {"type": "array", "items": {"type": "string"}}
                }
            }
        )

        # Register all tools
        self.bridge.register_sk_function_as_tool(privacy_harm_tool)
        self.bridge.register_sk_function_as_tool(privacy_harm_native_tool)
        self.bridge.register_sk_function_as_tool(factual_timeline_tool)
        self.bridge.register_sk_function_as_tool(causation_analysis_tool)
        self.bridge.register_sk_function_as_tool(citation_validation_tool)
        self.bridge.register_sk_function_as_tool(structure_validation_tool)

        logger.info(f"Registered {len(self.bridge.registered_functions)} SK tools for AutoGen")

    def get_autogen_function_map(self) -> Dict[str, Callable]:
        """Get function map for AutoGen agent registration."""

        function_map = {}
        for name in self.bridge.registered_functions.keys():
            function_map[name] = self.bridge.get_autogen_tool_function(name)

        return function_map

    def get_tool_descriptors(self) -> List[Dict[str, Any]]:
        """Get tool descriptors for AutoGen agent configuration."""
        return self.bridge.get_all_tool_descriptors()


class AutoGenAgentSKIntegration:
    """Integration helper for adding SK tools to AutoGen agents."""

    def __init__(self, sk_kernel: Kernel):
        self.tool_registry = SKToolRegistry(sk_kernel)

    def add_sk_tools_to_agent(self, autogen_agent) -> None:
        """Add SK tools to an AutoGen agent."""

        function_map = self.tool_registry.get_autogen_function_map()

        # Register functions with AutoGen agent
        for name, func in function_map.items():
            autogen_agent.register_function(func, name=name)

        logger.info(f"Added {len(function_map)} SK tools to AutoGen agent")

    def create_agent_with_sk_tools(self, agent_class, **kwargs) -> Any:
        """Create AutoGen agent with SK tools pre-registered."""

        # Create agent
        agent = agent_class(**kwargs)

        # Add SK tools
        self.add_sk_tools_to_agent(agent)

        return agent

    def get_sk_tool_usage_examples(self) -> Dict[str, str]:
        """Get usage examples for SK tools."""

        return {
            "draft_privacy_harm": """
# Draft privacy harm section using SK semantic function
result = await agent.run(
    task="Draft privacy harm analysis for this case",
    evidence='{"OGC_Email": "Sent", "PRC_Awareness": "Direct"}',
    posteriors='{"PrivacyHarm": 0.85, "Causation": 0.78}',
    case_summary="Case involving privacy violations..."
)
            """,

            "draft_factual_timeline": """
# Draft factual timeline section using SK functions
timeline = await agent.run(
    task="Create factual timeline for this case",
    evidence='{"OGC_Email_Apr18_2025": "Sent", "PRC_Awareness": "Direct"}',
    posteriors='{"TimelineEvent": 0.85, "Causation": 0.78}',
    case_summary="Case involving timeline of events..."
)
            """,

            "draft_causation_analysis": """
# Draft causation analysis section using SK functions
causation = await agent.run(
    task="Analyze causation for this case",
    evidence='{"OGC_Email": "Sent", "PrivacyViolation": "Confirmed"}',
    posteriors='{"Causation": 0.85, "LegalSuccess": 0.72}',
    case_summary="Case involving causation analysis..."
)
            """,

            "validate_citations": """
# Validate citations in a document
validation = await agent.run(
    task="Validate citations in this document",
    document="The privacy harm [OGC_Email:Sent] demonstrates..."
)
            """,

            "validate_structure": """
# Validate document structure
structure_check = await agent.run(
    task="Check if document has required sections",
    document="# Introduction\n## Analysis\n## Conclusion"
)
            """
        }


# Convenience functions for common integration patterns
def create_sk_enabled_autogen_agent(agent_class, sk_kernel: Kernel, **kwargs) -> Any:
    """Create AutoGen agent with SK tools enabled."""
    integration = AutoGenAgentSKIntegration(sk_kernel)
    return integration.create_agent_with_sk_tools(agent_class, **kwargs)


def get_sk_tool_function_map(sk_kernel: Kernel) -> Dict[str, Callable]:
    """Get SK tool function map for manual AutoGen integration."""
    tool_registry = SKToolRegistry(sk_kernel)
    return tool_registry.get_autogen_function_map()


# Export main classes and functions
__all__ = [
    "AutoGenSKBridge",
    "SKToolRegistry",
    "AutoGenAgentSKIntegration",
    "SKFunctionDescriptor",
    "create_sk_enabled_autogen_agent",
    "get_sk_tool_function_map"
]
