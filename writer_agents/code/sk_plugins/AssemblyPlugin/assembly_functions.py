"""Assembly plugin for Semantic Kernel with document stitching, exhibit linking, and filing formatter functions."""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

from ..base_plugin import (
    BaseSKPlugin,
    PluginMetadata,
    AssemblyFunction,
    FunctionResult,
    kernel_function,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentSection:
    """Individual document section."""
    title: str
    content: str
    order: int
    section_type: str
    citations: List[str]
    word_count: int


@dataclass
class Exhibit:
    """Document exhibit."""
    exhibit_id: str
    title: str
    description: str
    file_path: Optional[str] = None
    page_references: List[str] = None


@dataclass
class AssembledDocument:
    """Complete assembled document."""
    title: str
    sections: List[DocumentSection]
    exhibits: List[Exhibit]
    total_word_count: int
    filing_format: str
    metadata: Dict[str, Any]


class DocumentAssemblerFunction(AssemblyFunction):
    """Native function for document assembly."""

    def __init__(self):
        super().__init__(
            name="DocumentAssembler",
            description="Assemble document sections into complete document",
            assembly_type="document_stitching"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute document assembly."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: sections"
            )

        try:
            sections = kwargs["sections"]
            document_title = kwargs.get("document_title", "Legal Document")
            filing_format = kwargs.get("filing_format", "standard")

            # Assemble document
            assembled_doc = self._assemble_document(sections, document_title, filing_format)

            return FunctionResult(
                success=True,
                value=assembled_doc,
                metadata={"assembly_type": "document_stitching", "sections_count": len(sections)}
            )

        except Exception as e:
            logger.error(f"Error in DocumentAssembler: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _assemble_document(self, sections: List[Dict[str, Any]], document_title: str, filing_format: str) -> AssembledDocument:
        """Assemble document from sections."""

        # Convert sections to DocumentSection objects
        doc_sections = []
        for i, section_data in enumerate(sections):
            section = DocumentSection(
                title=section_data.get("title", f"Section {i+1}"),
                content=section_data.get("content", ""),
                order=section_data.get("order", i),
                section_type=section_data.get("section_type", "general"),
                citations=section_data.get("citations", []),
                word_count=len(section_data.get("content", "").split())
            )
            doc_sections.append(section)

        # Sort sections by order
        doc_sections.sort(key=lambda s: s.order)

        # Calculate total word count
        total_word_count = sum(section.word_count for section in doc_sections)

        # Create assembled document
        assembled_doc = AssembledDocument(
            title=document_title,
            sections=doc_sections,
            exhibits=[],  # Will be populated by exhibit linker
            total_word_count=total_word_count,
            filing_format=filing_format,
            metadata={
                "assembly_date": self._get_current_date(),
                "sections_count": len(doc_sections),
                "total_citations": sum(len(s.citations) for s in doc_sections)
            }
        )

        return assembled_doc

    def _get_current_date(self) -> str:
        """Get current date for metadata."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")


class ExhibitLinkerFunction(AssemblyFunction):
    """Native function for exhibit linking."""

    def __init__(self):
        super().__init__(
            name="ExhibitLinker",
            description="Link exhibits to document references",
            assembly_type="exhibit_linking"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute exhibit linking."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document, exhibits"
            )

        try:
            document = kwargs["document"]
            exhibits = kwargs["exhibits"]

            # Link exhibits to document
            linked_document = self._link_exhibits(document, exhibits)

            return FunctionResult(
                success=True,
                value=linked_document,
                metadata={"assembly_type": "exhibit_linking", "exhibits_count": len(exhibits)}
            )

        except Exception as e:
            logger.error(f"Error in ExhibitLinker: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _link_exhibits(self, document: AssembledDocument, exhibits: List[Dict[str, Any]]) -> AssembledDocument:
        """Link exhibits to document references."""

        # Convert exhibits to Exhibit objects
        exhibit_objects = []
        for exhibit_data in exhibits:
            exhibit = Exhibit(
                exhibit_id=exhibit_data.get("id", f"Exhibit_{len(exhibit_objects)+1}"),
                title=exhibit_data.get("title", "Untitled Exhibit"),
                description=exhibit_data.get("description", ""),
                file_path=exhibit_data.get("file_path"),
                page_references=exhibit_data.get("page_references", [])
            )
            exhibit_objects.append(exhibit)

        # Update document with exhibits
        document.exhibits = exhibit_objects

        # Add exhibit references to sections
        for section in document.sections:
            section.content = self._add_exhibit_references(section.content, exhibit_objects)

        # Update metadata
        document.metadata["exhibits_count"] = len(exhibit_objects)
        document.metadata["exhibit_references"] = self._count_exhibit_references(document.sections)

        return document

    def _add_exhibit_references(self, content: str, exhibits: List[Exhibit]) -> str:
        """Add exhibit references to content."""

        # Look for exhibit references in content
        exhibit_pattern = r'\[Exhibit\s+([A-Z]+)\]'
        matches = re.findall(exhibit_pattern, content)

        for exhibit_ref in matches:
            # Find matching exhibit
            matching_exhibit = next((e for e in exhibits if e.exhibit_id == exhibit_ref), None)
            if matching_exhibit:
                # Replace reference with formatted exhibit link
                exhibit_link = f"[Exhibit {exhibit_ref}: {matching_exhibit.title}]"
                content = content.replace(f"[Exhibit {exhibit_ref}]", exhibit_link)

        return content

    def _count_exhibit_references(self, sections: List[DocumentSection]) -> int:
        """Count total exhibit references in document."""
        total_refs = 0
        for section in sections:
            exhibit_pattern = r'\[Exhibit\s+[A-Z]+\]'
            refs = re.findall(exhibit_pattern, section.content)
            total_refs += len(refs)
        return total_refs


class FilingFormatterFunction(AssemblyFunction):
    """Native function for filing format formatting."""

    def __init__(self):
        super().__init__(
            name="FilingFormatter",
            description="Format document for court filing",
            assembly_type="filing_formatting"
        )

    async def execute(self, **kwargs) -> FunctionResult:
        """Execute filing formatting."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: document"
            )

        try:
            document = kwargs["document"]
            court_format = kwargs.get("court_format", "federal")
            jurisdiction = kwargs.get("jurisdiction", "US")

            # Format document for filing
            formatted_doc = self._format_for_filing(document, court_format, jurisdiction)

            return FunctionResult(
                success=True,
                value=formatted_doc,
                metadata={"assembly_type": "filing_formatting", "court_format": court_format}
            )

        except Exception as e:
            logger.error(f"Error in FilingFormatter: {e}")
            return FunctionResult(success=False, value=None, error=str(e))

    def _format_for_filing(self, document: AssembledDocument, court_format: str, jurisdiction: str) -> str:
        """Format document for court filing."""

        # Get formatting rules
        formatting_rules = self._get_formatting_rules(court_format, jurisdiction)

        # Build formatted document
        formatted_lines = []

        # Document header
        formatted_lines.extend(self._format_header(document.title, formatting_rules))

        # Table of contents
        if formatting_rules.get("include_toc", True):
            formatted_lines.extend(self._format_table_of_contents(document.sections))

        # Document sections
        for section in document.sections:
            formatted_lines.extend(self._format_section(section, formatting_rules))

        # Exhibits section
        if document.exhibits:
            formatted_lines.extend(self._format_exhibits_section(document.exhibits))

        # Document footer
        formatted_lines.extend(self._format_footer(document.metadata, formatting_rules))

        return "\n".join(formatted_lines)

    def _get_formatting_rules(self, court_format: str, jurisdiction: str) -> Dict[str, Any]:
        """Get formatting rules for court format."""

        rules = {
            "federal": {
                "font": "Times New Roman",
                "font_size": "12pt",
                "line_spacing": "double",
                "margins": "1 inch",
                "include_toc": True,
                "page_numbers": True,
                "header_format": "Case Name - Document Title"
            },
            "state": {
                "font": "Times New Roman",
                "font_size": "12pt",
                "line_spacing": "double",
                "margins": "1 inch",
                "include_toc": False,
                "page_numbers": True,
                "header_format": "Document Title"
            },
            "appellate": {
                "font": "Times New Roman",
                "font_size": "14pt",
                "line_spacing": "double",
                "margins": "1 inch",
                "include_toc": True,
                "page_numbers": True,
                "header_format": "Appellate Case Name"
            }
        }

        return rules.get(court_format, rules["federal"])

    def _format_header(self, title: str, rules: Dict[str, Any]) -> List[str]:
        """Format document header."""
        header_lines = []

        # Case caption (placeholder)
        header_lines.append("UNITED STATES DISTRICT COURT")
        header_lines.append("FOR THE DISTRICT OF [DISTRICT]")
        header_lines.append("")
        header_lines.append("[CASE CAPTION]")
        header_lines.append("")
        header_lines.append("")

        # Document title
        header_lines.append(f"{title.upper()}")
        header_lines.append("")

        return header_lines

    def _format_table_of_contents(self, sections: List[DocumentSection]) -> List[str]:
        """Format table of contents."""
        toc_lines = []

        toc_lines.append("TABLE OF CONTENTS")
        toc_lines.append("")

        for i, section in enumerate(sections, 1):
            toc_lines.append(f"{i}. {section.title} ................. {i}")

        toc_lines.append("")
        toc_lines.append("")

        return toc_lines

    def _format_section(self, section: DocumentSection, rules: Dict[str, Any]) -> List[str]:
        """Format individual section."""
        section_lines = []

        # Section title
        section_lines.append(f"{section.order}. {section.title.upper()}")
        section_lines.append("")

        # Section content
        content_lines = section.content.split('\n')
        for line in content_lines:
            if line.strip():
                section_lines.append(line)
            else:
                section_lines.append("")

        section_lines.append("")

        return section_lines

    def _format_exhibits_section(self, exhibits: List[Exhibit]) -> List[str]:
        """Format exhibits section."""
        exhibit_lines = []

        exhibit_lines.append("EXHIBITS")
        exhibit_lines.append("")

        for exhibit in exhibits:
            exhibit_lines.append(f"Exhibit {exhibit.exhibit_id}: {exhibit.title}")
            if exhibit.description:
                exhibit_lines.append(f"  {exhibit.description}")
            exhibit_lines.append("")

        return exhibit_lines

    def _format_footer(self, metadata: Dict[str, Any], rules: Dict[str, Any]) -> List[str]:
        """Format document footer."""
        footer_lines = []

        footer_lines.append("")
        footer_lines.append("")
        footer_lines.append("Respectfully submitted,")
        footer_lines.append("")
        footer_lines.append("[ATTORNEY NAME]")
        footer_lines.append("[BAR NUMBER]")
        footer_lines.append("[FIRM NAME]")
        footer_lines.append("[ADDRESS]")
        footer_lines.append("")
        footer_lines.append(f"Dated: {metadata.get('assembly_date', '')}")

        return footer_lines


class DocumentStitchingSemanticFunction(AssemblyFunction):
    """Semantic function for document stitching using LLM."""

    def __init__(self):
        super().__init__(
            name="DocumentStitchingSemantic",
            description="Stitch document sections using LLM with structured prompts",
            assembly_type="document_stitching"
        )
        self.prompt_template = self._get_prompt_template()

    def _get_prompt_template(self) -> str:
        """Get the prompt template for document stitching."""
        return """
You are a legal document assembly expert. Stitch together the following document sections into a cohesive legal document.

DOCUMENT TITLE: {{$document_title}}
COURT FORMAT: {{$court_format}}
JURISDICTION: {{$jurisdiction}}

SECTIONS TO ASSEMBLE:
{{$sections}}

REQUIREMENTS:
1. Maintain logical flow between sections
2. Add appropriate transitions
3. Ensure consistent formatting
4. Include proper legal document structure
5. Preserve all citations and references
6. Follow court formatting requirements

OUTPUT FORMAT:
Create a complete legal document with:
- Proper header and case caption
- Table of contents (if required)
- All sections in logical order
- Proper transitions between sections
- Exhibits section (if applicable)
- Proper footer and signature block

Ensure the document is ready for court filing.
        """

    async def execute(self, kernel: Kernel, **kwargs) -> FunctionResult:
        """Execute semantic document stitching."""
        if not self._validate_inputs(**kwargs):
            return FunctionResult(
                success=False,
                value=None,
                error="Missing required inputs: sections"
            )

        try:
            sections = kwargs["sections"]
            document_title = kwargs.get("document_title", "Legal Document")
            court_format = kwargs.get("court_format", "federal")
            jurisdiction = kwargs.get("jurisdiction", "US")

            # Create semantic function
            semantic_func = kernel.create_function_from_prompt(
                prompt_template=self.prompt_template,
                function_name=self.name,
                description=self.description
            )

            # Execute with context variables
            result = await kernel.invoke_function(
                semantic_func,
                variables={
                    "document_title": document_title,
                    "court_format": court_format,
                    "jurisdiction": jurisdiction,
                    "sections": json.dumps(sections, indent=2)
                }
            )

            return FunctionResult(
                success=True,
                value=result.value,
                metadata={
                    "assembly_type": "document_stitching",
                    "method": "semantic",
                    "tokens_used": getattr(result, 'usage_metadata', None)
                }
            )

        except Exception as e:
            logger.error(f"Error in DocumentStitchingSemantic: {e}")
            return FunctionResult(success=False, value=None, error=str(e))


class AssemblyPlugin(BaseSKPlugin):
    """Plugin for document assembly functions."""

    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.document_assembler = DocumentAssemblerFunction()
        self.exhibit_linker = ExhibitLinkerFunction()
        self.filing_formatter = FilingFormatterFunction()
        self.document_stitching_semantic = DocumentStitchingSemanticFunction()

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="AssemblyPlugin",
            description="Plugin for document assembly, exhibit linking, and filing formatting",
            version="1.0.0",
            functions=[
                "AssembleDocument",
                "LinkExhibits",
                "FormatForFiling",
                "DocumentStitchingSemantic"
            ]
        )

    async def _register_functions(self) -> None:
        """Register assembly functions with the kernel."""

        # Register document assembler
        @kernel_function(
            name="AssembleDocument",
            description="Assemble document sections into complete document"
        )
        async def assemble_document(
            sections: str,
            document_title: str = "Legal Document",
            filing_format: str = "standard"
        ) -> str:
            """Assemble document from sections."""
            result = await self.document_assembler.execute(
                sections=json.loads(sections),
                document_title=document_title,
                filing_format=filing_format
            )

            if result.success:
                doc = result.value
                return f"""
# {doc.title}

## Document Information
- Sections: {len(doc.sections)}
- Total Word Count: {doc.total_word_count}
- Filing Format: {doc.filing_format}

## Sections
{chr(10).join(f"### {section.title} (Order: {section.order})" for section in doc.sections)}

## Metadata
{json.dumps(doc.metadata, indent=2)}
                """.strip()
            else:
                raise RuntimeError(f"Document assembly failed: {result.error}")

        # Register exhibit linker
        @kernel_function(
            name="LinkExhibits",
            description="Link exhibits to document references"
        )
        async def link_exhibits(
            document: str,
            exhibits: str
        ) -> str:
            """Link exhibits to document."""
            result = await self.exhibit_linker.execute(
                document=json.loads(document),
                exhibits=json.loads(exhibits)
            )

            if result.success:
                doc = result.value
                return f"""
# Document with Linked Exhibits

## Exhibits Linked: {len(doc.exhibits)}
{chr(10).join(f"- {exhibit.exhibit_id}: {exhibit.title}" for exhibit in doc.exhibits)}

## Exhibit References: {doc.metadata.get('exhibit_references', 0)}
                """.strip()
            else:
                raise RuntimeError(f"Exhibit linking failed: {result.error}")

        # Register filing formatter
        @kernel_function(
            name="FormatForFiling",
            description="Format document for court filing"
        )
        async def format_for_filing(
            document: str,
            court_format: str = "federal",
            jurisdiction: str = "US"
        ) -> str:
            """Format document for filing."""
            result = await self.filing_formatter.execute(
                document=json.loads(document),
                court_format=court_format,
                jurisdiction=jurisdiction
            )

            if result.success:
                return result.value
            else:
                raise RuntimeError(f"Filing formatting failed: {result.error}")

        # Register semantic document stitching
        @kernel_function(
            name="DocumentStitchingSemantic",
            description="Stitch document sections using LLM with structured prompts"
        )
        async def document_stitching_semantic(
            sections: str,
            document_title: str = "Legal Document",
            court_format: str = "federal",
            jurisdiction: str = "US"
        ) -> str:
            """Semantic document stitching."""
            result = await self.document_stitching_semantic.execute(
                kernel=self.kernel,
                sections=json.loads(sections),
                document_title=document_title,
                court_format=court_format,
                jurisdiction=jurisdiction
            )

            if result.success:
                return result.value
            else:
                raise RuntimeError(f"Semantic document stitching failed: {result.error}")

        # Store function references
        self._functions["AssembleDocument"] = assemble_document
        self._functions["LinkExhibits"] = link_exhibits
        self._functions["FormatForFiling"] = format_for_filing
        self._functions["DocumentStitchingSemantic"] = document_stitching_semantic


# Export classes
__all__ = [
    "AssemblyPlugin",
    "DocumentAssemblerFunction",
    "ExhibitLinkerFunction",
    "FilingFormatterFunction",
    "DocumentStitchingSemanticFunction",
    "AssembledDocument",
    "DocumentSection",
    "Exhibit"
]
