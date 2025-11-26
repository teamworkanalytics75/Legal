"""Google Docs Formatter for Writer Agents System.

This module formats WriterDeliverable objects into Google Docs API format.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional

from tasks import WriterDeliverable, DraftSection, ReviewFindings

logger = logging.getLogger(__name__)


class GoogleDocsFormatter:
    """Formats WriterDeliverable objects for Google Docs API."""

    def __init__(self):
        """Initialize the formatter."""
        self.supported_formats = [
            "legal_memo",
            "motion",
            "brief",
            "analysis",
            "report"
        ]

    def format_deliverable(self, deliverable: WriterDeliverable, format_type: str = "legal_memo", validation_results: Optional[Dict[str, Any]] = None, outline_manager: Optional[Any] = None, detected_sections: Optional[Dict[str, int]] = None) -> List[Dict[str, str]]:
        """Format a WriterDeliverable for Google Docs API.

        ✅ NOW INTEGRATED WITH PERFECT OUTLINE STRUCTURE:
        - Reorders sections according to perfect outline order
        - Ensures critical transitions are maintained
        - Formats sections with proper hierarchy

        Args:
            deliverable: WriterDeliverable to format
            format_type: Type of document format
            validation_results: Optional validation results to include in document
            outline_manager: Optional OutlineManager for section reordering
            detected_sections: Optional dict of detected sections and positions

        Returns:
            List of formatted content elements for Google Docs API
        """
        content = []

        # Auto-select motion format for motion documents
        if (deliverable.metadata.get("document_type") or "").lower() == "motion":
            format_type = "motion"

        # Add document header
        content.extend(self._format_header(deliverable, format_type))

        # Add objective section
        # Suppress the objective banner for motions
        if format_type != "motion":
            content.extend(self._format_objective(deliverable))

        # If edited_document exists, use it directly (it's already formatted)
        if deliverable.edited_document:
            # Clean the document text first
            cleaned_text = self._clean_document_text(deliverable.edited_document, max_length=50000)
            # Split into paragraphs and add as content
            paragraphs = cleaned_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Check if it's a heading
                    if para.strip().startswith('#') or re.match(r'^[IVX]+\.\s+[A-Z]', para.strip()):
                        content.append({
                            "type": "heading2",
                            "text": para.strip().lstrip('#').strip()
                        })
                    else:
                        content.append({
                            "type": "paragraph",
                            "text": para.strip()
                        })
        else:
            # ✅ Reorder sections according to perfect outline structure if available
            sections_to_format = deliverable.sections
            if outline_manager and detected_sections:
                sections_to_format = self._reorder_sections_by_outline(
                    deliverable.sections,
                    outline_manager,
                    detected_sections
                )
                if sections_to_format != deliverable.sections:
                    logger.info(f"📋 Reordered {len(sections_to_format)} sections according to perfect outline structure")

            # Add sections (now in perfect outline order)
            for section in sections_to_format:
                content.extend(self._format_section(section))

        # Add reviews if available
        if deliverable.reviews:
            content.extend(self._format_reviews(deliverable.reviews))

        # Add metadata (with validation results if provided)
        # Suppress the metadata banner for motions
        if format_type != "motion":
            if not validation_results:
                validation_results = deliverable.metadata.get("validation_results")
            if not validation_results:
                validation_results = getattr(deliverable, 'validation_results', None)

            content.extend(self._format_metadata(deliverable, validation_results))

        return content

    def _clean_document_text(self, text: str, max_length: int = 50000) -> str:
        """Clean document text by removing code blocks, metadata, and limiting size.
        
        Args:
            text: Raw document text
            max_length: Maximum length to keep (default 50k chars, ~10-15 pages)
            
        Returns:
            Cleaned document text
        """
        if not text:
            return ""
        
        # Remove code blocks (```...```)
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove JSON metadata blocks
        text = re.sub(r'```json[\s\S]*?```', '', text)
        
        # Remove Python code blocks
        text = re.sub(r'```python[\s\S]*?```', '', text)
        
        # Remove lines that look like code (def, class, import, from)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip code-like lines
            if any(stripped.startswith(prefix) for prefix in ['def ', 'class ', 'import ', 'from ', '@', 'async def ']):
                continue
            # Skip lines that are just variable assignments with underscores (likely code)
            if re.match(r'^[a-z_]+_[a-z_]*\s*[:=]', stripped):
                continue
            # Skip lines with excessive underscores (likely code identifiers)
            if stripped.count('_') > 3 and not any(c.isupper() for c in stripped):
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Limit length
        if len(text) > max_length:
            # Try to cut at a paragraph boundary
            cut_point = text[:max_length].rfind('\n\n')
            if cut_point > max_length * 0.8:  # Only cut at paragraph if we keep at least 80%
                text = text[:cut_point] + "\n\n[Document truncated for length]"
            else:
                text = text[:max_length] + "\n\n[Document truncated for length]"
        
        return text.strip()

    def _format_header(self, deliverable: WriterDeliverable, format_type: str) -> List[Dict[str, str]]:
        """Format document header."""
        content = []

        # Document title
        title = self._get_document_title(deliverable, format_type)
        content.append({
            "type": "heading1",
            "text": title
        })

        # For motions, do not include the generic banner
        if format_type != "motion":
            # Document type
            content.append({
                "type": "paragraph",
                "text": f"Document Type: {format_type.upper().replace('_', ' ')}"
            })

            # Timestamp
            timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            content.append({
                "type": "paragraph",
                "text": f"Generated: {timestamp}"
            })

            # Separator
            content.append({
                "type": "paragraph",
                "text": "=" * 80
            })

        return content

    def _format_objective(self, deliverable: WriterDeliverable) -> List[Dict[str, str]]:
        """Format objective section."""
        content = []

        content.append({
            "type": "heading2",
            "text": "OBJECTIVE"
        })

        content.append({
            "type": "paragraph",
            "text": deliverable.plan.objective
        })

        if deliverable.plan.style_constraints:
            content.append({
                "type": "paragraph",
                "text": f"Style Constraints: {', '.join(deliverable.plan.style_constraints)}"
            })

        if deliverable.plan.citation_expectations:
            content.append({
                "type": "paragraph",
                "text": f"Citation Format: {deliverable.plan.citation_expectations}"
            })

        content.append({
            "type": "paragraph",
            "text": ""  # Empty line
        })

        return content

    def _format_section(self, section: DraftSection) -> List[Dict[str, str]]:
        """Format a draft section."""
        content = []

        # Section title
        content.append({
            "type": "heading2",
            "text": f"SECTION {section.section_id.upper()}: {section.title}"
        })

        section_body = section.body or ""
        section_body = self._clean_document_text(section_body, max_length=20000)

        # Section body
        content.append({
            "type": "paragraph",
            "text": section_body
        })

        # Empty line after section
        content.append({
            "type": "paragraph",
            "text": ""
        })

        return content

    def _format_reviews(self, reviews: List[ReviewFindings]) -> List[Dict[str, str]]:
        """Format review findings."""
        content = []

        content.append({
            "type": "heading2",
            "text": "REVIEW FINDINGS"
        })

        for review in reviews:
            severity_emoji = {
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
                "success": "✅"
            }.get(review.severity, "📝")

            content.append({
                "type": "paragraph",
                "text": f"{severity_emoji} [{review.severity.upper()}] {review.message}"
            })

            if review.suggestions:
                content.append({
                    "type": "paragraph",
                    "text": f"   Suggestion: {review.suggestions}"
                })

        content.append({
            "type": "paragraph",
            "text": ""
        })

        return content

    def _format_metadata(self, deliverable: WriterDeliverable, validation_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
        """Format metadata section with constraint validation results."""
        content = []

        content.append({
            "type": "heading2",
            "text": "DOCUMENT METADATA"
        })

        if deliverable.metadata:
            for key, value in deliverable.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    content.append({
                        "type": "paragraph",
                        "text": f"{key}: {value}"
                    })
                elif isinstance(value, dict):
                    # Serialize with proper handling of numpy types
                    content.append({
                        "type": "paragraph",
                        "text": f"{key}: {json.dumps(value, indent=2, default=str)}"
                    })

        # Add constraint validation results if available
        if validation_results:
            content.extend(self._format_validation_results(validation_results))

        return content

    def _format_validation_results(self, validation_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Format constraint validation results for display in Google Docs."""
        content = []

        content.append({
            "type": "heading2",
            "text": "CONSTRAINT VALIDATION RESULTS"
        })

        # Overall score
        overall_score = validation_results.get("overall_score", 0.0)
        passed = validation_results.get("passed", False)
        status_emoji = "✅" if passed else "⚠️"
        status_text = "PASSED" if passed else "NEEDS IMPROVEMENT"

        content.append({
            "type": "paragraph",
            "text": f"{status_emoji} Overall Validation Score: {overall_score:.2%} ({status_text})"
        })

        # Hierarchical scores if available
        if "scores" in validation_results:
            scores = validation_results["scores"]
            content.append({
                "type": "paragraph",
                "text": "\n📊 Hierarchical Scores:"
            })
            if "document_level" in scores:
                content.append({
                    "type": "paragraph",
                    "text": f"  • Document Level: {scores['document_level']:.2%}"
                })
            if "section_level" in scores:
                content.append({
                    "type": "paragraph",
                    "text": f"  • Section Level: {scores['section_level']:.2%}"
                })
            if "feature_level" in scores:
                content.append({
                    "type": "paragraph",
                    "text": f"  • Feature Level: {scores['feature_level']:.2%}"
                })

        # Constraint version if available
        if "constraint_version" in validation_results:
            content.append({
                "type": "paragraph",
                "text": f"Constraint System Version: {validation_results['constraint_version']}"
            })

        # Errors if any
        errors = validation_results.get("errors", [])
        if errors:
            content.append({
                "type": "paragraph",
                "text": "\n❌ Errors:"
            })
            for error in errors[:10]:  # Limit to first 10 errors
                error_text = error if isinstance(error, str) else error.get("message", str(error))
                content.append({
                    "type": "paragraph",
                    "text": f"  • {error_text}"
                })
            if len(errors) > 10:
                content.append({
                    "type": "paragraph",
                    "text": f"  ... and {len(errors) - 10} more errors"
                })

        # Warnings if any
        warnings = validation_results.get("warnings", [])
        if warnings:
            content.append({
                "type": "paragraph",
                "text": "\n⚠️ Warnings:"
            })
            for warning in warnings[:10]:  # Limit to first 10 warnings
                warning_text = warning if isinstance(warning, str) else warning.get("message", str(warning))
                content.append({
                    "type": "paragraph",
                    "text": f"  • {warning_text}"
                })
            if len(warnings) > 10:
                content.append({
                    "type": "paragraph",
                    "text": f"  ... and {len(warnings) - 10} more warnings"
                })

        # Key constraint details
        details = validation_results.get("details", [])
        if details:
            # Show top constraints (passing and failing)
            high_importance = [d for d in details if d.get("importance", 0) >= 5.0]
            if high_importance:
                content.append({
                    "type": "paragraph",
                    "text": "\n🔑 High-Importance Constraints:"
                })
                for detail in high_importance[:5]:  # Top 5
                    feature = detail.get("feature", "Unknown")
                    status = detail.get("status", "unknown")
                    actual = detail.get("actual_value", "N/A")
                    ideal = detail.get("ideal_value", "N/A")
                    status_icon = "✅" if status == "pass" else "❌"
                    content.append({
                        "type": "paragraph",
                        "text": f"  {status_icon} {feature}: {actual} (ideal: {ideal})"
                    })

        # Quality gate results if available
        gate_results = validation_results.get("gate_results", {})
        if gate_results:
            content.append({
                "type": "paragraph",
                "text": "\n🚪 Quality Gate Results:"
            })
            for gate_name, gate_data in list(gate_results.items())[:10]:  # Top 10 gates
                gate_score = gate_data.get("score", 0.0)
                gate_passed = gate_data.get("passed", False)
                gate_icon = "✅" if gate_passed else "❌"
                gate_desc = gate_data.get("details", "")
                desc_text = f" - {gate_desc}" if gate_desc else ""
                content.append({
                    "type": "paragraph",
                    "text": f"  {gate_icon} {gate_name}: {gate_score:.2%}{desc_text}"
                })

        content.append({
            "type": "paragraph",
            "text": ""  # Empty line
        })

        return content

    def _get_document_title(self, deliverable: WriterDeliverable, format_type: str) -> str:
        """Get document title based on deliverable and format type."""
        if format_type == "legal_memo":
            return "LEGAL MEMORANDUM"
        elif format_type == "motion":
            # Prefer explicit title from deliverable metadata
            title = deliverable.metadata.get("title")
            if title:
                return title
            return "Motion to Seal and Proceed Under Pseudonym"
        elif format_type == "brief":
            return "LEGAL BRIEF"
        elif format_type == "analysis":
            return "LEGAL ANALYSIS"
        elif format_type == "report":
            return "LEGAL REPORT"
        else:
            return "LEGAL DOCUMENT"

    def format_for_google_docs_api(self, deliverable: WriterDeliverable, format_type: str = "legal_memo", validation_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format deliverable for Google Docs API with requests and formatted text.

        Args:
            deliverable: WriterDeliverable to format
            format_type: Type of document format
            validation_results: Optional validation results to include

        Returns:
            Dictionary with 'requests' and 'formatted_text' keys
        """
        formatted_content = self.format_deliverable(deliverable, format_type, validation_results=validation_results)

        # Create formatted text version
        formatted_text = ""
        for element in formatted_content:
            if element["type"] == "heading1":
                formatted_text += f"\n{element['text']}\n"
                formatted_text += "=" * len(element['text']) + "\n"
            elif element["type"] == "heading2":
                formatted_text += f"\n{element['text']}\n"
                formatted_text += "-" * len(element['text']) + "\n"
            else:
                formatted_text += f"{element['text']}\n"

        return {
            "requests": formatted_content,
            "formatted_text": formatted_text.strip()
        }

    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text in [Node:State] format.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation dictionaries
        """
        import re

        citations = []
        pattern = r'\[([^:]+):([^\]]+)\]'

        for match in re.finditer(pattern, text):
            citations.append({
                "node": match.group(1),
                "state": match.group(2),
                "full_match": match.group(0),
                "start": match.start(),
                "end": match.end()
            })

        return citations

    def validate_format(self, deliverable: WriterDeliverable) -> Dict[str, Any]:
        """Validate the format of a deliverable.

        Args:
            deliverable: WriterDeliverable to validate

        Returns:
            Validation results dictionary
        """
        issues = []

        # Check required fields
        if not deliverable.plan.objective:
            issues.append("Missing objective")

        if not deliverable.sections:
            issues.append("No sections provided")

        # Check sections
        for i, section in enumerate(deliverable.sections):
            if not section.title:
                issues.append(f"Section {i+1} missing title")
            if not section.body:
                issues.append(f"Section {i+1} missing body")

        # Calculate statistics
        total_words = len(deliverable.edited_document.split()) if deliverable.edited_document else 0
        section_count = len(deliverable.sections)
        review_count = len(deliverable.reviews) if deliverable.reviews else 0

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "word_count": total_words,
                "section_count": section_count,
                "review_count": review_count
            }
        }

    def export_for_ml(self, deliverable: WriterDeliverable) -> Dict[str, Any]:
        """Export deliverable data for machine learning training.

        Args:
            deliverable: WriterDeliverable to export

        Returns:
            Dictionary with ML training data
        """
        return {
            "objective": deliverable.plan.objective,
            "format": deliverable.plan.deliverable_format,
            "tone": deliverable.plan.tone,
            "sections": [
                {
                    "id": section.section_id,
                    "title": section.title,
                    "body": section.body
                }
                for section in deliverable.sections
            ],
            "reviews": [
                {
                    "severity": review.severity,
                    "message": review.message,
                    "suggestions": review.suggestions
                }
                for review in deliverable.reviews
            ] if deliverable.reviews else [],
            "metadata": deliverable.metadata,
            "timestamp": datetime.now().isoformat()
        }

    def _reorder_sections_by_outline(self, sections: List[DraftSection], outline_manager: Any, detected_sections: Dict[str, int]) -> List[DraftSection]:
        """
        Reorder sections according to perfect outline structure.

        Args:
            sections: List of DraftSection objects
            outline_manager: OutlineManager instance
            detected_sections: Dict mapping section names to positions

        Returns:
            Reordered list of sections following perfect outline order
        """
        if not outline_manager or not detected_sections:
            return sections

        # Get perfect outline order
        perfect_order = outline_manager.get_section_order()

        # Create mapping: section_id -> DraftSection
        section_map = {section.section_id.lower(): section for section in sections}

        # Create mapping: detected section name -> DraftSection
        # Try to match detected sections to deliverable sections
        reordered = []
        used_section_ids = set()  # Use section IDs instead of objects (hashable)

        # First, try to match by perfect outline order
        for outline_section_name in perfect_order:
            # Try multiple matching strategies
            matched_section = None

            # Strategy 1: Exact match by section_id
            if outline_section_name in section_map:
                matched_section = section_map[outline_section_name]

            # Strategy 2: Match by detected section name
            elif outline_section_name in detected_sections:
                # Try to find section by title or section_id containing keywords
                for section in sections:
                    if section.section_id not in used_section_ids:  # Check ID instead of object
                        section_id_lower = section.section_id.lower()
                        title_lower = section.title.lower()
                        outline_name_lower = outline_section_name.lower()

                        # Check if section matches outline section
                        if (outline_name_lower in section_id_lower or
                            outline_name_lower in title_lower or
                            any(word in section_id_lower for word in outline_name_lower.split('_'))):
                            matched_section = section
                            break

            # Strategy 3: Match by section title keywords
            if not matched_section:
                outline_keywords = outline_section_name.replace('_', ' ').split()
                for section in sections:
                    if section.section_id not in used_section_ids:  # Check ID instead of object
                        title_lower = section.title.lower()
                        if any(keyword in title_lower for keyword in outline_keywords):
                            matched_section = section
                            break

            if matched_section and matched_section.section_id not in used_section_ids:
                reordered.append(matched_section)
                used_section_ids.add(matched_section.section_id)  # Add ID to set

        # Add any remaining sections that weren't matched
        for section in sections:
            if section.section_id not in used_section_ids:
                reordered.append(section)

        return reordered if reordered else sections


def format_writer_deliverable(deliverable: WriterDeliverable, format_type: str = "legal_memo", validation_results: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """Convenience function to format a WriterDeliverable."""
    formatter = GoogleDocsFormatter()
    return formatter.format_deliverable(deliverable, format_type, validation_results=validation_results)
