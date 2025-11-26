"""Document monitoring agent - watches directories and processes new documents."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.agent import BackgroundAgent, AgentConfig


class DocumentMonitorAgent(BackgroundAgent):
    """Monitors directories for new documents and extracts information."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.output_dir = Path("background_agents/outputs/document_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def process(self, task: Any) -> Any:
        """
        Process a document file.

        Args:
            task: Dict with 'file_path' key

        Returns:
            Dict with extracted information
        """
        # Validate input
        if not isinstance(task, dict) or 'file_path' not in task:
            self.logger.error(f"Invalid task data: {task}")
            return {'error': 'Task must be dict with file_path key'}

        file_path = Path(task['file_path'])

        if not file_path.exists():
            self.logger.error(f"File does not exist: {file_path}")
            return {'error': f'File not found: {file_path}'}

        self.logger.info(f"Processing document: {file_path.name}")

        # Read file (simplified - in production use proper PDF/DOCX readers)
        try:
            if file_path.suffix.lower() == '.pdf':
                content = await self._extract_pdf_text(file_path)
            elif file_path.suffix.lower() == '.json':
                with open(file_path) as f:
                    content = json.load(f)
                    content = str(content)  # Convert to string for analysis
            else:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return {'error': str(e)}

        # Extract information using LLM
        prompt = f"""Analyze this legal document and extract key information in JSON format:

Document: {content[:3000]}  # First 3000 chars

Extract:
1. Case name (if applicable)
2. Court and jurisdiction
3. Date filed
4. Key parties involved
5. Document type (petition, order, opinion, etc.)
6. Main legal issues
7. Citations to other cases or statutes
8. Outcome or ruling (if applicable)

Return ONLY valid JSON with these fields."""

        try:
            response = await self.llm_query(prompt, temperature=0.3)

            # Try to parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_data = json.loads(json_match.group())
            else:
                extracted_data = {'raw_response': response}

            # Add metadata
            result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'processed_at': datetime.now().isoformat(),
                'extracted_data': extracted_data,
                'content_length': len(content),
            }

            # Save result
            output_file = self.output_dir / f"{file_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            self.logger.info(f"Saved analysis to: {output_file}")

            return result

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {'error': str(e), 'file_path': str(file_path)}

    async def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF. Uses PyMuPDF if available."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            # Fallback: just return filename if PyMuPDF not available
            return f"[PDF file: {pdf_path.name}]"
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return f"[PDF extraction error: {e}]"

