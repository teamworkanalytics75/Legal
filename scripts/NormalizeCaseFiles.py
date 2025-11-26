#!/usr/bin/env python3
"""
Normalize Case Files - Add Missing Hash Metadata

This script processes all JSON files in the 1782_discovery folder and adds
the missing _matrix_case_hash and _matrix_primary_identifier fields to files
that don't have them, using the same logic as the downloader.
"""

import sys
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import CourtListener client directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "download_case_law",
    Path(__file__).parent.parent / "document_ingestion" / "download_case_law.py"
)
cl_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl_module)
CourtListenerClient = cl_module.CourtListenerClient

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CaseNormalizer:
    """Normalizes case files by adding missing hash metadata."""

    def __init__(self):
        self.client = CourtListenerClient()
        self.processed_count = 0
        self.updated_count = 0
        self.skipped_count = 0

    def _extract_case_name(self, opinion: Dict[str, Any]) -> str:
        """Extract case name from opinion metadata."""
        for key in (
            'case_name',
            'caseName',
            'caseNameFull',
            'case_name_full',
            'caption',
            'absolute_url'
        ):
            value = opinion.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "unnamed_case"

    def _compute_case_hash(self, opinion: Dict[str, Any]) -> Optional[str]:
        """Compute stable hash for an opinion using identifying metadata."""
        components: List[str] = []

        case_name = self._extract_case_name(opinion)
        if case_name:
            components.append(case_name.lower())

        # Citations may be a list or string
        citation = opinion.get('citation') or opinion.get('citations')
        if citation:
            if isinstance(citation, list):
                citation_repr = "|".join(str(item) for item in citation if item)
            else:
                citation_repr = str(citation)
            if citation_repr:
                components.append(citation_repr.lower())

        docket_number = opinion.get('docket_number') or opinion.get('docketNumber')
        if docket_number:
            components.append(str(docket_number).lower())

        court_id = opinion.get('court_id') or opinion.get('courtId')
        if court_id:
            components.append(str(court_id).lower())

        date_filed = opinion.get('date_filed') or opinion.get('dateFiled')
        if date_filed:
            components.append(str(date_filed).lower())

        if not components:
            return None

        # Create hash from components
        content = "|".join(components)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _determine_primary_identifier(self, opinion: Dict[str, Any], case_hash: Optional[str]) -> str:
        """Determine primary identifier for an opinion."""
        # Try to use existing ID fields first
        for key in ('id', 'opinion_id', 'cluster_id'):
            value = opinion.get(key)
            if value:
                return f"cluster:{value}"

        # Fall back to hash-based identifier
        if case_hash:
            return f"hash:{case_hash[:16]}"

        # Last resort - use case name
        case_name = self._extract_case_name(opinion)
        return f"name:{hashlib.md5(case_name.encode('utf-8')).hexdigest()[:16]}"

    def normalize_file(self, file_path: Path) -> bool:
        """Normalize a single JSON file by adding missing metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Skip if already has hash metadata
            if data.get('_matrix_case_hash') and data.get('_matrix_primary_identifier'):
                self.skipped_count += 1
                return False

            # Compute missing metadata
            case_hash = self._compute_case_hash(data)
            primary_identifier = self._determine_primary_identifier(data, case_hash)

            # Add metadata
            data['_matrix_case_hash'] = case_hash
            data['_matrix_primary_identifier'] = primary_identifier
            data['_matrix_normalized'] = True

            # Save updated file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.updated_count += 1
            logger.info(f"✅ Updated: {file_path.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return False

    def normalize_all_files(self):
        """Normalize all JSON files in the 1782_discovery folder."""
        logger.info("="*80)
        logger.info("NORMALIZING CASE FILES - ADDING MISSING HASH METADATA")
        logger.info("="*80)

        base_dir = Path("data/case_law/1782_discovery")

        # Find all JSON files
        json_files = list(base_dir.rglob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")

        # Process each file
        for file_path in json_files:
            self.processed_count += 1

            # Skip analysis/report files
            if any(skip in file_path.name.lower() for skip in ['report', 'analysis', 'index', 'backup']):
                logger.info(f"⏭️  Skipping analysis file: {file_path.name}")
                self.skipped_count += 1
                continue

            logger.info(f"Processing {self.processed_count}/{len(json_files)}: {file_path.name}")
            self.normalize_file(file_path)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("NORMALIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Files processed: {self.processed_count}")
        logger.info(f"Files updated: {self.updated_count}")
        logger.info(f"Files skipped: {self.skipped_count}")

        if self.updated_count > 0:
            logger.info("\n✅ Files have been normalized with hash metadata")
            logger.info("You can now run the checkpoint rebuild to get accurate counts")
        else:
            logger.info("\nℹ️  No files needed normalization")


if __name__ == "__main__":
    normalizer = CaseNormalizer()
    normalizer.normalize_all_files()
