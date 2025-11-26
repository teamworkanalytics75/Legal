"""Version management for master drafts with ML backup support."""

import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentVersion:
    """Represents a document version backup."""

    version_id: str
    document_id: str
    title: str
    content: str
    content_hash: str
    timestamp: str
    metadata: Dict[str, Any]
    ml_backup_path: Optional[str] = None


class VersionManager:
    """Manages document versions with ML backup support."""

    def __init__(
        self,
        backup_directory: Path = Path("outputs/master_drafts/versions"),
        ml_training_directory: Path = Path("outputs/ml_training_data/drafts"),
        max_versions_to_keep: int = 50
    ):
        """Initialize version manager.

        Args:
            backup_directory: Where to store version backups
            ml_training_directory: Where to store ML training data
            max_versions_to_keep: Maximum number of versions to keep
        """
        self.backup_directory = Path(backup_directory)
        self.ml_training_directory = Path(ml_training_directory)
        self.max_versions_to_keep = max_versions_to_keep

        # Create directories
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        self.ml_training_directory.mkdir(parents=True, exist_ok=True)

        # Version index file
        self.index_file = self.backup_directory / "version_index.json"
        self.versions = self._load_index()

    def _load_index(self) -> Dict[str, list]:
        """Load version index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load version index: {e}")
        return {}

    def _save_index(self):
        """Save version index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save version index: {e}")

    def _calculate_hash(self, content: str) -> str:
        """Calculate content hash for deduplication."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    def create_backup(
        self,
        document_id: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        save_for_ml: bool = True
    ) -> DocumentVersion:
        """Create a version backup before updating master draft.

        Args:
            document_id: Google Doc ID
            title: Document title
            content: Full document content
            metadata: Additional metadata
            save_for_ml: Whether to save as ML training data

        Returns:
            DocumentVersion object
        """
        # Calculate hash
        content_hash = self._calculate_hash(content)

        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{document_id}_{timestamp}"

        # Check if this is a duplicate
        doc_versions = self.versions.get(document_id, [])
        if doc_versions and doc_versions[-1].get("content_hash") == content_hash:
            logger.info(f"Skipping backup - content unchanged (hash: {content_hash})")
            return None

        # Create version object
        version = DocumentVersion(
            version_id=version_id,
            document_id=document_id,
            title=title,
            content=content,
            content_hash=content_hash,
            timestamp=timestamp,
            metadata=metadata or {}
        )

        # Save backup file
        backup_file = self.backup_directory / f"{version_id}.md"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Version ID:** {version_id}\n")
                f.write(f"**Timestamp:** {datetime.now().isoformat()}\n")
                f.write(f"**Content Hash:** {content_hash}\n\n")
                if metadata:
                    f.write(f"**Metadata:**\n```json\n{json.dumps(metadata, indent=2)}\n```\n\n")
                f.write("---\n\n")
                f.write(content)
            version.ml_backup_path = str(backup_file)
            logger.info(f"Saved backup: {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save backup file: {e}")

        # Save to ML training directory if requested
        if save_for_ml:
            ml_file = self.ml_training_directory / f"{version_id}.json"
            try:
                ml_data = {
                    "version_id": version_id,
                    "document_id": document_id,
                    "title": title,
                    "content": content,
                    "content_hash": content_hash,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                    "backup_path": str(backup_file)
                }
                with open(ml_file, 'w', encoding='utf-8') as f:
                    json.dump(ml_data, f, indent=2)
                logger.info(f"Saved ML training data: {ml_file}")
            except Exception as e:
                logger.error(f"Failed to save ML training data: {e}")

        # Add to index
        if document_id not in self.versions:
            self.versions[document_id] = []

        version_dict = {
            "version_id": version_id,
            "timestamp": timestamp,
            "content_hash": content_hash,
            "backup_path": str(backup_file),
            "ml_path": str(ml_file) if save_for_ml else None
        }

        self.versions[document_id].append(version_dict)

        # Prune old versions
        self._prune_versions(document_id)

        # Save index
        self._save_index()

        return version

    def _prune_versions(self, document_id: str):
        """Remove old versions if exceeding max_versions_to_keep."""
        if document_id not in self.versions:
            return

        versions = self.versions[document_id]
        if len(versions) <= self.max_versions_to_keep:
            return

        # Keep most recent N versions
        versions_to_keep = versions[-self.max_versions_to_keep:]
        versions_to_delete = versions[:-self.max_versions_to_keep]

        # Delete old backup files
        for version in versions_to_delete:
            backup_path = Path(version.get("backup_path", ""))
            if backup_path.exists():
                try:
                    backup_path.unlink()
                    logger.debug(f"Deleted old backup: {backup_path}")
                except Exception as e:
                    logger.warning(f"Could not delete old backup {backup_path}: {e}")

            ml_path = version.get("ml_path")
            if ml_path and Path(ml_path).exists():
                try:
                    Path(ml_path).unlink()
                    logger.debug(f"Deleted old ML data: {ml_path}")
                except Exception as e:
                    logger.warning(f"Could not delete old ML data {ml_path}: {e}")

        # Update index
        self.versions[document_id] = versions_to_keep

    def get_version_history(self, document_id: str) -> list:
        """Get version history for a document."""
        return self.versions.get(document_id, [])

    def get_latest_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent version of a document."""
        versions = self.versions.get(document_id, [])
        return versions[-1] if versions else None

    def get_version_count(self, document_id: str) -> int:
        """Get number of versions for a document."""
        return len(self.versions.get(document_id, []))

