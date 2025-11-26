"""Master Draft Manager for Google Docs Integration.

Centralizes master draft operations with fallback chain:
1. Database lookup (DocumentMetadataRecorder)
2. Google Drive API lookup (GoogleDocsBridge)
3. Create new document
"""

import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MasterDraftInfo:
    """Information about a master draft document."""
    doc_id: str
    doc_url: str
    exists: bool = True
    source: str = "unknown"  # "database", "drive", "created"


class MasterDraftManager:
    """Manages master draft documents with fallback lookup chain."""

    def __init__(
        self,
        document_tracker=None,
        google_docs_bridge=None
    ):
        """Initialize the Master Draft Manager.

        Args:
            document_tracker: DocumentMetadataRecorder instance
            google_docs_bridge: GoogleDocsBridge instance
        """
        self.document_tracker = document_tracker
        self.google_docs_bridge = google_docs_bridge

    def get_or_create_master_doc(
        self,
        title: str,
        folder_id: Optional[str] = None
    ) -> MasterDraftInfo:
        """Get existing master draft or create new one.

        Fallback chain:
        1. Check database (DocumentMetadataRecorder)
        2. Check Google Drive API (GoogleDocsBridge)
        3. Create new document

        Args:
            title: Master draft title
            folder_id: Optional Google Drive folder ID

        Returns:
            MasterDraftInfo with doc_id, doc_url, and source
        """
        # Step 1: Try database lookup
        if self.document_tracker:
            try:
                doc_record = self.document_tracker.get_doc_by_title(title, folder_id)
                if doc_record:
                    logger.info(f"Found master draft in database: {doc_record.google_doc_id}")
                    return MasterDraftInfo(
                        doc_id=doc_record.google_doc_id,
                        doc_url=doc_record.doc_url,
                        exists=True,
                        source="database"
                    )
            except Exception as e:
                logger.warning(f"Database lookup failed: {e}")

        # Step 2: Try Google Drive API lookup
        if self.google_docs_bridge:
            try:
                result = self.google_docs_bridge.find_document_by_title(title, folder_id)
                if result:
                    doc_id, doc_url = result
                    logger.info(f"Found master draft in Google Drive: {doc_id}")

                    # Store in database for future lookups
                    if self.document_tracker:
                        self.document_tracker.set_master_doc_id(title, doc_id, folder_id)

                    return MasterDraftInfo(
                        doc_id=doc_id,
                        doc_url=doc_url,
                        exists=True,
                        source="drive"
                    )
            except Exception as e:
                logger.warning(f"Google Drive lookup failed: {e}")

        # Step 3: Document not found - will need to be created
        logger.info(f"Master draft not found, will create new: {title}")
        return MasterDraftInfo(
            doc_id="",
            doc_url="",
            exists=False,
            source="not_found"
        )

    def get_master_doc_id(
        self,
        title: str,
        folder_id: Optional[str] = None
    ) -> Optional[str]:
        """Get master document ID if it exists.

        Args:
            title: Master draft title
            folder_id: Optional folder ID

        Returns:
            Google Doc ID if found, None otherwise
        """
        master_info = self.get_or_create_master_doc(title, folder_id)
        if master_info.exists:
            return master_info.doc_id
        return None

    def store_master_doc_id(
        self,
        title: str,
        doc_id: str,
        doc_url: str,
        folder_id: Optional[str] = None
    ) -> bool:
        """Store master document ID in database.

        Args:
            title: Master draft title
            doc_id: Google Doc ID
            doc_url: Google Doc URL
            folder_id: Optional folder ID

        Returns:
            True if successful, False otherwise
        """
        if not self.document_tracker:
            logger.warning("Document tracker not available, cannot store master doc ID")
            return False

        try:
            # Use the set_master_doc_id method
            success = self.document_tracker.set_master_doc_id(title, doc_id, folder_id)
            if success:
                logger.info(f"Stored master doc ID: {doc_id} for title: {title}")
            return success
        except Exception as e:
            logger.error(f"Failed to store master doc ID: {e}")
            return False

    def ensure_master_doc_record(
        self,
        title: str,
        doc_id: str,
        doc_url: str,
        folder_id: Optional[str] = None,
        case_id: Optional[str] = None,
        case_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Ensure master document record exists in database.

        Creates or updates the document record with master draft metadata.

        Args:
            title: Master draft title
            doc_id: Google Doc ID
            doc_url: Google Doc URL
            folder_id: Optional folder ID
            case_id: Optional case ID
            case_summary: Optional case summary
            metadata: Optional additional metadata

        Returns:
            True if successful, False otherwise
        """
        if not self.document_tracker:
            logger.warning("Document tracker not available, cannot ensure master doc record")
            return False

        try:
            # Check if record exists
            existing_doc = self.document_tracker.get_doc_by_title(title, folder_id)

            if existing_doc:
                # Update existing record
                update_metadata = {
                    **(existing_doc.metadata or {}),
                    "is_master_draft": True,
                    "master_draft_title": title,
                    **(metadata or {})
                }
                self.document_tracker.update_document(
                    doc_id,
                    metadata=update_metadata
                )
                logger.info(f"Updated master draft record: {doc_id}")
            else:
                # Create new record
                self.document_tracker.create_document_record(
                    case_id=case_id or f"master_{title.lower().replace(' ', '_')}",
                    google_doc_id=doc_id,
                    doc_url=doc_url,
                    folder_id=folder_id or "",
                    title=title,
                    case_summary=case_summary or "Master draft document",
                    metadata={
                        "is_master_draft": True,
                        "master_draft_title": title,
                        **(metadata or {})
                    }
                )
                logger.info(f"Created master draft record: {doc_id}")

            return True
        except Exception as e:
            logger.error(f"Failed to ensure master doc record: {e}")
            return False

