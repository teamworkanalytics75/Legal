"""Google Docs API Bridge for Writer Agents System.

This module provides integration with Google Docs API for creating, updating,
and managing documents in Google Drive. It supports both service account and OAuth authentication.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.auth.exceptions import DefaultCredentialsError
    import pickle
except ImportError as e:
    raise ImportError(
        "Google API client libraries not installed. "
        "Run: pip install google-api-python-client google-auth google-auth-oauthlib"
    ) from e

logger = logging.getLogger(__name__)

# OAuth 2.0 scopes
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


class GoogleDocsBridge:
    """Bridge for Google Docs API operations with OAuth and Service Account support."""

    def __init__(self, credentials_path: Optional[str] = None, use_oauth: bool = True):
        """Initialize the Google Docs bridge.

        Args:
            credentials_path: Path to credentials JSON file (OAuth or Service Account).
                             If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
            use_oauth: If True, use OAuth 2.0 authentication. If False, use service account.
        """
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.use_oauth = use_oauth

        if not self.credentials_path:
            raise ValueError(
                "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable or provide credentials_path."
            )

        if not Path(self.credentials_path).exists():
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")

        # Initialize services
        self._init_services()

    def _init_services(self):
        """Initialize Google API services with appropriate authentication."""
        try:
            if self.use_oauth:
                self.credentials = self._get_oauth_credentials()
            else:
                self.credentials = self._get_service_account_credentials()

            # Build API services
            self.docs_service = build("docs", "v1", credentials=self.credentials, cache_discovery=False)
            self.drive_service = build("drive", "v3", credentials=self.credentials, cache_discovery=False)

            logger.info("Google Docs Bridge initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Google Docs Bridge: {e}")
            raise

    def _get_oauth_credentials(self):
        """Get OAuth 2.0 credentials."""
        creds = None
        token_file = Path(self.credentials_path).parent / "token.pickle"

        # Load existing token if available
        if token_file.exists():
            with open(token_file, "rb") as token:
                creds = pickle.load(token)

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for next run
            with open(token_file, "wb") as token:
                pickle.dump(creds, token)

        return creds

    def _get_service_account_credentials(self):
        """Get service account credentials."""
        return service_account.Credentials.from_service_account_file(
            self.credentials_path, scopes=SCOPES
        )

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Google APIs."""
        try:
            # Test Drive API access
            about = self.drive_service.about().get(fields="user").execute()
            user_email = about.get("user", {}).get("emailAddress", "Unknown")

            return {
                "status": "success",
                "user_email": user_email,
                "auth_type": "oauth" if self.use_oauth else "service_account"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "auth_type": "oauth" if self.use_oauth else "service_account"
            }

    def create_document(self, title: str, folder_id: Optional[str] = None) -> Tuple[str, str]:
        """Create a new Google Doc.

        Args:
            title: Document title
            folder_id: Google Drive folder ID to create document in

        Returns:
            Tuple of (document_id, document_url)
        """
        try:
            if folder_id:
                # Create document in specific folder using Drive API
                file_metadata = {
                    "name": title,
                    "mimeType": "application/vnd.google-apps.document",
                    "parents": [folder_id]
                }
                file = self.drive_service.files().create(
                    body=file_metadata,
                    fields="id,webViewLink"
                ).execute()
                doc_id = file.get("id")
                doc_url = file.get("webViewLink")
            else:
                # Create document in root using Docs API
                document = {"title": title}
                doc = self.docs_service.documents().create(body=document).execute()
                doc_id = doc.get("documentId")
                doc_url = f"https://docs.google.com/document/d/{doc_id}/edit"

            logger.info(f"Created Google Doc: {doc_id}")
            return doc_id, doc_url

        except HttpError as e:
            logger.error(f"Failed to create Google Doc: {e}")
            raise

    def update_document(self, doc_id: str, content: List[Dict], title: Optional[str] = None) -> None:
        """Update a Google Doc with new content.

        Args:
            doc_id: Google Doc ID
            content: List of content elements (from formatter)
            title: Optional new title
        """
        try:
            # Fetch document to determine true character end index
            doc = self.docs_service.documents().get(documentId=doc_id).execute()
            body = doc.get("body", {})
            content_elements = body.get("content", [])

            # Compute reliable end index from element endIndex values, with body fallback
            end_index = 1
            max_end_index = 1
            for element in content_elements:
                element_end = element.get("endIndex", 1)
                if element_end and element_end > max_end_index:
                    max_end_index = element_end
            end_index = max_end_index
            body_end_index = body.get("endIndex", 1)
            if body_end_index and body_end_index > end_index:
                end_index = body_end_index

            requests: List[Dict[str, Any]] = []

            # Clear existing content using character indices (preserve final structural element)
            if end_index > 1:
                delete_end = end_index - 1
                if delete_end > 1:
                    requests.append({
                        "deleteContentRange": {
                            "range": {
                                "segmentId": "",
                                "startIndex": 1,
                                "endIndex": delete_end
                            }
                        }
                    })

            # Build insert requests (execute after delete) at index 1 in reverse order
            insert_requests: List[Dict[str, Any]] = []
            for element in content:
                if element.get("type") == "paragraph":
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": element.get("text", "") + "\n"
                        }
                    })
                elif element.get("type") == "heading1":
                    text = element.get("text", "")
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": text + "\n"
                        }
                    })
                    insert_requests.append({
                        "updateParagraphStyle": {
                            "range": {"segmentId": "", "startIndex": 1, "endIndex": len(text) + 1},
                            "paragraphStyle": {"namedStyleType": "HEADING_1"},
                            "fields": "*"
                        }
                    })
                elif element.get("type") == "heading2":
                    text = element.get("text", "")
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": text + "\n"
                        }
                    })
                    insert_requests.append({
                        "updateParagraphStyle": {
                            "range": {"segmentId": "", "startIndex": 1, "endIndex": len(text) + 1},
                            "paragraphStyle": {"namedStyleType": "HEADING_2"},
                            "fields": "*"
                        }
                    })

            insert_requests.reverse()
            requests.extend(insert_requests)

            # Execute batch update
            if requests:
                self.docs_service.documents().batchUpdate(
                    documentId=doc_id,
                    body={"requests": requests}
                ).execute()

            # Update title if provided
            if title:
                self.drive_service.files().update(
                    fileId=doc_id,
                    body={"name": title}
                ).execute()

            logger.info(f"Updated Google Doc: {doc_id}")

        except HttpError as e:
            logger.error(f"Failed to update Google Doc {doc_id}: {e}")
            raise

    def fetch_document_content(self, doc_id: str) -> str:
        """Fetch document content as plain text.

        Args:
            doc_id: Google Doc ID

        Returns:
            Document content as plain text
        """
        try:
            doc = self.docs_service.documents().get(documentId=doc_id).execute()
            content = ""

            for element in doc.get("body", {}).get("content", []):
                if "paragraph" in element:
                    for text_run in element["paragraph"]["elements"]:
                        if "textRun" in text_run:
                            content += text_run["textRun"]["content"]

            logger.info(f"Fetched content for Google Doc: {doc_id}")
            return content

        except HttpError as e:
            logger.error(f"Failed to fetch Google Doc content {doc_id}: {e}")
            raise

    def share_document(self, doc_id: str, email_address: str, role: str = "writer") -> None:
        """Share document with specified email address.

        Args:
            doc_id: Google Doc ID
            email_address: Email address to share with
            role: Permission role (reader, writer, owner)
        """
        try:
            permission_body = {
                "type": "user",
                "role": role,
                "emailAddress": email_address
            }

            self.drive_service.permissions().create(
                fileId=doc_id,
                body=permission_body,
                fields="id",
                sendNotificationEmail=False
            ).execute()

            logger.info(f"Shared Google Doc {doc_id} with {email_address} as {role}")

        except HttpError as e:
            logger.error(f"Failed to share Google Doc {doc_id} with {email_address}: {e}")
            raise

    def get_document_revisions(self, doc_id: str) -> List[Dict]:
        """Get document revision history.

        Args:
            doc_id: Google Doc ID

        Returns:
            List of revision information
        """
        try:
            revisions = self.drive_service.revisions().list(
                fileId=doc_id,
                fields="nextPageToken, revisions(id, modifiedTime, lastModifyingUser)"
            ).execute()

            logger.info(f"Fetched {len(revisions.get('revisions', []))} revisions for Google Doc: {doc_id}")
            return revisions.get("revisions", [])

        except HttpError as e:
            logger.error(f"Failed to get revisions for Google Doc {doc_id}: {e}")
            raise


def create_google_docs_bridge(credentials_path: Optional[str] = None, use_oauth: bool = True) -> GoogleDocsBridge:
    """Create a Google Docs bridge instance.

    Args:
        credentials_path: Path to credentials JSON file
        use_oauth: If True, use OAuth 2.0. If False, use service account.

    Returns:
        GoogleDocsBridge instance
    """
    return GoogleDocsBridge(credentials_path, use_oauth)
