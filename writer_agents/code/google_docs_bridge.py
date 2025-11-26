"""Google Docs API Bridge for Writer Agents System.

This module provides integration with Google Docs API for creating, updating,
and managing documents in Google Drive using OAuth 2.0 authentication.
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
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    import pickle
except ImportError as e:
    raise ImportError(
        "Google API client libraries not installed. "
        "Run: pip install google-api-python-client google-auth-oauthlib"
    ) from e

logger = logging.getLogger(__name__)

# OAuth 2.0 scopes
SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive",
]


class GoogleDocsBridge:
    """Bridge for Google Docs API operations with OAuth authentication."""

    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize the Google Docs bridge.

        Args:
            credentials_path: Path to OAuth credentials JSON file.
                             If None, uses GOOGLE_APPLICATION_CREDENTIALS env var.
        """
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

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
        """Initialize Google API services with OAuth authentication."""
        try:
            self.credentials = self._get_oauth_credentials()

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

    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to Google APIs."""
        try:
            # Test Drive API access
            about = self.drive_service.about().get(fields="user").execute()
            user_email = about.get("user", {}).get("emailAddress", "Unknown")

            return {
                "status": "success",
                "user_email": user_email,
                "auth_type": "oauth"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "auth_type": "oauth"
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
            # First, get the document to find its actual length
            doc = self.docs_service.documents().get(documentId=doc_id).execute()

            body = doc.get("body", {})
            content_elements = body.get("content", [])
            
            # Calculate the actual document end index from content elements
            # body.endIndex might not always be reliable, so we calculate from elements
            end_index = 1
            max_end_index = 1
            if content_elements:
                # Find the maximum endIndex from all content elements
                # This is the most reliable way to get document length
                for element in content_elements:
                    element_end = element.get("endIndex", 1)
                    if element_end > max_end_index:
                        max_end_index = element_end
                
                # Use max_end_index as primary source (most reliable)
                end_index = max_end_index
                
                # Also check body.endIndex as fallback, but prefer max_end_index
                body_end_index = body.get("endIndex", 1)
                if body_end_index > max_end_index:
                    end_index = body_end_index
                    logger.debug(f"Using body.endIndex ({body_end_index}) over max element endIndex ({max_end_index})")
            
            # Log what we found
            if end_index > 1:
                logger.debug(f"Document endIndex: {end_index} (from {len(content_elements)} elements, max element endIndex: {max_end_index})")

            requests = []

            # Clear existing content if document has content
            # Use max_end_index (from elements) as the authoritative source
            # end_index of 1 means empty document (just the structural element)
            if end_index > 1:
                # Delete from index 1 (after the structural element) to end_index - 1
                # This preserves the final structural element but removes all content
                # Note: end_index - 1 because we can't delete the final newline character
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
                    logger.info(f"Clearing document content from index 1 to {delete_end} (document endIndex: {end_index}, content elements: {len(content_elements)})")
                else:
                    logger.debug(f"Skipping delete - document appears empty (end_index: {end_index})")
            elif len(content_elements) > 1:
                # Fallback: if end_index is 1 but we have content elements, 
                # try to find the actual endIndex from the last element
                logger.warning(f"end_index is 1 but document has {len(content_elements)} elements - attempting to find actual endIndex")
                for element in reversed(content_elements):
                    if "endIndex" in element:
                        actual_end = element.get("endIndex", 1)
                        if actual_end > 1:
                            delete_end = actual_end - 1
                            requests.append({
                                "deleteContentRange": {
                                    "range": {
                                        "segmentId": "",
                                        "startIndex": 1,
                                        "endIndex": delete_end
                                    }
                                }
                            })
                            logger.info(f"Clearing document using fallback method: index 1 to {delete_end} (found endIndex: {actual_end})")
                            break

            # Insert new content AFTER deleting old content
            # Build insert requests (these will execute after delete)
            insert_requests = []
            for element in content:
                if element["type"] == "paragraph":
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": element["text"] + "\n"
                        }
                    })
                elif element["type"] == "heading1":
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": element["text"] + "\n"
                        }
                    })
                    insert_requests.append({
                        "updateParagraphStyle": {
                            "range": {
                                "segmentId": "",
                                "startIndex": 1,
                                "endIndex": len(element["text"]) + 1
                            },
                            "paragraphStyle": {"namedStyleType": "HEADING_1"},
                            "fields": "*"
                        }
                    })
                elif element["type"] == "heading2":
                    insert_requests.append({
                        "insertText": {
                            "location": {"index": 1},
                            "text": element["text"] + "\n"
                        }
                    })
                    insert_requests.append({
                        "updateParagraphStyle": {
                            "range": {
                                "segmentId": "",
                                "startIndex": 1,
                                "endIndex": len(element["text"]) + 1
                            },
                            "paragraphStyle": {"namedStyleType": "HEADING_2"},
                            "fields": "*"
                        }
                    })

            # Add insert requests to the main requests list
            # Reverse insert requests so they execute in correct order (last element first)
            insert_requests.reverse()
            requests.extend(insert_requests)

            # Execute batch update
            if requests:
                logger.info(f"Executing {len(requests)} batch update requests (delete + {len(insert_requests)} inserts)")
                try:
                    response = self.docs_service.documents().batchUpdate(
                        documentId=doc_id,
                        body={"requests": requests}
                    ).execute()
                    logger.info(f"Batch update completed successfully. Response: {response.get('replies', [])}")
                except Exception as e:
                    logger.error(f"Batch update failed: {e}")
                    # Log the requests for debugging
                    logger.error(f"Failed requests: {requests[:3]}...")  # First 3 for debugging
                    raise
            else:
                logger.warning("No update requests to execute - document may not have been cleared")

            # Update title if provided
            if title:
                self.drive_service.files().update(
                    fileId=doc_id,
                    body={"name": title}
                ).execute()

            logger.info(f"Updated Google Doc: {doc_id} with {len(content)} content elements")

        except HttpError as e:
            logger.error(f"Failed to update Google Doc {doc_id}: {e}")
            logger.error(f"HTTP Error details: {e.content if hasattr(e, 'content') else 'No details'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating Google Doc {doc_id}: {e}")
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

    def add_comment(self, doc_id: str, start_index: int, end_index: int, content: str, quoted_content: Optional[str] = None) -> Dict[str, Any]:
        """Add a comment to a document at a specific range.

        Args:
            doc_id: Google Doc ID
            start_index: Start index of the text to comment on (1-based)
            end_index: End index of the text to comment on (1-based)
            content: Comment text content
            quoted_content: Optional quoted content to highlight (defaults to text at range)

        Returns:
            Created comment information
        """
        try:
            request_body = {
                "requests": [{
                    "createComment": {
                        "comment": {
                            "content": [
                                {
                                    "text": content
                                }
                            ]
                        },
                        "location": {
                            "index": start_index,
                            "endIndex": end_index
                        }
                    }
                }]
            }

            if quoted_content:
                request_body["requests"][0]["createComment"]["quotedFileContent"] = {
                    "mimeType": "text/plain",
                    "value": quoted_content
                }

            response = self.docs_service.documents().batchUpdate(
                documentId=doc_id,
                body=request_body
            ).execute()

            comment_id = response.get("replies", [{}])[0].get("createComment", {}).get("comment", {}).get("commentId", "")
            logger.info(f"Added comment to Google Doc {doc_id} at range {start_index}-{end_index}")
            return {
                "comment_id": comment_id,
                "success": True
            }

        except HttpError as e:
            logger.error(f"Failed to add comment to Google Doc {doc_id}: {e}")
            raise

    def find_text_in_document(self, doc_id: str, search_text: str, case_sensitive: bool = False) -> List[Dict[str, int]]:
        """Find all occurrences of text in a document and return their indices.

        Args:
            doc_id: Google Doc ID
            search_text: Text to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of dictionaries with 'start' and 'end' indices for each match
        """
        try:
            doc = self.docs_service.documents().get(documentId=doc_id).execute()

            # Get full document text with indices
            full_text = ""
            index_map = []  # Maps character position to document index

            for element in doc.get("body", {}).get("content", []):
                if "paragraph" in element:
                    for text_run in element["paragraph"].get("elements", []):
                        if "textRun" in text_run:
                            text_content = text_run["textRun"].get("content", "")
                            start_index = text_run.get("startIndex", 0)

                            for i, char in enumerate(text_content):
                                full_text += char
                                index_map.append(start_index + i)

            # Search in full text
            matches = []
            search_lower = search_text.lower() if not case_sensitive else search_text
            full_text_search = full_text.lower() if not case_sensitive else full_text

            start = 0
            while True:
                pos = full_text_search.find(search_lower, start)
                if pos == -1:
                    break

                end_pos = pos + len(search_text)
                if pos < len(index_map) and end_pos <= len(index_map):
                    matches.append({
                        "start": index_map[pos],
                        "end": index_map[end_pos - 1] + 1,  # endIndex is exclusive
                        "text": full_text[pos:end_pos]
                    })

                start = pos + 1

            logger.info(f"Found {len(matches)} occurrences of '{search_text}' in document {doc_id}")
            return matches

        except HttpError as e:
            logger.error(f"Failed to search document {doc_id}: {e}")
            raise


def create_google_docs_bridge(credentials_path: Optional[str] = None) -> GoogleDocsBridge:
    """Create a Google Docs bridge instance.

    Args:
        credentials_path: Path to credentials JSON file

    Returns:
        GoogleDocsBridge instance
    """
    return GoogleDocsBridge(credentials_path)
