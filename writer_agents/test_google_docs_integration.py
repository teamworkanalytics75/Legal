"""Comprehensive test suite for Google Docs integration.

This module tests all components of the Google Docs integration system
including the bridge, formatter, version tracker, and document tracker.
"""

import asyncio
import json
import logging
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the modules to test
try:
    from google_docs_bridge import GoogleDocsBridge, create_google_docs_bridge
    from google_docs_formatter import GoogleDocsFormatter, format_writer_deliverable
    from version_history_tracker import VersionHistoryTracker, create_version_tracker, EditPattern
    from document_tracker import DocumentTracker, create_document_tracker, DocumentRecord
    from tasks import WriterDeliverable, DraftSection, PlanDirective, ReviewFindings
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google Docs modules not available for testing: {e}")
    MODULES_AVAILABLE = False


class TestGoogleDocsBridge(unittest.TestCase):
    """Test Google Docs Bridge functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Google Docs modules not available")

        # Create a temporary credentials file for testing
        self.temp_creds = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        self.temp_creds.write(json.dumps({
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-key\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "test-client-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com"
        }))
        self.temp_creds.close()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_creds.name):
            os.unlink(self.temp_creds.name)

    @patch('google_docs_bridge.build')
    @patch('google_docs_bridge.service_account.Credentials.from_service_account_file')
    def test_bridge_initialization(self, mock_credentials, mock_build):
        """Test bridge initialization."""
        # Mock the credentials and services
        mock_credentials.return_value = Mock()
        mock_build.return_value = Mock()

        bridge = GoogleDocsBridge(self.temp_creds.name)

        self.assertIsNotNone(bridge.docs_service)
        self.assertIsNotNone(bridge.drive_service)
        mock_credentials.assert_called_once()
        mock_build.assert_called()

    @patch('google_docs_bridge.build')
    @patch('google_docs_bridge.service_account.Credentials.from_service_account_file')
    def test_create_document(self, mock_credentials, mock_build):
        """Test document creation."""
        # Mock the services
        mock_credentials.return_value = Mock()
        mock_docs_service = Mock()
        mock_drive_service = Mock()
        mock_build.side_effect = [mock_docs_service, mock_drive_service]

        # Mock document creation response
        mock_docs_service.documents.return_value.create.return_value.execute.return_value = {
            "documentId": "test-doc-id"
        }

        bridge = GoogleDocsBridge(self.temp_creds.name)
        bridge.docs_service = mock_docs_service
        bridge.drive_service = mock_drive_service

        doc_id, doc_url = bridge.create_document("Test Document", "test-folder-id")

        self.assertEqual(doc_id, "test-doc-id")
        self.assertIn("test-doc-id", doc_url)
        mock_docs_service.documents.return_value.create.assert_called_once()

    @patch('google_docs_bridge.build')
    @patch('google_docs_bridge.service_account.Credentials.from_service_account_file')
    def test_fetch_document(self, mock_credentials, mock_build):
        """Test document fetching."""
        # Mock the services
        mock_credentials.return_value = Mock()
        mock_docs_service = Mock()
        mock_drive_service = Mock()
        mock_build.side_effect = [mock_docs_service, mock_drive_service]

        # Mock document fetch response
        mock_docs_service.documents.return_value.get.return_value.execute.return_value = {
            "body": {
                "content": [
                    {
                        "paragraph": {
                            "elements": [
                                {"textRun": {"content": "Test content"}}
                            ]
                        }
                    }
                ]
            }
        }

        mock_drive_service.files.return_value.get.return_value.execute.return_value = {
            "name": "Test Document",
            "createdTime": "2023-01-01T00:00:00Z",
            "modifiedTime": "2023-01-01T00:00:00Z",
            "webViewLink": "https://docs.google.com/document/d/test-doc-id/edit"
        }

        bridge = GoogleDocsBridge(self.temp_creds.name)
        bridge.docs_service = mock_docs_service
        bridge.drive_service = mock_drive_service

        result = bridge.fetch_document("test-doc-id")

        self.assertEqual(result["document_id"], "test-doc-id")
        self.assertEqual(result["title"], "Test Document")
        self.assertEqual(result["content"], "Test content")

    @patch('google_docs_bridge.build')
    @patch('google_docs_bridge.service_account.Credentials.from_service_account_file')
    def test_update_document(self, mock_credentials, mock_build):
        """Test document updating."""
        # Mock the services
        mock_credentials.return_value = Mock()
        mock_docs_service = Mock()
        mock_drive_service = Mock()
        mock_build.side_effect = [mock_docs_service, mock_drive_service]

        # Mock batch update response
        mock_docs_service.documents.return_value.batchUpdate.return_value.execute.return_value = {
            "revisionId": "test-revision-id"
        }

        bridge = GoogleDocsBridge(self.temp_creds.name)
        bridge.docs_service = mock_docs_service
        bridge.drive_service = mock_drive_service

        result = bridge.update_document("test-doc-id", "New content", "New Title")

        self.assertEqual(result["document_id"], "test-doc-id")
        self.assertEqual(result["revision_id"], "test-revision-id")
        mock_docs_service.documents.return_value.batchUpdate.assert_called_once()

    def test_credentials_not_found(self):
        """Test error handling when credentials are not found."""
        with self.assertRaises(ValueError):
            GoogleDocsBridge(None)

    def test_credentials_file_not_found(self):
        """Test error handling when credentials file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            GoogleDocsBridge("nonexistent-file.json")


class TestGoogleDocsFormatter(unittest.TestCase):
    """Test Google Docs Formatter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Google Docs modules not available")

        self.formatter = GoogleDocsFormatter()

        # Create test deliverable
        self.test_deliverable = WriterDeliverable(
            plan=PlanDirective(
                objective="Test legal memorandum",
                deliverable_format="Legal memorandum",
                tone="Professional and analytical",
                style_constraints=["Use proper legal citations", "Maintain formal tone"],
                citation_expectations="Use [Node:State] format"
            ),
            sections=[
                DraftSection(
                    section_id="intro",
                    title="Introduction",
                    body="This is a test introduction section with citations [Evidence1:True]."
                ),
                DraftSection(
                    section_id="analysis",
                    title="Legal Analysis",
                    body="This section contains legal analysis with multiple citations [Evidence1:True] and [Evidence2:False]."
                )
            ],
            edited_document="Test document content",
            reviews=[
                ReviewFindings(
                    section_id="intro",
                    severity="info",
                    message="Section looks good",
                    suggestions="Consider adding more detail"
                )
            ],
            metadata={"workflow_type": "test"}
        )

    def test_formatter_initialization(self):
        """Test formatter initialization."""
        self.assertIsInstance(self.formatter, GoogleDocsFormatter)
        self.assertIn("legal_memo", self.formatter.supported_formats)

    def test_format_deliverable(self):
        """Test deliverable formatting."""
        formatted = self.formatter.format_deliverable(self.test_deliverable)

        self.assertIsInstance(formatted, str)
        self.assertIn("LEGAL MEMORANDUM", formatted)
        self.assertIn("OBJECTIVE: Test legal memorandum", formatted)
        self.assertIn("SECTION 1: Introduction", formatted)
        self.assertIn("SECTION 2: Legal Analysis", formatted)
        self.assertIn("[Evidence1:True]", formatted)
        self.assertIn("REVIEW FINDINGS", formatted)

    def test_format_deliverable_different_types(self):
        """Test formatting with different document types."""
        for format_type in self.formatter.supported_formats:
            formatted = self.formatter.format_deliverable(self.test_deliverable, format_type)
            self.assertIsInstance(formatted, str)
            self.assertGreater(len(formatted), 100)

    def test_format_for_google_docs_api(self):
        """Test Google Docs API format generation."""
        api_format = self.formatter.format_for_google_docs_api(self.test_deliverable)

        self.assertIn("requests", api_format)
        self.assertIn("formatted_text", api_format)
        self.assertIsInstance(api_format["requests"], list)
        self.assertIsInstance(api_format["formatted_text"], str)

    def test_extract_citations(self):
        """Test citation extraction."""
        text = "This has citations [Node1:True] and [Node2:False] and [Node3:Maybe]."
        citations = self.formatter.extract_citations(text)

        self.assertEqual(len(citations), 3)
        self.assertEqual(citations[0]["node"], "Node1")
        self.assertEqual(citations[0]["state"], "True")
        self.assertEqual(citations[1]["node"], "Node2")
        self.assertEqual(citations[1]["state"], "False")

    def test_validate_format(self):
        """Test format validation."""
        validation = self.formatter.validate_format(self.test_deliverable)

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["stats"]["section_count"], 2)
        self.assertEqual(validation["stats"]["review_count"], 1)
        self.assertGreater(validation["stats"]["word_count"], 0)

    def test_validate_format_invalid(self):
        """Test validation with invalid deliverable."""
        invalid_deliverable = WriterDeliverable(
            plan=PlanDirective(objective="", deliverable_format="", tone=""),
            sections=[],
            edited_document=""
        )

        validation = self.formatter.validate_format(invalid_deliverable)

        self.assertFalse(validation["valid"])
        self.assertGreater(len(validation["issues"]), 0)


class TestVersionHistoryTracker(unittest.TestCase):
    """Test Version History Tracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Google Docs modules not available")

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.temp_db.close()

        self.tracker = VersionHistoryTracker(self.temp_db.name)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertIsInstance(self.tracker, VersionHistoryTracker)
        self.assertTrue(os.path.exists(self.temp_db.name))

    def test_store_and_retrieve_edit_pattern(self):
        """Test storing and retrieving edit patterns."""
        pattern = EditPattern(
            pattern_id="test-pattern-1",
            document_id="test-doc-1",
            edit_type="citation_fix",
            before_text="[Node1:False]",
            after_text="[Node1:True]",
            context="Legal analysis section",
            confidence_score=0.9,
            timestamp=datetime.now().isoformat(),
            user="test-user",
            section_id="analysis"
        )

        # Store pattern
        self.tracker._store_edit_pattern(pattern)

        # Retrieve patterns
        patterns = self.tracker.get_edit_patterns(document_id="test-doc-1")

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].pattern_id, "test-pattern-1")
        self.assertEqual(patterns[0].edit_type, "citation_fix")

    def test_generate_statistics(self):
        """Test statistics generation."""
        # Add some test patterns
        for i in range(5):
            pattern = EditPattern(
                pattern_id=f"test-pattern-{i}",
                document_id=f"test-doc-{i % 2}",
                edit_type="citation_fix" if i % 2 == 0 else "tone_adjustment",
                before_text=f"Before text {i}",
                after_text=f"After text {i}",
                context=f"Context {i}",
                confidence_score=0.8,
                timestamp=datetime.now().isoformat(),
                user="test-user"
            )
            self.tracker._store_edit_pattern(pattern)

        stats = self.tracker.generate_statistics()

        self.assertEqual(stats.total_edits, 5)
        self.assertEqual(stats.document_count, 2)
        self.assertIn("citation_fix", stats.edit_types)
        self.assertIn("tone_adjustment", stats.edit_types)

    def test_export_patterns_for_ml(self):
        """Test ML export functionality."""
        # Add test pattern
        pattern = EditPattern(
            pattern_id="test-pattern-1",
            document_id="test-doc-1",
            edit_type="citation_fix",
            before_text="[Node1:False]",
            after_text="[Node1:True]",
            context="Legal analysis",
            confidence_score=0.9,
            timestamp=datetime.now().isoformat(),
            user="test-user"
        )
        self.tracker._store_edit_pattern(pattern)

        # Export to temporary file
        temp_export = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        temp_export.close()

        try:
            result = self.tracker.export_patterns_for_ml(temp_export.name)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["patterns_exported"], 1)

            # Verify exported data
            with open(temp_export.name, 'r') as f:
                exported_data = json.load(f)

            self.assertEqual(len(exported_data), 1)
            self.assertEqual(exported_data[0]["edit_type"], "citation_fix")
            self.assertEqual(exported_data[0]["input_text"], "[Node1:False]")
            self.assertEqual(exported_data[0]["output_text"], "[Node1:True]")

        finally:
            if os.path.exists(temp_export.name):
                os.unlink(temp_export.name)

    def test_classify_edit_type(self):
        """Test edit type classification."""
        # Citation fix
        edit_type = self.tracker._classify_edit_type("[Node1:False]", "[Node1:True]")
        self.assertEqual(edit_type, "citation_fix")

        # Argument improvement
        edit_type = self.tracker._classify_edit_type("Short text", "This is a much longer text with more details")
        self.assertEqual(edit_type, "argument_improvement")

        # Tone adjustment
        edit_type = self.tracker._classify_edit_type("This is gonna be good", "This will be excellent")
        self.assertEqual(edit_type, "tone_adjustment")

        # Formatting
        edit_type = self.tracker._classify_edit_type("  Text with spaces  ", "Text with spaces")
        self.assertEqual(edit_type, "formatting")


class TestDocumentTracker(unittest.TestCase):
    """Test Document Tracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Google Docs modules not available")

        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.temp_db.close()

        self.tracker = DocumentTracker(self.temp_db.name)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        self.assertIsInstance(self.tracker, DocumentTracker)
        self.assertTrue(os.path.exists(self.temp_db.name))

    def test_create_document_record(self):
        """Test creating document records."""
        record = self.tracker.create_document_record(
            case_id="test-case-1",
            google_doc_id="test-doc-1",
            doc_url="https://docs.google.com/document/d/test-doc-1/edit",
            folder_id="test-folder-1",
            title="Test Document",
            case_summary="Test case summary",
            metadata={"test": "value"}
        )

        self.assertIsInstance(record, DocumentRecord)
        self.assertEqual(record.case_id, "test-case-1")
        self.assertEqual(record.google_doc_id, "test-doc-1")
        self.assertEqual(record.status, "active")

    def test_get_doc_for_case(self):
        """Test retrieving document for case."""
        # Create a document record
        self.tracker.create_document_record(
            case_id="test-case-1",
            google_doc_id="test-doc-1",
            doc_url="https://docs.google.com/document/d/test-doc-1/edit",
            folder_id="test-folder-1",
            title="Test Document",
            case_summary="Test case summary"
        )

        # Retrieve document
        doc = self.tracker.get_doc_for_case("test-case-1")

        self.assertIsNotNone(doc)
        self.assertEqual(doc.google_doc_id, "test-doc-1")
        self.assertEqual(doc.case_id, "test-case-1")

    def test_list_all_docs(self):
        """Test listing all documents."""
        # Create multiple document records
        for i in range(3):
            self.tracker.create_document_record(
                case_id=f"test-case-{i}",
                google_doc_id=f"test-doc-{i}",
                doc_url=f"https://docs.google.com/document/d/test-doc-{i}/edit",
                folder_id="test-folder-1",
                title=f"Test Document {i}",
                case_summary=f"Test case summary {i}"
            )

        docs = self.tracker.list_all_docs()

        self.assertEqual(len(docs), 3)
        self.assertEqual(docs[0].google_doc_id, "test-doc-2")  # Most recent first

    def test_update_document(self):
        """Test updating document information."""
        # Create document record
        self.tracker.create_document_record(
            case_id="test-case-1",
            google_doc_id="test-doc-1",
            doc_url="https://docs.google.com/document/d/test-doc-1/edit",
            folder_id="test-folder-1",
            title="Original Title",
            case_summary="Original summary"
        )

        # Update document
        success = self.tracker.update_document(
            "test-doc-1",
            title="Updated Title",
            case_summary="Updated summary",
            metadata={"updated": True}
        )

        self.assertTrue(success)

        # Verify update
        doc = self.tracker.get_doc_for_case("test-case-1")
        self.assertEqual(doc.title, "Updated Title")

    def test_archive_document(self):
        """Test archiving document."""
        # Create document record
        self.tracker.create_document_record(
            case_id="test-case-1",
            google_doc_id="test-doc-1",
            doc_url="https://docs.google.com/document/d/test-doc-1/edit",
            folder_id="test-folder-1",
            title="Test Document",
            case_summary="Test case summary"
        )

        # Archive document
        success = self.tracker.archive_document("test-doc-1", "Test archiving")

        self.assertTrue(success)

        # Verify archive
        doc = self.tracker.get_doc_for_case("test-case-1")
        self.assertEqual(doc.status, "archived")

    def test_generate_statistics(self):
        """Test statistics generation."""
        # Create test documents
        for i in range(5):
            self.tracker.create_document_record(
                case_id=f"test-case-{i}",
                google_doc_id=f"test-doc-{i}",
                doc_url=f"https://docs.google.com/document/d/test-doc-{i}/edit",
                folder_id="test-folder-1",
                title=f"Test Document {i}",
                case_summary=f"Test case summary {i}"
            )

        # Archive one document
        self.tracker.archive_document("test-doc-0")

        stats = self.tracker.generate_statistics()

        self.assertEqual(stats.total_documents, 5)
        self.assertEqual(stats.active_documents, 4)
        self.assertEqual(stats.archived_documents, 1)
        self.assertEqual(stats.total_cases, 5)

    def test_search_documents(self):
        """Test document search functionality."""
        # Create test documents
        self.tracker.create_document_record(
            case_id="test-case-1",
            google_doc_id="test-doc-1",
            doc_url="https://docs.google.com/document/d/test-doc-1/edit",
            folder_id="test-folder-1",
            title="Privacy Analysis Document",
            case_summary="Analysis of privacy violations"
        )

        self.tracker.create_document_record(
            case_id="test-case-2",
            google_doc_id="test-doc-2",
            doc_url="https://docs.google.com/document/d/test-doc-2/edit",
            folder_id="test-folder-1",
            title="Contract Review Document",
            case_summary="Review of contract terms"
        )

        # Search for privacy-related documents
        results = self.tracker.search_documents("privacy")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].google_doc_id, "test-doc-1")


class TestIntegration(unittest.TestCase):
    """Test integration between components."""

    def setUp(self):
        """Set up test fixtures."""
        if not MODULES_AVAILABLE:
            self.skipTest("Google Docs modules not available")

        # Create temporary databases
        self.temp_version_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.temp_version_db.close()

        self.temp_doc_db = tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False)
        self.temp_doc_db.close()

        self.version_tracker = VersionHistoryTracker(self.temp_version_db.name)
        self.document_tracker = DocumentTracker(self.temp_doc_db.name)

    def tearDown(self):
        """Clean up test fixtures."""
        for temp_file in [self.temp_version_db.name, self.temp_doc_db.name]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_end_to_end_workflow(self):
        """Test end-to-end workflow integration."""
        # Create test deliverable
        deliverable = WriterDeliverable(
            plan=PlanDirective(
                objective="Test integration",
                deliverable_format="Legal memorandum",
                tone="Professional",
                citation_expectations="Use [Node:State] format"
            ),
            sections=[
                DraftSection(
                    section_id="test",
                    title="Test Section",
                    body="Test content with citation [TestNode:True]"
                )
            ],
            edited_document="Test document",
            metadata={"workflow_type": "integration_test"}
        )

        # Test formatter
        formatter = GoogleDocsFormatter()
        formatted_content = formatter.format_deliverable(deliverable)

        self.assertIsInstance(formatted_content, str)
        self.assertIn("LEGAL MEMORANDUM", formatted_content)

        # Test document tracker
        doc_record = self.document_tracker.create_document_record(
            case_id="integration-test-case",
            google_doc_id="integration-test-doc",
            doc_url="https://docs.google.com/document/d/integration-test-doc/edit",
            folder_id="test-folder",
            title="Integration Test Document",
            case_summary="Test case for integration"
        )

        self.assertIsInstance(doc_record, DocumentRecord)

        # Test version tracker
        pattern = EditPattern(
            pattern_id="integration-pattern-1",
            document_id="integration-test-doc",
            edit_type="citation_fix",
            before_text="[TestNode:False]",
            after_text="[TestNode:True]",
            context="Integration test",
            confidence_score=0.95,
            timestamp=datetime.now().isoformat(),
            user="integration-test"
        )

        self.version_tracker._store_edit_pattern(pattern)
        patterns = self.version_tracker.get_edit_patterns(document_id="integration-test-doc")

        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].edit_type, "citation_fix")


def run_tests():
    """Run all tests."""
    if not MODULES_AVAILABLE:
        print("Google Docs modules not available. Skipping tests.")
        return

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_classes = [
        TestGoogleDocsBridge,
        TestGoogleDocsFormatter,
        TestVersionHistoryTracker,
        TestDocumentTracker,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
