"""File system monitoring for document processing."""

import asyncio
import logging
from pathlib import Path
from typing import Callable, List, Set

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None


class DocumentFileHandler(FileSystemEventHandler):
    """Handler for file system events on document directories."""

    def __init__(self, extensions: List[str], callback: Callable[[Path], None]):
        self.extensions = [ext.lower() for ext in extensions]
        self.callback = callback
        self.logger = logging.getLogger("file_monitor")
        self.processed_files: Set[str] = set()

    def on_created(self, event):
        """Called when a file is created."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Check if it's an extension we care about
        if file_path.suffix.lower() not in self.extensions:
            return

        # Avoid processing the same file multiple times
        if str(file_path) in self.processed_files:
            return

        self.logger.info(f"New file detected: {file_path.name}")
        self.processed_files.add(str(file_path))

        # Call the callback
        try:
            self.callback(file_path)
        except Exception as e:
            self.logger.error(f"Error in callback for {file_path}: {e}")

    def on_modified(self, event):
        """Called when a file is modified."""
        # For now, treat modified files the same as created
        # (some systems trigger modified instead of created)
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        if file_path.suffix.lower() not in self.extensions:
            return

        if str(file_path) in self.processed_files:
            return

        self.logger.info(f"Modified file detected: {file_path.name}")
        self.processed_files.add(str(file_path))

        try:
            self.callback(file_path)
        except Exception as e:
            self.logger.error(f"Error in callback for {file_path}: {e}")


class FileMonitor:
    """Monitors directories for new/modified files."""

    def __init__(self):
        if not WATCHDOG_AVAILABLE:
            raise RuntimeError("watchdog library is required. Install with: pip install watchdog")

        self.observer = Observer()
        self.logger = logging.getLogger("file_monitor")
        self.handlers = []

    def watch_directory(
        self,
        directory: Path,
        extensions: List[str],
        callback: Callable[[Path], None],
        recursive: bool = True
    ):
        """
        Start watching a directory for files with specific extensions.

        Args:
            directory: Path to watch
            extensions: List of extensions (e.g., ['.pdf', '.docx'])
            callback: Function to call when a matching file is found
            recursive: Whether to watch subdirectories
        """
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return

        handler = DocumentFileHandler(extensions, callback)
        self.handlers.append(handler)

        self.observer.schedule(handler, str(directory), recursive=recursive)
        self.logger.info(f"Watching directory: {directory} (recursive={recursive})")
        self.logger.info(f"  Extensions: {', '.join(extensions)}")

    def start(self):
        """Start the file monitoring."""
        if not self.handlers:
            self.logger.warning("No directories are being watched")
            return

        self.observer.start()
        self.logger.info("File monitoring started")

    def stop(self):
        """Stop the file monitoring."""
        self.observer.stop()
        self.observer.join()
        self.logger.info("File monitoring stopped")

    def scan_existing_files(
        self,
        directory: Path,
        extensions: List[str],
        callback: Callable[[Path], None],
        recursive: bool = True,
        max_files: int = 10
    ):
        """
        Scan directory for existing files that match criteria.
        Useful for initial processing of files that exist before monitoring starts.

        Args:
            directory: Directory to scan
            extensions: Extensions to look for
            callback: Function to call for each file
            recursive: Whether to scan subdirectories
            max_files: Maximum number of files to process initially
        """
        if not directory.exists():
            self.logger.warning(f"Directory does not exist: {directory}")
            return

        processed_count = 0

        # Use glob pattern for extensions
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"

            for file_path in directory.glob(pattern):
                if not file_path.is_file():
                    continue

                if processed_count >= max_files:
                    self.logger.info(f"Reached max_files limit ({max_files})")
                    return

                try:
                    self.logger.info(f"Processing existing file: {file_path.name}")
                    callback(file_path)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")

        self.logger.info(f"Processed {processed_count} existing files")

