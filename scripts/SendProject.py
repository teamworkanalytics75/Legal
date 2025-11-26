#!/usr/bin/env python3
"""
Sender script for Agents project transfer
Run this on the laptop to send files to gaming PC (192.168.1.2)
"""

import socket
import os
import zlib
import hashlib
import struct
import json
from pathlib import Path
import time
import sys

DEFAULT_TARGET_IP = "192.168.1.2"

if hasattr(sys.stdout, "reconfigure"):
    # Ensure emojis or UTF-8 output do not crash on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class ProjectSender:
    def __init__(self, target_ip=DEFAULT_TARGET_IP, port=9999, source_dir="."):
        self.target_ip = target_ip
        self.port = port
        self.source_dir = Path(source_dir)
        self.socket = None

        # Files/directories to exclude
        self.exclude_patterns = {
            '__pycache__',
            'node_modules',
            '.venv',
            '.git',
            'dist',
            'build',
            '.pytest_cache',
            '*.pyc',
            '*.pyo',
            '*.log',
            '.DS_Store',
            'Thumbs.db',
            '*.tmp',
            '*.temp',
            '*.csv',
            '*.bz2',
            '*.zip',
            '*.tar.gz',
            '*.sqlite',
            '*.db',
            'data/bulk_downloads',
            'Court-Cases-Data-Scrapping',
            'awesome-legal-data.zip',
            'legal-ml-datasets.zip',
            'Indian_SC_Judgment_database.zip'
        }

    def should_exclude(self, path):
        """Check if file/directory should be excluded"""
        path_str = str(path)
        name = path.name
        
        # Check exact matches
        if name in self.exclude_patterns:
            return True
        
        # Check patterns
        for pattern in self.exclude_patterns:
            if pattern.startswith('*') and path_str.endswith(pattern[1:]):
                return True
        
        # Check directory patterns
        for pattern in self.exclude_patterns:
            if '/' in pattern and pattern in path_str:
                return True
        
        # Exclude files larger than 50MB
        try:
            if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
                return True
        except (OSError, FileNotFoundError):
            pass
        
        return False

    def scan_files(self):
        """Scan source directory and return list of files to transfer"""
        files_to_transfer = []

        print("🔍 Scanning project directory...")

        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)

            # Remove excluded directories from dirs list (in-place modification)
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]

            for file in files:
                file_path = root_path / file

                if not self.should_exclude(file_path):
                    # Calculate relative path from source directory
                    rel_path = file_path.relative_to(self.source_dir)
                    files_to_transfer.append(rel_path)

        print(f"📋 Found {len(files_to_transfer)} files to transfer")
        return files_to_transfer

    def calculate_file_info(self, file_path):
        """Calculate file size, checksum, and compression info"""
        full_path = self.source_dir / file_path

        with open(full_path, 'rb') as f:
            file_data = f.read()

        # Calculate checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        # Compress data
        compressed_data = zlib.compress(file_data, level=6)

        return {
            'original_size': len(file_data),
            'compressed_size': len(compressed_data),
            'checksum': checksum,
            'data': compressed_data
        }

    def connect_to_server(self):
        """Connect to the receiver server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            print(f"🔌 Connecting to {self.target_ip}:{self.port}...")
            self.socket.connect((self.target_ip, self.port))
            print("✅ Connected to gaming PC")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False

    def send_file_metadata(self, metadata):
        """Send file metadata to receiver"""
        meta_json = json.dumps(metadata).encode('utf-8')
        meta_len = len(meta_json)

        # Send metadata length
        self.socket.send(struct.pack('!I', meta_len))
        # Send metadata
        self.socket.send(meta_json)

    def send_file(self, file_path, file_info):
        """Send a single file"""
        # Send metadata
        metadata = {
            'path': str(file_path),
            'compressed_size': file_info['compressed_size'],
            'checksum': file_info['checksum'],
            'original_size': file_info['original_size']
        }
        self.send_file_metadata(metadata)

        # Send compressed data
        self.socket.send(file_info['data'])

        # Wait for acknowledgment
        response = self.socket.recv(2)
        if response != b'OK':
            raise ConnectionError("Receiver did not acknowledge file")

    def transfer_files(self):
        """Main transfer loop"""
        try:
            # Scan files
            files_to_transfer = self.scan_files()

            if not files_to_transfer:
                print("❌ No files found to transfer")
                return False

            # Send total file count
            self.socket.send(struct.pack('!I', len(files_to_transfer)))

            transferred_files = 0
            total_bytes = 0
            start_time = time.time()

            for file_path in files_to_transfer:
                print(f"📤 Sending: {file_path}")

                # Calculate file info
                file_info = self.calculate_file_info(file_path)

                # Send file
                self.send_file(file_path, file_info)

                transferred_files += 1
                total_bytes += file_info['original_size']

                # Progress update
                elapsed = time.time() - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0
                progress = (transferred_files / len(files_to_transfer)) * 100

                print(f"✅ {transferred_files}/{len(files_to_transfer)} files ({progress:.1f}%) - "
                      f"{speed/1024/1024:.1f} MB/s")

            print(f"\n🎉 Transfer complete!")
            print(f"📊 Total: {transferred_files} files, {total_bytes/1024/1024:.1f} MB")

        except Exception as e:
            print(f"❌ Transfer error: {e}")
            return False

        return True

    def cleanup(self):
        """Close connection"""
        if self.socket:
            self.socket.close()

def main():
    print("💻 Agents Project Sender")
    print("=" * 50)

    sender = ProjectSender()

    try:
        if sender.connect_to_server():
            sender.transfer_files()
    except KeyboardInterrupt:
        print("\n⏹️  Transfer cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        sender.cleanup()
        print("🔌 Connection closed")

if __name__ == "__main__":
    main()
