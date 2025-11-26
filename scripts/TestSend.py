#!/usr/bin/env python3
"""
Simple test sender - sends just a few small files to test ethernet connection
"""

import socket
import os
import zlib
import hashlib
import struct
import json
from pathlib import Path
import time

class TestSender:
    def __init__(self, target_ip="192.168.1.2", port=9999):
        self.target_ip = target_ip
        self.port = port
        self.socket = None

        # Just send a few small important files
        self.test_files = [
            "README.md",
            "config/openai_config.json",
            "config/agent_prompts.json",
            "analyze_1782_pdfs.py",
            "cli.py"
        ]

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

    def send_file(self, file_path):
        """Send a single file"""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
        except FileNotFoundError:
            print(f"⚠️  File not found: {file_path}")
            return False

        # Calculate checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        # Compress data
        compressed_data = zlib.compress(file_data, level=6)

        # Send metadata
        metadata = {
            'path': str(file_path),
            'compressed_size': len(compressed_data),
            'checksum': checksum,
            'original_size': len(file_data)
        }

        meta_json = json.dumps(metadata).encode('utf-8')
        meta_len = len(meta_json)

        # Send metadata length
        self.socket.send(struct.pack('!I', meta_len))
        # Send metadata
        self.socket.send(meta_json)

        # Send compressed data
        self.socket.send(compressed_data)

        # Wait for acknowledgment
        response = self.socket.recv(2)
        if response != b'OK':
            raise ConnectionError("Receiver did not acknowledge file")

        return True

    def transfer_test_files(self):
        """Send test files"""
        try:
            # Send total file count
            self.socket.send(struct.pack('!I', len(self.test_files)))

            transferred_files = 0
            total_bytes = 0
            start_time = time.time()

            for file_path in self.test_files:
                if os.path.exists(file_path):
                    print(f"📤 Sending: {file_path}")

                    if self.send_file(file_path):
                        transferred_files += 1
                        file_size = os.path.getsize(file_path)
                        total_bytes += file_size

                        # Progress update
                        elapsed = time.time() - start_time
                        speed = total_bytes / elapsed if elapsed > 0 else 0

                        print(f"✅ {transferred_files}/{len(self.test_files)} files - "
                              f"{speed/1024:.1f} KB/s")
                    else:
                        print(f"❌ Failed to send: {file_path}")
                else:
                    print(f"⚠️  File not found: {file_path}")

            print(f"\n🎉 Test transfer complete!")
            print(f"📊 Total: {transferred_files} files, {total_bytes/1024:.1f} KB")

        except Exception as e:
            print(f"❌ Transfer error: {e}")
            return False

        return True

    def cleanup(self):
        """Close connection"""
        if self.socket:
            self.socket.close()

def main():
    print("🧪 Test Transfer - Small Files Only")
    print("=" * 50)
    print("🎯 Testing ethernet connection with small files")

    sender = TestSender()

    try:
        if sender.connect_to_server():
            sender.transfer_test_files()
    except KeyboardInterrupt:
        print("\n⏹️  Transfer cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        sender.cleanup()
        print("🔌 Connection closed")

if __name__ == "__main__":
    main()
