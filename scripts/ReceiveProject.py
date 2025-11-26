#!/usr/bin/env python3
"""
Receiver script for Agents project transfer
Run this on the gaming PC (192.168.1.2) to receive files from laptop
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

if hasattr(sys.stdout, "reconfigure"):
    # Keep Unicode status icons from crashing Windows consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

class ProjectReceiver:
    def __init__(self, port=9999, dest_dir="Agents_Transfer"):
        self.port = port
        self.dest_dir = Path(dest_dir)
        self.socket = None
        self.connection = None

    def start_server(self):
        """Start the TCP server and wait for connection"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(1)
            print(f"🚀 Server listening on port {self.port}")
            print(f"📁 Files will be saved to: {self.dest_dir.absolute()}")
            print("⏳ Waiting for connection from laptop...")

            self.connection, addr = self.socket.accept()
            print(f"✅ Connected to {addr[0]}:{addr[1]}")
            return True

        except Exception as e:
            print(f"❌ Server error: {e}")
            return False

    def receive_data(self, size):
        """Receive exactly 'size' bytes from connection"""
        data = b''
        while len(data) < size:
            chunk = self.connection.recv(min(size - len(data), 4096))
            if not chunk:
                raise ConnectionError("Connection lost")
            data += chunk
        return data

    def receive_file_metadata(self):
        """Receive file metadata (path, size, checksum)"""
        # Receive metadata length
        meta_len = struct.unpack('!I', self.receive_data(4))[0]
        # Receive metadata JSON
        meta_data = self.receive_data(meta_len).decode('utf-8')
        return json.loads(meta_data)

    def receive_file(self, file_path, expected_size, expected_checksum):
        """Receive a single file"""
        file_path = self.dest_dir / file_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Receive compressed data
        compressed_data = self.receive_data(expected_size)

        # Decompress
        try:
            file_data = zlib.decompress(compressed_data)
        except zlib.error as e:
            raise ValueError(f"Decompression failed: {e}")

        # Verify checksum
        actual_checksum = hashlib.sha256(file_data).hexdigest()
        if actual_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")

        # Write file
        with open(file_path, 'wb') as f:
            f.write(file_data)

        return len(file_data)

    def transfer_files(self):
        """Main transfer loop"""
        try:
            # Receive total file count
            total_files = struct.unpack('!I', self.receive_data(4))[0]
            print(f"📊 Total files to receive: {total_files}")

            received_files = 0
            total_bytes = 0
            start_time = time.time()

            while received_files < total_files:
                # Receive file metadata
                metadata = self.receive_file_metadata()
                file_path = metadata['path']
                file_size = metadata['compressed_size']
                checksum = metadata['checksum']
                original_size = metadata['original_size']

                print(f"📄 Receiving: {file_path} ({original_size:,} bytes)")

                # Receive file
                bytes_received = self.receive_file(file_path, file_size, checksum)
                total_bytes += bytes_received
                received_files += 1

                # Send acknowledgment
                self.connection.send(b'OK')

                # Progress update
                elapsed = time.time() - start_time
                speed = total_bytes / elapsed if elapsed > 0 else 0
                progress = (received_files / total_files) * 100

                print(f"✅ {received_files}/{total_files} files ({progress:.1f}%) - "
                      f"{speed/1024/1024:.1f} MB/s")

            print(f"\n🎉 Transfer complete!")
            print(f"📁 Files saved to: {self.dest_dir.absolute()}")
            print(f"📊 Total: {received_files} files, {total_bytes/1024/1024:.1f} MB")

        except Exception as e:
            print(f"❌ Transfer error: {e}")
            return False

        return True

    def cleanup(self):
        """Close connections"""
        if self.connection:
            self.connection.close()
        if self.socket:
            self.socket.close()

def main():
    print("🖥️  Agents Project Receiver")
    print("=" * 50)

    receiver = ProjectReceiver()

    try:
        if receiver.start_server():
            receiver.transfer_files()
    except KeyboardInterrupt:
        print("\n⏹️  Transfer cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        receiver.cleanup()
        print("🔌 Connection closed")

if __name__ == "__main__":
    main()
