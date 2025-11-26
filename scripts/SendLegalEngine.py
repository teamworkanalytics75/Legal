#!/usr/bin/env python3
"""
Priority sender script for Agents project transfer
Focuses on legal reasoning engine components first
"""

import socket
import os
import zlib
import hashlib
import struct
import json
from pathlib import Path
import time

class PriorityProjectSender:
    def __init__(self, target_ip="192.168.1.2", port=9999, source_dir="."):
        self.target_ip = target_ip
        self.port = port
        self.source_dir = Path(source_dir)
        self.socket = None
        
        # Priority patterns for legal reasoning engine
        self.priority_patterns = {
            # Core legal reasoning files
            'analyze_1782_pdfs.py',
            'section_1782_mining',
            'writer_agents',
            'bayesian_network',
            'nlp_analysis',
            'ml_system',
            'factuality_filter',
            'document_ingestion',
            
            # Databases and data
            '*.db',
            'harvard_crimson_intelligence.db',
            'precedent_database.db',
            'jobs.db',
            'vida_data.db',
            
            # Configuration and prompts
            'config',
            '1782_agent_query_examples.json',
            'agent_prompts.json',
            'langchain_capability_seeds.json',
            
            # Documentation
            'CHATGPT_PROJECT_DOCS',
            '*.md',
            'README.md',
            'PROJECT_ORGANIZATION_V2.2.md',
            'MARKET_VALUE_ASSESSMENT_V2.2.md',
            
            # Core Python files
            '*.py',
            'cli.py',
            'pipeline_orchestrator.py',
            
            # Case law and legal data
            'case_law',
            'data/section_1782',
            'data/top_50_section_1782_cases.json',
            'data/section_1782_top_50_cases.json'
        }
        
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
            'data/bulk_downloads',
            'Court-Cases-Data-Scrapping',
            'awesome-legal-data.zip',
            'legal-ml-datasets.zip',
            'Indian_SC_Judgment_database.zip',
            'node_modules',
            'sprites',
            'witchweb_3d',
            'witchweb_ui',
            'design_explorer',
            'dynamo_generator',
            'revit_agent',
            'voice_system'
        }
    
    def is_priority_file(self, path):
        """Check if file matches priority patterns"""
        path_str = str(path)
        name = path.name
        
        for pattern in self.priority_patterns:
            if pattern == name:
                return True
            if pattern.endswith('*') and name.endswith(pattern[:-1]):
                return True
            if pattern in path_str:
                return True
        
        return False
    
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
        
        # Exclude files larger than 100MB (increased limit for databases)
        try:
            if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
                return True
        except (OSError, FileNotFoundError):
            pass
        
        return False
    
    def scan_files(self):
        """Scan source directory and return prioritized list of files to transfer"""
        priority_files = []
        other_files = []
        
        print("🔍 Scanning project directory for legal reasoning engine components...")
        
        for root, dirs, files in os.walk(self.source_dir):
            root_path = Path(root)
            
            # Remove excluded directories from dirs list
            dirs[:] = [d for d in dirs if not self.should_exclude(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                
                if not self.should_exclude(file_path):
                    rel_path = file_path.relative_to(self.source_dir)
                    
                    if self.is_priority_file(file_path):
                        priority_files.append(rel_path)
                    else:
                        other_files.append(rel_path)
        
        # Combine with priority files first
        all_files = priority_files + other_files
        
        print(f"📋 Found {len(priority_files)} priority files (legal reasoning engine)")
        print(f"📋 Found {len(other_files)} other files")
        print(f"📋 Total: {len(all_files)} files to transfer")
        
        return all_files
    
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
    print("⚖️  Legal Reasoning Engine Transfer")
    print("=" * 50)
    print("🎯 Priority: 1782 PDFs, databases, NLP/ML components")
    
    sender = PriorityProjectSender()
    
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
