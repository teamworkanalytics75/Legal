#!/usr/bin/env python3
"""
Setup script for CourtListener PDF Downloader
Installs Playwright and required dependencies
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r",
            "scripts/pdf_downloader_requirements.txt"
        ])
        print("✅ Python dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False

    return True

def install_playwright_browsers():
    """Install Playwright browser binaries."""
    print("🌐 Installing Playwright browser binaries...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "playwright", "install", "chromium"
        ])
        print("✅ Playwright browsers installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Playwright browsers: {e}")
        return False

    return True

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")

    directories = [
        "data/case_law/pdfs",
        "data/case_law/logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function."""
    print("🚀 Setting up CourtListener PDF Downloader")
    print("=" * 50)

    # Install Python dependencies
    if not install_requirements():
        print("❌ Setup failed at Python dependencies")
        return False

    # Install Playwright browsers
    if not install_playwright_browsers():
        print("❌ Setup failed at Playwright browsers")
        return False

    # Create directories
    create_directories()

    print("\n✅ Setup complete!")
    print("\nTo run the PDF downloader:")
    print("python scripts/courtlistener_pdf_downloader.py --topic 1782_discovery --limit 10")
    print("\nFor headless mode:")
    print("python scripts/courtlistener_pdf_downloader.py --topic 1782_discovery --limit 10 --headless")

if __name__ == "__main__":
    main()
