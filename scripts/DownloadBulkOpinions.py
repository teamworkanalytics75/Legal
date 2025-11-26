"""
The Matrix - Download CourtListener Bulk Opinions Data

Downloads the complete opinions CSV (49.9 GB), then filters for Massachusetts cases.
This is MUCH faster than API searches!
"""

import requests
import os
from pathlib import Path
from tqdm import tqdm

# Latest opinions file (Oct 2025)
BULK_DATA_URL = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data/opinions-2025-10-09.csv.bz2"
DOWNLOAD_PATH = Path("data/bulk_downloads")
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DOWNLOAD_PATH / "opinions-2025-10-09.csv.bz2"

print("="*60)
print("WITCHWEB - Bulk Opinions Download")
print("="*60)
print(f"\nDownloading from: {BULK_DATA_URL}")
print(f"Size: ~50 GB compressed")
print(f"Output: {OUTPUT_FILE}")
print("\nThis will take 1-3 hours depending on your internet speed.")
print("The download can be resumed if interrupted.")
print("\n" + "="*60 + "\n")

def download_file_with_resume(url, output_path):
    """Download large file with resume capability."""

    # Check if file partially downloaded
    if output_path.exists():
        existing_size = output_path.stat().st_size
        headers = {'Range': f'bytes={existing_size}-'}
        mode = 'ab'
        print(f"Resuming download from {existing_size / 1e9:.2f} GB...")
    else:
        headers = {}
        mode = 'wb'
        existing_size = 0

    # Start download
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    total_size = int(response.headers.get('content-length', 0)) + existing_size

    with open(output_path, mode) as f:
        with tqdm(
            total=total_size,
            initial=existing_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading opinions"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"\nDownload complete! Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1e9:.2f} GB")

if __name__ == "__main__":
    try:
        download_file_with_resume(BULK_DATA_URL, OUTPUT_FILE)

        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Extract the file:")
        print(f" bzip2 -d {OUTPUT_FILE}")
        print("\n2. Filter for Massachusetts:")
        print(" py filter_massachusetts_cases.py")
        print("\n3. Import to database:")
        print(" py run_case_law_pipeline.py --import")
        print("\n" + "="*60)

    except KeyboardInterrupt:
        print("\n\nDownload paused.")
        print("Run this script again to resume from where you left off.")
    except Exception as e:
        print(f"\n\nError: {e}")
        print("You can resume by running this script again.")

