#!/usr/bin/env python3
"""
Download PDFs for existing §1782 cases.

Phase 2 of PDF processing pipeline - downloads PDFs from URLs extracted
in Phase 1 and saves them to organized directory structure.
"""

import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/pdf_download.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def download_pdf(url: str, output_path: Path, timeout: int = 30, retries: int = 3) -> bool:
    """
    Download PDF file with retries and error handling.

    Args:
        url: PDF download URL
        output_path: Local path to save PDF
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        True if download successful, False otherwise
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for attempt in range(retries):
        try:
            logger.info(f"Downloading PDF (attempt {attempt + 1}/{retries}): {url[:80]}...")

            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with streaming
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # Verify download
            if output_path.stat().st_size == 0:
                logger.error(f"Downloaded file is empty: {output_path}")
                output_path.unlink()  # Remove empty file
                return False

            logger.info(f"Successfully downloaded: {output_path.name} ({output_path.stat().st_size:,} bytes)")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt + 1}: {url}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                logger.error(f"All retry attempts failed for timeout: {url}")
                return False

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"PDF not found (404): {url}")
                return False
            elif e.response.status_code == 403:
                logger.error(f"Access forbidden (403): {url}")
                return False
            else:
                logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1}: {url}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"All retry attempts failed for HTTP error: {url}")
                    return False

        except Exception as e:
            logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                logger.error(f"All retry attempts failed: {url} - {e}")
                return False

    return False


def get_domain_from_url(url: str) -> str:
    """Extract domain from URL for rate limiting tracking."""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "unknown"


def download_all_pdfs(mapping_file: Path, output_dir: Path) -> Dict:
    """
    Download all PDFs from the mapping file.

    Args:
        mapping_file: Path to PDF URLs mapping JSON
        output_dir: Directory to save downloaded PDFs

    Returns:
        Dictionary with download results
    """
    logger.info(f"Loading PDF URLs from: {mapping_file}")

    if not mapping_file.exists():
        logger.error(f"Mapping file not found: {mapping_file}")
        return {'error': 'Mapping file not found'}

    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)

    pdf_mappings = mapping_data.get('pdf_mappings', {})
    if not pdf_mappings:
        logger.error("No PDF mappings found in file")
        return {'error': 'No PDF mappings found'}

    logger.info(f"Found {len(pdf_mappings)} PDF URLs to download")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = {
        'download_timestamp': str(Path().cwd()),
        'mapping_file': str(mapping_file),
        'output_directory': str(output_dir),
        'total_urls': len(pdf_mappings),
        'successful_downloads': 0,
        'failed_downloads': 0,
        'downloads_by_domain': {},
        'alignment_mapping': {},
        'errors': []
    }

    # Download each PDF
    for i, (cluster_id, pdf_info) in enumerate(pdf_mappings.items(), 1):
        logger.info(f"[{i}/{len(pdf_mappings)}] Processing cluster {cluster_id}: {pdf_info['case_name']}")

        pdf_url = pdf_info['pdf_url']
        case_name = pdf_info['case_name']
        safe_case_name = pdf_info['safe_case_name']

        # Create output filename
        pdf_filename = f"{cluster_id}_{safe_case_name}.pdf"
        pdf_path = output_dir / pdf_filename

        # Skip if already downloaded
        if pdf_path.exists():
            logger.info(f"  [SKIP] PDF already exists: {pdf_filename}")
            results['successful_downloads'] += 1
            results['alignment_mapping'][cluster_id] = {
                'json_path': pdf_info['json_path'],
                'pdf_path': str(pdf_path),
                'case_name': case_name,
                'download_status': 'already_exists',
                'pdf_url': pdf_url
            }
            continue

        # Download PDF
        success = download_pdf(pdf_url, pdf_path)

        # Track domain for rate limiting
        domain = get_domain_from_url(pdf_url)
        if domain not in results['downloads_by_domain']:
            results['downloads_by_domain'][domain] = {'success': 0, 'failed': 0}

        if success:
            results['successful_downloads'] += 1
            results['downloads_by_domain'][domain]['success'] += 1
            results['alignment_mapping'][cluster_id] = {
                'json_path': pdf_info['json_path'],
                'pdf_path': str(pdf_path),
                'case_name': case_name,
                'download_status': 'success',
                'pdf_url': pdf_url,
                'file_size': pdf_path.stat().st_size
            }
            logger.info(f"  [OK] Downloaded: {pdf_filename}")
        else:
            results['failed_downloads'] += 1
            results['downloads_by_domain'][domain]['failed'] += 1
            results['alignment_mapping'][cluster_id] = {
                'json_path': pdf_info['json_path'],
                'pdf_path': None,
                'case_name': case_name,
                'download_status': 'failed',
                'pdf_url': pdf_url,
                'error': 'Download failed after retries'
            }
            results['errors'].append({
                'cluster_id': cluster_id,
                'case_name': case_name,
                'url': pdf_url,
                'error': 'Download failed'
            })
            logger.error(f"  [FAIL] Failed: {case_name}")

        # Rate limiting - wait between downloads
        if i < len(pdf_mappings):  # Don't wait after last download
            logger.info(f"  [WAIT] Waiting 2 seconds before next download...")
            time.sleep(2)

    # Save alignment mapping
    alignment_file = Path("data/case_law/json_pdf_alignment.json")
    with open(alignment_file, 'w', encoding='utf-8') as f:
        json.dump(results['alignment_mapping'], f, indent=2, ensure_ascii=False)

    logger.info(f"Alignment mapping saved to: {alignment_file}")

    # Save download report
    report_file = Path("data/case_law/pdf_download_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Download report saved to: {report_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"PDF DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total URLs:        {results['total_urls']}")
    print(f"Successful:       {results['successful_downloads']}")
    print(f"Failed:           {results['failed_downloads']}")
    print(f"Success Rate:     {(results['successful_downloads']/results['total_urls'])*100:.1f}%")
    print(f"Output Directory: {output_dir}")
    print(f"Alignment File:   {alignment_file}")
    print(f"Report File:      {report_file}")
    print(f"{'='*60}")

    if results['downloads_by_domain']:
        print(f"\nDownloads by Domain:")
        for domain, stats in results['downloads_by_domain'].items():
            print(f"  {domain}: {stats['success']} success, {stats['failed']} failed")

    if results['errors']:
        print(f"\nFailed Downloads:")
        for error in results['errors'][:5]:  # Show first 5 errors
            print(f"  {error['cluster_id']}: {error['case_name']}")

    return results


def main():
    """Main execution function."""
    print("Starting PDF download for §1782 cases...")

    # Define paths
    mapping_file = Path("data/case_law/pdf_urls_mapping.json")
    output_dir = Path("data/case_law/pdfs/1782_discovery")

    # Run downloads
    results = download_all_pdfs(mapping_file, output_dir)

    if 'error' in results:
        logger.error(f"Download failed: {results['error']}")
        sys.exit(1)

    logger.info("PDF download completed")
    print(f"\n[SUCCESS] Download complete! Check {output_dir} for PDFs.")


if __name__ == "__main__":
    main()

