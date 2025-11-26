#!/usr/bin/env python3
"""
Extract PDF URLs from existing §1782 case JSONs.

Phase 1 of PDF processing pipeline - scans all JSONs in The Art of War - Database/
and extracts download_url or local_path fields to build a mapping for PDF downloads.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/pdf_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def extract_pdf_urls_from_json(json_path: Path) -> Optional[Dict]:
    """
    Extract PDF URL information from a single JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary with PDF URL info or None if no URL found
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get basic case info
        cluster_id = data.get('cluster_id')
        case_name = data.get('caseName', 'Unknown')

        if not cluster_id:
            logger.warning(f"No cluster_id in {json_path.name}")
            return None

        # Look for PDF URL in opinions array
        pdf_url = None
        pdf_source = None

        opinions = data.get('opinions', [])
        if opinions and len(opinions) > 0:
            first_opinion = opinions[0]
            pdf_url = first_opinion.get('download_url')
            if pdf_url:
                pdf_source = "opinions[0].download_url"

        # Fallback: check top-level fields
        if not pdf_url:
            pdf_url = data.get('download_url')
            if pdf_url:
                pdf_source = "download_url"

        if not pdf_url:
            pdf_url = data.get('local_path')
            if pdf_url:
                pdf_source = "local_path"

        if not pdf_url:
            logger.debug(f"No PDF URL found in {json_path.name}")
            return None

        # Create safe filename
        safe_case_name = case_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        safe_case_name = ''.join(c for c in safe_case_name if c.isalnum() or c in '_-')
        safe_case_name = safe_case_name[:50]  # Limit length

        return {
            'cluster_id': cluster_id,
            'case_name': case_name,
            'safe_case_name': safe_case_name,
            'pdf_url': pdf_url,
            'pdf_source': pdf_source,
            'json_path': str(json_path),
            'json_size': json_path.stat().st_size
        }

    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return None


def extract_all_pdf_urls(source_dir: Path, output_file: Path) -> Dict:
    """
    Extract PDF URLs from all JSON files in source directory.

    Args:
        source_dir: Directory containing JSON files
        output_file: Path to save the mapping JSON

    Returns:
        Dictionary with extraction results
    """
    logger.info(f"Scanning JSON files in: {source_dir}")

    if not source_dir.exists():
        logger.error(f"Source directory does not exist: {source_dir}")
        return {'error': 'Source directory not found'}

    # Find all JSON files
    json_files = list(source_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files")

    if len(json_files) == 0:
        logger.warning("No JSON files found in source directory")
        return {'error': 'No JSON files found'}

    # Process each JSON file
    pdf_mappings = {}
    processed_count = 0
    success_count = 0
    error_count = 0

    for json_file in json_files:
        processed_count += 1
        logger.info(f"[{processed_count}/{len(json_files)}] Processing: {json_file.name}")

        pdf_info = extract_pdf_urls_from_json(json_file)

        if pdf_info:
            cluster_id = pdf_info['cluster_id']
            pdf_mappings[str(cluster_id)] = pdf_info
            success_count += 1
            logger.info(f"  [OK] Found PDF URL: {pdf_info['pdf_url'][:80]}...")
        else:
            error_count += 1
            logger.warning(f"  [X] No PDF URL found")

    # Save results
    results = {
        'extraction_timestamp': str(Path().cwd()),
        'source_directory': str(source_dir),
        'total_json_files': len(json_files),
        'processed_files': processed_count,
        'successful_extractions': success_count,
        'failed_extractions': error_count,
        'pdf_mappings': pdf_mappings
    }

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"PDF URL EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Source Directory: {source_dir}")
    print(f"Total JSON Files: {len(json_files)}")
    print(f"Processed Files:  {processed_count}")
    print(f"PDF URLs Found:   {success_count}")
    print(f"Missing URLs:     {error_count}")
    print(f"Success Rate:     {(success_count/processed_count)*100:.1f}%")
    print(f"Output File:      {output_file}")
    print(f"{'='*60}")

    if success_count > 0:
        print(f"\nSample PDF URLs found:")
        for i, (cluster_id, info) in enumerate(list(pdf_mappings.items())[:3]):
            print(f"  {cluster_id}: {info['case_name']}")
            print(f"    URL: {info['pdf_url'][:100]}...")

    return results


def main():
    """Main execution function."""
    print("Starting PDF URL extraction from §1782 case JSONs...")

    # Define paths
    source_dir = Path("data/case_law/The Art of War - Database")
    output_file = Path("data/case_law/pdf_urls_mapping.json")

    # Run extraction
    results = extract_all_pdf_urls(source_dir, output_file)

    if 'error' in results:
        logger.error(f"Extraction failed: {results['error']}")
        sys.exit(1)

    logger.info("PDF URL extraction completed successfully")
    print(f"\n[SUCCESS] Extraction complete! Check {output_file} for results.")


if __name__ == "__main__":
    main()
