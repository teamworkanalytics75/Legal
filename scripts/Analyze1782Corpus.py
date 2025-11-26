#!/usr/bin/env python3
"""
Master §1782 Corpus Analyzer

Orchestrates the complete extraction pipeline across all 13 PDFs,
populates the SQL database, and prepares data for summary report generation.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import time

# Import our enhanced extractor
from enhanced_1782_extractor import analyze_case_comprehensive
from setup_1782_database import create_1782_database, populate_from_enhanced_extraction

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/corpus_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def load_alignment_mapping(alignment_file: Path) -> Dict:
    """Load the JSON-PDF alignment mapping."""
    logger.info(f"Loading alignment mapping from: {alignment_file}")

    if not alignment_file.exists():
        logger.error(f"Alignment file not found: {alignment_file}")
        return {}

    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)

    # Filter for successful downloads
    successful_cases = {
        cluster_id: info for cluster_id, info in alignment_data.items()
        if info.get('download_status') == 'success' and info.get('pdf_path')
    }

    logger.info(f"Found {len(successful_cases)} successful PDF downloads")
    return successful_cases


def process_corpus(alignment_file: Path, output_dir: Path, db_path: Path) -> Dict:
    """
    Process all PDFs in the corpus through comprehensive analysis.

    Args:
        alignment_file: Path to JSON-PDF alignment mapping
        output_dir: Directory to save individual analysis results
        db_path: Path to SQLite database

    Returns:
        Dictionary with processing results
    """
    logger.info("Starting comprehensive §1782 corpus analysis...")

    # Load alignment mapping
    successful_cases = load_alignment_mapping(alignment_file)
    if not successful_cases:
        logger.error("No successful cases found to analyze")
        return {'error': 'No cases to analyze'}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create database
    logger.info("Creating/updating database schema...")
    if not create_1782_database(db_path):
        logger.error("Failed to create database")
        return {'error': 'Database creation failed'}

    # Process each case
    results = {
        'processing_timestamp': str(Path().cwd()),
        'alignment_file': str(alignment_file),
        'output_directory': str(output_dir),
        'database_path': str(db_path),
        'total_cases': len(successful_cases),
        'successful_analyses': 0,
        'failed_analyses': 0,
        'analyses': [],
        'errors': []
    }

    logger.info(f"Processing {len(successful_cases)} cases...")

    for i, (cluster_id, case_info) in enumerate(successful_cases.items(), 1):
        logger.info(f"[{i}/{len(successful_cases)}] Processing cluster {cluster_id}: {case_info['case_name']}")

        pdf_path = Path(case_info['pdf_path'])

        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            results['failed_analyses'] += 1
            results['errors'].append({
                'cluster_id': cluster_id,
                'error': 'PDF file not found',
                'pdf_path': str(pdf_path)
            })
            continue

        # Analyze case
        start_time = time.time()
        analysis = analyze_case_comprehensive(pdf_path, case_info)
        duration = time.time() - start_time

        if analysis:
            results['successful_analyses'] += 1
            results['analyses'].append(analysis)

            # Save individual analysis
            analysis_file = output_dir / f"{cluster_id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"  [OK] Analysis complete in {duration:.2f}s: {analysis_file.name}")
            logger.info(f"    - Outcome: {analysis['outcome']['outcome']} (confidence: {analysis['outcome']['confidence']:.2f})")
            logger.info(f"    - Intel factors: {sum(1 for f in analysis['intel_factors'].values() if f['detected'])}/4")
            logger.info(f"    - Citations: {len(analysis['citations'])}")
            logger.info(f"    - Pages: {analysis['structural']['page_count']}")
        else:
            results['failed_analyses'] += 1
            results['errors'].append({
                'cluster_id': cluster_id,
                'error': 'Analysis failed',
                'pdf_path': str(pdf_path)
            })
            logger.error(f"  [FAIL] Analysis failed for {case_info['case_name']}")

    # Populate database
    if results['analyses']:
        logger.info("Populating database with analysis results...")
        if populate_from_enhanced_extraction(db_path, results['analyses']):
            logger.info("Database population successful")
        else:
            logger.error("Database population failed")
            results['errors'].append({'error': 'Database population failed'})

    # Save combined results
    results_file = output_dir / "combined_corpus_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Combined results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"§1782 CORPUS ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Cases:        {results['total_cases']}")
    print(f"Successful:        {results['successful_analyses']}")
    print(f"Failed:            {results['failed_analyses']}")
    print(f"Success Rate:      {(results['successful_analyses']/results['total_cases'])*100:.1f}%")
    print(f"Output Directory:  {output_dir}")
    print(f"Database:          {db_path}")
    print(f"Results File:      {results_file}")
    print(f"{'='*60}")

    if results['analyses']:
        print(f"\nOutcome Distribution:")
        outcomes = {}
        for analysis in results['analyses']:
            outcome = analysis['outcome']['outcome']
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        for outcome, count in outcomes.items():
            print(f"  {outcome}: {count}")

        print(f"\nIntel Factor Detection:")
        factor_counts = {'factor_1': 0, 'factor_2': 0, 'factor_3': 0, 'factor_4': 0}
        for analysis in results['analyses']:
            for factor_name, factor_data in analysis['intel_factors'].items():
                if factor_data.get('detected'):
                    factor_counts[factor_name] += 1

        for factor, count in factor_counts.items():
            print(f"  {factor}: {count}/{results['successful_analyses']} cases")

        print(f"\nCitation Summary:")
        citation_counts = {}
        for analysis in results['analyses']:
            for citation in analysis['citations']:
                case_name = citation['case_name']
                citation_counts[case_name] = citation_counts.get(case_name, 0) + 1

        for case_name, count in sorted(citation_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {case_name}: {count} mentions")

    return results


def main():
    """Main execution function."""
    print("Starting §1782 Corpus Analysis Pipeline...")

    # Define paths
    alignment_file = Path("data/case_law/json_pdf_alignment.json")
    output_dir = Path("data/case_law/corpus_analysis_results")
    db_path = Path("data/case_law/1782_analysis.db")

    # Run analysis
    results = process_corpus(alignment_file, output_dir, db_path)

    if 'error' in results:
        logger.error(f"Analysis failed: {results['error']}")
        sys.exit(1)

    logger.info("Corpus analysis completed successfully")
    print(f"\n[SUCCESS] Analysis complete! Check {output_dir} for results.")
    print(f"[SUCCESS] Database ready at: {db_path}")


if __name__ == "__main__":
    main()
