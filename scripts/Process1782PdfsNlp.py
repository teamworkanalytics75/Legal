#!/usr/bin/env python3
"""
Process downloaded §1782 PDFs through NLP analysis pipeline.

Extracts text from PDFs, runs entity extraction, causal analysis, and
Bayesian evidence extraction to identify patterns and rules.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# PDF text extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[WARN] pdfplumber not installed. PDF text extraction will be limited.")

# NLP pipeline - fix relative import issues
import os
import sys

# Add the nlp_analysis directory to Python path
nlp_path = os.path.abspath('nlp_analysis')
if nlp_path not in sys.path:
    sys.path.insert(0, nlp_path)

# Add the code subdirectory
code_path = os.path.abspath('nlp_analysis/code')
if code_path not in sys.path:
    sys.path.insert(0, code_path)

try:
    # Import the pipeline module directly
    import pipeline
    NLPAnalysisPipeline = pipeline.NLPAnalysisPipeline
    HAS_NLP_PIPELINE = True
    print("[INFO] NLP pipeline loaded successfully")
except ImportError as e:
    HAS_NLP_PIPELINE = False
    print(f"[WARN] NLP pipeline not available: {e}")
    print("[WARN] Will extract text only.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/pdf_nlp_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract text from PDF file using pdfplumber.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text or None if extraction fails
    """
    if not HAS_PDFPLUMBER:
        logger.error("pdfplumber not available for text extraction")
        return None

    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"--- Page {page_num} ---\n{page_text}")

        full_text = '\n\n'.join(text_parts)

        if len(full_text.strip()) < 100:
            logger.warning(f"Extracted text too short from {pdf_path.name}: {len(full_text)} chars")
            return None

        logger.info(f"Extracted {len(full_text)} characters from {pdf_path.name}")
        return full_text

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return None


def analyze_pdf_with_nlp(pdf_path: Path, case_info: Dict) -> Optional[Dict]:
    """
    Analyze PDF through NLP pipeline.

    Args:
        pdf_path: Path to PDF file
        case_info: Case metadata from alignment mapping

    Returns:
        NLP analysis results or None if analysis fails
    """
    logger.info(f"Analyzing PDF: {pdf_path.name}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.error(f"Could not extract text from {pdf_path.name}")
        return None

    try:
        if not HAS_NLP_PIPELINE:
            logger.warning("NLP pipeline not available - extracting text only")
            return {
                'case_metadata': {
                    'cluster_id': case_info.get('cluster_id'),
                    'case_name': case_info.get('case_name'),
                    'pdf_path': str(pdf_path),
                    'json_path': case_info.get('json_path'),
                    'text_length': len(text)
                },
                'extracted_text': text[:1000] + "..." if len(text) > 1000 else text,
                'summary': {
                    'num_entities': 0,
                    'num_relations': 0,
                    'num_causal_links': 0,
                    'num_bn_nodes': 0,
                    'num_evidence_statements': 0
                }
            }

        # Initialize NLP pipeline
        pipeline = NLPAnalysisPipeline()

        # Run analysis (disable coreference resolution for speed)
        analysis = pipeline.analyze_text(text, resolve_coref=False)

        # Add case metadata
        analysis['case_metadata'] = {
            'cluster_id': case_info.get('cluster_id'),
            'case_name': case_info.get('case_name'),
            'pdf_path': str(pdf_path),
            'json_path': case_info.get('json_path'),
            'text_length': len(text)
        }

        logger.info(f"Analysis complete for {pdf_path.name}:")
        logger.info(f"  - Entities: {analysis['summary']['num_entities']}")
        logger.info(f"  - Relations: {analysis['summary']['num_relations']}")
        logger.info(f"  - Causal links: {analysis['summary']['num_causal_links']}")
        logger.info(f"  - BN nodes: {analysis['summary']['num_bn_nodes']}")

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing {pdf_path.name}: {e}")
        return None


def process_all_pdfs(alignment_file: Path, output_dir: Path) -> Dict:
    """
    Process all successfully downloaded PDFs through NLP analysis.

    Args:
        alignment_file: Path to JSON-PDF alignment mapping
        output_dir: Directory to save analysis results

    Returns:
        Dictionary with processing results
    """
    logger.info(f"Loading alignment mapping from: {alignment_file}")

    if not alignment_file.exists():
        logger.error(f"Alignment file not found: {alignment_file}")
        return {'error': 'Alignment file not found'}

    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)

    # Filter for successful downloads
    successful_cases = {
        cluster_id: info for cluster_id, info in alignment_data.items()
        if info.get('download_status') == 'success' and info.get('pdf_path')
    }

    logger.info(f"Found {len(successful_cases)} PDFs to analyze")

    if len(successful_cases) == 0:
        logger.warning("No successful PDF downloads found to analyze")
        return {'error': 'No PDFs to analyze'}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each PDF
    results = {
        'processing_timestamp': str(Path().cwd()),
        'alignment_file': str(alignment_file),
        'output_directory': str(output_dir),
        'total_pdfs': len(successful_cases),
        'successful_analyses': 0,
        'failed_analyses': 0,
        'analyses': {},
        'errors': []
    }

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

        # Analyze PDF
        analysis = analyze_pdf_with_nlp(pdf_path, case_info)

        if analysis:
            results['successful_analyses'] += 1
            results['analyses'][cluster_id] = analysis

            # Save individual analysis
            analysis_file = output_dir / f"{cluster_id}_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)

            logger.info(f"  [OK] Analysis saved: {analysis_file.name}")
        else:
            results['failed_analyses'] += 1
            results['errors'].append({
                'cluster_id': cluster_id,
                'error': 'Analysis failed',
                'pdf_path': str(pdf_path)
            })
            logger.error(f"  [FAIL] Analysis failed for {case_info['case_name']}")

    # Save combined results
    results_file = output_dir / "combined_nlp_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Combined results saved to: {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"PDF NLP ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total PDFs:        {results['total_pdfs']}")
    print(f"Successful:       {results['successful_analyses']}")
    print(f"Failed:           {results['failed_analyses']}")
    print(f"Success Rate:     {(results['successful_analyses']/results['total_pdfs'])*100:.1f}%")
    print(f"Output Directory: {output_dir}")
    print(f"Results File:     {results_file}")
    print(f"{'='*60}")

    if results['analyses']:
        print(f"\nAnalysis Results:")
        for cluster_id, analysis in results['analyses'].items():
            summary = analysis['summary']
            print(f"  {cluster_id}: {analysis['case_metadata']['case_name']}")
            print(f"    Entities: {summary['num_entities']}, Relations: {summary['num_relations']}")
            print(f"    Causal: {summary['num_causal_links']}, BN Nodes: {summary['num_bn_nodes']}")

    return results


def main():
    """Main execution function."""
    print("Starting PDF NLP analysis for §1782 cases...")

    # Define paths
    alignment_file = Path("data/case_law/json_pdf_alignment.json")
    output_dir = Path("data/case_law/nlp_analysis_results")

    # Run analysis
    results = process_all_pdfs(alignment_file, output_dir)

    if 'error' in results:
        logger.error(f"Analysis failed: {results['error']}")
        sys.exit(1)

    logger.info("PDF NLP analysis completed")
    print(f"\n[SUCCESS] Analysis complete! Check {output_dir} for results.")


if __name__ == "__main__":
    main()
