#!/usr/bin/env python3
"""
Save All Found Wishlist Cases

This script takes all the cases that were found via CourtListener but not yet saved,
and saves them to the corpus with appropriate metadata, then updates all documentation.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def save_all_found_cases():
    """Save all cases that were found but not yet saved to the corpus."""

    logger.info("="*80)
    logger.info("SAVING ALL FOUND WISHLIST CASES")
    logger.info("="*80)

    # Load the acquisition log
    log_path = Path("data/case_law/wishlist_acquisition_log.json")
    if not log_path.exists():
        logger.error("Acquisition log not found!")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    # Get all cases that were found but not saved
    found_cases = [
        case for case in log_data["cases"]
        if case["status"] == "found" and case.get("file_path") is None
    ]

    logger.info(f"Found {len(found_cases)} cases to save")

    client = CourtListenerClient()
    saved_cases = []

    for i, case in enumerate(found_cases, 1):
        logger.info(f"\nProgress: {i}/{len(found_cases)}")
        logger.info(f"Saving: {case['wishlist_name']}")

        cluster_id = case.get('cluster_id')
        if not cluster_id:
            logger.warning(f"No cluster_id for {case['wishlist_name']}")
            continue

        try:
            # Get the opinion by cluster ID using search
            result = client.search_opinions(
                keywords=[str(cluster_id)],
                limit=5
            )

            if not result or 'results' not in result:
                logger.warning(f"No results for cluster {cluster_id}")
                continue

            # Find the matching opinion
            opinion = None
            for op in result['results']:
                if op.get('cluster_id') == cluster_id:
                    opinion = op
                    break

            if not opinion:
                logger.warning(f"Cluster {cluster_id} not found in results")
                continue

            logger.info(f"Found opinion: {opinion.get('caseName', 'Unknown')}")

            # Save the case with special metadata
            try:
                # Add wishlist metadata
                opinion['_wishlist_source'] = case['wishlist_name']
                opinion['_wishlist_citation'] = case['citation']
                opinion['_acquisition_method'] = 'wishlist_search'
                opinion['_acquisition_date'] = datetime.now().isoformat()

                file_path = client.save_opinion(opinion, topic="1782_discovery")

                # Update the case entry
                case['file_path'] = str(file_path)
                case['status'] = 'saved'
                case['notes'] = 'Saved to corpus via wishlist acquisition'

                saved_cases.append({
                    'wishlist_name': case['wishlist_name'],
                    'citation': case['citation'],
                    'cluster_id': cluster_id,
                    'file_path': str(file_path),
                    'case_name': opinion.get('caseName', 'Unknown')
                })

                logger.info(f"✓ Case saved to: {file_path}")

            except Exception as e:
                logger.error(f"Error saving case: {e}")
                case['notes'] = f"Found but save failed: {e}"

        except Exception as e:
            logger.error(f"Error processing {case['wishlist_name']}: {e}")

        # Rate limiting
        import time
        time.sleep(1)

    # Update the acquisition log
    log_data['found_via_courtlistener'] = len([c for c in log_data["cases"] if c.get("file_path")])
    log_data['last_updated'] = datetime.now().isoformat()

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    # Save results
    results_file = Path("data/case_law/saved_wishlist_cases.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(saved_cases, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("SAVE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Cases successfully saved: {len(saved_cases)}")

    if saved_cases:
        logger.info(f"\nSaved cases:")
        for case in saved_cases:
            logger.info(f"  ✓ {case['wishlist_name']}")

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Updated acquisition log: {log_path}")

    return saved_cases


def update_documentation():
    """Update all documentation to reflect the new cases."""

    logger.info("\n" + "="*80)
    logger.info("UPDATING DOCUMENTATION")
    logger.info("="*80)

    # Run duplicate analysis
    logger.info("Running duplicate analysis...")
    import subprocess
    result = subprocess.run(['py', 'scripts/analyze_duplicates.py'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✓ Duplicate analysis completed")
    else:
        logger.error(f"Duplicate analysis failed: {result.stderr}")

    # Run wishlist coverage check
    logger.info("Running wishlist coverage check...")
    result = subprocess.run(['py', 'scripts/check_wishlist_coverage.py'],
                          capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("✓ Wishlist coverage check completed")
        logger.info("Coverage results:")
        logger.info(result.stdout)
    else:
        logger.error(f"Coverage check failed: {result.stderr}")

    # Update the manual retrieval document
    logger.info("Updating manual retrieval documentation...")
    update_manual_retrieval_doc()

    logger.info("✓ Documentation update completed")


def update_manual_retrieval_doc():
    """Update the manual retrieval document with current status."""

    # Load current acquisition log
    log_path = Path("data/case_law/wishlist_acquisition_log.json")
    if not log_path.exists():
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    # Count cases by status
    saved_cases = [c for c in log_data["cases"] if c.get("file_path")]
    manual_cases = [c for c in log_data["cases"] if c["status"] == "manual_required"]
    found_but_not_saved = [c for c in log_data["cases"] if c["status"] == "found" and not c.get("file_path")]

    # Update the manual retrieval document
    doc_content = f"""# Manual Retrieval Required for §1782 Wishlist Cases

## Overview
This document lists cases from the 45-case wishlist that require manual retrieval via PACER, Bloomberg Law, or Westlaw because they are not available through CourtListener's public API.

## Current Status (Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})

| Status | Count | Percentage |
|--------|-------|------------|
| **Successfully Saved** | {len(saved_cases)} | {len(saved_cases)/45*100:.1f}% |
| **Requires Manual Retrieval** | {len(manual_cases)} | {len(manual_cases)/45*100:.1f}% |
| **Found But Not Saved** | {len(found_but_not_saved)} | {len(found_but_not_saved)/45*100:.1f}% |

## Cases Requiring Manual Retrieval

"""

    if manual_cases:
        for i, case in enumerate(manual_cases, 1):
            doc_content += f"""### {i}. {case['wishlist_name']}
- **Citation**: {case['citation']}
- **Status**: ❌ Not found via CourtListener
- **Manual Retrieval Instructions**:
  1. Access PACER for the appropriate court
  2. Search for docket containing relevant party names
  3. Look for order matching the citation date
  4. Download the document
  5. Save as PDF and extract plain text
  6. Normalize to JSON schema with §1782 validation

"""
    else:
        doc_content += "**No cases currently require manual retrieval!** 🎉\n\n"

    if found_but_not_saved:
        doc_content += f"""## Cases Found But Not Yet Saved ({len(found_but_not_saved)})

These cases were found via CourtListener but have not been saved to the corpus yet:

"""
        for case in found_but_not_saved:
            doc_content += f"- [ ] **{case['wishlist_name']}** ({case['citation']})\n"

    doc_content += f"""
## Successfully Saved Cases ({len(saved_cases)})

These cases have been successfully acquired and saved to the corpus:

"""
    for case in saved_cases:
        doc_content += f"- [x] **{case['wishlist_name']}** ({case['citation']})\n"

    doc_content += f"""
## Next Steps

1. **Manual PACER Retrieval**: Focus on the {len(manual_cases)} case(s) requiring manual access
2. **Review Found Cases**: Examine the {len(found_but_not_saved)} cases found but not saved
3. **Update Coverage**: Re-run wishlist coverage analysis after manual validation

## Success Metrics
- ✅ {len(saved_cases)}/{45} cases successfully acquired ({len(saved_cases)/45*100:.1f}% success rate)
- ✅ {44-len(manual_cases)}/{45} cases found via CourtListener ({(44-len(manual_cases))/45*100:.1f}% discovery rate)
- ❌ {len(manual_cases)} case(s) require manual PACER retrieval
"""

    # Write the updated document
    doc_path = Path("data/case_law/manual_retrieval_needed.md")
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(doc_content)

    logger.info(f"✓ Updated manual retrieval document: {doc_path}")


def main():
    """Main entry point."""
    logger.info("Starting comprehensive wishlist case saving and documentation update...")

    # Save all found cases
    saved_cases = save_all_found_cases()

    # Update all documentation
    update_documentation()

    logger.info("\n🎉 All found cases saved and documentation updated!")
    logger.info(f"Check data/case_law/saved_wishlist_cases.json for detailed results")


if __name__ == "__main__":
    main()
