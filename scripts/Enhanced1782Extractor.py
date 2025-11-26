#!/usr/bin/env python3
"""
Enhanced §1782 Pattern Extractor

Extracts structural features, outcomes, Intel factors, statutory prerequisites,
and citations from §1782 case PDFs using hybrid detection methods.
"""

import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# PDF text extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("[WARN] pdfplumber not installed. PDF text extraction will be limited.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/case_law/enhanced_extraction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from PDF file using pdfplumber."""
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


def count_pages(pdf_path: Path) -> int:
    """Count pages in PDF file."""
    if not HAS_PDFPLUMBER:
        return 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def extract_headers(text: str) -> List[str]:
    """Extract section headers from text."""
    # Common legal document headers
    header_patterns = [
        r'^BACKGROUND\s*$',
        r'^DISCUSSION\s*$',
        r'^ANALYSIS\s*$',
        r'^CONCLUSION\s*$',
        r'^ORDER\s*$',
        r'^MEMORANDUM\s*$',
        r'^APPLICATION\s*$',
        r'^ARGUMENT\s*$',
        r'^FACTS\s*$',
        r'^PROCEDURAL\s+HISTORY\s*$'
    ]

    headers = []
    lines = text.split('\n')

    for line in lines:
        line = line.strip()
        for pattern in header_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                headers.append(line)
                break

    return headers


def extract_structural_features(text: str, pdf_path: Path) -> Dict:
    """Extract structural features from PDF text."""
    return {
        'page_count': count_pages(pdf_path),
        'word_count': len(text.split()),
        'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
        'section_headers': extract_headers(text),
        'num_section_headers': len(extract_headers(text)),
        'has_proposed_order': 'proposed order' in text.lower(),
        'has_memorandum': 'memorandum' in text.lower(),
        'has_affidavit': 'affidavit' in text.lower(),
        'has_exhibits': 'exhibit' in text.lower(),
        'has_declaration': 'declaration' in text.lower()
    }


def extract_outcome(text: str) -> Dict:
    """Extract case outcome using regex patterns."""
    # Search in last 20% of document (where orders typically appear)
    tail_start = int(len(text) * 0.8)
    tail = text[tail_start:]

    patterns = {
        'granted': [
            r'(?:IS\s+)?GRANTED',
            r'GRANT(?:S|ING)',
            r'ALLOW(?:S|ING)',
            r'PERMIT(?:S|TING)'
        ],
        'denied': [
            r'(?:IS\s+)?DENIED',
            r'DENY(?:S|ING)',
            r'REJECT(?:S|ING)',
            r'REFUSE(?:S|ING)'
        ],
        'partial': [
            r'GRANTED\s+IN\s+PART',
            r'DENIED\s+IN\s+PART',
            r'PARTIALLY\s+GRANTED',
            r'PARTIALLY\s+DENIED'
        ]
    }

    outcome_scores = {'granted': 0, 'denied': 0, 'partial': 0}
    contexts = []

    for outcome_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, tail, re.IGNORECASE)
            for match in matches:
                outcome_scores[outcome_type] += 1

                # Extract context window (3 lines around match)
                lines = tail.split('\n')
                match_line_idx = tail[:match.start()].count('\n')

                context_start = max(0, match_line_idx - 1)
                context_end = min(len(lines), match_line_idx + 2)
                context = '\n'.join(lines[context_start:context_end])

                contexts.append({
                    'outcome': outcome_type,
                    'pattern': pattern,
                    'context': context.strip(),
                    'confidence': 0.8 if 'IS' in pattern else 0.6
                })

    # Determine final outcome
    if outcome_scores['partial'] > 0:
        final_outcome = 'partial'
        confidence = 0.9
    elif outcome_scores['granted'] > outcome_scores['denied']:
        final_outcome = 'granted'
        confidence = min(0.9, 0.6 + (outcome_scores['granted'] * 0.1))
    elif outcome_scores['denied'] > outcome_scores['granted']:
        final_outcome = 'denied'
        confidence = min(0.9, 0.6 + (outcome_scores['denied'] * 0.1))
    else:
        final_outcome = 'unclear'
        confidence = 0.3

    return {
        'outcome': final_outcome,
        'confidence': confidence,
        'scores': outcome_scores,
        'contexts': contexts,
        'disposition_text': contexts[0]['context'] if contexts else None
    }


def find_pattern(pattern: str, text: str) -> Dict:
    """Find pattern in text and return match details."""
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return {'found': False, 'count': 0, 'contexts': []}

    contexts = []
    for match in matches:
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end]
        contexts.append(context.strip())

    return {
        'found': True,
        'count': len(matches),
        'contexts': contexts,
        'weight': len(matches) * 0.1  # Simple weight based on frequency
    }


def extract_intel_factors(text: str) -> Dict:
    """Extract Intel factors using hybrid detection."""

    # Exact pattern matching for explicit factor mentions
    explicit_patterns = {
        'factor_1': [
            r'first\s+(?:Intel\s+)?factor',
            r'factor\s+one',
            r'first\s+discretionary\s+factor'
        ],
        'factor_2': [
            r'second\s+(?:Intel\s+)?factor',
            r'factor\s+two',
            r'second\s+discretionary\s+factor'
        ],
        'factor_3': [
            r'third\s+(?:Intel\s+)?factor',
            r'factor\s+three',
            r'third\s+discretionary\s+factor'
        ],
        'factor_4': [
            r'fourth\s+(?:Intel\s+)?factor',
            r'factor\s+four',
            r'fourth\s+discretionary\s+factor'
        ]
    }

    # Semantic patterns for factor content
    semantic_patterns = {
        'foreign_participant': [
            r'foreign\s+(?:participant|party)',
            r'person\s+or\s+entity\s+in\s+foreign\s+proceeding',
            r'foreign\s+tribunal\s+participant'
        ],
        'receptivity': [
            r'receptiv(?:ity|e)',
            r'foreign\s+tribunal\s+receptiv',
            r'willingness\s+to\s+consider'
        ],
        'circumvention': [
            r'circumvent(?:ion|ing)',
            r'evasion\s+of\s+foreign\s+proof',
            r'bypass(?:ing)?\s+foreign\s+discovery'
        ],
        'unduly_intrusive': [
            r'unduly\s+intrusive',
            r'excessive\s+burden',
            r'unreasonable\s+intrusion',
            r'overly\s+broad'
        ]
    }

    # Extract explicit mentions
    explicit_results = {}
    for factor_name, patterns in explicit_patterns.items():
        best_match = {'found': False, 'count': 0, 'contexts': [], 'weight': 0}
        for pattern in patterns:
            match_result = find_pattern(pattern, text)
            if match_result['found'] and match_result['count'] > best_match['count']:
                best_match = match_result
        explicit_results[factor_name] = best_match

    # Extract semantic mentions
    semantic_results = {}
    for factor_name, patterns in semantic_patterns.items():
        best_match = {'found': False, 'count': 0, 'contexts': [], 'weight': 0}
        for pattern in patterns:
            match_result = find_pattern(pattern, text)
            if match_result['found'] and match_result['count'] > best_match['count']:
                best_match = match_result
        semantic_results[factor_name] = best_match

    # Combine results
    combined_results = {}
    for factor_name in explicit_patterns.keys():
        explicit = explicit_results[factor_name]
        semantic_key = factor_name.replace('factor_', '').replace('1', 'foreign_participant').replace('2', 'receptivity').replace('3', 'circumvention').replace('4', 'unduly_intrusive')
        semantic = semantic_results.get(semantic_key, {'found': False, 'count': 0, 'contexts': [], 'weight': 0})

        combined_results[factor_name] = {
            'explicit': explicit,
            'semantic': semantic,
            'detected': explicit['found'] or semantic['found'],
            'detection_method': 'explicit' if explicit['found'] else 'semantic' if semantic['found'] else 'none',
            'total_weight': explicit['weight'] + semantic['weight'],
            'all_contexts': explicit['contexts'] + semantic['contexts']
        }

    return combined_results


def extract_statutory_prereqs(text: str) -> Dict:
    """Extract statutory prerequisites from text."""

    prereq_patterns = {
        'foreign_tribunal': [
            r'foreign\s+(?:tribunal|proceeding|court)',
            r'international\s+(?:tribunal|proceeding)',
            r'foreign\s+judicial\s+proceeding'
        ],
        'interested_person': [
            r'interested\s+person',
            r'person\s+interested',
            r'party\s+to\s+foreign\s+proceeding'
        ],
        'resides_in_district': [
            r'resides?\s+(?:in|within)\s+(?:this\s+)?district',
            r'found\s+in\s+(?:this\s+)?district',
            r'located\s+in\s+(?:this\s+)?district'
        ],
        'for_use_in_foreign': [
            r'for\s+use\s+in\s+(?:a\s+)?foreign',
            r'use\s+in\s+foreign\s+proceeding',
            r'assistance\s+for\s+foreign'
        ]
    }

    results = {}
    for prereq_name, patterns in prereq_patterns.items():
        best_match = {'found': False, 'count': 0, 'contexts': [], 'satisfied': False}

        for pattern in patterns:
            match_result = find_pattern(pattern, text)
            if match_result['found']:
                best_match['found'] = True
                best_match['count'] += match_result['count']
                best_match['contexts'].extend(match_result['contexts'])

        # Determine if prerequisite is satisfied (simple heuristic)
        if best_match['found']:
            # Look for negative indicators
            negative_indicators = ['not', 'lacks', 'fails', 'insufficient', 'missing']
            satisfied = True

            for context in best_match['contexts']:
                context_lower = context.lower()
                for neg in negative_indicators:
                    if neg in context_lower and prereq_name.replace('_', ' ') in context_lower:
                        satisfied = False
                        break
                if not satisfied:
                    break

            best_match['satisfied'] = satisfied

        results[prereq_name] = best_match

    return results


def extract_context_window(text: str, search_term: str, window: int = 100) -> str:
    """Extract context window around search term."""
    search_lower = search_term.lower()
    text_lower = text.lower()

    start_idx = text_lower.find(search_lower)
    if start_idx == -1:
        return ""

    context_start = max(0, start_idx - window)
    context_end = min(len(text), start_idx + len(search_term) + window)

    return text[context_start:context_end].strip()


def extract_citations(text: str) -> List[Dict]:
    """Extract citations to key precedents."""

    key_cases = [
        'Intel Corp',
        'Intel Corporation',
        'Brandi-Dohrn',
        'Euromepa',
        'ZF Automotive',
        'AlixPartners',
        'Macquarie',
        'del Valle Ruiz',
        'ZF Automotive U.S.'
    ]

    citations = []

    for case in key_cases:
        if case.lower() in text.lower():
            context = extract_context_window(text, case, window=150)

            # Determine if citation is favorable (simple heuristic)
            favorable_indicators = ['supports', 'favors', 'consistent', 'applies', 'follows']
            unfavorable_indicators = ['distinguishable', 'inapplicable', 'rejects', 'declines']

            context_lower = context.lower()
            favorable_score = sum(1 for indicator in favorable_indicators if indicator in context_lower)
            unfavorable_score = sum(1 for indicator in unfavorable_indicators if indicator in context_lower)

            favorable = favorable_score > unfavorable_score if (favorable_score + unfavorable_score) > 0 else None

            citations.append({
                'case_name': case,
                'context': context,
                'mentioned': True,
                'favorable': favorable,
                'confidence': min(1.0, (favorable_score + unfavorable_score) * 0.2)
            })

    return citations


def extract_judge_info(text: str) -> Dict:
    """Extract judge information from text."""

    # Common judge title patterns
    judge_patterns = [
        r'Judge\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Honorable\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Hon\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'Magistrate\s+Judge\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
    ]

    judges = []
    for pattern in judge_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            judge_name = match.group(1).strip()
            if judge_name not in judges:
                judges.append(judge_name)

    return {
        'judges': judges,
        'primary_judge': judges[0] if judges else None
    }


def extract_date_info(text: str) -> Dict:
    """Extract date information from text."""

    # Date patterns
    date_patterns = [
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}'
    ]

    dates = []
    for pattern in date_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            date_str = match.group(0).strip()
            if date_str not in dates:
                dates.append(date_str)

    return {
        'dates_found': dates,
        'decision_date': dates[-1] if dates else None  # Assume last date is decision date
    }


def analyze_case_comprehensive(pdf_path: Path, case_info: Dict) -> Dict:
    """Perform comprehensive analysis of a single case."""

    logger.info(f"Analyzing case: {case_info['case_name']}")

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.error(f"Could not extract text from {pdf_path.name}")
        return None

    # Extract all features
    structural = extract_structural_features(text, pdf_path)
    outcome = extract_outcome(text)
    intel_factors = extract_intel_factors(text)
    statutory_prereqs = extract_statutory_prereqs(text)
    citations = extract_citations(text)
    judge_info = extract_judge_info(text)
    date_info = extract_date_info(text)

    # Combine results
    analysis = {
        'cluster_id': case_info.get('cluster_id'),
        'case_name': case_info.get('case_name'),
        'pdf_path': str(pdf_path),
        'json_path': case_info.get('json_path'),
        'text_length': len(text),
        'structural': structural,
        'outcome': outcome,
        'intel_factors': intel_factors,
        'statutory_prereqs': statutory_prereqs,
        'citations': citations,
        'judge_info': judge_info,
        'date_info': date_info,
        'analysis_timestamp': str(Path().cwd())
    }

    logger.info(f"Analysis complete for {case_info['case_name']}:")
    logger.info(f"  - Outcome: {outcome['outcome']} (confidence: {outcome['confidence']:.2f})")
    logger.info(f"  - Intel factors detected: {sum(1 for f in intel_factors.values() if f['detected'])}")
    logger.info(f"  - Citations: {len(citations)}")
    logger.info(f"  - Pages: {structural['page_count']}")

    return analysis


def main():
    """Test the enhanced extractor on a single PDF."""
    print("Testing Enhanced §1782 Pattern Extractor...")

    # Test on one PDF
    test_pdf = Path("data/case_law/pdfs/1782_discovery/10370545_In_Re_Ex_Parte_Application_of_Gregory_Gliner.pdf")

    if not test_pdf.exists():
        logger.error(f"Test PDF not found: {test_pdf}")
        return

    case_info = {
        'cluster_id': '10370545',
        'case_name': 'In Re Ex Parte Application of Gregory Gliner',
        'json_path': 'data/case_law/The Art of War - Database/10370545_In_Re_Ex_Parte_Application_of_Gregory_Gliner.json'
    }

    analysis = analyze_case_comprehensive(test_pdf, case_info)

    if analysis:
        # Save test results
        output_file = Path("data/case_law/test_extraction_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Test extraction complete!")
        print(f"Results saved to: {output_file}")
        print(f"\nQuick Summary:")
        print(f"- Outcome: {analysis['outcome']['outcome']}")
        print(f"- Intel factors: {sum(1 for f in analysis['intel_factors'].values() if f['detected'])}/4")
        print(f"- Citations: {len(analysis['citations'])}")
        print(f"- Pages: {analysis['structural']['page_count']}")
    else:
        print("[ERROR] Test extraction failed")


if __name__ == "__main__":
    main()
