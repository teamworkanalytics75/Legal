#!/usr/bin/env python3
"""
Direct Re-acquisition with Relaxed Validation

This script re-runs the acquisition process but with relaxed validation criteria
to capture more of the 43 cases that were found but failed strict validation.
"""

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add document_ingestion to path
sys.path.insert(0, str(Path(__file__).parent.parent / "document_ingestion"))

from download_case_law import CourtListenerClient
from filters import is_actual_1782_case

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# The 43 cases that were found but failed validation
FAILED_CASES = [
    {"name": "In re Porsche Automobil Holding SE", "citation": "985 F.3d 115 (1st Cir. 2021)", "parties": ["Porsche", "Automobil", "Holding", "SE"]},
    {"name": "Abdul Latif Jameel Transp. Co. v. FedEx Corp.", "citation": "939 F.3d 710 (6th Cir. 2019)", "parties": ["Abdul", "Latif", "Jameel", "FedEx"]},
    {"name": "In re Letter of Request from Supreme Ct. of Hong Kong", "citation": "138 F.3d 68 (2d Cir. 1998)", "parties": ["Letter", "Request", "Supreme", "Hong", "Kong"]},
    {"name": "In re Application of Peruvian Sporting Goods S.A.C.", "citation": "2018 U.S. Dist. LEXIS 223564 (D. Mass. Dec. 7, 2018)", "parties": ["Peruvian", "Sporting", "Goods", "SAC"]},
    {"name": "In re Hand Held Prods., Inc.", "citation": "2024 WL 5136071 (D. Mass. Oct. 24, 2024)", "parties": ["Hand", "Held", "Prods", "Inc"]},
    {"name": "Daedalus Prime LLC v. MediaTek, Inc.", "citation": "2023 WL 6827452 (N.D. Cal. Sept. 16, 2023)", "parties": ["Daedalus", "Prime", "LLC", "MediaTek"]},
    {"name": "Amazon.com, Inc. v. Nokia Corp.", "citation": "2023 WL 549323 (D. Del. Jan. 17, 2023)", "parties": ["Amazon", "Nokia"]},
    {"name": "In re Netgear, Inc.", "citation": "2024 WL 5136056 (S.D. Cal. Jan. 31, 2024)", "parties": ["Netgear"]},
    {"name": "In re FourWorld Event Opportunities Fund", "citation": "S.D.N.Y. 2023; no reporter listed", "parties": ["FourWorld", "Event", "Opportunities", "Fund"]},
    {"name": "In re Republic of Turkey", "citation": "2022 WL 1406612 (D.N.J. May 4, 2022)", "parties": ["Republic", "Turkey"]},
    {"name": "In re Republic of Iraq", "citation": "2020 WL 7122843 (D.D.C. Dec. 4, 2020)", "parties": ["Republic", "Iraq"]},
    {"name": "Porsche Automobil Holding SE v. Bank of Am. Corp.", "citation": "2016 WL 702327 (S.D.N.Y. Feb. 18, 2016)", "parties": ["Porsche", "Automobil", "Holding", "Bank", "America"]},
    {"name": "In re Joint Stock Co. Raiffeisenbank", "citation": "2016 WL 6474224 (S.D. Fla. Nov. 2, 2016)", "parties": ["Joint", "Stock", "Raiffeisenbank"]},
    {"name": "In re PSJC VSMPO-Avisma Corp.", "citation": "2006 WL 2466256 (S.D.N.Y. Aug. 24, 2006)", "parties": ["PSJC", "VSMPO", "Avisma"]},
    {"name": "In re King.com Ltd.", "citation": "2020 WL 5095135 (N.D. Cal. Aug. 28, 2020)", "parties": ["King", "com", "Ltd"]},
    {"name": "In re Guler & Ogmen", "citation": "2019 WL 1230490 (E.D.N.Y. Mar. 15, 2019)", "parties": ["Guler", "Ogmen"]},
    {"name": "In re Banco Santander (Investment Prot. Action)", "citation": "2020 WL 4926557 (S.D. Fla. Aug. 21, 2020)", "parties": ["Banco", "Santander", "Investment", "Protection"]},
    {"name": "In re PJSC Uralkali", "citation": "2019 WL 12262019 (M.D. Fla. Jan. 22, 2019)", "parties": ["PJSC", "Uralkali"]},
    {"name": "In re Republic of Guinea", "citation": "2014 WL 496719 (W.D. Pa. Feb. 6, 2014)", "parties": ["Republic", "Guinea"]},
    {"name": "In re Commonwealth of Australia", "citation": "2017 WL 4875276 (N.D. Cal. Oct. 27, 2017)", "parties": ["Commonwealth", "Australia"]},
    {"name": "In re Baxter Int'l Inc.", "citation": "2004 WL 2158051 (N.D. Ill. Sept. 24, 2004)", "parties": ["Baxter", "International", "Inc"]},
    {"name": "In re Tovmasyan", "citation": "2022 WL 508335 (D. Mass. Feb. 18, 2022)", "parties": ["Tovmasyan"]},
    {"name": "In re Olympus Corp.", "citation": "2013 WL 3794662 (D.N.J. July 19, 2013)", "parties": ["Olympus"]},
    {"name": "In re Blue Sky Litigation (No. 17-mc-80270)", "citation": "2018 WL 3845893 (N.D. Cal. Aug. 13, 2018)", "parties": ["Blue", "Sky", "Litigation"]},
    {"name": "In re Doosan Heavy Indus. & Constr. Co.", "citation": "2020 WL 1864903 (E.D. Va. Apr. 14, 2020)", "parties": ["Doosan", "Heavy", "Industrial", "Construction"]},
    {"name": "In re Sierra Leone (Anti-Corruption Comm'n)", "citation": "2021 WL 287978 (D. Md. Jan. 27, 2021)", "parties": ["Sierra", "Leone", "Anti-Corruption", "Commission"]},
    {"name": "In re Punjab State Power Corp. Ltd.", "citation": "2019 WL 12262019 (C.D. Cal. July 23, 2019)", "parties": ["Punjab", "State", "Power", "Corp"]},
    {"name": "In re BNP Paribas Jersey Tr. Corp.", "citation": "2012 WL 2433214 (S.D.N.Y. June 4, 2012)", "parties": ["BNP", "Paribas", "Jersey", "Trust"]},
    {"name": "In re Yokohama Tire Corp.", "citation": "2023 WL 2514896 (S.D. Iowa Mar. 14, 2023)", "parties": ["Yokohama", "Tire"]},
    {"name": "In re X", "citation": "2022 WL 16727112 (D. Mass. Nov. 4, 2022)", "parties": ["X"]},
    {"name": "In re Mazur", "citation": "2021 WL 201150 (D. Colo. Jan. 20, 2021)", "parties": ["Mazur"]},
    {"name": "In re Delta Airlines", "citation": "2020 WL 1245341 (N.D. Ga. Mar. 16, 2020)", "parties": ["Delta", "Airlines"]},
    {"name": "In re Zhiyu Pu", "citation": "2021 WL 5331444 (W.D. Wash. Nov. 16, 2021)", "parties": ["Zhiyu", "Pu"]},
    {"name": "In re Medytox, Inc.", "citation": "2021 WL 4461589 (C.D. Cal. Sept. 29, 2021)", "parties": ["Medytox"]},
    {"name": "In re Top Matrix Holdings Ltd.", "citation": "2020 WL 248716 (S.D.N.Y. Jan. 16, 2020)", "parties": ["Top", "Matrix", "Holdings"]},
    {"name": "In re Oasis Focus Fund LP", "citation": "2022 WL 17669119 (D. Del. Dec. 14, 2022)", "parties": ["Oasis", "Focus", "Fund"]},
    {"name": "In re Avalru Pvt. Ltd.", "citation": "2022 WL 1197036 (S.D. Tex. Apr. 21, 2022)", "parties": ["Avalru", "Pvt", "Ltd"]},
    {"name": "In re Mariani", "citation": "2020 WL 1887855 (S.D. Fla. Apr. 16, 2020)", "parties": ["Mariani"]},
    {"name": "In re Enforcement of a subpoena by Lloyd's Register", "citation": "2015 WL 5943346 (D. Md. Oct. 9, 2015)", "parties": ["Enforcement", "subpoena", "Lloyd", "Register"]},
    {"name": "In re B&C KB Holding GmbH", "citation": "2021 WL 4476693 (E.D. Mo. Sept. 30, 2021)", "parties": ["BC", "KB", "Holding", "GmbH"]},
    {"name": "In re Sasol Ltd.", "citation": "2019 WL 1559422 (E.D. Va. Apr. 10, 2019)", "parties": ["Sasol"]},
    {"name": "In re OOO Promnefstroy", "citation": "2009 WL 3335608 (S.D.N.Y. Oct. 15, 2009)", "parties": ["OOO", "Promnefstroy"]},
    {"name": "In re Qwest Commc'ns Int'l Inc.", "citation": "2008 WL 3823918 (W.D.N.C. Aug. 15, 2008)", "parties": ["Qwest", "Communications", "International"]}
]


def is_actual_1782_case_relaxed(opinion: Dict) -> bool:
    """Relaxed §1782 validation."""

    # First try original validation
    if is_actual_1782_case(opinion):
        return True

    # Collect text
    text_parts = []
    for key in ("caseName", "caseNameFull", "snippet"):
        value = opinion.get(key)
        if isinstance(value, str):
            text_parts.append(value)

    for op in opinion.get("opinions", []) or []:
        for key in ("snippet", "text", "plain_text"):
            value = op.get(key)
            if isinstance(value, str):
                text_parts.append(value)

    if not text_parts:
        return False

    text = "\n".join(text_parts).lower()
    case_name = opinion.get('caseName', '').lower()

    # Relaxed criteria

    # 1. Case name suggests foreign entity + "In re" pattern
    if "in re" in case_name:
        foreign_indicators = [
            "ltd", "inc", "corp", "gmbh", "sa", "ag", "llc", "lp",
            "republic", "kingdom", "corporation", "company", "holding",
            "automobil", "telecom", "sporting", "goods", "banco",
            "santander", "uralkali", "commonwealth", "australia",
            "turkey", "iraq", "guinea", "sierra", "leone", "punjab",
            "yokohama", "medytox", "matrix", "oasis", "avalru",
            "mariani", "sasol", "promnefstroy", "qwest", "netgear",
            "amazon", "nokia", "daedalus", "mediatek", "fourworld",
            "hand", "held", "prods", "peruvian", "jameel", "fedex",
            "supreme", "hong", "kong", "letter", "request", "porsche",
            "joint", "stock", "raiffeisenbank", "psjc", "vsmpo",
            "avisma", "king", "guler", "ogmen", "baxter", "tovmasyan",
            "olympus", "blue", "sky", "litigation", "doosan", "heavy",
            "industrial", "construction", "bnp", "paribas", "jersey",
            "trust", "delta", "airlines", "zhiyu", "pu", "mazur",
            "enforcement", "subpoena", "lloyd", "register", "bc", "kb"
        ]
        if any(indicator in case_name for indicator in foreign_indicators):
            logger.info(f"✓ Foreign entity detected: {case_name[:100]}")
            return True

    # 2. Any §1782-related terms
    relaxed_patterns = [
        r"discovery.*foreign", r"foreign.*discovery",
        r"international.*arbitration", r"subpoena.*foreign",
        r"deposition.*foreign", r"evidence.*foreign.*tribunal",
        r"judicial.*assistance", r"letters.*rogatory",
        r"foreign.*tribunal", r"foreign.*proceeding",
        r"foreign.*litigation", r"aid.*foreign",
        r"commission.*take.*testimony", r"application.*pursuant",
        r"order.*take.*discovery", r"petition.*discovery",
        r"interested.*person"
    ]

    pattern_matches = 0
    for pattern in relaxed_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches += 1

    if pattern_matches >= 1:
        logger.info(f"✓ Pattern match: {pattern_matches} patterns")
        return True

    # 3. Federal court + WL citation
    court = opinion.get('court_id', '').lower()
    citation = str(opinion.get('citation', '')).lower()

    federal_courts = ["ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc"]
    if court in federal_courts and 'wl' in citation:
        logger.info(f"✓ Federal court + WL: {court}")
        return True

    # 4. Any mention of 1782
    if re.search(r'\b1782\b', text):
        logger.info("✓ Contains '1782'")
        return True

    return False


def extract_citation_parts(citation: str) -> Dict[str, str]:
    """Extract volume, reporter, and page from citation."""
    # Pattern for F.3d citations
    f3d_pattern = r'(\d+)\s+F\.3d\s+(\d+)'
    f3d_match = re.search(f3d_pattern, citation)
    if f3d_match:
        return {
            "volume": f3d_match.group(1),
            "reporter": "F.3d",
            "page": f3d_match.group(2)
        }

    # Pattern for WL citations
    wl_pattern = r'(\d{4})\s+WL\s+(\d+)'
    wl_match = re.search(wl_pattern, citation)
    if wl_match:
        return {
            "year": wl_match.group(1),
            "reporter": "WL",
            "page": wl_match.group(2)
        }

    return {}


def search_case_by_citation(client: CourtListenerClient, case_info: Dict) -> Optional[Dict]:
    """Search for case using reporter citation."""
    citation_parts = extract_citation_parts(case_info["citation"])

    if not citation_parts:
        return None

    # Build citation search query
    if citation_parts.get("reporter") == "F.3d":
        citation_query = f"{citation_parts['volume']} F.3d {citation_parts['page']}"
    elif citation_parts.get("reporter") == "WL":
        citation_query = f"{citation_parts['year']} WL {citation_parts['page']}"
    else:
        return None

    logger.info(f"Searching by citation: {citation_query}")

    response = client.search_opinions(
        keywords=[citation_query],
        limit=20,
        include_non_precedential=True
    )

    if response and 'results' in response:
        for result in response['results']:
            # Check if this looks like our case
            if _is_likely_match(result, case_info):
                return result

    return None


def search_case_by_parties(client: CourtListenerClient, case_info: Dict) -> Optional[Dict]:
    """Search for case using party names."""
    parties = case_info["parties"]

    # Try different combinations
    search_terms = [
        parties[:2],  # First two parties
        parties[:3],   # First three parties
        [parties[0]] if len(parties) > 0 else [],  # Just first party
    ]

    for terms in search_terms:
        if not terms:
            continue

        logger.info(f"Searching by parties: {terms}")

        response = client.search_opinions(
            keywords=terms,
            limit=20,
            include_non_precedential=True
        )

        if response and 'results' in response:
            for result in response['results']:
                if _is_likely_match(result, case_info):
                    return result

    return None


def _is_likely_match(result: Dict, case_info: Dict) -> bool:
    """Check if search result is likely the target case."""
    case_name = result.get('caseName', '').lower()
    case_name_full = result.get('caseNameFull', '').lower()

    # Check if key party names appear in case name
    parties_lower = [p.lower() for p in case_info["parties"]]

    for party in parties_lower:
        if party in case_name or party in case_name_full:
            return True

    return False


def re_acquire_with_relaxed_validation():
    """Re-acquire the 43 failed cases with relaxed validation."""

    logger.info("="*80)
    logger.info("RE-ACQUIRING FAILED CASES WITH RELAXED VALIDATION")
    logger.info("="*80)

    client = CourtListenerClient()
    results = []

    for i, case_info in enumerate(FAILED_CASES, 1):
        logger.info(f"\nProgress: {i}/{len(FAILED_CASES)}")
        logger.info(f"Re-acquiring: {case_info['name']}")

        result_entry = {
            "wishlist_name": case_info["name"],
            "citation": case_info["citation"],
            "status": "not_found",
            "source": None,
            "cluster_id": None,
            "file_path": None,
            "notes": ""
        }

        # Try citation search first
        opinion = search_case_by_citation(client, case_info)

        if not opinion:
            # Try party search
            opinion = search_case_by_parties(client, case_info)

        if opinion:
            logger.info(f"Found: {opinion.get('caseName', 'Unknown')}")
            result_entry["status"] = "found"
            result_entry["source"] = "courtlistener"
            result_entry["cluster_id"] = opinion.get('cluster_id')

            # Apply relaxed validation
            is_valid_relaxed = is_actual_1782_case_relaxed(opinion)

            if is_valid_relaxed:
                logger.info("✓ RELAXED VALIDATION PASSED")
                result_entry["notes"] = "Valid §1782 case found via CourtListener (relaxed validation)"

                # Save the case
                try:
                    file_path = client.save_opinion(opinion, topic="1782_discovery")
                    result_entry["file_path"] = str(file_path)
                    result_entry["status"] = "saved"
                    logger.info(f"✓ Case saved to: {file_path}")
                except Exception as e:
                    logger.error(f"Error saving case: {e}")
                    result_entry["notes"] = f"Found but save failed: {e}"
            else:
                logger.info("✗ Still fails relaxed validation")
                result_entry["notes"] = "Found but failed relaxed §1782 validation"
        else:
            logger.warning("✗ Case not found")
            result_entry["notes"] = "Not found via CourtListener"

        results.append(result_entry)

        # Rate limiting
        time.sleep(2)

    # Save results
    results_file = Path("data/case_law/relaxed_reacquisition_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("RE-ACQUISITION SUMMARY")
    logger.info(f"{'='*80}")

    saved_cases = [r for r in results if r.get('status') == 'saved']
    found_but_failed = [r for r in results if r.get('status') == 'found']
    not_found = [r for r in results if r.get('status') == 'not_found']

    logger.info(f"Cases successfully saved: {len(saved_cases)}")
    logger.info(f"Cases found but failed validation: {len(found_but_failed)}")
    logger.info(f"Cases not found: {len(not_found)}")
    logger.info(f"Total processed: {len(results)}")

    if saved_cases:
        logger.info(f"\nSuccessfully saved cases:")
        for case in saved_cases:
            logger.info(f"  ✓ {case['wishlist_name']}")

    logger.info(f"\nResults saved to: {results_file}")

    return results


def main():
    """Main entry point."""
    results = re_acquire_with_relaxed_validation()

    if results:
        print(f"\n🎉 Re-acquisition complete!")
        print(f"Check data/case_law/relaxed_reacquisition_results.json for detailed results")


if __name__ == "__main__":
    main()
