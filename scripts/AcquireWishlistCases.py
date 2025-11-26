#!/usr/bin/env python3
"""
Acquire 45 Remaining §1782 Wishlist Cases

This script systematically searches for and acquires 45 missing §1782 cases
from the wishlist using CourtListener API, RECAP documents, and manual retrieval tracking.
"""

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

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

# The 45 remaining wishlist cases
WISHLIST_CASES = [
    {
        "name": "In re Porsche Automobil Holding SE",
        "citation": "985 F.3d 115 (1st Cir. 2021)",
        "parties": ["Porsche", "Automobil", "Holding", "SE"]
    },
    {
        "name": "Abdul Latif Jameel Transp. Co. v. FedEx Corp.",
        "citation": "939 F.3d 710 (6th Cir. 2019)",
        "parties": ["Abdul", "Latif", "Jameel", "FedEx"]
    },
    {
        "name": "Consorcio Ecuatoriano de Telecom. S.A. v. JAS Forwarding (USA), Inc.",
        "citation": "747 F.3d 1262 (11th Cir. 2014)",
        "parties": ["Consorcio", "Ecuatoriano", "Telecom", "JAS", "Forwarding"]
    },
    {
        "name": "In re Letter of Request from Supreme Ct. of Hong Kong",
        "citation": "138 F.3d 68 (2d Cir. 1998)",
        "parties": ["Letter", "Request", "Supreme", "Hong", "Kong"]
    },
    {
        "name": "In re Application of Peruvian Sporting Goods S.A.C.",
        "citation": "2018 U.S. Dist. LEXIS 223564 (D. Mass. Dec. 7, 2018)",
        "parties": ["Peruvian", "Sporting", "Goods", "SAC"]
    },
    {
        "name": "In re Hand Held Prods., Inc.",
        "citation": "2024 WL 5136071 (D. Mass. Oct. 24, 2024)",
        "parties": ["Hand", "Held", "Prods", "Inc"]
    },
    {
        "name": "Daedalus Prime LLC v. MediaTek, Inc.",
        "citation": "2023 WL 6827452 (N.D. Cal. Sept. 16, 2023)",
        "parties": ["Daedalus", "Prime", "LLC", "MediaTek"]
    },
    {
        "name": "Amazon.com, Inc. v. Nokia Corp.",
        "citation": "2023 WL 549323 (D. Del. Jan. 17, 2023)",
        "parties": ["Amazon", "Nokia"]
    },
    {
        "name": "In re Netgear, Inc.",
        "citation": "2024 WL 5136056 (S.D. Cal. Jan. 31, 2024)",
        "parties": ["Netgear"]
    },
    {
        "name": "In re FourWorld Event Opportunities Fund",
        "citation": "S.D.N.Y. 2023; no reporter listed",
        "parties": ["FourWorld", "Event", "Opportunities", "Fund"]
    },
    {
        "name": "In re Republic of Turkey",
        "citation": "2022 WL 1406612 (D.N.J. May 4, 2022)",
        "parties": ["Republic", "Turkey"]
    },
    {
        "name": "In re Republic of Iraq",
        "citation": "2020 WL 7122843 (D.D.C. Dec. 4, 2020)",
        "parties": ["Republic", "Iraq"]
    },
    {
        "name": "Porsche Automobil Holding SE v. Bank of Am. Corp.",
        "citation": "2016 WL 702327 (S.D.N.Y. Feb. 18, 2016)",
        "parties": ["Porsche", "Automobil", "Holding", "Bank", "America"]
    },
    {
        "name": "In re Joint Stock Co. Raiffeisenbank",
        "citation": "2016 WL 6474224 (S.D. Fla. Nov. 2, 2016)",
        "parties": ["Joint", "Stock", "Raiffeisenbank"]
    },
    {
        "name": "In re PSJC VSMPO-Avisma Corp.",
        "citation": "2006 WL 2466256 (S.D.N.Y. Aug. 24, 2006)",
        "parties": ["PSJC", "VSMPO", "Avisma"]
    },
    {
        "name": "In re King.com Ltd.",
        "citation": "2020 WL 5095135 (N.D. Cal. Aug. 28, 2020)",
        "parties": ["King", "com", "Ltd"]
    },
    {
        "name": "In re Guler & Ogmen",
        "citation": "2019 WL 1230490 (E.D.N.Y. Mar. 15, 2019)",
        "parties": ["Guler", "Ogmen"]
    },
    {
        "name": "In re Banco Santander (Investment Prot. Action)",
        "citation": "2020 WL 4926557 (S.D. Fla. Aug. 21, 2020)",
        "parties": ["Banco", "Santander", "Investment", "Protection"]
    },
    {
        "name": "In re PJSC Uralkali",
        "citation": "2019 WL 12262019 (M.D. Fla. Jan. 22, 2019)",
        "parties": ["PJSC", "Uralkali"]
    },
    {
        "name": "In re Republic of Guinea",
        "citation": "2014 WL 496719 (W.D. Pa. Feb. 6, 2014)",
        "parties": ["Republic", "Guinea"]
    },
    {
        "name": "In re Commonwealth of Australia",
        "citation": "2017 WL 4875276 (N.D. Cal. Oct. 27, 2017)",
        "parties": ["Commonwealth", "Australia"]
    },
    {
        "name": "In re Baxter Int'l Inc.",
        "citation": "2004 WL 2158051 (N.D. Ill. Sept. 24, 2004)",
        "parties": ["Baxter", "International", "Inc"]
    },
    {
        "name": "In re Tovmasyan",
        "citation": "2022 WL 508335 (D. Mass. Feb. 18, 2022)",
        "parties": ["Tovmasyan"]
    },
    {
        "name": "In re Olympus Corp.",
        "citation": "2013 WL 3794662 (D.N.J. July 19, 2013)",
        "parties": ["Olympus"]
    },
    {
        "name": "In re Madhya Pradesh v. Getit Infoservices",
        "citation": "2020 WL 7695053 (C.D. Cal. Nov. 30, 2020)",
        "parties": ["Madhya", "Pradesh", "Getit", "Infoservices"]
    },
    {
        "name": "In re Blue Sky Litigation (No. 17-mc-80270)",
        "citation": "2018 WL 3845893 (N.D. Cal. Aug. 13, 2018)",
        "parties": ["Blue", "Sky", "Litigation"]
    },
    {
        "name": "In re Doosan Heavy Indus. & Constr. Co.",
        "citation": "2020 WL 1864903 (E.D. Va. Apr. 14, 2020)",
        "parties": ["Doosan", "Heavy", "Industrial", "Construction"]
    },
    {
        "name": "In re Sierra Leone (Anti-Corruption Comm'n)",
        "citation": "2021 WL 287978 (D. Md. Jan. 27, 2021)",
        "parties": ["Sierra", "Leone", "Anti-Corruption", "Commission"]
    },
    {
        "name": "In re Punjab State Power Corp. Ltd.",
        "citation": "2019 WL 12262019 (C.D. Cal. July 23, 2019)",
        "parties": ["Punjab", "State", "Power", "Corp"]
    },
    {
        "name": "In re BNP Paribas Jersey Tr. Corp.",
        "citation": "2012 WL 2433214 (S.D.N.Y. June 4, 2012)",
        "parties": ["BNP", "Paribas", "Jersey", "Trust"]
    },
    {
        "name": "In re Yokohama Tire Corp.",
        "citation": "2023 WL 2514896 (S.D. Iowa Mar. 14, 2023)",
        "parties": ["Yokohama", "Tire"]
    },
    {
        "name": "In re X",
        "citation": "2022 WL 16727112 (D. Mass. Nov. 4, 2022)",
        "parties": ["X"]
    },
    {
        "name": "In re Mazur",
        "citation": "2021 WL 201150 (D. Colo. Jan. 20, 2021)",
        "parties": ["Mazur"]
    },
    {
        "name": "In re Delta Airlines",
        "citation": "2020 WL 1245341 (N.D. Ga. Mar. 16, 2020)",
        "parties": ["Delta", "Airlines"]
    },
    {
        "name": "In re Zhiyu Pu",
        "citation": "2021 WL 5331444 (W.D. Wash. Nov. 16, 2021)",
        "parties": ["Zhiyu", "Pu"]
    },
    {
        "name": "In re Medytox, Inc.",
        "citation": "2021 WL 4461589 (C.D. Cal. Sept. 29, 2021)",
        "parties": ["Medytox"]
    },
    {
        "name": "In re Top Matrix Holdings Ltd.",
        "citation": "2020 WL 248716 (S.D.N.Y. Jan. 16, 2020)",
        "parties": ["Top", "Matrix", "Holdings"]
    },
    {
        "name": "In re Oasis Focus Fund LP",
        "citation": "2022 WL 17669119 (D. Del. Dec. 14, 2022)",
        "parties": ["Oasis", "Focus", "Fund"]
    },
    {
        "name": "In re Avalru Pvt. Ltd.",
        "citation": "2022 WL 1197036 (S.D. Tex. Apr. 21, 2022)",
        "parties": ["Avalru", "Pvt", "Ltd"]
    },
    {
        "name": "In re Mariani",
        "citation": "2020 WL 1887855 (S.D. Fla. Apr. 16, 2020)",
        "parties": ["Mariani"]
    },
    {
        "name": "In re Enforcement of a subpoena by Lloyd's Register",
        "citation": "2015 WL 5943346 (D. Md. Oct. 9, 2015)",
        "parties": ["Enforcement", "subpoena", "Lloyd", "Register"]
    },
    {
        "name": "In re B&C KB Holding GmbH",
        "citation": "2021 WL 4476693 (E.D. Mo. Sept. 30, 2021)",
        "parties": ["BC", "KB", "Holding", "GmbH"]
    },
    {
        "name": "In re Sasol Ltd.",
        "citation": "2019 WL 1559422 (E.D. Va. Apr. 10, 2019)",
        "parties": ["Sasol"]
    },
    {
        "name": "In re OOO Promnefstroy",
        "citation": "2009 WL 3335608 (S.D.N.Y. Oct. 15, 2009)",
        "parties": ["OOO", "Promnefstroy"]
    },
    {
        "name": "In re Qwest Commc'ns Int'l Inc.",
        "citation": "2008 WL 3823918 (W.D.N.C. Aug. 15, 2008)",
        "parties": ["Qwest", "Communications", "International"]
    }
]


class WishlistCaseAcquirer:
    """Acquire wishlist cases using CourtListener API and RECAP documents."""

    def __init__(self):
        self.client = CourtListenerClient()
        self.tracking_log = {
            "acquisition_date": datetime.now().isoformat(),
            "total_cases": len(WISHLIST_CASES),
            "found_via_courtlistener": 0,
            "found_via_recap": 0,
            "requires_manual_retrieval": 0,
            "cases": []
        }

    def extract_citation_parts(self, citation: str) -> Dict[str, str]:
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

    def search_case_by_citation(self, case_info: Dict) -> Optional[Dict]:
        """Search for case using reporter citation."""
        citation_parts = self.extract_citation_parts(case_info["citation"])

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

        response = self.client.search_opinions(
            keywords=[citation_query],
            limit=20,
            include_non_precedential=True
        )

        if response and 'results' in response:
            for result in response['results']:
                # Check if this looks like our case
                if self._is_likely_match(result, case_info):
                    return result

        return None

    def search_case_by_parties(self, case_info: Dict) -> Optional[Dict]:
        """Search for case using party names."""
        parties = case_info["parties"]

        # Try different combinations of party names
        search_terms = [
            parties[:2],  # First two parties
            parties[:3],   # First three parties
            [parties[0]] if len(parties) > 0 else [],  # Just first party
        ]

        for terms in search_terms:
            if not terms:
                continue

            logger.info(f"Searching by parties: {terms}")

            response = self.client.search_opinions(
                keywords=terms,
                limit=20,
                include_non_precedential=True
            )

            if response and 'results' in response:
                for result in response['results']:
                    if self._is_likely_match(result, case_info):
                        return result

        return None

    def search_case_combined(self, case_info: Dict) -> Optional[Dict]:
        """Search using both citation and party information."""
        citation_parts = self.extract_citation_parts(case_info["citation"])
        parties = case_info["parties"]

        if not citation_parts or not parties:
            return None

        # Build combined search
        search_terms = []

        if citation_parts.get("reporter") == "F.3d":
            search_terms.append(f"{citation_parts['volume']} F.3d {citation_parts['page']}")
        elif citation_parts.get("reporter") == "WL":
            search_terms.append(f"{citation_parts['year']} WL {citation_parts['page']}")

        # Add first party name
        if parties:
            search_terms.append(parties[0])

        logger.info(f"Searching combined: {search_terms}")

        response = self.client.search_opinions(
            keywords=search_terms,
            limit=20,
            include_non_precedential=True
        )

        if response and 'results' in response:
            for result in response['results']:
                if self._is_likely_match(result, case_info):
                    return result

        return None

    def _is_likely_match(self, result: Dict, case_info: Dict) -> bool:
        """Check if search result is likely the target case."""
        case_name = result.get('caseName', '').lower()
        case_name_full = result.get('caseNameFull', '').lower()

        # Check if key party names appear in case name
        parties_lower = [p.lower() for p in case_info["parties"]]

        for party in parties_lower:
            if party in case_name or party in case_name_full:
                return True

        return False

    def search_for_docket(self, case_info: Dict) -> Optional[Dict]:
        """Search for a docket using case information."""
        try:
            # Search for dockets using case name
            search_terms = [case_info["name"]]

            # Also try with party names
            if case_info["parties"]:
                search_terms.extend(case_info["parties"][:2])

            for term in search_terms:
                if not term:
                    continue

                logger.info(f"Searching for docket: {term}")

                # Search dockets endpoint
                response = self.client._make_request("/dockets/", params={
                    'q': term,
                    'format': 'json',
                    'order_by': 'date_filed desc'
                })

                if response and 'results' in response and response['results']:
                    # Return first docket result
                    return response['results'][0]

        except Exception as e:
            logger.error(f"Error searching for docket: {e}")

        return None

    def fetch_recap_documents(self, docket_id: str) -> Optional[Dict]:
        """Fetch RECAP documents for a docket."""
        try:
            endpoint = f"/dockets/{docket_id}/"
            response = self.client._make_request(endpoint)

            if response and 'recap_documents' in response:
                recap_docs = response['recap_documents']

                # Look for opinion or order documents
                for doc in recap_docs:
                    doc_type = doc.get('document_type', '').lower()
                    if doc_type in ['opinion', 'order', 'memorandum']:
                        logger.info(f"Found RECAP document: {doc.get('document_number', 'unknown')}")
                        return doc

        except Exception as e:
            logger.error(f"Error fetching RECAP documents for docket {docket_id}: {e}")

        return None

    def create_opinion_from_recap(self, recap_doc: Dict, docket_data: Dict) -> Optional[Dict]:
        """Create a mock opinion object from RECAP document data."""
        try:
            # Extract text from RECAP document
            text = recap_doc.get('plain_text', '')
            if not text:
                text = recap_doc.get('ocr_text', '')
            if not text:
                # Try to extract from HTML
                html = recap_doc.get('html', '')
                if html:
                    import re
                    text = re.sub(r'<[^>]+>', '', html)

            if not text or len(text.strip()) < 50:
                logger.warning("RECAP document has insufficient text content")
                return None

            # Create mock opinion object
            mock_opinion = {
                'caseName': docket_data.get('case_name', 'Unknown Case'),
                'citation': docket_data.get('absolute_url', ''),
                'court': docket_data.get('court', ''),
                'dateFiled': docket_data.get('date_filed', ''),
                'text': text,
                'docket_id': docket_data.get('id'),
                'recap_document': recap_doc,
                'source': 'recap'
            }

            return mock_opinion

        except Exception as e:
            logger.error(f"Error creating opinion from RECAP document: {e}")
            return None

    def acquire_case(self, case_info: Dict) -> Dict[str, Any]:
        """Acquire a single case using multiple search strategies."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Acquiring: {case_info['name']}")
        logger.info(f"Citation: {case_info['citation']}")
        logger.info(f"{'='*60}")

        case_result = {
            "wishlist_name": case_info["name"],
            "citation": case_info["citation"],
            "status": "not_found",
            "source": None,
            "cluster_id": None,
            "file_path": None,
            "search_queries_attempted": [],
            "notes": ""
        }

        # Strategy 1: Search by citation
        case_result["search_queries_attempted"].append("citation_search")
        result = self.search_case_by_citation(case_info)

        if result:
            logger.info(f"Found via citation search: {result.get('caseName', 'Unknown')}")
            case_result["status"] = "found"
            case_result["source"] = "courtlistener"
            case_result["cluster_id"] = result.get('cluster_id')

            # Check if it's a valid §1782 case
            if is_actual_1782_case(result):
                logger.info("✓ Confirmed as §1782 case")
                case_result["notes"] = "Valid §1782 case found via CourtListener"

                # Save the case
                try:
                    file_path = self.client.save_opinion(result, topic="1782_discovery")
                    case_result["file_path"] = str(file_path)
                    self.tracking_log["found_via_courtlistener"] += 1
                except Exception as e:
                    logger.error(f"Error saving case: {e}")
                    case_result["notes"] = f"Found but save failed: {e}"
            else:
                logger.warning("✗ Not a valid §1782 case")
                case_result["notes"] = "Found but failed §1782 validation"

            return case_result

        # Strategy 2: Search by parties
        case_result["search_queries_attempted"].append("party_search")
        result = self.search_case_by_parties(case_info)

        if result:
            logger.info(f"Found via party search: {result.get('caseName', 'Unknown')}")
            case_result["status"] = "found"
            case_result["source"] = "courtlistener"
            case_result["cluster_id"] = result.get('cluster_id')

            if is_actual_1782_case(result):
                logger.info("✓ Confirmed as §1782 case")
                case_result["notes"] = "Valid §1782 case found via CourtListener"

                try:
                    file_path = self.client.save_opinion(result, topic="1782_discovery")
                    case_result["file_path"] = str(file_path)
                    self.tracking_log["found_via_courtlistener"] += 1
                except Exception as e:
                    logger.error(f"Error saving case: {e}")
                    case_result["notes"] = f"Found but save failed: {e}"
            else:
                logger.warning("✗ Not a valid §1782 case")
                case_result["notes"] = "Found but failed §1782 validation"

            return case_result

        # Strategy 3: Combined search
        case_result["search_queries_attempted"].append("combined_search")
        result = self.search_case_combined(case_info)

        if result:
            logger.info(f"Found via combined search: {result.get('caseName', 'Unknown')}")
            case_result["status"] = "found"
            case_result["source"] = "courtlistener"
            case_result["cluster_id"] = result.get('cluster_id')

            if is_actual_1782_case(result):
                logger.info("✓ Confirmed as §1782 case")
                case_result["notes"] = "Valid §1782 case found via CourtListener"

                try:
                    file_path = self.client.save_opinion(result, topic="1782_discovery")
                    case_result["file_path"] = str(file_path)
                    self.tracking_log["found_via_courtlistener"] += 1
                except Exception as e:
                    logger.error(f"Error saving case: {e}")
                    case_result["notes"] = f"Found but save failed: {e}"
            else:
                logger.warning("✗ Not a valid §1782 case")
                case_result["notes"] = "Found but failed §1782 validation"

            return case_result

        # Strategy 4: Try RECAP documents for cases with dockets but no opinion text
        case_result["search_queries_attempted"].append("recap_search")

        # Try to find a docket by searching for the case name
        docket_result = self.search_for_docket(case_info)
        if docket_result:
            logger.info(f"Found docket: {docket_result.get('case_name', 'Unknown')}")

            # Try to fetch RECAP documents
            recap_doc = self.fetch_recap_documents(docket_result.get('id'))
            if recap_doc:
                logger.info("✓ Found RECAP document")

                # Create a mock opinion object from RECAP document
                mock_opinion = self.create_opinion_from_recap(recap_doc, docket_result)

                if mock_opinion and is_actual_1782_case(mock_opinion):
                    logger.info("✓ Confirmed as §1782 case via RECAP")
                    case_result["status"] = "found"
                    case_result["source"] = "recap"
                    case_result["cluster_id"] = docket_result.get('id')
                    case_result["notes"] = "Valid §1782 case found via RECAP documents"

                    try:
                        file_path = self.client.save_opinion(mock_opinion, topic="1782_discovery")
                        case_result["file_path"] = str(file_path)
                        self.tracking_log["found_via_recap"] += 1
                    except Exception as e:
                        logger.error(f"Error saving RECAP case: {e}")
                        case_result["notes"] = f"Found via RECAP but save failed: {e}"

                    return case_result
                else:
                    logger.warning("✗ RECAP document not a valid §1782 case")
                    case_result["notes"] = "Found via RECAP but failed §1782 validation"
            else:
                logger.info("ℹ️ No RECAP documents found for docket")

        # If all strategies failed, mark as requiring manual retrieval
        logger.warning("✗ Case not found via CourtListener or RECAP")
        case_result["status"] = "manual_required"
        case_result["source"] = "pacer"
        case_result["notes"] = "Not available via CourtListener or RECAP - requires manual PACER/Bloomberg retrieval"
        self.tracking_log["requires_manual_retrieval"] += 1

        return case_result

    def acquire_all_cases(self) -> None:
        """Acquire all wishlist cases."""
        logger.info(f"Starting acquisition of {len(WISHLIST_CASES)} wishlist cases")

        for i, case_info in enumerate(WISHLIST_CASES, 1):
            logger.info(f"\nProgress: {i}/{len(WISHLIST_CASES)}")

            case_result = self.acquire_case(case_info)
            self.tracking_log["cases"].append(case_result)

            # Rate limiting
            time.sleep(2)

        # Save tracking log
        self.save_tracking_log()

        # Generate summary
        self.print_summary()

    def save_tracking_log(self) -> None:
        """Save the tracking log to JSON file."""
        log_path = Path("data/case_law/wishlist_acquisition_log.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.tracking_log, f, indent=2, ensure_ascii=False)

        logger.info(f"Tracking log saved to: {log_path}")

    def print_summary(self) -> None:
        """Print acquisition summary."""
        logger.info(f"\n{'='*80}")
        logger.info("ACQUISITION SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total cases processed: {self.tracking_log['total_cases']}")
        logger.info(f"Found via CourtListener: {self.tracking_log['found_via_courtlistener']}")
        logger.info(f"Found via RECAP: {self.tracking_log['found_via_recap']}")
        logger.info(f"Requires manual retrieval: {self.tracking_log['requires_manual_retrieval']}")

        # Show found cases
        found_cases = [c for c in self.tracking_log["cases"] if c["status"] == "found"]
        if found_cases:
            logger.info(f"\nFound cases ({len(found_cases)}):")
            for case in found_cases:
                logger.info(f"  ✓ {case['wishlist_name']}")

        # Show cases requiring manual retrieval
        manual_cases = [c for c in self.tracking_log["cases"] if c["status"] == "manual_required"]
        if manual_cases:
            logger.info(f"\nCases requiring manual retrieval ({len(manual_cases)}):")
            for case in manual_cases:
                logger.info(f"  ✗ {case['wishlist_name']} - {case['citation']}")


def main():
    """Main entry point."""
    acquirer = WishlistCaseAcquirer()
    acquirer.acquire_all_cases()


if __name__ == "__main__":
    main()
