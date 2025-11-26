#!/usr/bin/env python3
"""
Process the comprehensive petition analysis into structured features
Convert the detailed analysis into machine-learning ready format
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PetitionFeatureProcessor:
    def __init__(self):
        self.petitions = []
        self.feature_schema = {
            # Identity & venue
            'district': str,
            'case_no': str,
            'file_date': str,
            'ex_parte': bool,
            'applicant': str,
            'respondents': List[str],

            # Foreign proceeding
            'foreign_country': str,
            'foreign_tribunal': str,
            'posture': str,
            'area_of_law': str,

            # Statutory prongs
            'found_in_district': bool,
            'for_use': str,
            'interested_person': bool,

            # Intel factors
            'intel_non_party': bool,
            'intel_receptivity': bool,
            'intel_no_circumvention': bool,
            'intel_not_unduly_burdensome': bool,

            # Discovery request
            'discovery_type': str,
            'scope_summary': str,
            'protective_order_requested': str,
            'attachments': List[str],

            # Authority signals
            'intel_cited': bool,
            'local_grants_cited': int,
            'other_citations_count': int,

            # Drafting/format signals
            'has_toa_toc': bool,
            'intel_headings': bool,
            'pages': int,
            'tone_keywords': List[str],

            # Provenance
            'source_pdf': str,
            'source_citation': str,

            # Derived features for ML
            'request_narrowness': str,  # single-item vs multi-category
            'residency_proof_style': str,  # SEC 10-K vs registration vs prior §1782
            'receptivity_evidence': str,  # explicit vs tribunal named vs none
            'sector_tag': str,  # Patent/FRAND vs Financial vs Government
            'local_precedent_density': int,  # number of same-district grants cited
            'not_burdensome_phrasing': int,  # count of "not burdensome" mentions
            'has_proposed_subpoena': bool,
            'bank_record_petition': bool,  # third-party banks + tracing
        }

    def load_petition_data(self):
        """Load the 5 analyzed petitions."""
        self.petitions = [
            {
                "district": "N.D. Cal.",
                "case_no": "5:25-mc-80042 (Doc. 1)",
                "file_date": "2025-02-20",
                "ex_parte": True,
                "applicant": "HMD Global Oy",
                "respondents": ["Samsung Electronics Co., Ltd.", "Samsung Electronics America, Inc.", "Samsung Semiconductor, Inc."],
                "foreign_country": "Germany",
                "foreign_tribunal": "Munich District Court / Munich Higher Regional Court",
                "posture": "two appeals pending; one case pending on validity",
                "area_of_law": "Patent (FRAND / exhaustion)",
                "found_in_district": True,
                "for_use": "License terms to support FRAND and exhaustion defenses",
                "interested_person": True,
                "intel_factors": {
                    "non_party": True,
                    "receptivity": True,
                    "no_circumvention": True,
                    "not_unduly_burdensome": True
                },
                "discovery": {
                    "type": "documents",
                    "scope_summary": "All agreements between Samsung and Huawei, incl. 2019 settlement referenced in Huawei White Paper",
                    "protective_order_requested": "Not specified",
                    "attachments": ["Proposed subpoena (Ex. A)", "von Falck decl.", "Warren decl."]
                },
                "authority": {
                    "intel_cited": True,
                    "local_grants_cited": ["Qualcomm §1782 line", "IPCom", "Illumina Cambridge", "BMW"],
                    "other_citations_count": 10
                },
                "drafting": {
                    "has_toa_toc": True,
                    "intel_headings": True,
                    "pages": 21,
                    "tone_keywords": ["narrowly tailored", "not burdensome"]
                },
                "source_pdf": "gov.uscourts.cand.444930.1.0.pdf",
                "source_citation": "See pp. 1–3 (caption, ex parte, single request), TOC p.4; Intel analysis pp. 10–13; burdensomeness p. 20."
            },
            {
                "district": "N.D. Fla.",
                "case_no": "4:25-mc-00015-MW-MAF (Doc. 1)",
                "file_date": "2025-03-17",
                "ex_parte": True,
                "applicant": ["LM Property Development Limited", "Mirko Kovats"],
                "respondents": ["GreenPointe Holdings, LLC"],
                "foreign_country": "Bahamas",
                "foreign_tribunal": "Bahamas proceedings (unspecified division)",
                "posture": "pending",
                "area_of_law": "Commercial dispute (documents for foreign litigation)",
                "found_in_district": True,
                "for_use": "Production of documents for Bahamas litigations",
                "interested_person": True,
                "intel_factors": {
                    "non_party": True,
                    "receptivity": True,
                    "no_circumvention": True,
                    "not_unduly_burdensome": True
                },
                "discovery": {
                    "type": "documents",
                    "scope_summary": "Production tied to Applicants' claims; proposed order attached",
                    "protective_order_requested": "Not stated",
                    "attachments": ["Proposed order", "Declarations", "Memorandum of Law"]
                },
                "authority": {
                    "intel_cited": True,
                    "local_grants_cited": 0,
                    "other_citations_count": 3
                },
                "drafting": {
                    "has_toa_toc": False,
                    "intel_headings": True,
                    "pages": 3,
                    "tone_keywords": ["meets three statutory requirements"]
                },
                "source_pdf": "gov.uscourts.flnd.530243.1.0.pdf",
                "source_citation": "See pp.1–2 for request and Intel factors; p.3 contains local Rule 7.1(F) word-count certification."
            },
            {
                "district": "S.D.N.Y.",
                "case_no": "1:24-mc-00575-LJL (Doc. 1)",
                "file_date": "2024-12-10",
                "ex_parte": True,
                "applicant": "Navios South American Logistics Inc.",
                "respondents": ["AmEx", "Banco Santander", "Bank of New York Mellon", "Boca World LLC", "Citibank", "Citimortgage", "Standard Bank", "UBS", "Wells Fargo"],
                "foreign_country": "Republic of the Marshall Islands",
                "foreign_tribunal": "High Court of the RMI",
                "posture": "civil action pending (breach of duty/fraud/etc.)",
                "area_of_law": "Corporate fraud / tracing",
                "found_in_district": True,
                "for_use": "Tracing payments/accounts/real estate tied to alleged fraud",
                "interested_person": True,
                "intel_factors": {
                    "non_party": True,
                    "receptivity": True,
                    "no_circumvention": True,
                    "not_unduly_burdensome": True
                },
                "discovery": {
                    "type": "documents + potential testimony",
                    "scope_summary": "Bank/credit/real-estate records tied to defendants and alleged transactions",
                    "protective_order_requested": "Not stated",
                    "attachments": ["Lennon declaration + complaint excerpts"]
                },
                "authority": {
                    "intel_cited": True,
                    "local_grants_cited": 2,
                    "other_citations_count": 5
                },
                "drafting": {
                    "has_toa_toc": False,
                    "intel_headings": True,
                    "pages": 9,
                    "tone_keywords": ["efficient assistance", "minimally relevant standard"]
                },
                "source_pdf": "gov.uscourts.nysd.633161.1.0.pdf",
                "source_citation": "See pp.1–2 for parties/tribunal; pp.3–7 for §1782 and Intel analysis; p.8 conclusion/signature."
            },
            {
                "district": "S.D. Cal.",
                "case_no": "3:24-mc-02144-JLB (Doc. 1)",
                "file_date": "2024-12-30",
                "ex_parte": True,
                "applicant": "HMD Global Oy",
                "respondents": ["Qualcomm Incorporated"],
                "foreign_country": "Germany",
                "foreign_tribunal": "Munich District Court / Munich Higher Regional Court",
                "posture": "two infringements dismissed; appeals pending; one case awaiting validity",
                "area_of_law": "Patent (FRAND / exhaustion)",
                "found_in_district": True,
                "for_use": "License/covenant terms to prove exhaustion and FRAND defenses",
                "interested_person": True,
                "intel_factors": {
                    "non_party": True,
                    "receptivity": True,
                    "no_circumvention": True,
                    "not_unduly_burdensome": True
                },
                "discovery": {
                    "type": "documents",
                    "scope_summary": "IP licenses, covenants not to sue, or similar agreements between Huawei and Qualcomm for asserted/comparable patents",
                    "protective_order_requested": "Not specified",
                    "attachments": ["Proposed subpoena", "von Falck decl.", "Warren decl."]
                },
                "authority": {
                    "intel_cited": True,
                    "local_grants_cited": 3,
                    "other_citations_count": 9
                },
                "drafting": {
                    "has_toa_toc": True,
                    "intel_headings": True,
                    "pages": 15,
                    "tone_keywords": ["narrowly tailored", "easily searchable universe"]
                },
                "source_pdf": "gov.uscourts.casd.801286.1.0.pdf",
                "source_citation": "See TOC p.2, Intro ¶¶1–2 pp.5–6, statutory & Intel analysis pp.6–11."
            },
            {
                "district": "S.D.N.Y.",
                "case_no": "1:24-mc-00557 (Doc. 1)",
                "file_date": "2024-12-03",
                "ex_parte": True,
                "applicant": "Republic of Türkiye",
                "respondents": ["Bank of America, N.A.", "Wells Fargo Bank, N.A."],
                "foreign_country": "Türkiye",
                "foreign_tribunal": "Criminal prosecution + money-laundering investigation",
                "posture": "ongoing criminal case + investigation",
                "area_of_law": "Criminal (insider trading / money laundering)",
                "found_in_district": True,
                "for_use": "USD-denominated transaction records tied to named suspects",
                "interested_person": True,
                "intel_factors": {
                    "non_party": True,
                    "receptivity": True,
                    "no_circumvention": True,
                    "not_unduly_burdensome": True
                },
                "discovery": {
                    "type": "documents",
                    "scope_summary": "Transactions to/from/through respondent banks for specified individuals and dates",
                    "protective_order_requested": "Not stated",
                    "attachments": ["Proposed order", "Subpoena exhibits", "Prosecutor declaration"]
                },
                "authority": {
                    "intel_cited": True,
                    "local_grants_cited": 0,
                    "other_citations_count": 2
                },
                "drafting": {
                    "has_toa_toc": False,
                    "intel_headings": True,
                    "pages": 4,
                    "tone_keywords": ["tailored requests", "expeditiously grant"]
                },
                "source_pdf": "gov.uscourts.nysd.632759.1.0.pdf",
                "source_citation": "See pp.1–3 for purpose/targets and §1782 prongs."
            }
        ]

    def extract_derived_features(self, petition: Dict[str, Any]) -> Dict[str, Any]:
        """Extract derived features for machine learning."""
        features = {}

        # Request narrowness
        scope = petition['discovery']['scope_summary'].lower()
        if 'single' in scope or 'narrow' in scope or 'easily searchable' in scope:
            features['request_narrowness'] = 'single-item'
        else:
            features['request_narrowness'] = 'multi-category'

        # Residency proof style
        citation = petition['source_citation'].lower()
        if 'sec' in citation or '10-k' in citation:
            features['residency_proof_style'] = 'SEC_filing'
        elif 'prior' in citation and '1782' in citation:
            features['residency_proof_style'] = 'prior_1782_estoppel'
        elif 'registration' in citation:
            features['residency_proof_style'] = 'registration'
        else:
            features['residency_proof_style'] = 'judicial_notice'

        # Receptivity evidence
        tribunal = petition['foreign_tribunal'].lower()
        if 'receptive' in petition['source_citation'].lower() or 'consider' in petition['source_citation'].lower():
            features['receptivity_evidence'] = 'explicit'
        elif 'court' in tribunal or 'tribunal' in tribunal:
            features['receptivity_evidence'] = 'tribunal_named'
        else:
            features['receptivity_evidence'] = 'none'

        # Sector tag
        area = petition['area_of_law'].lower()
        if 'patent' in area or 'frand' in area:
            features['sector_tag'] = 'Patent_FRAND'
        elif 'bank' in area or 'financial' in area or 'fraud' in area:
            features['sector_tag'] = 'Financial'
        elif 'criminal' in area or 'government' in area:
            features['sector_tag'] = 'Government'
        else:
            features['sector_tag'] = 'Commercial'

        # Local precedent density
        features['local_precedent_density'] = petition['authority']['local_grants_cited'] if isinstance(petition['authority']['local_grants_cited'], int) else len(petition['authority']['local_grants_cited'])

        # Not burdensome phrasing count
        tone_keywords = petition['drafting']['tone_keywords']
        features['not_burdensome_phrasing'] = sum(1 for keyword in tone_keywords if 'burden' in keyword.lower() or 'narrow' in keyword.lower())

        # Has proposed subpoena
        attachments = petition['discovery']['attachments']
        features['has_proposed_subpoena'] = any('subpoena' in att.lower() for att in attachments)

        # Bank record petition
        respondents = petition['respondents'] if isinstance(petition['respondents'], list) else [petition['respondents']]
        bank_keywords = ['bank', 'financial', 'credit', 'wells fargo', 'bank of america', 'citibank', 'santander']
        features['bank_record_petition'] = any(any(keyword in resp.lower() for keyword in bank_keywords) for resp in respondents)

        return features

    def process_all_petitions(self):
        """Process all petitions into ML-ready format."""
        logger.info("🔍 Processing petition features for machine learning")
        logger.info("=" * 60)

        processed_petitions = []

        for i, petition in enumerate(self.petitions, 1):
            logger.info(f"📄 Processing petition {i}: {petition['applicant']} → {petition['respondents'][0] if isinstance(petition['respondents'], list) else petition['respondents']}")

            # Extract derived features
            derived_features = self.extract_derived_features(petition)

            # Flatten the petition data
            flat_petition = {
                # Identity & venue
                'district': petition['district'],
                'case_no': petition['case_no'],
                'file_date': petition['file_date'],
                'ex_parte': petition['ex_parte'],
                'applicant': petition['applicant'] if isinstance(petition['applicant'], str) else ', '.join(petition['applicant']),
                'respondents': petition['respondents'],

                # Foreign proceeding
                'foreign_country': petition['foreign_country'],
                'foreign_tribunal': petition['foreign_tribunal'],
                'posture': petition['posture'],
                'area_of_law': petition['area_of_law'],

                # Statutory prongs
                'found_in_district': petition['found_in_district'],
                'for_use': petition['for_use'],
                'interested_person': petition['interested_person'],

                # Intel factors
                'intel_non_party': petition['intel_factors']['non_party'],
                'intel_receptivity': petition['intel_factors']['receptivity'],
                'intel_no_circumvention': petition['intel_factors']['no_circumvention'],
                'intel_not_unduly_burdensome': petition['intel_factors']['not_unduly_burdensome'],

                # Discovery request
                'discovery_type': petition['discovery']['type'],
                'scope_summary': petition['discovery']['scope_summary'],
                'protective_order_requested': petition['discovery']['protective_order_requested'],
                'attachments': petition['discovery']['attachments'],

                # Authority signals
                'intel_cited': petition['authority']['intel_cited'],
                'local_grants_cited': petition['authority']['local_grants_cited'],
                'other_citations_count': petition['authority']['other_citations_count'],

                # Drafting/format signals
                'has_toa_toc': petition['drafting']['has_toa_toc'],
                'intel_headings': petition['drafting']['intel_headings'],
                'pages': petition['drafting']['pages'],
                'tone_keywords': petition['drafting']['tone_keywords'],

                # Provenance
                'source_pdf': petition['source_pdf'],
                'source_citation': petition['source_citation'],

                # Derived features
                **derived_features
            }

            processed_petitions.append(flat_petition)

            logger.info(f"✓ Processed: {flat_petition['district']} - {flat_petition['sector_tag']} - {flat_petition['pages']} pages")

        return processed_petitions

    def save_results(self, processed_petitions):
        """Save the processed petition features."""
        # Save as JSON
        json_file = Path("data/case_law/petition_features_comprehensive.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': '2025-10-19',
                'total_petitions': len(processed_petitions),
                'feature_schema': list(self.feature_schema.keys()),
                'petitions': processed_petitions
            }, f, indent=2, ensure_ascii=False)

        # Save as JSONL (one petition per line)
        jsonl_file = Path("data/case_law/petition_features.jsonl")
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for petition in processed_petitions:
                f.write(json.dumps(petition, ensure_ascii=False) + '\n')

        # Create summary report
        report_file = Path("data/case_law/petition_features_report.md")
        report_content = f"""# Comprehensive Petition Features Report

## Summary
- **Total Petitions Analyzed**: {len(processed_petitions)}
- **Analysis Date**: 2025-10-19
- **Feature Schema**: {len(self.feature_schema)} features per petition

## Feature Categories

### Identity & Venue
- District, case number, file date, ex parte status
- Applicant and respondent information

### Foreign Proceeding
- Country, tribunal, posture, area of law

### Statutory Prongs
- Found in district, for use, interested person

### Intel Factors
- Non-party, receptivity, no circumvention, not unduly burdensome

### Discovery Request
- Type, scope, protective order, attachments

### Authority Signals
- Intel cited, local grants cited, other citations

### Drafting/Format Signals
- TOC, headings, pages, tone keywords

### Derived Features
- Request narrowness, residency proof style, receptivity evidence
- Sector tag, local precedent density, bank record petition

## Individual Petitions

"""

        for i, petition in enumerate(processed_petitions, 1):
            report_content += f"""### Petition {i}: {petition['applicant']} → {petition['respondents'][0] if isinstance(petition['respondents'], list) else petition['respondents']}

- **District**: {petition['district']}
- **Sector**: {petition['sector_tag']}
- **Pages**: {petition['pages']}
- **Request Type**: {petition['discovery_type']}
- **Intel Cited**: {petition['intel_cited']}
- **Local Precedent**: {petition['local_precedent_density']} cases
- **Bank Record Petition**: {petition['bank_record_petition']}

---
"""

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"📊 Results saved to: {json_file}")
        logger.info(f"📄 JSONL saved to: {jsonl_file}")
        logger.info(f"📄 Report saved to: {report_file}")

def main():
    processor = PetitionFeatureProcessor()
    processor.load_petition_data()
    processed_petitions = processor.process_all_petitions()
    processor.save_results(processed_petitions)

    logger.info(f"\n✅ Processing complete!")
    logger.info(f"📊 Processed {len(processed_petitions)} petitions with comprehensive features")
    logger.info(f"📁 Check results in: data/case_law/")

if __name__ == "__main__":
    main()
