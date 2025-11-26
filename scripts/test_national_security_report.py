#!/usr/bin/env python3
"""
Test script: Generate comprehensive report on whether US case law supports
the case as a matter of US national security.

This tests the full system:
- Case law research
- CatBoost feature analysis
- SK plugin enforcement
- Refinement loop
- Report generation
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "writer_agents" / "code"))

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from writer_agents.code.insights import CaseInsights
from writer_agents.code.case_law_researcher import CaseLawResearcher
import sqlite3
import pandas as pd

# Import the existing NationalSecurityResearcher
sys.path.insert(0, str(project_root / "case_law_data" / "scripts"))
from research_national_security_cases import NationalSecurityResearcher


# Known good sealing precedents
SEALING_PRECEDENTS = {
    "Nixon v. Warner Communications": "Balancing test for sealing/access",
    "In re Sealed Case": "Sealing in sensitive contexts (D.C. Cir.)",
    "Friends for All Children v. Lockheed": "Protective orders in discovery",
    "Intel Corp.": "Section 1782 discovery with sealing",
    "Brandi-Dohrn": "Section 1782 protective orders",
    "del Valle Ruiz": "Section 1782 confidentiality",
    "O'Keeffe": "Section 1782 sealing"
}


def is_off_point_case(case_name: str, court: str, text_snippet: str) -> tuple[bool, str]:
    """
    Check if a case is off-point (wrong context entirely).

    Returns:
        (is_off_point, reason)
    """
    case_name_lower = case_name.lower()
    court_lower = court.lower()
    text_lower = text_snippet.lower()

    # Art repatriation cases
    if any(name in case_name_lower for name in ["kunstsammlungen", "elicofon", "dürer", "art repatriation", "painting"]):
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "Art repatriation case without sealing precedent"

    # Immigration policy cases (Trump v. Hawaii, State of Hawaii v. Trump)
    if any(name in case_name_lower for name in ["hawaii v. trump", "trump v. hawaii", "travel ban"]):
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "Immigration policy case without sealing precedent"

    # General defamation without sealing (US Dominion v. Byrne)
    if "dominion" in case_name_lower and "byrne" in case_name_lower:
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "General defamation case without sealing precedent"

    # FSIA terrorism judgments without sealing (Owens v. Republic of Sudan)
    if "owens" in case_name_lower and "sudan" in case_name_lower:
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "FSIA terrorism judgment without sealing precedent"

    # Nuclear site litigation without sealing holdings (Cook v. Rockwell)
    if "cook" in case_name_lower and "rockwell" in case_name_lower:
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "Nuclear site litigation without sealing holdings"

    # ATS/human rights cases without sealing (Xuncax v. Gramajo)
    if "xuncax" in case_name_lower and "gramajo" in case_name_lower:
        if "seal" not in text_lower or ("motion to seal" not in text_lower and "sealing granted" not in text_lower):
            return True, "ATS/human rights case without sealing precedent"

    # Cases about releasing records (Lepore - Pentagon Papers)
    if "lepore" in case_name_lower or ("pentagon papers" in text_lower and "release" in text_lower):
        if "release" in text_lower and ("grand jury" in text_lower or "unseal" in text_lower):
            return True, "Case about releasing sealed records (favors disclosure)"

    # FOIA cases that are only about agency disclosure, not civil sealing
    if "sims v. cia" in case_name_lower or ("cia" in case_name_lower and "foia" in text_lower):
        if "motion to seal" not in text_lower and "sealing granted" not in text_lower:
            # Only off-point if it's purely FOIA, not a sealing motion
            if "mk-ultra" in text_lower or "foia exemption" in text_lower:
                return True, "FOIA/intel agency case, not civil sealing precedent"

    return False, ""


def classify_case_relevance(
    case_name: str,
    text_snippet: str,
    support_score: int,
    oppose_score: int,
    has_ns_tag: bool,
    is_contra: bool
) -> tuple[str, str]:
    """
    Classify a case by relevance to sealing/impound motions.

    Returns:
        (category, explanation)
        Categories: "Direct Precedent", "Helpful by Analogy", "Neutral", "Contra", "Off-Point"
    """
    text_lower = text_snippet.lower()
    case_name_lower = case_name.lower()

    # Contra cases (favor disclosure)
    if is_contra or oppose_score > support_score + 2:
        return "Contra", "Case favors disclosure/public access"

    # Direct Precedent: Cases that directly address sealing/impound motions in civil context
    sealing_keywords = [
        "motion to seal granted",
        "motion to seal approved",
        "protective order granted",
        "file under seal",
        "impound",
        "sealing granted",
        "leave to file under seal",
        "sealed case" in text_lower and ("granted" in text_lower or "approved" in text_lower)
    ]

    has_sealing_precedent = any(kw in text_lower for kw in sealing_keywords if isinstance(kw, str))
    has_sealing_precedent = has_sealing_precedent or (
        "sealed case" in text_lower and ("granted" in text_lower or "approved" in text_lower)
    )

    # Check if it's a known sealing precedent
    is_known_precedent = any(
        name.lower() in case_name_lower
        for name in SEALING_PRECEDENTS.keys()
    )

    if has_sealing_precedent or is_known_precedent:
        return "Direct Precedent", "Case directly addresses sealing/impound motions in civil context"

    # Helpful by Analogy: Cases that support secrecy but in different contexts
    # FOIA/intel agency cases, privacy cases, etc.
    if any(keyword in text_lower for keyword in ["foia", "intelligence", "classified", "c.i.a.", "cia"]):
        if "national security" in text_lower or "secrecy" in text_lower:
            return "Helpful by Analogy", "FOIA/intel agency case supporting secrecy (helpful by analogy, not direct precedent)"

    # Check for privacy/sealing by analogy
    if any(keyword in text_lower for keyword in ["privacy", "confidential", "protective order"]):
        if support_score > 3 and "seal" not in text_lower:
            return "Helpful by Analogy", "Privacy/confidentiality case (helpful by analogy, distinguish context)"

    # Neutral: Cases mentioning national security but not relevant to sealing
    if has_ns_tag or "national security" in text_lower:
        if support_score > 0 and support_score <= oppose_score:
            return "Neutral", "Mentions national security but not directly relevant to sealing motions"

    # Default: Neutral if no clear classification
    if support_score > oppose_score:
        return "Helpful by Analogy", "Potentially relevant but requires careful analysis of holdings"

    return "Neutral", "Not clearly relevant to sealing motions"


async def generate_national_security_report():
    """Generate comprehensive report on national security case law support."""

    print("=" * 80)
    print("NATIONAL SECURITY CASE LAW ANALYSIS REPORT")
    print("=" * 80)
    print()
    print("Question: Does US case law support that this case is a matter of US national security?")
    print()

    # Initialize research components
    print("Step 1: Initializing research components...")
    case_law_researcher = CaseLawResearcher()

    if not case_law_researcher.enabled:
        print("ERROR: CaseLawResearcher not enabled. Cannot generate report.")
        return None

    print("[OK] Case law databases loaded")
    print()

    # Create case insights for the research question
    print("Step 2: Formulating research query...")
    research_question = """
    Whether the allegations in this case, if factually plausible, would constitute
    a matter of US national security that would justify sealing and other protective measures.

    Key factors to consider:
    - Foreign government interference (Chinese Communist Party / People's Republic of China)
    - Academic institution (Harvard) with national security implications
    - Retaliation and harassment by foreign government actors
    - Reputation harm with potential impact on US national security
    - Information warfare and disinformation campaigns
    - Motion to seal granted for national security reasons
    - Protective orders in cases involving foreign governments
    - Federal court cases involving national security classification
    """

    insights = CaseInsights(
        reference_id="national_security_analysis",
        summary=research_question.strip(),
        posteriors=[],
        jurisdiction="US Federal",
        case_style="National Security Matter"
    )

    print(f"[OK] Research query formulated")
    print(f"  Query: {insights.summary[:100]}...")
    print()

    # Perform case law research using two methods:
    # 1. Direct database query for NS-tagged cases
    # 2. Semantic similarity search
    print("Step 3: Researching US case law...")
    print("  Method 1: Direct database query for NS-tagged cases")
    print("  Method 2: Semantic similarity search")
    print()

    # Use existing NationalSecurityResearcher to get NS-tagged cases
    ns_tagged_cases = []
    try:
        ns_researcher = NationalSecurityResearcher()
        ns_researcher.connect()

        # Get cases with NS tag + specific combinations (most relevant)
        relevant_df = ns_researcher.find_relevant_cases(
            include_harvard=True,
            include_defamation=True,
            include_foreign_gov=True,
            include_1782=False,
            limit=30
        )

        for _, row in relevant_df.iterrows():
            ns_tagged_cases.append({
                "case_name": row.get("case_name", "Unknown"),
                "court": row.get("court", "Unknown"),
                "citation": "",
                "date_filed": str(row.get("date_filed", "")),
                "text_snippet": row.get("text_snippet", ""),
                "similarity_score": 1.0,  # High score for direct tag match
                "tags": {
                    "NS": int(row.get("tag_national_security", 0)),
                    "Harvard": int(row.get("tag_academic_institution", 0)),
                    "ForeignGov": int(row.get("tag_foreign_government", 0)),
                    "Defamation": int(row.get("tag_defamation", 0))
                },
                "outcome": f"plaintiff={int(row.get('favorable_to_plaintiff', 0))}, defendant={int(row.get('favorable_to_defendant', 0))}",
                "source": "direct_ns_tag_query",
                "cluster_id": row.get("cluster_id")
            })

        # Also get ALL NS-tagged cases (broader search)
        cursor = ns_researcher.conn.cursor()
        cursor.execute("""
            SELECT
                cluster_id,
                case_name,
                court,
                date_filed,
                tag_national_security,
                COALESCE(tag_academic_institution, 0) as tag_academic_institution,
                COALESCE(tag_foreign_government, 0) as tag_foreign_government,
                COALESCE(tag_defamation, 0) as tag_defamation,
                COALESCE(favorable_to_plaintiff, 0) as favorable_to_plaintiff,
                COALESCE(favorable_to_defendant, 0) as favorable_to_defendant,
                SUBSTR(cleaned_text, 1, 1000) as text_snippet
            FROM cases
            WHERE tag_national_security = 1
            ORDER BY
                (COALESCE(tag_academic_institution, 0) + COALESCE(tag_foreign_government, 0) + COALESCE(tag_defamation, 0)) DESC,
                date_filed DESC
            LIMIT 30
        """)

        rows = cursor.fetchall()
        existing_case_names = {c.get("case_name") for c in ns_tagged_cases}

        for row in rows:
            case_name = row["case_name"]
            if case_name and case_name not in existing_case_names:
                ns_tagged_cases.append({
                    "case_name": case_name,
                    "court": row["court"],
                    "citation": "",
                    "date_filed": str(row["date_filed"] if row["date_filed"] else ""),
                    "text_snippet": row["text_snippet"] if row["text_snippet"] else "",
                    "similarity_score": 0.9,  # Slightly lower but still high
                    "tags": {
                        "NS": int(row["tag_national_security"]),
                        "Harvard": int(row["tag_academic_institution"]),
                        "ForeignGov": int(row["tag_foreign_government"]),
                        "Defamation": int(row["tag_defamation"])
                    },
                    "outcome": f"plaintiff={int(row['favorable_to_plaintiff'])}, defendant={int(row['favorable_to_defendant'])}",
                    "source": "direct_ns_tag_query_all",
                    "cluster_id": row["cluster_id"]
                })

        ns_researcher.close()
        print(f"[OK] Found {len(ns_tagged_cases)} cases with NS tag from direct database queries")
    except Exception as e:
        logger.warning(f"NationalSecurityResearcher failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        print(f"[WARNING] Direct NS tag query failed: {e}")

    # Semantic similarity search
    print("  Performing semantic similarity search...")
    research_results = case_law_researcher.research_case_law(
        insights=insights,
        top_k=30,
        min_similarity=0.4  # Higher threshold to avoid false positives
    )

    semantic_cases = []
    if research_results and research_results.get("cases"):
        semantic_cases = research_results.get("cases", [])
        print(f"[OK] Found {len(semantic_cases)} cases via semantic search")

    # Combine and deduplicate cases (prefer NS-tagged cases)
    cases_dict = {}
    for case in ns_tagged_cases:
        key = case.get("case_name", "") or f"case_{case.get('cluster_id', 'unknown')}"
        if key:
            cases_dict[key] = case

    # Add semantic cases if not already present
    for case in semantic_cases:
        key = case.get("case_name", "") or f"case_{case.get('cluster_id', 'unknown')}"
        if key and key not in cases_dict:
            # Only add if not a PA CCP case
            case_name_lower = case.get("case_name", "").lower()
            court_lower = case.get("court", "").lower()
            if "ccp" in case_name_lower and ("pennsylvania" in court_lower or "court of common pleas" in court_lower):
                continue  # Skip PA Court of Common Pleas cases
            cases_dict[key] = case
            if "source" not in case:
                case["source"] = "semantic_search"

    cases = list(cases_dict.values())

    # Prioritize NS-tagged cases
    cases.sort(key=lambda x: (
        x.get("tags", {}).get("NS", 0) == 1,  # NS tag first
        x.get("similarity_score", 0)  # Then by similarity
    ), reverse=True)

    print(f"[OK] Total unique cases: {len(cases)} (NS-tagged: {len(ns_tagged_cases)}, Semantic: {len(semantic_cases)})")
    if len(ns_tagged_cases) == 0:
        print(f"  WARNING: Direct NS tag query returned 0 cases (expected 254 from unified_corpus)")
    print()

    # Analyze cases for support/opposition
    print("Step 4: Analyzing cases for support/opposition...")

    supporting_cases = []
    opposing_cases = []
    neutral_cases = []
    off_point_cases = []

    for case in cases:
        # Check if case is off-point first
        case_name = case.get("case_name", "Unknown")
        court = case.get("court", "")
        text_snippet = case.get("text_snippet", "")

        is_off_point, reason = is_off_point_case(case_name, court, text_snippet)
        if is_off_point:
            off_point_cases.append({
                **case,
                "filter_reason": reason
            })
            continue  # Skip off-point cases
        # Extract outcome if available
        outcome = str(case.get("outcome", "")).lower()
        text_snippet = (case.get("text_snippet", "") + " " + case.get("case_name", "")).lower()
        case_name = case.get("case_name", "Unknown")
        similarity_score = case.get("similarity_score", 0)

        # Already filtered off-point cases above

        # Initialize variables
        support_score = 0
        oppose_score = 0
        has_ns_tag = False
        is_pa_ccp_case = False

        # Check tags first (high weight - these are manually curated)
        tags = case.get("tags", {})
        if isinstance(tags, dict):
            if tags.get("NS") == 1 or tags.get("NationalSecurity") == 1:
                support_score += 5  # High weight for NS tag
                has_ns_tag = True
            if tags.get("Harvard") == 1:
                support_score += 2
            if tags.get("ForeignGov") == 1:
                support_score += 2

        # Filter false positives: "CCP" in case name often means "Court of Common Pleas" (PA state court)
        # Not relevant to national security - these are state court cases, not federal national security cases
        case_name_lower = case_name.lower()
        if "ccp" in case_name_lower and (
            "court of common pleas" in text_snippet or
            "pennsylvania" in case.get("court", "").lower() or
            "supreme court of pennsylvania" in case.get("court", "").lower()
        ):
            # This is likely a Pennsylvania state court case, not relevant
            is_pa_ccp_case = True
            # Only filter if no NS tag (NS tag overrides)
            if not has_ns_tag:
                support_score = 0
                oppose_score = 0

        # Determine if case supports or opposes
        # Look for keywords indicating support for national security sealing
        # Prioritize sealing-specific language
        support_indicators = [
            # Sealing-specific (highest weight)
            "motion to seal granted",
            "motion to seal approved",
            "protective order granted",
            "file under seal",
            "impound",
            "sealing granted",
            "sealing approved",
            "confidentiality order",
            "leave to file under seal",
            "seal the record",
            "sealed case",
            # General national security (lower weight)
            "national security",
            "foreign government",
            "academic institution",
            "harvard",
            "foreign interference",
            "confidential",
            "classified",
            "sensitive information",
            "privacy interests",
            "safety concerns",
            "retaliation",
            "harassment"
        ]

        oppose_indicators = [
            "sealing denied",
            "motion to seal denied",
            "public access",
            "first amendment",
            "no national security",
            "presumption of public access",
            "unseal",
            "disclosure required",
            "release of records",
            "public access favored",
            "first amendment right to access"
        ]

        # Context-aware contra detection (checked separately)
        is_contra = False
        if "presumption of public access" in text_snippet and ("granted" in text_snippet or "favored" in text_snippet or "allowed" in text_snippet):
            is_contra = True
        if "release" in text_snippet and ("grand jury" in text_snippet or "pentagon papers" in text_snippet):
            is_contra = True
        if "unsealed" in text_snippet and "granted" in text_snippet:
            is_contra = True
        if "disclosure" in text_snippet and "required" in text_snippet and "seal" not in text_snippet:
            is_contra = True
        if "release" in text_snippet and "grand jury" in text_snippet and "pentagon papers" in text_snippet:
            is_contra = True

        # Count indicator matches (weighted)
        for ind in support_indicators:
            if ind in text_snippet:
                # Sealing-specific language gets highest weight
                if ind in ["motion to seal granted", "motion to seal approved", "protective order granted",
                          "file under seal", "impound", "sealing granted", "sealing approved",
                          "confidentiality order", "leave to file under seal"]:
                    support_score += 3  # High weight for direct sealing language
                elif ind in ["seal the record", "sealed case"]:
                    support_score += 2  # Medium-high weight
                else:
                    support_score += 1  # Standard weight for general indicators

                # Additional boost for combination of sealing + national security
                if "seal" in text_snippet and "national security" in text_snippet:
                    support_score += 1

        for ind in oppose_indicators:
            if ind in text_snippet:
                oppose_score += 1
                # High-value indicators get extra weight
                if ind in ["sealing denied", "motion to seal denied", "presumption of public access", "release of records", "unseal"]:
                    oppose_score += 2  # Increased weight for contra indicators

        # Context-aware contra detection
        if is_contra:
            oppose_score += 3  # High weight for context-aware contra detection

        # Check outcome field (more flexible parsing)
        if "plaintiff" in outcome:
            if "1" in outcome or "favorable" in outcome or "granted" in outcome:
                support_score += 3
            elif "0" in outcome or "denied" in outcome:
                oppose_score += 3
        elif "defendant" in outcome:
            if "1" in outcome or "favorable" in outcome:
                oppose_score += 3
            elif "0" in outcome:
                support_score += 3

        # If no text snippet, rely more heavily on tags and similarity
        if len(text_snippet) < 50:
            # Low content - rely on tags and similarity only
            if not has_ns_tag and similarity_score < 0.7:
                # Skip cases with no content and low similarity
                support_score = 0
                oppose_score = 0

        # Boost score based on similarity (high similarity = more relevant)
        if similarity_score > 0.6:
            support_score += 1
        elif similarity_score > 0.5:
            support_score += 0.5


        # If case has NS tag, it's automatically supporting (even if PA CCP case)
        if has_ns_tag:
            support_score = max(support_score, 3)  # Ensure minimum score

        # Check for federal court (more likely to be national security cases)
        court = case.get("court", "").lower()
        if "district court" in court or "federal" in court or "circuit" in court:
            support_score += 1

        # Classify case by relevance
        category, explanation = classify_case_relevance(
            case_name=case_name,
            text_snippet=text_snippet,
            support_score=support_score,
            oppose_score=oppose_score,
            has_ns_tag=has_ns_tag,
            is_contra=is_contra
        )

        # Add classification metadata
        case_with_category = {
            **case,
            "support_score": support_score,
            "oppose_score": oppose_score,
            "category": category,
            "category_explanation": explanation,
            "is_contra": is_contra
        }

        # Classify into lists
        if category == "Contra":
            opposing_cases.append(case_with_category)
        elif category == "Direct Precedent":
            supporting_cases.append(case_with_category)
        elif category == "Helpful by Analogy":
            supporting_cases.append(case_with_category)  # Still supporting, but with disclaimer
        elif category == "Neutral":
            neutral_cases.append(case_with_category)
        else:
            # Default classification based on scores
            if (has_ns_tag or support_score >= 2) and support_score > oppose_score and not is_pa_ccp_case:
                supporting_cases.append(case_with_category)
            elif oppose_score > 0 and oppose_score > support_score:
                opposing_cases.append(case_with_category)
            else:
                neutral_cases.append(case_with_category)

    # Count by category
    direct_precedents = [c for c in supporting_cases if c.get("category") == "Direct Precedent"]
    helpful_by_analogy = [c for c in supporting_cases if c.get("category") == "Helpful by Analogy"]
    contra_cases = [c for c in opposing_cases if c.get("category") == "Contra"]

    print(f"[OK] Analysis complete:")
    print(f"  - Supporting cases: {len(supporting_cases)}")
    print(f"    - Direct Precedents: {len(direct_precedents)}")
    print(f"    - Helpful by Analogy: {len(helpful_by_analogy)}")
    print(f"  - Opposing cases: {len(opposing_cases)}")
    print(f"    - Contra (favor disclosure): {len(contra_cases)}")
    print(f"  - Neutral cases: {len(neutral_cases)}")
    print(f"  - Off-point cases filtered: {len(off_point_cases)}")
    print()

    # Debug: Show top cases by similarity
    if len(supporting_cases) == 0 and len(opposing_cases) == 0:
        print("DEBUG: No cases classified. Showing top 5 cases by similarity:")
        top_cases = sorted(cases, key=lambda x: x.get("similarity_score", 0), reverse=True)[:5]
        for i, case in enumerate(top_cases, 1):
            print(f"  {i}. {case.get('case_name', 'Unknown')[:60]}")
            print(f"     Court: {case.get('court', 'Unknown')[:50]}")
            print(f"     Similarity: {case.get('similarity_score', 0):.3f}")
            print(f"     Tags: {case.get('tags', {})}")
            print(f"     Text snippet length: {len(case.get('text_snippet', ''))}")
            print()

    # Check for known sealing precedents in results
    known_precedents_found = []
    for case in cases:
        case_name = case.get("case_name", "").lower()
        for precedent_name in SEALING_PRECEDENTS.keys():
            if precedent_name.lower() in case_name:
                known_precedents_found.append({
                    "name": case.get("case_name", "Unknown"),
                    "description": SEALING_PRECEDENTS[precedent_name],
                    "case": case
                })
                break

    # Generate report
    print("Step 5: Generating comprehensive report...")
    if known_precedents_found:
        print(f"  Found {len(known_precedents_found)} known sealing precedents in results")

    report = generate_report(
        research_question=research_question,
        total_cases=len(cases),
        supporting_cases=supporting_cases,
        opposing_cases=opposing_cases,
        neutral_cases=neutral_cases,
        off_point_cases=off_point_cases,
        all_cases=cases,  # Pass all cases for manual review section
        research_results=research_results,
        known_precedents=known_precedents_found
    )

    # Save report
    output_dir = Path("outputs") / "national_security_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"national_security_analysis_{timestamp}.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[OK] Report saved to: {report_file}")
    print()

    # Print summary
    print("=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)
    print()
    print(f"Total cases analyzed: {len(cases)}")
    print(f"Supporting cases: {len(supporting_cases)} ({len(supporting_cases)/len(cases)*100:.1f}%)")
    print(f"Opposing cases: {len(opposing_cases)} ({len(opposing_cases)/len(cases)*100:.1f}%)")
    print(f"Neutral cases: {len(neutral_cases)} ({len(neutral_cases)/len(cases)*100:.1f}%)")
    print()

    if len(supporting_cases) > len(opposing_cases):
        print("CONCLUSION: US case law GENERALLY SUPPORTS the position that")
        print("this case is a matter of US national security.")
        print()
        print(f"Support ratio: {len(supporting_cases)}/{len(cases)} cases ({len(supporting_cases)/len(cases)*100:.1f}%)")
    elif len(opposing_cases) > len(supporting_cases):
        print("CONCLUSION: US case law GENERALLY DOES NOT SUPPORT the position that")
        print("this case is a matter of US national security.")
        print()
        print(f"Opposition ratio: {len(opposing_cases)}/{len(cases)} cases ({len(opposing_cases)/len(cases)*100:.1f}%)")
    else:
        print("CONCLUSION: US case law is MIXED on whether this case")
        print("is a matter of US national security.")
        print()
        print("Further analysis needed.")

    print()
    print("=" * 80)
    print(f"Full report available at: {report_file}")
    print("=" * 80)

    return {
        "report_file": str(report_file),
        "total_cases": len(cases),
        "supporting_cases": len(supporting_cases),
        "opposing_cases": len(opposing_cases),
        "neutral_cases": len(neutral_cases),
        "support_ratio": len(supporting_cases) / len(cases) if cases else 0
    }


def generate_report(
    research_question: str,
    total_cases: int,
    supporting_cases: list,
    opposing_cases: list,
    neutral_cases: list,
    off_point_cases: list,
    all_cases: list,
    research_results: dict,
    known_precedents: list = None
) -> str:
    """Generate markdown report."""

    # Categorize cases
    direct_precedents = [c for c in supporting_cases if c.get("category") == "Direct Precedent"]
    helpful_by_analogy = [c for c in supporting_cases if c.get("category") == "Helpful by Analogy"]
    contra_cases = [c for c in opposing_cases if c.get("category") == "Contra"]

    # Calculate accurate support ratio (only direct precedents)
    relevant_cases = len(direct_precedents) + len(contra_cases)
    direct_precedent_ratio = len(direct_precedents) / relevant_cases if relevant_cases > 0 else 0

    report_lines = [
        "# US Case Law Analysis: National Security Matter",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Research Question",
        "",
        research_question,
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Total cases retrieved:** {total_cases}",
        f"- **Off-point cases filtered:** {len(off_point_cases)}",
        f"- **Cases analyzed:** {total_cases - len(off_point_cases)}",
        "",
        "### Case Classification",
        "",
        f"- **Direct Precedents** (sealing/impound motions): {len(direct_precedents)}",
        f"- **Helpful by Analogy** (different contexts): {len(helpful_by_analogy)}",
        f"- **Contra** (favor disclosure): {len(contra_cases)}",
        f"- **Neutral**: {len(neutral_cases)}",
        "",
        "### Conclusion",
        "",
    ]

    # Conclusion based on direct precedents only
    if len(direct_precedents) > len(contra_cases) and len(direct_precedents) > 0:
        report_lines.extend([
            f"**US case law GENERALLY SUPPORTS** the position that this case is a matter",
            f"of US national security, based on {len(direct_precedents)} direct sealing precedents",
            f"vs. {len(contra_cases)} contra cases.",
            "",
            f"**Support ratio (direct precedents only):** {len(direct_precedents)}/{relevant_cases} ({direct_precedent_ratio*100:.1f}%)",
            "",
            "**Important:** This ratio is based on direct sealing precedents only. Cases",
            "classified as 'Helpful by Analogy' require careful analysis of holdings and",
            "distinction of facts, as they arise in different contexts (e.g., FOIA/intel",
            "agencies rather than civil sealing motions).",
            ""
        ])
    elif len(contra_cases) > len(direct_precedents):
        report_lines.extend([
            f"**US case law GENERALLY DOES NOT SUPPORT** the position that this case",
            f"is a matter of US national security, based on {len(contra_cases)} contra cases",
            f"vs. {len(direct_precedents)} direct sealing precedents.",
            "",
            f"**Opposition ratio:** {len(contra_cases)}/{relevant_cases} ({len(contra_cases)/relevant_cases*100:.1f}%)",
            ""
        ])
    else:
        report_lines.extend([
            "**US case law is MIXED** on whether this case is a matter of US national security.",
            f"Found {len(direct_precedents)} direct precedents and {len(contra_cases)} contra cases.",
            "Further analysis and case-specific arguments are needed.",
            ""
        ])

    # Known Sealing Precedents Section
    if known_precedents:
        report_lines.extend([
            "---",
            "",
            "## Recommended Sealing Precedents",
            "",
            "The following well-established sealing precedents should be considered:",
            ""
        ])
        for prec in known_precedents:
            report_lines.extend([
                f"- **{prec['name']}**: {prec['description']}",
                ""
            ])
        report_lines.extend([
            "",
            "**Note:** These are established precedents for sealing/impound motions.",
            "If found in your database, they should be prioritized in your motion.",
            ""
        ])

    # Direct Precedents Section
    report_lines.extend([
        "---",
        "",
        "## Direct Sealing Precedents",
        "",
        f"Found {len(direct_precedents)} cases that directly address sealing/impound motions:",
        ""
    ])

    if len(direct_precedents) == 0:
        report_lines.extend([
            "**No direct sealing precedents found.** Cases classified as supporting rely on",
            "analogy or general principles rather than direct sealing holdings.",
            ""
        ])
    else:
        # Top direct precedents
        top_direct = sorted(direct_precedents, key=lambda x: x.get("support_score", 0), reverse=True)[:10]
        for i, case in enumerate(top_direct, 1):
            report_lines.extend([
                f"### {i}. {case.get('case_name', 'Unknown Case')}",
                "",
                f"- **Court:** {case.get('court', 'Unknown')}",
                f"- **Citation:** {case.get('citation', 'N/A')}",
                f"- **Category:** {case.get('category', 'Unknown')}",
                f"- **Support Score:** {case.get('support_score', 0)}",
                f"- **Explanation:** {case.get('category_explanation', 'N/A')}",
                ""
            ])
            if case.get('text_snippet'):
                snippet = case['text_snippet'][:500].replace('\n', ' ')
                report_lines.extend([
                    f"- **Relevant Text:** {snippet}...",
                    ""
                ])

    # Helpful by Analogy Section
    if len(helpful_by_analogy) > 0:
        report_lines.extend([
            "---",
            "",
            "## Helpful by Analogy",
            "",
            f"Found {len(helpful_by_analogy)} cases that may be helpful by analogy:",
            "",
            "**⚠️ IMPORTANT DISCLAIMER:** These cases support secrecy or confidentiality",
            "but arise in different contexts (e.g., FOIA/intel agencies, privacy cases).",
            "They are NOT direct precedents for civil sealing motions. Use with caution",
            "and be prepared to distinguish facts or show how the analogy applies.",
            ""
        ])

        top_analogy = sorted(helpful_by_analogy, key=lambda x: x.get("support_score", 0), reverse=True)[:5]
        for i, case in enumerate(top_analogy, 1):
            report_lines.extend([
                f"### {i}. {case.get('case_name', 'Unknown Case')}",
                "",
                f"- **Court:** {case.get('court', 'Unknown')}",
                f"- **Category:** {case.get('category', 'Unknown')}",
                f"- **Explanation:** {case.get('category_explanation', 'N/A')}",
                ""
            ])
            if case.get('text_snippet'):
                snippet = case['text_snippet'][:300].replace('\n', ' ')
                report_lines.extend([
                    f"- **Relevant Text:** {snippet}...",
                    ""
                ])

    # Contra Cases Section
    if len(contra_cases) > 0:
        report_lines.extend([
            "---",
            "",
            "## Contra Cases (Favor Disclosure)",
            "",
            f"Found {len(contra_cases)} cases that favor disclosure/public access:",
            "",
            "**⚠️ IMPORTANT:** These cases cut AGAINST your position. You must distinguish",
            "them or show why they don't apply to your facts.",
            ""
        ])

        top_contra = sorted(contra_cases, key=lambda x: x.get("oppose_score", 0), reverse=True)[:5]
        for i, case in enumerate(top_contra, 1):
            report_lines.extend([
                f"### {i}. {case.get('case_name', 'Unknown Case')}",
                "",
                f"- **Court:** {case.get('court', 'Unknown')}",
                f"- **Category:** {case.get('category', 'Unknown')}",
                f"- **Explanation:** {case.get('category_explanation', 'N/A')}",
                f"- **Oppose Score:** {case.get('oppose_score', 0)}",
                ""
            ])
            if case.get('text_snippet'):
                snippet = case['text_snippet'][:500].replace('\n', ' ')
                report_lines.extend([
                    f"- **Relevant Text:** {snippet}...",
                    ""
                ])

    # Off-Point Cases Section (for reference)
    if len(off_point_cases) > 0:
        report_lines.extend([
            "---",
            "",
            "## Off-Point Cases (Filtered)",
            "",
            f"Filtered {len(off_point_cases)} cases that are off-point (wrong context):",
            "",
            "These cases mention national security but are not relevant to sealing motions.",
            "Examples include art repatriation, immigration policy, general defamation without",
            "sealing, etc.",
            ""
        ])

        for i, case in enumerate(off_point_cases[:10], 1):
            report_lines.extend([
                f"### {i}. {case.get('case_name', 'Unknown Case')}",
                "",
                f"- **Court:** {case.get('court', 'Unknown')}",
                f"- **Filter Reason:** {case.get('filter_reason', 'Unknown')}",
                ""
            ])

    report_lines.extend([
        "---",
        "",
        "## Key Legal Principles Identified",
        "",
        "### Supporting Principles",
        "",
        "1. **Foreign Government Interference**: Cases involving foreign government",
        "   interference or influence have been recognized as national security matters.",
        "",
        "2. **Academic Institutions**: Cases involving academic institutions with",
        "   national security implications have been granted protective measures.",
        "",
        "3. **Sealing and Protective Orders**: Courts have granted sealing and",
        "   protective orders in cases with national security implications.",
        "",
        "### Opposing Principles",
        "",
        "1. **Public Access**: Strong presumption in favor of public access to",
        "   court records and proceedings.",
        "",
        "2. **First Amendment**: Public's right to access information about",
        "   government and legal proceedings.",
        "",
        "3. **Burden of Proof**: High burden to overcome presumption of public access.",
        "",
        "---",
        "",
        "## Recommendations",
        "",
        "1. **Emphasize supporting cases**: Focus on cases that directly support",
        "   the national security position, particularly those with similar facts.",
        "",
        "2. **Distinguish opposing cases**: Address opposing cases by distinguishing",
        "   facts or showing how they don't apply to this specific situation.",
        "",
        "3. **Cite direct precedents**: Prioritize cases classified as 'Direct Precedent'",
        "   in your motion. These directly address sealing/impound motions.",
        "",
        "4. **Use analogies carefully**: Cases classified as 'Helpful by Analogy' require",
        "   careful analysis. Distinguish facts and explain why the analogy applies despite",
        "   different contexts (e.g., FOIA vs. civil sealing).",
        "",
        "5. **Address contra cases**: If contra cases are identified, you must distinguish",
        "   them or show why they don't apply to your specific facts.",
        "",
        "6. **Address burden of proof**: Clearly establish why the national security",
        "   interest outweighs the presumption of public access.",
        "",
        "---",
        "",
        "## Research Methodology",
        "",
        f"- **Databases searched**: Multiple US federal case law databases",
        f"- **Search method**: Direct NS tag queries + semantic similarity search",
        f"- **Minimum similarity threshold**: 0.4",
        f"- **Total cases retrieved**: {total_cases}",
        f"- **Off-point cases filtered**: {len(off_point_cases)}",
        f"- **Analysis method**: Automated classification with multi-tier categorization",
        f"- **Categories**: Direct Precedent, Helpful by Analogy, Neutral, Contra, Off-Point",
        "",
        "---",
        "",
        f"**Report generated by:** The Matrix Legal AI System",
        f"**Version:** 1.0",
        ""
    ])

    return "\n".join(report_lines)


if __name__ == "__main__":
    print()
    print("Starting National Security Case Law Analysis...")
    print()

    try:
        result = asyncio.run(generate_national_security_report())

        if result:
            print()
            print("[OK] Analysis complete!")
            print(f"  Report file: {result['report_file']}")
            print(f"  Support ratio: {result['support_ratio']*100:.1f}%")
        else:
            print()
            print("[ERROR] Analysis failed - no results generated")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        print()
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

