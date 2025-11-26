"""Utilities for enforcing required motion sections using personal case facts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SECTION_PATTERNS: Dict[str, str] = {
    "INTRODUCTION": r"(?im)^\s*(?:#{2,3}\s+|I\.\s+|1\.\s+)?INTRODUCTION\b",
    "FACTUAL BACKGROUND": r"(?im)^\s*(?:#{2,3}\s+|II\.\s+|2\.\s+)?FACTUAL\s+BACKGROUND\b",
    "PRIVACY HARM ANALYSIS": r"(?im)^\s*(?:#{2,3}\s+|IV\.\s+|4\.\s+)?PRIVACY\s+HARM\s+ANALYSIS\b",
    "LEGAL STANDARD": r"(?im)^\s*(?:#{2,3}\s+|II\.\s+|2\.\s+)?LEGAL\s+STANDARD\b",
    "ARGUMENT": r"(?im)^\s*(?:#{2,3}\s+|III\.\s+|3\.\s+)?ARGUMENT\b",
    "CONCLUSION": r"(?im)^\s*(?:#{2,3}\s+|V\.\s+|5\.\s+)?CONCLUSION\b",
}

SECTION_ORDER: List[str] = [
    "INTRODUCTION",
    "FACTUAL BACKGROUND",
    "PRIVACY HARM ANALYSIS",
    "LEGAL STANDARD",
    "ARGUMENT",
    "CONCLUSION",
]


def enforce_section_structure(document: str, sk_context: Dict[str, Any]) -> str:
    """Ensure the draft contains every required section and augment factual background with case facts."""
    if not document:
        return document

    flags = re.IGNORECASE | re.MULTILINE
    missing = [name for name, pattern in SECTION_PATTERNS.items() if not re.search(pattern, document, flags)]
    facts = collect_structured_facts()

    updated_doc = document

    # Add any missing sections
    if missing:
        prefix_sections: List[str] = []
        suffix_sections: List[str] = []

        if "INTRODUCTION" in missing:
            intro = build_section_text("INTRODUCTION", sk_context, facts)
            if intro:
                prefix_sections.append(intro)

        for section_name in SECTION_ORDER:
            if section_name == "INTRODUCTION":
                continue
            if section_name in missing:
                section_text = build_section_text(section_name, sk_context, facts)
                if section_text:
                    suffix_sections.append(section_text)

        if prefix_sections or suffix_sections:
            updated_parts = [part.strip() for part in prefix_sections + [updated_doc.strip()] + suffix_sections if part and part.strip()]
            updated_doc = "\n\n".join(updated_parts)
            logger.info(f"Added structured sections: {[sec for sec in SECTION_ORDER if sec in missing]}")

    # Augment existing FACTUAL BACKGROUND with case-specific facts
    try:
        updated_doc = _augment_factual_background(updated_doc, facts)
    except Exception as _exc:
        logger.debug(f"[SECTION_HELPER] Could not augment factual background: {_exc}")

    return updated_doc


def collect_structured_facts() -> Dict[str, str]:
    """Return structured case facts via CaseFactsProvider or fallback JSON."""
    logger.debug("[SECTION_HELPER] Collecting structured facts...")

    # Try CaseFactsProvider first
    try:
        from .sk_plugins.FeaturePlugin.CaseFactsProvider import get_case_facts_provider

        provider = get_case_facts_provider()
        if provider:
            facts = provider.get_all_structured_facts()
            if facts:
                logger.info(f"[SECTION_HELPER] Loaded {len(facts)} facts from CaseFactsProvider")
                return facts
            else:
                logger.debug("[SECTION_HELPER] CaseFactsProvider returned no facts")
        else:
            logger.debug("[SECTION_HELPER] CaseFactsProvider not available")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug(f"[SECTION_HELPER] CaseFactsProvider not available for section enforcement: {exc}")

    # Fallback to JSON file
    logger.debug("[SECTION_HELPER] Attempting fallback: loading from case_insights.json...")
    json_facts = _load_facts_from_case_insights()
    if json_facts:
        logger.info(f"[SECTION_HELPER] Loaded {len(json_facts)} facts from case_insights.json fallback")
        return json_facts
    else:
        logger.warning("[SECTION_HELPER] No facts found in CaseFactsProvider or case_insights.json!")
        return {}


def build_section_text(section_name: str, sk_context: Dict[str, Any], facts: Dict[str, str]) -> str:
    """Build section text grounded in personal corpus facts."""
    section_name = section_name.upper()
    summary = _clean(sk_context.get("case_summary"))
    jurisdiction = _clean(sk_context.get("jurisdiction")) or "D. Massachusetts"

    hk_defamation = _clean(facts.get("hk_allegation_defamation"))
    hk_ccp = _clean(facts.get("hk_allegation_ccp_family"))
    competitor = _clean(facts.get("hk_allegation_competitor"))
    harvard_retaliation = _clean(facts.get("harvard_retaliation_events"))
    ogc_chain = _clean(facts.get("ogc_email_allegations"))
    privacy_leak = _clean(facts.get("privacy_leak_events"))
    safety = _clean(facts.get("safety_concerns"))

    if section_name == "INTRODUCTION":
        lines = [
            "## INTRODUCTION",
            "",
            "Plaintiff respectfully moves for leave to file the supporting materials under seal and to proceed under a pseudonym.",
        ]
        if summary:
            lines.append(summary)
        if ogc_chain:
            lines.append(f"Harvard's Office of the General Counsel was warned ({ogc_chain}) yet allowed the retaliatory publications to remain online.")
        if harvard_retaliation:
            lines.append(f"The Harvard Club publications summarized here show the ongoing retaliation: {harvard_retaliation}")
        return "\n".join(lines)

    if section_name == "PRIVACY HARM ANALYSIS":
        lines = ["## PRIVACY HARM ANALYSIS", ""]
        if privacy_leak:
            lines.append(f"The record includes direct disclosures of sensitive material: {privacy_leak}")
        if safety or hk_ccp:
            risk_line = safety or "These disclosures expose Plaintiff and witnesses to arrest, torture, or retaliation by PRC security services."
            if hk_ccp:
                lines.append(hk_ccp)
            lines.append(risk_line)
        if not (privacy_leak or safety or hk_ccp):
            lines.append("The supporting declarations detail how disclosure of identifying data would immediately reveal the plaintiff to hostile foreign actors.")
        return "\n".join(lines)

    if section_name == "LEGAL STANDARD":
        lines = [
            "## LEGAL STANDARD",
            "",
            f"Federal Rule of Civil Procedure 5.2(d) and Local Rule 7.2 authorize {jurisdiction} to seal records when privacy and safety interests outweigh the presumption of public access.",
            "Courts balance the First Amendment right of access against concrete harms, considering whether disclosure would chill whistleblowing, reveal confidential medical issues, or expose parties to foreign retaliation.",
            "Petitions supported by sworn allegations of retaliation, government harassment, or national-security sensitivities routinely proceed under seal and pseudonym to protect litigants."
        ]
        return "\n".join(lines)

    if section_name == "ARGUMENT":
        lines = ["## ARGUMENT", ""]
        if hk_defamation:
            lines.append("### A. Harvard-directed publications weaponized false accusations")
            lines.append(hk_defamation)
        if hk_ccp:
            lines.append("### B. Foreign-interference risk elevates the sealing interest")
            lines.append(hk_ccp)
        if competitor:
            lines.append("### C. Business retaliation amplifies the harm")
            lines.append(competitor)
        if ogc_chain:
            lines.append("### D. Harvard's notice history shows sealing is the only safeguard")
            lines.append(ogc_chain)
        if not any([hk_defamation, hk_ccp, competitor, ogc_chain]):
            lines.append("The balancing test favors sealing because disclosure would further the retaliation summarized in the HK High Court pleadings.")
        return "\n".join(lines)

    if section_name == "CONCLUSION":
        lines = [
            "## CONCLUSION",
            "",
            "For these reasons, Plaintiff asks the Court to:",
            "1. Permit the continued use of the pseudonym Jane Doe;",
            "2. Seal the supporting exhibits that describe foreign-interference threats and personal identifiers; and",
            "3. Grant any further relief needed to prevent retaliation against Plaintiff and witnesses."
        ]
        if harvard_retaliation:
            lines.append(f"The relief is necessary because the retaliation remains ongoing: {harvard_retaliation}")
        return "\n".join(lines)

    return ""


def _augment_factual_background(document: str, facts: Dict[str, str]) -> str:
    """Append a case-specific facts block to an existing FACTUAL BACKGROUND section.

    Heuristic: if the factual background does not already mention core entities (Harvard, OGC, Xi,
    Hong Kong, 2019/2025), append a concise bullet list grounded in fact blocks.
    """
    flags = re.IGNORECASE | re.MULTILINE
    fb_match = re.search(SECTION_PATTERNS["FACTUAL BACKGROUND"], document, flags)
    if not fb_match:
        return document

    # Find section bounds: start at match, end at next heading (LEGAL STANDARD or ARGUMENT or CONCLUSION)
    start = fb_match.start()
    tail_pattern = r"(?im)^(?:#{2,3}\s+|III?\.|IV\.|V\.|\d+\.)\s+(LEGAL\s+STANDARD|ARGUMENT|CONCLUSION|PRIVACY\s+HARM\s+ANALYSIS)\b"
    tail = re.search(tail_pattern, document[start:], flags)
    end = start + tail.start() if tail else len(document)
    section_text = document[start:end]

    # Check if already specific
    if re.search(r"Harvard|OGC|Xi|Hong\s*Kong|2019|2025|Statement of Claim", section_text, flags):
        return document  # Already includes case-specific details

    # Build facts block
    bullets: List[str] = []
    def _add(label: str, key: str):
        val = facts.get(key)
        if val:
            bullets.append(f"- {label}: {val}")

    _add("Defamation and publications", "hk_allegation_defamation")
    _add("Harvard OGC notice chain", "ogc_email_allegations")
    _add("Privacy leak events", "privacy_leak_events")
    _add("Safety concerns", "safety_concerns")
    _add("Retaliation timeline", "harvard_retaliation_events")

    if not bullets:
        return document

    facts_block = "\n".join([
        "",
        "### Case-Specific Facts (Grounded)",
        *bullets,
        ""
    ])

    augmented_section = section_text.rstrip() + "\n" + facts_block + "\n"
    return document[:start] + augmented_section + document[end:]


def _load_facts_from_case_insights() -> Dict[str, str]:
    """Fallback: load fact blocks from outputs/case_insights.json."""
    try:
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).resolve().parents[1] / "outputs" / "case_insights.json",
            Path(__file__).resolve().parents[2] / "outputs" / "case_insights.json",
            Path.cwd() / "outputs" / "case_insights.json",
        ]

        insights_path = None
        for path in possible_paths:
            if path.exists():
                insights_path = path
                logger.debug(f"[SECTION_HELPER] Found case_insights.json at: {insights_path}")
                break

        if not insights_path:
            logger.debug("[SECTION_HELPER] case_insights.json not found in any expected location")
            return {}

        data = json.loads(insights_path.read_text(encoding="utf-8"))
        fact_blocks = data.get("fact_blocks") or {}
        if not isinstance(fact_blocks, dict):
            logger.warning("[SECTION_HELPER] fact_blocks in JSON is not a dictionary")
            return {}

        logger.debug(f"[SECTION_HELPER] Loaded {len(fact_blocks)} fact blocks from JSON")

        ogc_chain = " ".join(
            _clean(fact_blocks.get(key))
            for key in ["ogc_email_1_threat", "ogc_email_2_non_response", "ogc_email_3_meet_confer"]
            if _clean(fact_blocks.get(key))
        ).strip()

        fallback = {key: _clean(value) for key, value in fact_blocks.items() if _clean(value)}
        if ogc_chain:
            fallback["ogc_email_allegations"] = ogc_chain
            logger.debug("[SECTION_HELPER] Combined OGC email allegations")

        return fallback
    except Exception as exc:  # pragma: no cover - best-effort fallback
        logger.warning(f"[SECTION_HELPER] Failed to load case insights for section enforcement: {exc}")
        return {}


def _clean(value: Any) -> str:
    """Return a clean string representation."""
    if not value:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
