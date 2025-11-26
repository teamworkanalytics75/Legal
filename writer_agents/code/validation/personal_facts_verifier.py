"""
Utilities for verifying that generated motions reference personal corpus facts.

This module exposes a single entry point,
`verify_motion_uses_personal_facts`, which scans a motion for mandatory
lawsuit-specific references (HK Statement, OGC emails, critical dates, and
allegations) and reports which facts were found or missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Pattern, Tuple

try:
    from writer_agents.code.validation.fact_graph_query import FactGraphQuery
except Exception:  # pragma: no cover - optional dependency
    FactGraphQuery = None  # type: ignore[misc]

logger = logging.getLogger(__name__)


@dataclass
class FactRule:
    """Represents a single fact requirement with compiled regex patterns."""

    name: str
    description: str
    patterns: Iterable[str]
    optional: bool = False
    is_negative: bool = False
    compiled_patterns: List[Pattern[str]] = field(init=False)

    def __post_init__(self) -> None:
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE) for pattern in self.patterns
        ]

    def match(
        self,
        text: str,
        original_text: str,
        aliases: Optional[Iterable[str]] = None,
    ) -> Optional[Tuple[str, bool]]:
        """Return snippet if rule satisfied.

        Returns (snippet, is_violation) where is_violation indicates a negative rule hit.
        """
        for pattern in self.compiled_patterns:
            match = pattern.search(text)
            if match:
                start, end = match.span()
                snippet = original_text[start:end]
                return snippet, self.is_negative

        if aliases:
            for alias in aliases:
                if not alias:
                    continue
                alias_pattern = re.compile(re.escape(alias), re.IGNORECASE)
                match = alias_pattern.search(original_text)
                if match:
                    start, end = match.span()
                    snippet = original_text[start:end]
                    return snippet, self.is_negative
        return None


# Cluster-based fact taxonomy for seal/pseudonym motions
# Each cluster requires "at least one" fact to be present (not all facts)

# Cluster A - Core case posture & foreign proceeding
_CLUSTER_A_RULES = (
    FactRule(
        name="hk_statement_of_claim",
        description="References the Hong Kong Statement of Claim",
        patterns=(
            r"hong\s+kong\s+statement\s+of\s+claim",
            r"\bhk\s+statement\s+of\s+claim\b",
            r"action\s+no\.?\s*771",
            r"action\s+no\.?\s*771\s+of\s+2025",
            r"court\s+of\s+first\s+instance",
        ),
    ),
    FactRule(
        name="hk_writ",
        description="References the Writ of Summons and its sealed filing",
        patterns=(
            r"writ\s+of\s+summons",
            r"the\s+writ\s+was\s+sealed",
            r"high\s+court\s+of\s+the\s+hong\s+kong\s+special\s+administrative\s+region",
        ),
    ),
    FactRule(
        name="foreign_proceeding_1782",
        description="References the existence of the foreign proceeding and U.S. § 1782 posture",
        patterns=(
            r"28\s+u\.?\s*s\.?\s*c\.?\s*§?\s*1782",
            r"judicial\s+assistance",
            r"foreign\s+tribunal",
            r"hong\s+kong\s+high\s+court",
        ),
    ),
)

# Cluster B - Identity, pseudonymity, sealing
_CLUSTER_B_RULES = (
    FactRule(
        name="pseudonym_request",
        description="Express, explicit request to proceed under pseudonym",
        patterns=(
            r"proceed\s+under\s+a\s+pseudonym",
            r"permission\s+to\s+litigate\s+under\s+a\s+pseudonym",
            r"litigate\s+pseudonymously",
        ),
    ),
    FactRule(
        name="seal_request",
        description="Explicit request to seal filings or materials",
        patterns=(
            r"file\s+under\s+seal",
            r"impoundment",
            r"impound\s+these\s+materials",
            r"seal\s+the\s+docket",
        ),
    ),
    FactRule(
        name="sensitive_identifying_info",
        description="Mentions the risk that disclosure of name, address, personal identifiers would increase danger",
        patterns=(
            r"personally\s+identifying\s+information",
            r"name,\s+address,\s+contact\s+information",
            r"re-?identification",
        ),
    ),
)

# Cluster C - PRC risk, Xi context, and EsuWiki
_CLUSTER_C_RULES = (
    FactRule(
        name="xi_slide",
        description="References the Xi slides (father slide, daughter slide)",
        patterns=(
            r"xi\s+mingze",
            r"xi\s+jinping'?s\s+daughter",
            r"china'?s\s+new\s+princess",
            r"xi\s+zeming",
            r"xi\s+mingze.*harvard",
            r"photograph.*xi\s+mingze",
            r"xi\s+jinping.*daughter.*harvard",
        ),
    ),
    FactRule(
        name="esuwiki_case",
        description="References EsuWiki case 1902136 and its crackdown",
        patterns=(
            r"esuwiki",
            r"恶俗维基",
            r"case\s+1902136",
            r"esuwiki\s+crackdown",
        ),
    ),
    FactRule(
        name="esuwiki_torture",
        description="References torture/coercion of EsuWiki defendants",
        patterns=(
            r"niu\s+tengyu",
            r"\btorture\b",
            r"coercion\s+severity",
        ),
    ),
    FactRule(
        name="state_dept_china_advisory",
        description="References U.S. State Dept Level 2 advisory for China",
        patterns=(
            r"exercise\s+increased\s+caution\s+in\s+china",
            r"arbitrary\s+enforcement\s+of\s+local\s+laws",
            r"level\s+2\s+travel\s+advisory",
        ),
    ),
    FactRule(
        name="harvard_gss_china_risks",
        description="Harvard GSS / OSAC warnings on China travel risk",
        patterns=(
            r"harvard\s+global\s+support\s+services",
            r"\bgss\b",
            r"overseas\s+security\s+advisory\s+council",
            r"\bosac\b",
            r"avoid\s+sensitive\s+topics\s+such\s+as\s+tiananmen\s+square",
        ),
    ),
)

# Cluster D - Publication chain (Statements, Monkey, Résumé)
_CLUSTER_D_RULES = (
    FactRule(
        name="statement1_publication",
        description="Publication of Statement 1 (19 April 2019)",
        patterns=(
            r"statement\s+1\s+was\s+published\s+on\s+19\s+april\s+2019",
            r"19\s+april\s+2019\s+pdf\s+notice",
        ),
    ),
    FactRule(
        name="statement2_publication_backdating",
        description="Publication + backdating of Statement 2 (around 29–30 April 2019)",
        patterns=(
            r"statement\s+2",
            r"29[–-]30\s+april\s+2019",
            r"subsequently\s+re-?dated",
        ),
    ),
    FactRule(
        name="monkey_article",
        description="The Monkey article mocking your race, Harvard credentials, and chess club",
        patterns=(
            r"monkey\s+article",
            r"黑猩猩",
            r"explicitly\s+mocked.*harvard\s+chess\s+club",
            r"mocked.*harvard.*credential",
            r"monkey.*article.*harvard",
            r"monkey.*article.*race",
        ),
    ),
    FactRule(
        name="resume_article",
        description="The Résumé article repeating/weaponizing your résumé",
        patterns=(
            r"résumé\s+article",
            r"wechat\s+résumé\s+article",
        ),
    ),
    FactRule(
        name="wechat_pdf_persistence",
        description="Persistence of Statement 1 PDF on WeChat",
        patterns=(
            r"posted\s+as\s+a\s+pdf\s+in\s+wechat",
            r"re-?downloaded.*statement\s+1.*pdf",
            r"wechat.*pdf",
            r"statement\s+1.*pdf.*wechat",
            r"pdf.*wechat.*2025",
            r"wechat.*statement\s+1",
        ),
    ),
)

# Cluster E - Harm: economic, reputational, safety
_CLUSTER_E_RULES = (
    FactRule(
        name="economic_loss_contracts",
        description="Loss of high-value contracts and work",
        patterns=(
            r"lost\s+contracts",
            r"no\s+longer\s+secured.*usd\s+50,000",
            r"usd\s+50,000.*consulting",
            r"lost.*consulting\s+contracts",
            r"economic\s+loss.*contracts",
            r"50,000.*contracts",
        ),
    ),
    FactRule(
        name="economic_loss_cancellations_refunds",
        description="Cancellations and refund demands",
        patterns=(
            r"cancelled\s+classes",
            r"parents\s+requested\s+refunds",
        ),
    ),
    FactRule(
        name="reputational_harm",
        description="Damage to reputation in China and the Harvard community",
        patterns=(
            r"significant\s+harm\s+to\s+reputation",
            r"destroyed\s+his\s+professional\s+reputation",
        ),
    ),
    FactRule(
        name="mental_health_decline",
        description="Mental health deterioration due to harassment/defamation",
        patterns=(
            r"insufferable\s+emotional\s+pain",
            r"deteriorating\s+mental\s+health",
            r"psychiatric\s+injury",
        ),
    ),
    FactRule(
        name="physical_sickness",
        description="Stress-related or psychiatric injury leading to physical illness",
        patterns=(
            r"physical\s+sickness\s+owing\s+to\s+psychiatric\s+injury",
            r"stress-?related\s+medical\s+harm",
        ),
    ),
    FactRule(
        name="unable_to_leave_china_2019_2022",
        description="You could not safely exit China until July 2022",
        patterns=(
            r"trapped\s+in\s+china",
            r"could\s+not\s+safely\s+leave",
            r"unable\s+to\s+exit.*china",
            r"stuck\s+in\s+china",
            r"could\s+not\s+leave\s+the\s+prc",
            r"unable\s+to\s+leave\s+china.*2022",
            r"july\s+2022.*leave.*china",
        ),
    ),
    FactRule(
        name="prc_surveillance_handlers",
        description="PRC-linked handlers approaching you (2017, 2020)",
        patterns=(
            r"handler.*shanghai",
            r"asked\s+targeted\s+questions\s+consistent\s+with\s+surveillance",
        ),
    ),
)

# Cluster F - Harvard knowledge, non-response, and concealment
_CLUSTER_F_RULES = (
    FactRule(
        name="notice_wang_april_2019",
        description="You warned Yi Wang in April 2019",
        patterns=(
            r"april\s+2019.*yi\s+wang",
            r"yi\s+wang.*april\s+2019",
            r"emailed\s+yi\s+wang",
            r"warned.*yi\s+wang",
            r"april\s+23.*2019.*wang",
            r"attached.*monkey\s+article",
            r"monkey\s+article.*wang",
        ),
    ),
    FactRule(
        name="notice_mcgrath_may_july_2019",
        description="You warned Marlyn McGrath and she closed the matter",
        patterns=(
            r"email\s+to\s+marlyn\s+mcgrath",
            r"thank\s+you\s+for\s+this\s+information",
            r"matter\s+closed",
        ),
    ),
    FactRule(
        name="notice_haa_2019",
        description="HAA was notified in 2019",
        patterns=(
            r"email\s+to\s+haa",
            r"harvard\s+alumni\s+association",
        ),
    ),
    FactRule(
        name="notice_haa_2024_berry",
        description="Anonymous November 4, 2024 inquiry about suing the Clubs",
        patterns=(
            r"aad\s+service\s+desk",
            r"alumni\s+clubs\s+are\s+separate\s+legal\s+entities\s+and\s+are\s+not\s+part\s+of\s+the\s+haa",
        ),
    ),
    FactRule(
        name="notice_ogc_2025",
        description="April & August 2025 OGC emails",
        patterns=(
            r"without\s+prejudice.*settlement",
            r"office\s+of\s+general\s+counsel",
            r"\bogc\b",
            r"exposure\s+to\s+torture",
            r"april\s+2025.*ogc",
            r"ogc.*april\s+2025",
            r"harvard\s+ogc.*2025",
            r"ogc.*email.*2025",
        ),
    ),
    FactRule(
        name="harvard_silence_pattern",
        description="Pattern of Harvard not responding or correcting",
        patterns=(
            r"harvard\s+did\s+not\s+correct\s+or\s+retract",
            r"no\s+substantive\s+reply",
        ),
    ),
)

# Cluster G - Spoliation & reactive deletions (OPTIONAL)
_CLUSTER_G_RULES = (
    FactRule(
        name="backdating_statement2",
        description="Statement 2 backdated to April 19",
        patterns=(
            r"statement\s+2\s+was\s+never\s+published\s+with\s+its\s+real\s+date",
            r"took\s+the\s+place\s+of\s+statement\s+1\s+but\s+kept\s+the\s+april\s+19\s+date",
        ),
        optional=True,
    ),
    FactRule(
        name="2024_shanghai_site_disappearance",
        description="HCS website disappears after HAA/Berry reply",
        patterns=(
            r"harvard\s+club\s+of\s+shanghai\s+website\s+disappeared",
            r"within\s+approximately\s+one\s+month\s+of\s+the\s+haa\s+response",
        ),
        optional=True,
    ),
    FactRule(
        name="2025_hk_toggle",
        description="HK statement disappears and reappears around April 7/18 emails",
        patterns=(
            r"hk\s+club\s+statement\s+was\s+visible\s+on\s+april\s+7,\s+2025.*gone\s+on\s+april\s+18.*back\s+on\s+april\s+19",
        ),
        optional=True,
    ),
)

# Cluster H - Doxxing & privacy/exposure (OPTIONAL)
_CLUSTER_H_RULES = (
    FactRule(
        name="doxxing_monkey_resume_articles",
        description="Monkey and Résumé articles function as doxxing",
        patterns=(
            r"doxxing",
            r"public\s+disclosure\s+of\s+personal\s+information",
            r"targeted\s+his\s+race\s+and\s+immigration\s+status",
        ),
        optional=True,
    ),
    FactRule(
        name="wechat_groups_exposure",
        description="Exposure via Harvard WeChat groups",
        patterns=(
            r"harvard\s+alumni\s+wechat\s+group",
            r"pdf\s+circulated\s+in\s+wechat\s+groups",
        ),
        optional=True,
    ),
    FactRule(
        name="harvard_center_shanghai_privacy_policy",
        description="HCS privacy disclosures vis-à-vis your data",
        patterns=(
            r"harvard\s+center\s+shanghai\s+personal\s+information\s+privacy\s+disclosures",
            r"\bpipl\b",
        ),
        optional=True,
    ),
)

# Cluster I - Disability & vulnerability (OPTIONAL)
_CLUSTER_I_RULES = (
    FactRule(
        name="disability_notice_rockefeller",
        description="Disclosure of disability in Rockefeller Fellowship essay",
        patterns=(
            r"2014\s+rockefeller\s+fellowship\s+application\s+essay",
            r"disclosed\s+his\s+disability",
        ),
        optional=True,
    ),
    FactRule(
        name="harvard_knew_vulnerability",
        description="Harvard knew you were vulnerable and disabled",
        patterns=(
            r"harvard\s+had\s+notice\s+of\s+his\s+disability",
            r"disability/ vulnerability",
        ),
        optional=True,
    ),
)

# Backward compatibility rules for old fact names
# These map old names to new cluster-based facts
_BACKWARD_COMPAT_RULES = (
    FactRule(
        name="hk_statement",
        description="References the Hong Kong Statement of Claim (backward compatibility)",
        patterns=(
            r"hong\s+kong\s+statement\s+of\s+claim",
            r"\bhk\s+statement\b",
            r"action\s+no\.?\s*771",
        ),
        optional=True,  # Optional since hk_statement_of_claim is the primary rule
    ),
    FactRule(
        name="ogc_emails",
        description="References the Harvard OGC emails (backward compatibility)",
        patterns=(
            r"office\s+of\s+general\s+counsel",
            r"\bogc\b",
            r"ogc\s+email",
        ),
        optional=True,  # Optional since notice_ogc_2025 is the primary rule
    ),
    FactRule(
        name="allegation_defamation",
        description="Explains defamation allegation",
        patterns=(
            r"\bdefamation\b",
            r"\bdefamatory\b",
        ),
        # Not optional - defamation is a core allegation
    ),
)

# Combine all clusters into a flat list for backward compatibility
# Required clusters: A (base), B (identity), C (PRC risk), D (publication), E (harm), F (knowledge)
# Optional clusters: G (spoliation), H (doxxing), I (vulnerability)
DEFAULT_FACT_RULES: Tuple[FactRule, ...] = (
    *_CLUSTER_A_RULES,
    *_CLUSTER_B_RULES,
    *_CLUSTER_C_RULES,
    *_CLUSTER_D_RULES,
    *_CLUSTER_E_RULES,
    *_CLUSTER_F_RULES,
    *_CLUSTER_G_RULES,  # Optional but recommended
    *_CLUSTER_H_RULES,  # Optional but recommended
    *_CLUSTER_I_RULES,  # Optional but recommended
    *_BACKWARD_COMPAT_RULES,  # Backward compatibility for old fact names
)

# Cluster definitions for cluster-based verification
FACT_CLUSTERS: Dict[str, Tuple[FactRule, ...]] = {
    "cluster_base": _CLUSTER_A_RULES,
    "cluster_identity": _CLUSTER_B_RULES,
    "cluster_prc_risk": _CLUSTER_C_RULES,
    "cluster_publication": _CLUSTER_D_RULES,
    "cluster_harm": _CLUSTER_E_RULES,
    "cluster_knowledge": _CLUSTER_F_RULES,
    "cluster_spoliation": _CLUSTER_G_RULES,  # Optional
    "cluster_doxxing": _CLUSTER_H_RULES,  # Optional
    "cluster_vulnerability": _CLUSTER_I_RULES,  # Optional
}

# Required clusters (must have at least one fact from each)
REQUIRED_CLUSTERS: List[str] = [
    "cluster_base",
    "cluster_identity",
    "cluster_prc_risk",
    "cluster_publication",
    "cluster_harm",
    "cluster_knowledge",
]

RULE_TO_GRAPH_TYPE: Dict[str, str] = {
    # Cluster A - Core case posture
    "hk_statement_of_claim": "document_reference",
    "hk_statement": "document_reference",  # Backward compatibility
    "hk_writ": "document_reference",
    "foreign_proceeding_1782": "legal_proceeding",
    # Cluster B - Identity
    "pseudonym_request": "relief_request",
    "seal_request": "relief_request",
    "sensitive_identifying_info": "privacy_risk",
    # Cluster C - PRC risk
    "xi_slide": "safety_risk",
    "esuwiki_case": "safety_risk",
    "esuwiki_torture": "safety_risk",
    "state_dept_china_advisory": "safety_risk",
    "harvard_gss_china_risks": "safety_risk",
    # Cluster D - Publication
    "statement1_publication": "publication_event",
    "statement2_publication_backdating": "publication_event",
    "monkey_article": "publication_event",
    "resume_article": "publication_event",
    "wechat_pdf_persistence": "publication_event",
    # Cluster E - Harm
    "economic_loss_contracts": "harm",
    "economic_loss_cancellations_refunds": "harm",
    "reputational_harm": "harm",
    "mental_health_decline": "harm",
    "physical_sickness": "harm",
    "unable_to_leave_china_2019_2022": "harm",
    "prc_surveillance_handlers": "harm",
    # Cluster F - Knowledge
    "notice_wang_april_2019": "notice_event",
    "notice_mcgrath_may_july_2019": "notice_event",
    "notice_haa_2019": "notice_event",
    "notice_haa_2024_berry": "notice_event",
    "notice_ogc_2025": "notice_event",
    "ogc_emails": "notice_event",  # Backward compatibility
    "harvard_silence_pattern": "notice_event",
    # Backward compatibility - allegations
    "allegation_defamation": "allegation",
    # Cluster G - Spoliation
    "backdating_statement2": "spoliation",
    "2024_shanghai_site_disappearance": "spoliation",
    "2025_hk_toggle": "spoliation",
    # Cluster H - Doxxing
    "doxxing_monkey_resume_articles": "privacy_breach",
    "wechat_groups_exposure": "privacy_breach",
    "harvard_center_shanghai_privacy_policy": "privacy_breach",
    # Cluster I - Vulnerability
    "disability_notice_rockefeller": "vulnerability",
    "harvard_knew_vulnerability": "vulnerability",
}


def _gather_aliases(personal_corpus_facts: Dict[str, Any]) -> Dict[str, List[str]]:
    """Return aliases per rule name derived from user-provided data."""
    aliases: Dict[str, List[str]] = {}
    provided_aliases = personal_corpus_facts.get("aliases", {})
    if isinstance(provided_aliases, dict):
        for key, values in provided_aliases.items():
            if isinstance(values, (list, tuple, set)):
                aliases[key] = [str(value) for value in values if value]
            elif isinstance(values, str):
                aliases[key] = [values]

    # If case_insights style dict passed, use fact_blocks text to enhance
    fact_blocks = personal_corpus_facts.get("fact_blocks")
    if isinstance(fact_blocks, dict):
        # Cluster A - HK Statement
        hk_text = fact_blocks.get("hk_retaliation_events") or fact_blocks.get("hk_allegation_defamation") or fact_blocks.get("hk_statement_of_claim")
        if hk_text:
            aliases.setdefault("hk_statement_of_claim", []).append(hk_text[:120])
            aliases.setdefault("hk_statement", []).append(hk_text[:120])  # Backward compatibility
        
        # Cluster F - OGC emails
        ogc_text = fact_blocks.get("ogc_email_1_threat") or fact_blocks.get("ogc_email_allegations")
        if ogc_text:
            aliases.setdefault("notice_ogc_2025", []).append(ogc_text[:120])
            aliases.setdefault("ogc_emails", []).append(ogc_text[:120])  # Backward compatibility

    return aliases


def _normalize_motion_text(motion_text: str) -> Tuple[str, str]:
    if not isinstance(motion_text, str):
        raise TypeError("motion_text must be a string")
    stripped = motion_text.strip()
    return stripped, stripped.lower()


def _extract_aliases_from_case_insights(case_insights: Dict[str, Any]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    fact_blocks = case_insights.get("fact_blocks") or {}
    case_summary = case_insights.get("case_summary", "")

    action_pattern = re.compile(r"action\s+no\.?\s*\d+", re.IGNORECASE)
    hk_phrase_pattern = re.compile(r"(hk|hong\s+kong)\s+statement", re.IGNORECASE)

    def _append_alias(key: str, value: str) -> None:
        normalized = value.strip()
        if not normalized:
            return
        aliases.setdefault(key, [])
        if normalized not in aliases[key]:
            aliases[key].append(normalized[:240])

    def add_alias(key: str, text: Optional[str]) -> None:
        if not text:
            return
        snippet = text.strip()
        if not snippet:
            return
        _append_alias(key, snippet)
        for candidate in re.split(r"[.;]", snippet):
            candidate = candidate.strip()
            if 5 <= len(candidate) <= 160:
                _append_alias(key, candidate)
        match = action_pattern.search(snippet)
        if match:
            _append_alias(key, match.group(0))
        match = hk_phrase_pattern.search(snippet)
        if match:
            _append_alias(key, match.group(0))

    # Cluster A - HK Statement
    add_alias("hk_statement_of_claim", fact_blocks.get("hk_allegation_defamation"))
    add_alias("hk_statement_of_claim", fact_blocks.get("hk_retaliation_events"))
    add_alias("hk_statement", fact_blocks.get("hk_allegation_defamation"))  # Backward compatibility
    add_alias("hk_statement", fact_blocks.get("hk_retaliation_events"))  # Backward compatibility
    
    # Cluster F - OGC emails
    add_alias("notice_ogc_2025", fact_blocks.get("ogc_email_1_threat"))
    add_alias("notice_ogc_2025", fact_blocks.get("ogc_email_2_non_response"))
    add_alias("notice_ogc_2025", fact_blocks.get("ogc_email_3_meet_confer"))
    add_alias("ogc_emails", fact_blocks.get("ogc_email_1_threat"))  # Backward compatibility
    add_alias("ogc_emails", fact_blocks.get("ogc_email_2_non_response"))  # Backward compatibility
    add_alias("ogc_emails", fact_blocks.get("ogc_email_3_meet_confer"))  # Backward compatibility
    
    # Cluster C - PRC risk
    add_alias("xi_slide", fact_blocks.get("hk_allegation_ccp_family"))
    
    # Cluster E - Harm
    add_alias("unable_to_leave_china_2019_2022", fact_blocks.get("safety_concerns"))
    add_alias("prc_surveillance_handlers", fact_blocks.get("harvard_retaliation_events"))

    if "Hong Kong Statement of Claim" in case_summary:
        add_alias("hk_statement", "Hong Kong Statement of Claim")
    if "Action No. 771" in case_summary:
        add_alias("hk_statement", "Action No. 771")
    if "Office of General Counsel" in case_summary:
        add_alias("ogc_emails", "Office of General Counsel")

    return aliases


def _aliases_from_corpus_dir(corpus_dir: Optional[Path]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}
    if not corpus_dir:
        return aliases

    try:
        if not corpus_dir.exists():
            return aliases
    except OSError:
        return aliases

    for path in sorted(corpus_dir.glob("*.txt")):
        stem = path.stem.strip()
        if not stem:
            continue
        lowered = stem.lower()
        if any(keyword in lowered for keyword in ("statement", "claim")):
            aliases.setdefault("hk_statement", []).append(stem)
        if "ogc" in lowered or "email" in lowered:
            aliases.setdefault("ogc_emails", []).append(stem)
            aliases.setdefault("timeline_april_ogc_emails", []).append(stem)
        if "june" in lowered and ("arrest" in lowered or "detention" in lowered):
            aliases.setdefault("timeline_june_2025_arrests", []).append(stem)
    return aliases


def verify_motion_uses_personal_facts(
    motion_text: str,
    personal_corpus_facts: Optional[Dict[str, Any]] = None,
    required_rules: Optional[Iterable[FactRule]] = None,
    negative_rules: Optional[Iterable[FactRule]] = None,
    fact_graph_query: Optional["FactGraphQuery"] = None,
) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """
    Verify that the motion references lawsuit-specific personal facts.

    Args:
        motion_text: Generated motion text to inspect.
        personal_corpus_facts: Optional dict with metadata (case_insights,
            alias overrides, etc.).
        required_rules: Optional iterable of FactRule overrides for advanced use.

    Returns:
        Tuple of (is_valid, missing_fact_names, verification_details)
    """
    personal_corpus_facts = personal_corpus_facts or {}
    rules = tuple(required_rules) if required_rules else DEFAULT_FACT_RULES
    original_text, normalized_text = _normalize_motion_text(motion_text)
    aliases_map = _gather_aliases(personal_corpus_facts)

    matches: Dict[str, str] = {}
    evaluated_rules: List[Dict[str, Any]] = []
    missing: List[str] = []
    violations: List[str] = []
    negative_rules = tuple(negative_rules) if negative_rules else NEGATIVE_FACT_RULES

    motion_lower = original_text.lower()
    graph_fact_coverage: Dict[str, Any] = {}
    graph_missing_pairs: List[Dict[str, str]] = []
    graph_suggestions: List[str] = []

    graph_summary: Optional[Dict[str, Any]] = None
    if fact_graph_query and FactGraphQuery:
        graph_fact_coverage = _graph_fact_coverage(motion_lower, fact_graph_query)
        if graph_fact_coverage:
            graph_summary = _summarize_graph_coverage(graph_fact_coverage)
            graph_missing_pairs = graph_summary.get("missing_facts", [])
            graph_suggestions = graph_summary.get("suggestions", [])

    for rule in rules:
        alias_values = aliases_map.get(rule.name, [])
        match_result = rule.match(normalized_text, original_text, alias_values)
        matched = match_result is not None and not (match_result and match_result[1])
        graph_snippet: Optional[str] = None
        if not matched and fact_graph_query and FactGraphQuery:
            graph_snippet = _match_fact_via_graph(rule.name, motion_lower, fact_graph_query)
            matched = bool(graph_snippet)
        evaluated_rules.append(
            {
                "name": rule.name,
                "description": rule.description,
                "matched": matched,
                "optional": rule.optional,
                "alias_count": len(alias_values),
            }
        )
        if match_result:
            snippet, _ = match_result
            matches[rule.name] = snippet
        elif graph_snippet:
            matches[rule.name] = graph_snippet
        elif not rule.optional:
            missing.append(rule.name)

    # Evaluate negative rules (violations)
    evaluated_negative: List[Dict[str, Any]] = []
    for rule in negative_rules:
        match_result = rule.match(normalized_text, original_text, aliases_map.get(rule.name))
        violation = bool(match_result)
        evaluated_negative.append(
            {
                "name": rule.name,
                "description": rule.description,
                "violated": violation,
            }
        )
        if violation:
            snippet, _ = match_result  # Negative rules always treat match as violation
            violations.append(rule.name)
            matches[rule.name] = snippet

    # Calculate cluster-based coverage
    cluster_coverage = _calculate_cluster_coverage(matches, evaluated_rules)

    total_required = len([rule for rule in rules if not rule.optional])
    coverage = (total_required - len(missing)) / total_required if total_required else 1.0

    details: Dict[str, Any] = {
        "matches": matches,
        "evaluated_rules": evaluated_rules,
        "negative_rules": evaluated_negative,
        "coverage": round(coverage, 4),
        "total_required": total_required,
        "violations": violations,
        "cluster_coverage": cluster_coverage,
    }
    if graph_fact_coverage:
        details["graph_fact_coverage"] = graph_fact_coverage
        details["graph_missing_facts"] = graph_missing_pairs
        if graph_summary is None:
            graph_summary = _summarize_graph_coverage(graph_fact_coverage)
        details["graph_coverage_summary"] = graph_summary

    # Cluster-based validation: require at least one fact from each required cluster
    cluster_coverage_data = details.get("cluster_coverage", {})
    missing_clusters = cluster_coverage_data.get("missing_clusters", [])
    cluster_satisfaction_rate = cluster_coverage_data.get("cluster_satisfaction_rate", 0.0)
    
    # Motion is valid if:
    # 1. All required clusters have at least one fact (cluster_satisfaction_rate == 1.0)
    # 2. No violations (negative rules)
    # 3. For backward compatibility, also check if using old flat list approach
    cluster_valid = cluster_satisfaction_rate >= 1.0 and not violations
    flat_valid = len(missing) == 0 and not violations
    
    # Prefer cluster-based validation, but fall back to flat if clusters not available
    is_valid = cluster_valid if cluster_coverage_data else flat_valid
    
    # Update missing list to include missing clusters for better reporting
    if missing_clusters:
        details["missing_clusters"] = missing_clusters
        details["cluster_validation_failed"] = True
    
    return is_valid, missing, violations, details


def _match_fact_via_graph(rule_name: str, motion_lower: str, fact_query: "FactGraphQuery") -> Optional[str]:
    graph_type = RULE_TO_GRAPH_TYPE.get(rule_name)
    if not graph_type:
        return None
    try:
        candidates = fact_query.get_all_facts_by_type(graph_type)
    except Exception as exc:  # pragma: no cover - defensive
        logger = logging.getLogger(__name__)
        logger.debug("KnowledgeGraph lookup failed for %s: %s", rule_name, exc)
        candidates = []

    for candidate in candidates:
        value = (candidate.get("value") or candidate.get("fact_value") or "").strip()
        if value and value.lower() in motion_lower:
            return value

    semantic_match = _match_fact_semantically(motion_lower, graph_type, fact_query)
    return semantic_match


def _match_fact_semantically(
    motion_text: str,
    fact_type: str,
    fact_query: "FactGraphQuery",
) -> Optional[str]:
    try:
        matches = fact_query.find_similar_facts(
            motion_text,
            fact_type=fact_type,
            top_k=1,
            similarity_threshold=0.4,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger = logging.getLogger(__name__)
        logger.debug("KnowledgeGraph semantic match failed for %s: %s", fact_type, exc)
        return None
    if matches:
        return matches[0].fact_value
    return None


def _graph_fact_coverage(
    motion_lower: str,
    fact_query: "FactGraphQuery",
) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    try:
        hierarchy = fact_query.get_fact_hierarchy()
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("KnowledgeGraph hierarchy lookup failed: %s", exc)
        return coverage

    for fact_type, values in hierarchy.items():
        normalized_values = [
            str(value).strip()
            for value in values
            if isinstance(value, str) and str(value).strip()
        ]
        if not normalized_values:
            continue
        present: List[str] = []
        missing: List[str] = []
        for value in normalized_values:
            token = value.lower()
            if token and token in motion_lower:
                present.append(value)
            else:
                missing.append(value)
        coverage[fact_type] = {
            "total": len(normalized_values),
            "present": present,
            "missing": missing,
            "coverage": round(len(present) / len(normalized_values), 4),
        }
    return coverage


def _calculate_cluster_coverage(
    matches: Dict[str, str],
    evaluated_rules: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate cluster-based coverage (at least one fact per required cluster)."""
    cluster_stats: Dict[str, Dict[str, Any]] = {}
    
    # Build a map of rule name to cluster
    rule_to_cluster: Dict[str, str] = {}
    for cluster_name, cluster_rules in FACT_CLUSTERS.items():
        for rule in cluster_rules:
            rule_to_cluster[rule.name] = cluster_name
    
    # Count matches per cluster (only non-optional facts count for satisfaction)
    for cluster_name, cluster_rules in FACT_CLUSTERS.items():
        matched_facts = []
        matched_required_facts = []
        total_facts = len(cluster_rules)
        total_required_facts = len([r for r in cluster_rules if not r.optional])
        
        for rule in cluster_rules:
            if rule.name in matches:
                matched_facts.append(rule.name)
                if not rule.optional:
                    matched_required_facts.append(rule.name)
        
        # Cluster is satisfied if at least one non-optional fact is matched
        # (or if all facts in cluster are optional, then any match satisfies it)
        is_satisfied = (
            len(matched_required_facts) > 0
            if total_required_facts > 0
            else len(matched_facts) > 0
        )
        
        cluster_stats[cluster_name] = {
            "total_facts": total_facts,
            "total_required_facts": total_required_facts,
            "matched_facts": matched_facts,
            "matched_required_facts": matched_required_facts,
            "matched_count": len(matched_facts),
            "coverage": len(matched_facts) / total_facts if total_facts > 0 else 0.0,
            "satisfied": is_satisfied,
            "required": cluster_name in REQUIRED_CLUSTERS,
        }
    
    # Calculate overall cluster satisfaction
    required_clusters_satisfied = sum(
        1 for cluster_name in REQUIRED_CLUSTERS
        if cluster_stats.get(cluster_name, {}).get("satisfied", False)
    )
    total_required_clusters = len(REQUIRED_CLUSTERS)
    cluster_satisfaction_rate = (
        required_clusters_satisfied / total_required_clusters
        if total_required_clusters > 0
        else 1.0
    )
    
    missing_clusters = [
        cluster_name
        for cluster_name in REQUIRED_CLUSTERS
        if not cluster_stats.get(cluster_name, {}).get("satisfied", False)
    ]
    
    return {
        "cluster_stats": cluster_stats,
        "required_clusters_satisfied": required_clusters_satisfied,
        "total_required_clusters": total_required_clusters,
        "cluster_satisfaction_rate": round(cluster_satisfaction_rate, 4),
        "missing_clusters": missing_clusters,
        "rule_to_cluster": rule_to_cluster,
    }


def _summarize_graph_coverage(
    graph_coverage: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_types": len(graph_coverage),
        "missing_facts": [],
        "suggestions": [],
    }
    if not graph_coverage:
        summary["average_coverage"] = 0.0
        return summary

    coverage_values = [section.get("coverage", 0.0) for section in graph_coverage.values()]
    summary["average_coverage"] = round(sum(coverage_values) / len(coverage_values), 4)

    missing_entries: List[Dict[str, str]] = []
    suggestions: List[str] = []
    for fact_type, section in graph_coverage.items():
        for value in section.get("missing", [])[:3]:
            entry = {"fact_type": fact_type, "value": value}
            missing_entries.append(entry)
            if len(suggestions) < 5:
                suggestions.append(f"Add detail for {value} ({fact_type})")
    summary["missing_facts"] = missing_entries
    summary["suggestions"] = suggestions
    return summary


def verify_motion_with_case_insights(
    motion_text: str,
    case_insights_path: Optional[Path] = None,
    corpus_dir: Optional[Path] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Convenience wrapper that loads lawsuit_facts_extracted.json and extracts aliases automatically.

    Args:
        motion_text: Generated motion text to verify.
        case_insights_path: Optional path to lawsuit_facts_extracted.json
            (defaults to writer_agents/outputs/lawsuit_facts_extracted.json, also supports case_insights.json for backward compatibility).
        corpus_dir: Personal corpus directory (defaults to case_law_data/lawsuit_source_documents).

    Returns:
        Tuple of (is_valid, missing_fact_names, verification_details).
    """
    # Try new name first, then old name for backward compatibility
    default_path = Path("writer_agents/outputs/lawsuit_facts_extracted.json")
    if not case_insights_path:
        case_insights_path = default_path if default_path.exists() else Path("writer_agents/outputs/case_insights.json")
    elif not case_insights_path.exists() and case_insights_path.name == "case_insights.json":
        # If old name provided but doesn't exist, try new name
        new_path = case_insights_path.parent / "lawsuit_facts_extracted.json"
        if new_path.exists():
            case_insights_path = new_path
    if not case_insights_path.exists():
        raise FileNotFoundError(f"Case insights file not found: {case_insights_path}")

    try:
        case_insights = json.loads(case_insights_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to parse case insights JSON: {case_insights_path}") from exc

    personal_data: Dict[str, Any] = {
        "fact_blocks": case_insights.get("fact_blocks") or {},
    }

    alias_map = _extract_aliases_from_case_insights(case_insights)
    default_corpus = Path("case_law_data/lawsuit_source_documents")
    old_corpus = Path("case_law_data/tmp_corpus")
    corpus_path = corpus_dir or (default_corpus if default_corpus.exists() else old_corpus)
    corpus_aliases = _aliases_from_corpus_dir(corpus_path)
    for key, values in corpus_aliases.items():
        alias_map.setdefault(key, [])
        for value in values:
            if value not in alias_map[key]:
                alias_map[key].append(value)
    if alias_map:
        personal_data["aliases"] = alias_map

    return verify_motion_uses_personal_facts(motion_text, personal_data)


__all__ = [
    "verify_motion_uses_personal_facts",
    "verify_motion_with_case_insights",
    "FactRule",
    "DEFAULT_FACT_RULES",
    "FACT_CLUSTERS",
    "REQUIRED_CLUSTERS",
]
NEGATIVE_FACT_RULES: Tuple[FactRule, ...] = (
    FactRule(
        name="not_prc_citizen",
        description="Must NOT claim PRC citizenship when source documents state US citizenship",
        patterns=(
            r"home\s+country\s+of\s+prc",
            r"prc\s+citizen",
            r"citizen\s+of\s+prc",
            r"prc\s+national",
            r"national\s+of\s+prc",
        ),
        is_negative=True,
    ),
    # Example extensibility: forbid fabricated courthouse locations
    FactRule(
        name="not_wrong_court_location",
        description="Must NOT relocate the courthouse/case venue to an unsupported city",
        patterns=(
            r"district\s+of\s+hong\s+kong",
            r"beijing\s+district\s+court",
        ),
        is_negative=True,
    ),
)
