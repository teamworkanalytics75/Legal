"""Template library for translating raw fact snippets into canonical propositions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern


@dataclass
class FactTemplate:
    """Represents a reusable regex-based extraction template."""

    name: str
    category: str
    pattern: Pattern[str]
    proposition_format: str
    metadata_defaults: Dict[str, str]

    def extract(self, text: str) -> Optional[Dict[str, str]]:
        """Attempt to match template against text and build proposition metadata."""
        match = self.pattern.search(text)
        if not match:
            return None

        groups = {k: v.strip() if isinstance(v, str) else v for k, v in match.groupdict().items()}
        try:
            proposition = self.proposition_format.format(**groups)
        except KeyError:
            proposition = text.strip()

        metadata = dict(self.metadata_defaults)
        if groups.get("date"):
            metadata.setdefault("EventDate", groups["date"])
        if groups.get("location"):
            metadata.setdefault("EventLocation", groups["location"])
        if groups.get("actor"):
            metadata.setdefault("Speaker", groups["actor"])
        if groups.get("source"):
            metadata.setdefault("SourceDocument", groups["source"])

        return {
            "proposition": proposition.strip().rstrip(".") + ".",
            "metadata": metadata,
        }


def _compile(pattern: str, flags: int = re.IGNORECASE) -> Pattern[str]:
    return re.compile(pattern, flags)


def get_fact_templates() -> List[FactTemplate]:
    """Return the curated set of fact extraction templates."""
    templates: List[FactTemplate] = [
        FactTemplate(
            name="communication_email",
            category="communication",
            pattern=_compile(
                r"on\s+(?P<date>[^,]+),\s+(?P<actor>[^,]+?)\s+sent\s+(?P<target>[^,]+?)\s+(?P<document>[^.]+)"
            ),
            proposition_format="{actor} sent {document} to {target}",
            metadata_defaults={"EvidenceType": "Email", "TruthStatus": "Alleged"},
        ),
        FactTemplate(
            name="filing_event",
            category="filing",
            pattern=_compile(
                r"(?P<actor>[^,]+?)\s+filed\s+(?P<document>[^,]+?)\s+on\s+(?P<date>[^.]+)"
            ),
            proposition_format="{actor} filed {document}",
            metadata_defaults={"EvidenceType": "USFiling", "TruthStatus": "True"},
        ),
        FactTemplate(
            name="publication_event",
            category="publication",
            pattern=_compile(
                r"(?P<platform>WeChat|news|article)[^,]*\s+published\s+(?P<article>[^,]+?)\s+on\s+(?P<date>[^,]+)"
            ),
            proposition_format="{platform} published {article}",
            metadata_defaults={"EvidenceType": "WeChatArticle", "TruthStatus": "HostileFalseClaim"},
        ),
        FactTemplate(
            name="allegation_event",
            category="allegation",
            pattern=_compile(
                r"The\s+(?P<actor>[^,]+?)\s+alleges\s+(?P<claim>[^.]+)"
            ),
            proposition_format="{actor} alleges {claim}",
            metadata_defaults={"TruthStatus": "Alleged"},
        ),
        FactTemplate(
            name="prc_presence",
            category="prc_risk",
            pattern=_compile(
                r"(?P<actor>[^,]+?)\s+was\s+in\s+(?P<location>PRC|China|People's Republic of China)\s+(?P<period>[^.]+)"
            ),
            proposition_format="{actor} was in {location} {period}",
            metadata_defaults={"TruthStatus": "Alleged", "Speaker": "Plaintiff"},
        ),
        FactTemplate(
            name="ogc_action",
            category="ogc",
            pattern=_compile(
                r"Harvard\s+OGC\s+(?P<action>[^,]+?)\s+on\s+(?P<date>[^.]+)"
            ),
            proposition_format="Harvard OGC {action}",
            metadata_defaults={"Speaker": "Harvard", "ActorRole": "Harvard", "EvidenceType": "Email"},
        ),
        FactTemplate(
            name="travel_event",
            category="travel",
            pattern=_compile(
                r"(?P<actor>[^,]+?)\s+traveled\s+to\s+(?P<location>[^,]+?)\s+on\s+(?P<date>[^.]+)"
            ),
            proposition_format="{actor} traveled to {location}",
            metadata_defaults={"TruthStatus": "Alleged"},
        ),
        FactTemplate(
            name="threat_event",
            category="threat",
            pattern=_compile(
                r"(?P<actor>[^,]+?)\s+faced\s+(?P<threat>[^,]+?)\s+due\s+to\s+(?P<cause>[^.]+)"
            ),
            proposition_format="{actor} faced {threat} due to {cause}",
            metadata_defaults={"TruthStatus": "HostileFalseClaim"},
        ),
        FactTemplate(
            name="procedural_event",
            category="procedural",
            pattern=_compile(
                r"(?P<court>[^,]+?)\s+(?P<action>entered|issued|granted|denied)\s+(?P<order>[^,]+?)\s+on\s+(?P<date>[^.]+)"
            ),
            proposition_format="{court} {action} {order}",
            metadata_defaults={"TruthStatus": "True", "EvidenceType": "CourtOrder"},
        ),
        FactTemplate(
            name="wechat_article_quote",
            category="publication",
            pattern=_compile(
                r"The\s+(?P<article>[^,]+?)\s+article\s+described\s+(?P<subject>[^,]+?)\s+as\s+['\"](?P<quote>[^'\"]+)['\"]"
            ),
            proposition_format="{article} article described {subject} as '{quote}'",
            metadata_defaults={"EvidenceType": "WeChatArticle", "TruthStatus": "HostileFalseClaim"},
        ),
        # Communication/Statement templates for correspondence
        FactTemplate(
            name="email_from_to",
            category="communication",
            pattern=_compile(
                r"From:\s+(?P<from>[^<\n]+?)(?:\s*<[^>]+>)?\s+To:\s+(?P<to>[^<\n]+?)(?:\s*<[^>]+>)?"
            ),
            proposition_format="{from} sent email to {to}",
            metadata_defaults={"EvidenceType": "Email", "EventType": "Communication"},
        ),
        FactTemplate(
            name="person_said_statement",
            category="communication",
            pattern=_compile(
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|stated|wrote|noted|indicated|asserted|claimed|mentioned)\s+(?:that\s+)?(?P<statement>[^.!?]+[.!?])"
            ),
            proposition_format="{person} said that {statement}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email"},
        ),
        FactTemplate(
            name="org_said_statement",
            category="communication",
            pattern=_compile(
                r"(?P<org>Harvard\s+(?:OGC|Office\s+of\s+General\s+Counsel|Club|University)|Vivien\s+Chan\s+&\s+Co|Harvard\s+Global\s+Support\s+Services)\s+(?:said|stated|wrote|noted|indicated|asserted|claimed|mentioned|confirmed|reiterated)\s+(?:that\s+)?(?P<statement>[^.!?]+[.!?])"
            ),
            proposition_format="{org} said that {statement}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email", "ActorRole": "Harvard"},
        ),
        FactTemplate(
            name="email_subject_statement",
            category="communication",
            pattern=_compile(
                r"Subject:\s+(?P<subject>[^\n]+)"
            ),
            proposition_format="Email subject: {subject}",
            metadata_defaults={"EvidenceType": "Email", "EventType": "Communication"},
        ),
        FactTemplate(
            name="dated_statement",
            category="communication",
            pattern=_compile(
                r"(?:On|Date:)\s+(?P<date>[^,]+?),\s+(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|stated|wrote|noted|indicated)\s+(?:that\s+)?(?P<statement>[^.!?]+[.!?])"
            ),
            proposition_format="On {date}, {person} said that {statement}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email"},
        ),
        FactTemplate(
            name="non_response_failure",
            category="communication",
            pattern=_compile(
                r"(?P<org>Harvard\s+(?:OGC|Office\s+of\s+General\s+Counsel)|Your\s+office|Harvard)\s+(?:has\s+not|did\s+not|never|failed\s+to)\s+(?:acknowledged?|responded?|respond|acknowledge|reply|replied|confirm|confirmed)\s+(?:receipt|to|that)?"
            ),
            proposition_format="{org} did not respond or acknowledge",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email", "ActorRole": "Harvard", "TruthStatus": "True"},
        ),
        FactTemplate(
            name="no_response_stated",
            category="communication",
            pattern=_compile(
                r"(?:I\s+)?(?:did\s+not|was\s+prepared\s+to\s+delay\s+if\s+I\s+received\s+a\s+timely\s+response\.\s+I\s+did\s+not|has\s+not\s+acknowledged|never\s+responded)"
            ),
            proposition_format="Harvard OGC did not respond to the communication",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email", "ActorRole": "Harvard", "TruthStatus": "True"},
        ),
        FactTemplate(
            name="letter_demand",
            category="communication",
            pattern=_compile(
                r"(?P<org>Harvard\s+University|Vivien\s+Chan\s+&\s+Co)\s+(?:demanded|requested|required)\s+(?:that\s+)?(?P<demand>[^.!?]+[.!?])"
            ),
            proposition_format="{org} demanded that {demand}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Letter", "ActorRole": "Harvard", "TruthStatus": "True"},
        ),
        FactTemplate(
            name="email_warning",
            category="communication",
            pattern=_compile(
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:warned|notified|informed)\s+(?P<recipient>[^,]+?)\s+(?:that\s+)?(?P<warning>[^.!?]+[.!?])"
            ),
            proposition_format="{person} warned {recipient} that {warning}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email"},
        ),
        FactTemplate(
            name="email_deadline",
            category="communication",
            pattern=_compile(
                r"(?P<person>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:set|gave|provided)\s+(?:a\s+)?deadline\s+(?:of\s+)?(?P<date>[^,]+?)\s+(?:for\s+)?(?P<action>[^.!?]+[.!?])"
            ),
            proposition_format="{person} set deadline of {date} for {action}",
            metadata_defaults={"EventType": "Communication", "EvidenceType": "Email"},
        ),
    ]
    return templates


__all__ = ["FactTemplate", "get_fact_templates"]
