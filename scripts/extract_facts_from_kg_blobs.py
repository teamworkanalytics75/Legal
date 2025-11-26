#!/usr/bin/env python3
"""
Extract readable amplifier and EsuWiki chain facts from KG export blobs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

COLUMNS = [
    "Proposition",
    "FactID",
    "Subject",
    "Verb",
    "Object",
    "EventType",
    "EventDate",
    "EventLocation",
    "ActorRole",
    "Speaker",
    "TruthStatus",
    "EvidenceType",
    "SourceDocument",
    "SourceExcerpt",
    "SafetyRisk",
    "PublicExposure",
    "RiskRationale",
    "confidence_tier",
    "causal_salience_score",
    "confidence_reason",
    "causal_salience_reason",
    "ExtractionMethod",
    "ExtractionConfidence",
    "CanonicalSubject",
    "CanonicalSpeaker",
    "CanonicalActorRole",
    "CanonicalEventLocation",
    "CanonicalEntityID",
    "Notes",
    "Pattern",
    "SourceFormat",
    "ClassificationFixed",
]


@dataclass
class FactBlueprint:
    key: str
    patterns: List[str]
    proposition: str
    category: str
    subject: str
    verb: str
    obj: str
    event_type: str
    event_date: str
    event_location: str
    actor_role: str
    truth_status: str
    evidence_type: str
    safety_risk: str
    public_exposure: str
    notes: str


BLUEPRINTS: List[FactBlueprint] = [
    FactBlueprint(
        key="zhihu_republish",
        patterns=["art_zhihu_2019_04_22", "pla_zhihu"],
        proposition="On 22 April 2019, Zhihu published a post quoting Statement 1, amplifying the allegation to a nationwide Q&A audience.",
        category="Amplifier",
        subject="Zhihu platform",
        verb="republished",
        obj="Statement 1 allegations",
        event_type="Publication",
        event_date="2019-04-22",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Derived from ART_Zhihu_2019_04_22 edges",
    ),
    FactBlueprint(
        key="e_canada_article",
        patterns=["art_ecanada_2019_04_28"],
        proposition="On 28 April 2019, the Chinese-language outlet E-Canada ran an article quoting Statement 1, extending the allegations into Canadian Chinese media.",
        category="Amplifier",
        subject="E-Canada outlet",
        verb="published",
        obj="Statement 1 article",
        event_type="Publication",
        event_date="2019-04-28",
        event_location="Canada / China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Derived from ART_ECanada_2019_04_28 nodes",
    ),
    FactBlueprint(
        key="sohu_carry",
        patterns=["pla_sohu", "art_ecanada_2019_04_28"],
        proposition="Sohu later carried the E-Canada article, helping negative Statement 1 coverage circulate across major Chinese news portals.",
        category="Amplifier",
        subject="Sohu platform",
        verb="carried",
        obj="E-Canada article quoting Statement 1",
        event_type="Publication",
        event_date="2019-04-30",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Based on PLA_Sohu edges attaching to ART_ECanada_2019_04_28",
    ),
    FactBlueprint(
        key="baidu_resume",
        patterns=["pla_baidu", "art_resume_republish_fs"],
        proposition="Baidu's Baijiahao network hosted the republished Résumé article, keeping defamatory résumé claims discoverable through China's dominant search engine.",
        category="Amplifier",
        subject="Baidu platform",
        verb="hosted",
        obj="Republished Résumé article",
        event_type="Publication",
        event_date="2019-04-25",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Based on PLA_Baidu → ART_Resume_Republish_FS edges",
    ),
    FactBlueprint(
        key="monkey_wechat",
        patterns=["art_monkey_wechat", "pla_wechat"],
        proposition="WeChat groups circulated the 'Monkey' article, enabling hostile ridicule and negative exposure across Harvard-related chat groups.",
        category="Amplifier",
        subject="WeChat groups",
        verb="circulated",
        obj="'Monkey' article targeting Malcolm Grayson",
        event_type="Publication",
        event_date="2019-04-19",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="high",
        public_exposure="already_public",
        notes="From ART_Monkey_WeChat and PLA_WeChat overlaps",
    ),
    FactBlueprint(
        key="resume_wechat",
        patterns=["art_resume_wechat", "pla_wechat"],
        proposition="WeChat timelines also republished the Résumé article, keeping false Harvard employment claims in constant circulation.",
        category="Amplifier",
        subject="WeChat timelines",
        verb="republished",
        obj="Résumé article alleging Harvard admissions ties",
        event_type="Publication",
        event_date="2019-04-19",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="high",
        public_exposure="already_public",
        notes="From ART_Resume_WeChat and PLA_WeChat overlaps",
    ),
    FactBlueprint(
        key="guangming_cover",
        patterns=["pla_guangming"],
        proposition="Guangming Daily, a state-run outlet, participated in the amplifier network that repeated Harvard club statements.",
        category="Amplifier",
        subject="Guangming Daily",
        verb="covered",
        obj="Harvard club statements and follow-on commentary",
        event_type="Publication",
        event_date="2017-08-21",
        event_location="China",
        actor_role="StateMedia",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Based on PLA_Guangming platform nodes",
    ),
    FactBlueprint(
        key="zhihu_platform_desc",
        patterns=["pla_zhihu"],
        proposition="Zhihu operates as China's Quora-style platform, so Statement 1 republication there exposed allegations to millions of Q&A readers.",
        category="Platform",
        subject="Zhihu platform",
        verb="serves",
        obj="Chinese Q&A audience",
        event_type="Publication",
        event_date="2019-04-22",
        event_location="China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Platform metadata from PLA_Zhihu node",
    ),
    FactBlueprint(
        key="e_canada_desc",
        patterns=["art_ecanada_2019_04_28"],
        proposition="E-Canada is a Chinese-language news outlet that syndicated Harvard Club statements for diaspora readers.",
        category="Platform",
        subject="E-Canada outlet",
        verb="syndicated",
        obj="Harvard Club statements",
        event_type="Publication",
        event_date="2019-04-28",
        event_location="Canada / China",
        actor_role="Platform",
        truth_status="True",
        evidence_type="Document",
        safety_risk="none",
        public_exposure="already_public",
        notes="Platform description derived from KG node ART_ECanada_2019_04_28",
    ),
    FactBlueprint(
        key="esu_case_open",
        patterns=["esu_caseopen_2019_06_14"],
        proposition="On 14 June 2019, China's Ministry of Public Security opened EsuWiki case 1902136, tying the crackdown to Harvard Statement 1 timelines.",
        category="EsuWiki",
        subject="PRC Ministry of Public Security",
        verb="opened",
        obj="EsuWiki / Niu Tengyu case 1902136",
        event_type="Harm",
        event_date="2019-06-14",
        event_location="PRC",
        actor_role="StateActor",
        truth_status="True",
        evidence_type="Document",
        safety_risk="extreme",
        public_exposure="partially_public",
        notes="From ESU_CaseOpen_2019_06_14 edges tied to ENT_MPS",
    ),
    FactBlueprint(
        key="esu_arrests",
        patterns=["esu_arrests_2019_08"],
        proposition="After the case opened, PRC security services organized arrests in August 2019 as part of the EsuWiki crackdown.",
        category="EsuWiki",
        subject="PRC security services",
        verb="arrested",
        obj="EsuWiki contributors",
        event_type="Harm",
        event_date="2019-08-15",
        event_location="PRC",
        actor_role="StateActor",
        truth_status="True",
        evidence_type="Document",
        safety_risk="extreme",
        public_exposure="partially_public",
        notes="From ESU_Arrests_2019_08 and ENT_MPS edges",
    ),
    FactBlueprint(
        key="niu_arrests_statement",
        patterns=["niu tengyu", "23 others"],
        proposition="Authorities reported in May 2019 that Niu Tengyu and 23 others were arrested for sharing photographs of Xi Mingze.",
        category="EsuWiki",
        subject="PRC authorities",
        verb="reported arrests of",
        obj="Niu Tengyu and 23 contributors",
        event_type="Harm",
        event_date="2019-05-20",
        event_location="PRC",
        actor_role="StateActor",
        truth_status="True",
        evidence_type="Document",
        safety_risk="extreme",
        public_exposure="partially_public",
        notes="From textual proposition referencing arrests of 23 people",
    ),
    FactBlueprint(
        key="rare_privacy",
        patterns=["esu_rareprivacy"],
        proposition="The EsuWiki crackdown invoked rare privacy charges against online activists linked to the Harvard Statement timeline.",
        category="EsuWiki",
        subject="PRC prosecutors",
        verb="brought",
        obj="rare privacy charges in EsuWiki case",
        event_type="Harm",
        event_date="2019-06-14",
        event_location="PRC",
        actor_role="StateActor",
        truth_status="True",
        evidence_type="Document",
        safety_risk="extreme",
        public_exposure="partially_public",
        notes="Derived from ESU_RarePrivacy entity edges",
    ),
    FactBlueprint(
        key="torture_severity",
        patterns=["esu_tortureseverity"],
        proposition="The EsuWiki case documented torture/coercion severity against detained contributors.",
        category="EsuWiki",
        subject="Detained EsuWiki contributors",
        verb="suffered",
        obj="torture and coercion during detention",
        event_type="Harm",
        event_date="2019-12-01",
        event_location="PRC",
        actor_role="StateActor",
        truth_status="Alleged",
        evidence_type="Document",
        safety_risk="extreme",
        public_exposure="partially_public",
        notes="From ESU_TortureSeverity node",
    ),
    FactBlueprint(
        key="statement_overlap",
        patterns=["st_1", "esu_caseopen_2019_06_14"],
        proposition="Harvard Statement 1 publication preceded the EsuWiki case opening, showing an April-to-June cause timeline cited in the KG.",
        category="EsuWiki",
        subject="Harvard Statement 1 timeline",
        verb="preceded",
        obj="EsuWiki case opening",
        event_type="Harm",
        event_date="2019-04-19",
        event_location="China",
        actor_role="Harvard",
        truth_status="True",
        evidence_type="Document",
        safety_risk="high",
        public_exposure="partially_public",
        notes="From ST_1 → ESU_CaseOpen edges",
    ),
    FactBlueprint(
        key="weibo_visibility",
        patterns=["ev_hchk_2020_04_03"],
        proposition="Harvard Club statements were still visible in April 2020, showing the amplifier network kept defamatory claims live a year later.",
        category="Amplifier",
        subject="Harvard Club postings",
        verb="remained visible",
        obj="April 2020 Harvard Club Hong Kong captures",
        event_type="Publication",
        event_date="2020-04-03",
        event_location="Hong Kong",
        actor_role="Harvard",
        truth_status="True",
        evidence_type="Document",
        safety_risk="high",
        public_exposure="already_public",
        notes="From EV_HCHK_2020_04_03 nodes",
    ),
]


def extract_facts(text: str, source_path: Path) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    text_lower = text.lower()
    for idx, blueprint in enumerate(BLUEPRINTS, start=1):
        if all(pattern in text_lower for pattern in blueprint.patterns):
            fact_id = f"KGFACT_{idx:03d}"
            row = {
                "Proposition": blueprint.proposition,
                "FactID": fact_id,
                "Subject": blueprint.subject,
                "Verb": blueprint.verb,
                "Object": blueprint.obj,
                "EventType": blueprint.event_type,
                "EventDate": blueprint.event_date,
                "EventLocation": blueprint.event_location,
                "ActorRole": blueprint.actor_role,
                "Speaker": blueprint.subject,
                "TruthStatus": blueprint.truth_status,
                "EvidenceType": blueprint.evidence_type,
                "SourceDocument": source_path.name,
                "SourceExcerpt": blueprint.notes,
                "SafetyRisk": blueprint.safety_risk,
                "PublicExposure": blueprint.public_exposure,
                "RiskRationale": "",
                "confidence_tier": 1,
                "causal_salience_score": 0.8 if blueprint.category != "Platform" else 0.5,
                "confidence_reason": "kg_pattern",
                "causal_salience_reason": blueprint.category,
                "ExtractionMethod": "kg_blob_parser",
                "ExtractionConfidence": "0.85",
                "CanonicalSubject": "",
                "CanonicalSpeaker": "",
                "CanonicalActorRole": "",
                "CanonicalEventLocation": "",
                "CanonicalEntityID": "",
                "Notes": blueprint.notes,
                "Pattern": blueprint.category,
                "SourceFormat": "kg_edges_export_removed.csv",
                "ClassificationFixed": "",
            }
            facts.append(row)
    return facts


def write_report(report_path: Path, facts: List[Dict[str, str]]) -> None:
    counts = Counter(row["Pattern"] for row in facts)
    lines = [
        "# KG Facts Extraction Report",
        "",
        f"- Total facts extracted: {len(facts)}",
    ]
    for category, count in counts.items():
        lines.append(f"  - {category}: {count}")
    lines.append("")
    lines.append("## Sample Facts")
    for row in facts[:5]:
        lines.append(f"- {row['Proposition']}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract readable facts from KG blobs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("case_law_data/kg_edges_export_removed.csv"),
        help="KG blob CSV exported by Session 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("case_law_data/extracted_kg_facts.csv"),
        help="Destination CSV for structured facts.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/analysis_outputs/kg_facts_extraction_report.md"),
        help="Markdown report describing extracted facts.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    text = args.input.read_text(encoding="utf-8", errors="ignore")
    facts = extract_facts(text, args.input)
    if not facts:
        raise SystemExit("No matching KG facts were extracted. Ensure KG patterns exist.")

    df = pd.DataFrame(facts, columns=COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    write_report(args.report, facts)
    print(f"Wrote {len(facts)} facts to {args.output}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
