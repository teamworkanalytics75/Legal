"""
Fact extraction utilities for STORM-inspired research.
Identifies key events (date, actor, narrative, enforcement) from search snippets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

DATE_PATTERNS = [
    re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b"),
    re.compile(r"\b(\d{1,2} [A-Za-z]+ \d{4})\b"),
    re.compile(r"\b([A-Za-z]+ \d{1,2}, \d{4})\b"),
    re.compile(r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})\b"),
    re.compile(r"\b(\d{4})\b"),
]

ACTOR_PATTERNS = [
    re.compile(r"(Central Propaganda Department|Cyberspace Administration of China|CAC|Ministry of Public Security|MPS|United Front Work Department|UFWD|State Council|Public Security Bureau|PSB)", re.IGNORECASE),
    re.compile(r"(WeChat|Weibo|Zhihu|Douban|Baidu|Tencent|ByteDance|TikTok|Douyin|Xiaohongshu|Little Red Book)", re.IGNORECASE),
    re.compile(r"(overseas Chinese media|diaspora media|Chinese-language media|foreign media)", re.IGNORECASE),
]

ENFORCEMENT_PATTERNS = [
    re.compile(r"(arrested|detained|prosecuted|charged|summoned|investigated|crackdown|campaign|censored|blocked|deleted|sanctioned|raid)", re.IGNORECASE),
    re.compile(r"(directive|notice|instruction|order|guidelines|proclamation)", re.IGNORECASE),
]


KNOWN_CASES = [
    {"keyword": "EsuWiki", "label": "EsuWiki crackdown (2024)", "category": "Elite narrative leak"},
    {"keyword": "Feminist Voices", "label": "Feminist Voices takedown (2018)", "category": "Gender activism"},
    {"keyword": "campus nationalism", "label": "Campus nationalism campaigns", "category": "Student mobilization"},
    {"keyword": "Xi Mingze", "label": "Xi Mingze overseas narratives", "category": "Elite family exposure"},
    {"keyword": "709 crackdown", "label": "709 human rights lawyer arrests", "category": "Rights defense"},
    {"keyword": "Hong Kong protests", "label": "Hong Kong protest coverage", "category": "Transnational unrest"},
    {"keyword": "Uyghur", "label": "Xinjiang/Uyghur information control", "category": "Ethnic sensitivity"},
]


@dataclass
class ExtractedFact:
    title: str
    url: str
    date: Optional[str] = None
    actor: Optional[str] = None
    action: Optional[str] = None
    narrative: Optional[str] = None
    source_snippet: Optional[str] = None
    domain: Optional[str] = None
    case_reference: Optional[str] = None
    category: Optional[str] = None


def extract_date(snippet: str) -> Optional[str]:
    for pattern in DATE_PATTERNS:
        match = pattern.search(snippet)
        if match:
            return match.group(1)
    return None


def extract_actor(snippet: str) -> Optional[str]:
    for pattern in ACTOR_PATTERNS:
        match = pattern.search(snippet)
        if match:
            return match.group(1)
    return None


def extract_action(snippet: str) -> Optional[str]:
    for pattern in ENFORCEMENT_PATTERNS:
        match = pattern.search(snippet)
        if match:
            return match.group(1)
    return None


def summarize_snippet(snippet: str, max_length: int = 240) -> str:
    snippet = snippet.replace("\n", " ").strip()
    if len(snippet) <= max_length:
        return snippet
    return snippet[: max_length - 3] + "..."


def extract_fact(result: Dict[str, Any]) -> ExtractedFact:
    title = result.get("title", "Unnamed Source")
    url = result.get("href", "")
    snippet = result.get("body", "")
    domain = result.get("domain", "")

    fact = ExtractedFact(
        title=title,
        url=url,
        domain=domain,
        source_snippet=summarize_snippet(snippet),
    )

    fact.date = extract_date(snippet)
    fact.actor = extract_actor(snippet)
    fact.action = extract_action(snippet)

    for keyword in ["narrative", "content", "post", "leak", "exposure", "campaign", "arrest", "lawsuit"]:
        idx = snippet.lower().find(keyword)
        if idx != -1:
            window = snippet[max(0, idx - 80): idx + 120]
            fact.narrative = summarize_snippet(window, 200)
            break

    for case in KNOWN_CASES:
        if case["keyword"].lower() in snippet.lower() or case["keyword"].lower() in title.lower():
            fact.case_reference = case["label"]
            fact.category = case["category"]
            if not fact.narrative:
                fact.narrative = summarize_snippet(snippet, 200)
            break

    return fact


def extract_facts(results: List[Dict[str, Any]], limit: int = 6) -> List[ExtractedFact]:
    facts: List[ExtractedFact] = []
    seen_urls = set()

    for result in results:
        url = result.get("href")
        if not url or url in seen_urls:
            continue

        fact = extract_fact(result)
        if not (fact.date or fact.actor or fact.action or fact.narrative):
            continue

        facts.append(fact)
        seen_urls.add(url)

        if len(facts) >= limit:
            break

    return facts
