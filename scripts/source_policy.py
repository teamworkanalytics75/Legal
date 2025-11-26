"""
Source policy configuration for STORM-inspired research.
Defines approved domains, blocked domains, and primary-source detection.
"""

from __future__ import annotations

from typing import Set, Dict, Any
from pathlib import Path
import json
from urllib.parse import urlparse

# Domains published by PRC authorities or state media that qualify as primary sources.
def normalize_domain(domain: str) -> str:
    """Normalize domain for consistent matching."""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


PRIMARY_SOURCE_DOMAINS: Set[str] = {
    "cac.gov.cn",
    "www.cac.gov.cn",
    "mps.gov.cn",
    "www.mps.gov.cn",
    "gov.cn",
    "www.gov.cn",
    "chinacourt.org",
    "www.chinacourt.org",
    "court.gov.cn",
    "www.court.gov.cn",
    "people.com.cn",
    "www.people.com.cn",
    "xinhuanet.com",
    "www.xinhuanet.com",
    "cctv.com",
    "www.cctv.com",
    "cctv.cn",
    "www.cctv.cn",
    "caixin.com",
    "www.caixin.com",
    "news.cn",
    "www.news.cn",
    "paper.people.com.cn",
    "legalinfo.gov.cn",
    "www.legalinfo.gov.cn",
    "china.com.cn",
    "www.china.com.cn",
    "sohu.com",  # PRC portals often mirror primary docs
    "www.sohu.com",
    "ccdi.gov.cn",
    "www.ccdi.gov.cn",
    "moj.gov.cn",
    "www.moj.gov.cn",
    "gqb.gov.cn",
    "www.gqb.gov.cn",
}

# High-quality secondary sources (think tanks, academic outlets, reputable news desks).
APPROVED_DOMAINS: Set[str] = {
    "scmp.com",
    "www.scmp.com",
    "thediplomat.com",
    "www.thediplomat.com",
    "asia.nikkei.com",
    "chinadigitaltimes.net",
    "www.chinadigitaltimes.net",
    "jamestown.org",
    "www.jamestown.org",
    "chinapower.csis.org",
    "csis.org",
    "www.csis.org",
    "carnegieendowment.org",
    "www.carnegieendowment.org",
    "brookings.edu",
    "www.brookings.edu",
    "chinafile.com",
    "www.chinafile.com",
    "stacks.stanford.edu",
    "www.stacks.stanford.edu",
    "merics.org",
    "www.merics.org",
    "nikkei.com",
    "www.nikkei.com",
    "reuters.com",
    "www.reuters.com",
    "apnews.com",
    "www.apnews.com",
    "aljazeera.com",
    "www.aljazeera.com",
    "ft.com",
    "www.ft.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "foreignpolicy.com",
    "www.foreignpolicy.com",
    "globalvoices.org",
    "www.globalvoices.org",
}

# Domains that can be tolerated when evidence is scarce, but rank lower.
SECONDARY_DOMAINS: Set[str] = {
    "theguardian.com",
    "www.theguardian.com",
    "nytimes.com",
    "www.nytimes.com",
    "washingtonpost.com",
    "www.washingtonpost.com",
    "vox.com",
    "www.vox.com",
    "deutsche-welle.com",
    "www.deutsche-welle.com",
    "hk01.com",
    "www.hk01.com",
}

# Low-quality or generic domains to always exclude.
BLOCKED_DOMAINS: Set[str] = {
    "thefreedictionary.com",
    "vocabulary.com",
    "collinsdictionary.com",
    "dictionary.com",
    "merriam-webster.com",
    "britannica.com",
    "thesaurus.com",
    "speedtest.net",
    "www.speedtest.net",
    "fast.com",
    "speed.cloudflare.com",
    "highspeedinternet.com",
    "www.highspeedinternet.com",
    "speedsmart.net",
    "www.speedsmart.net",
}

# Minimum number of primary-source entries expected per report.
MIN_PRIMARY_SOURCES: int = 2


def _load_overrides() -> Dict[str, Any]:
    overrides_path = Path(__file__).resolve().parent.parent / "config" / "source_policy_overrides.json"
    if not overrides_path.exists():
        return {}
    try:
        with overrides_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _apply_overrides() -> None:
    overrides = _load_overrides()
    if not overrides:
        return

    PRIMARY_SOURCE_DOMAINS.update(map(normalize_domain, overrides.get("primary", [])))
    APPROVED_DOMAINS.update(map(normalize_domain, overrides.get("approved", [])))
    SECONDARY_DOMAINS.update(map(normalize_domain, overrides.get("secondary", [])))
    BLOCKED_DOMAINS.update(map(normalize_domain, overrides.get("blocked", [])))

    global MIN_PRIMARY_SOURCES
    if "min_primary_sources" in overrides:
        try:
            MIN_PRIMARY_SOURCES = int(overrides["min_primary_sources"])
        except (ValueError, TypeError):
            pass


_apply_overrides()


def normalize_domain(domain: str) -> str:
    """Normalize domain for consistent matching."""
    domain = domain.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def extract_domain(url: str) -> str:
    """Extract and normalize the domain from a URL."""
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path.split("/")[0]
    return normalize_domain(host)


def is_blocked_domain(domain: str) -> bool:
    return normalize_domain(domain) in BLOCKED_DOMAINS


def is_primary_source_domain(domain: str) -> bool:
    return normalize_domain(domain) in PRIMARY_SOURCE_DOMAINS


def domain_priority(domain: str) -> int:
    """
    Compute priority: 0 (primary source), 1 (approved), 2 (secondary),
    3 (other), 5 (blocked).
    """
    norm = normalize_domain(domain)
    if norm in PRIMARY_SOURCE_DOMAINS:
        return 0
    if norm in APPROVED_DOMAINS:
        return 1
    if norm in SECONDARY_DOMAINS:
        return 2
    if norm in BLOCKED_DOMAINS:
        return 5
    return 3


def is_allowed_domain(domain: str) -> bool:
    """Return True when a domain is acceptable (not blocked)."""
    return domain_priority(domain) < 5
