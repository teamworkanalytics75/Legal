"""
Batch runner for STORM-inspired research topics.
Executes multiple high-depth research prompts sequentially.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Ensure the scripts directory is on the import path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from STORMInspiredResearch import STORMInspiredResearch  # noqa: E402


TOPICS: List[Tuple[str, str]] = [
    (
        "Political Signaling and Narrative Control in Chinese Crackdown Campaigns",
        """Investigate how the Chinese Communist Party initiates domestic crackdown campaigns in response to perceived narrative threats or contradictions in elite or foreign discourse. Analyze case studies (e.g., EsuWiki, feminist movements, campus nationalism) to identify common precursors such as foreign media coverage, online exposure of Party families, or reputational challenges to ideological legitimacy.

Goal: Understand the mechanism of narrative conflict → propaganda response → enforcement escalation chain that often precedes campaign-style crackdowns.""",
    ),
    (
        "Information Flows Between Overseas Chinese Media and Domestic Crackdowns",
        """Study how information originating in overseas Chinese-language or bilingual media (e.g., WeChat, Zhihu reposts, or Western social platforms) can influence domestic CCP risk perception, censorship priorities, or public-security responses. Identify documented cases where foreign or diaspora online content preceded targeted enforcement actions within the PRC.

Goal: Map the cross-border media relay between diaspora platforms and Chinese security operations, showing how foreign narratives can be treated as domestic triggers.""",
    ),
    (
        "Institutional Decision-Making and the Timing of Campaign-Style Crackdowns",
        """Examine how the PRC’s internal bureaucratic structure (e.g., Central Propaganda Department, United Front Work Department, Cyberspace Administration, and Public Security) coordinates campaign-style crackdowns on ideological or reputational grounds. What are the decision-making patterns, timing cues, and institutional triggers preceding events such as the EsuWiki crackdown?

Goal: Surface organizational sequences and lag times (weeks to months) between public ‘signals’ and enforcement outcomes.""",
    ),
    (
        "Foreign Education Narratives as Sensitivity Triggers in CCP Information Policy",
        """Analyze historical cases in which foreign universities or academic narratives intersected with Chinese political sensitivities—especially involving elite families, overseas study, or criticism of leadership figures. Determine how such narratives have previously prompted image-protection campaigns or targeted censorship actions within China.

Goal: Identify precedents where cross-border academic or reputational stories led to CCP information-control responses, helping to generalize your own context.""",
    ),
    (
        "Comparative Analysis of ‘Personal Privacy’ or ‘Defamation’ Charges in Political Crackdowns",
        """Investigate the legal and rhetorical use of ‘personal privacy,’ ‘defamation,’ or ‘reputation protection’ charges in PRC political prosecutions (2015–2025). Compare these cases to determine whether such charges function as proxy tools for political narrative control, and assess how severity, timing, and coordination patterns align with larger propaganda campaigns.

Goal: Understand how nominally civil or reputational offenses are used to mask politically motivated crackdowns, as seen in EsuWiki-type cases.""",
    ),
]


def main() -> None:
    """Run STORM-inspired research for all configured topics."""
    researcher = STORMInspiredResearch()

    for title, prompt in TOPICS:
        topic_text = f"""{title}

RESEARCH PROMPT:
{prompt}
"""
        print("\n" + "#" * 120)
        print(f"Starting research run for: {title}")
        print("#" * 120 + "\n")
        researcher.run_comprehensive_research(topic_text)


if __name__ == "__main__":
    main()
