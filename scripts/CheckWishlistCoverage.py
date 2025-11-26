#!/usr/bin/env python3
"""
Cross-reference the Top 100 § 1782 wish list against the curated corpus.

Reads every JSON file in data/case_law/1782_discovery/, gathers the stored case
names, and reports which of the wish-list authorities are present or missing.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = ROOT / "data" / "case_law" / "1782_discovery"

# The wish list captured as a raw string so it can be reparsed if updated later.
TOP_100_TEXT = r"""
1. Intel Corp. v. Advanced Micro Devices, Inc., 542 U.S. 241 (2004) - Supreme Court.
2. ZF Automotive US, Inc. v. Luxshare, Ltd., 142 S. Ct. 2078 (2022) - Supreme Court.
3. AlixPartners, LLP v. Fund for Prot. of Inv. Rights, 142 S. Ct. 2078 (2022) - Supreme Court.
4. In re Application of Asta Medica, S.A., 981 F.2d 1 (1st Cir. 1992) - 1st Circuit.
5. In re Schlich (George W. Schlich), 893 F.3d 40 (1st Cir. 2018) - 1st Circuit.
6. In re Porsche Automobil Holding SE, 985 F.3d 115 (1st Cir. 2021) - 1st Circuit.
7. Nat'l Broadcasting Co. v. Bear Stearns & Co., 165 F.3d 184 (2d Cir. 1999) - 2d Circuit.
8. Republic of Kazakhstan v. Biedermann Int'l, 168 F.3d 880 (5th Cir. 1999) - 5th Circuit.
9. Abdul Latif Jameel Transp. Co. v. FedEx Corp., 939 F.3d 710 (6th Cir. 2019) - 6th Circuit.
10. Servotronics, Inc. v. Boeing Co., 954 F.3d 209 (4th Cir. 2020) - 4th Circuit.
11. Servotronics, Inc. v. Rolls-Royce PLC, 975 F.3d 689 (7th Cir. 2020) - 7th Circuit.
12. Consorcio Ecuatoriano de Telecom. S.A. v. JAS Forwarding (USA), Inc., 747 F.3d 1262 (11th Cir. 2014) - 11th Circuit.
13. Euromepa S.A. v. R. Esmerian, Inc. (Euromepa I), 51 F.3d 1095 (2d Cir. 1995) - 2d Circuit.
14. Euromepa S.A. v. Esmerian, Inc. (Euromepa II), 154 F.3d 24 (2d Cir. 1998) - 2d Circuit.
15. Brandi-Dohrn v. IKB Deutsche Industriebank AG, 673 F.3d 76 (2d Cir. 2012) - 2d Circuit.
16. Mees v. Buiter, 793 F.3d 291 (2d Cir. 2015) - 2d Circuit.
17. Certain Funds, Accounts &/or Inv. Vehicles v. KPMG, LLP, 798 F.3d 113 (2d Cir. 2015) - 2d Circuit.
18. In re del Valle Ruiz, 939 F.3d 520 (2d Cir. 2019) - 2d Circuit.
19. In re Guo (Hanwei Guo v. Deutsche Bank), 965 F.3d 96 (2d Cir. 2020) - 2d Circuit.
20. In re Bayer AG, 146 F.3d 188 (3d Cir. 1998) - 3d Circuit.
21. In re Application of Gianoli Aldunate, 3 F.3d 54 (2d Cir. 1993) - 2d Circuit.
22. In re Metallgesellschaft AG, 121 F.3d 77 (2d Cir. 1997) - 2d Circuit.
23. In re Letter of Request from Supreme Ct. of Hong Kong, 138 F.3d 68 (2d Cir. 1998) - 2d Circuit.
24. In re Clerici (Patricio Clerici), 481 F.3d 1324 (11th Cir. 2007) - 11th Circuit.
25. Sergeeva v. Tripleton Int'l Ltd., 834 F.3d 1194 (11th Cir. 2016) - 11th Circuit.
26. Furstenberg Finance SAS v. Litai Assets LLC, 877 F.3d 1031 (11th Cir. 2017) - 11th Circuit.
27. Republic of Ecuador v. Hinchee, 741 F.3d 1185 (11th Cir. 2013) - 11th Circuit.
28. Heraeus Kulzer GmbH v. Biomet, Inc., 633 F.3d 591 (7th Cir. 2011) - 7th Circuit.
29. In re Application of Malev Hungarian Airlines, 964 F.2d 97 (2d Cir. 1992) - 2d Circuit.
30. In re Application of Esses, 101 F.3d 873 (2d Cir. 1996) - 2d Circuit.
31. Four Pillars Enter. Co. v. Avery Dennison Corp., 308 F.3d 1075 (9th Cir. 2002) - 9th Circuit.
32. In re Letters Rogatory from Tokyo Dist. Court, 539 F.2d 1216 (9th Cir. 1976) - 9th Circuit.
33. In re Premises Located at 840 140th Ave. NE, Bellevue, Wash., 634 F.3d 557 (9th Cir. 2011) - 9th Circuit.
34. In re CPC Patent Techs. Pty Ltd., 34 F.4th 801 (9th Cir. 2022) - 9th Circuit.
35. Khrapunov v. Prosyankin, 931 F.3d 922 (9th Cir. 2019) - 9th Circuit.
36. In re Application of Republic of Ecuador, 735 F.3d 1179 (10th Cir. 2013) - 10th Circuit.
37. Andover Healthcare, Inc. v. 3M Co., 817 F.3d 621 (8th Cir. 2016) - 8th Circuit.
38. Kiobel by Samkalden v. Cravath, Swaine & Moore LLP, 895 F.3d 238 (2d Cir. 2018) - 2d Circuit.
39. In re Biomet Orthopaedics Switzerland GmbH, 742 F. App'x 690 (3d Cir. 2018) - 3d Circuit.
40. In re Accent Delight Int'l Ltd., 869 F.3d 121 (2d Cir. 2017) - 2d Circuit.
41. Schmitz v. Bernstein Liebhard LLP, 376 F.3d 79 (2d Cir. 2004) - 2d Circuit.
42. In re Sarrio, S.A., 119 F.3d 143 (2d Cir. 1997) - 2d Circuit.
43. In re Edelman, 295 F.3d 171 (2d Cir. 2002) - 2d Circuit.
44. In re O'Keeffe, 646 F. App'x 263 (3d Cir. 2016) - 3d Circuit.
45. In re Roz Trading Ltd., 469 F. Supp. 2d 1221 (N.D. Ga. 2006) - District.
46. In re Application of Babcock Borsig AG, 583 F. Supp. 2d 233 (D. Mass. 2008) - District.
47. In re Schlich, No. 16-mc-91278, 2016 WL 7209565 (D. Mass. Dec. 9, 2016) - District.
48. In re Chevron Corp., 762 F. Supp. 2d 242 (D. Mass. 2010) - District.
49. In re Application of Peruvian Sporting Goods S.A.C., 2018 U.S. Dist. LEXIS 223564 (D. Mass. Dec. 7, 2018) - District.
50. In re Hand Held Prods., Inc., 2024 WL 5136071 (D. Mass. Oct. 24, 2024) - District.
51. Daedalus Prime LLC v. MediaTek, Inc., 2023 WL 6827452 (N.D. Cal. Sept. 16, 2023) - District.
52. Amazon.com, Inc. v. Nokia Corp., 2023 WL 549323 (D. Del. Jan. 17, 2023) - District.
53. In re Netgear, Inc., 2024 WL 5136056 (S.D. Cal. Jan. 31, 2024) - District.
54. In re FourWorld Event Opportunities Fund, No. 1:21-mc-00466 (S.D.N.Y. 2023) - District.
55. In re Sveaas, 249 F.R.D. 96 (S.D.N.Y. 2008) - District.
56. In re Caratube Int'l Oil Co., 730 F. Supp. 2d 101 (D.D.C. 2010) - District.
57. In re Republic of Turkey, 2022 WL 1406612 (D.N.J. May 4, 2022) - District.
58. In re Republic of Iraq, 2020 WL 7122843 (D.D.C. Dec. 4, 2020) - District.
59. In re Financialright GmbH, 2017 WL 2879696 (S.D.N.Y. July 6, 2017) - District.
60. Porsche Automobil Holding SE v. Bank of Am. Corp., 2016 WL 702327 (S.D.N.Y. Feb. 18, 2016) - District.
61. In re Joint Stock Co. Raiffeisenbank, 2016 WL 6474224 (S.D. Fla. Nov. 2, 2016) - District.
62. In re Kleimar N.V., 220 F. Supp. 3d 517 (S.D.N.Y. 2016) - District.
63. In re PSJC VSMPO-Avisma Corp., 2006 WL 2466256 (S.D.N.Y. Aug. 24, 2006) - District.
64. In re XPO Logistics, Inc., 2017 WL 2226593 (D. Kan. May 22, 2017) - District.
65. In re King.com Ltd., 2020 WL 5095135 (N.D. Cal. Aug. 28, 2020) - District.
66. In re Guler & Ogmen, 2019 WL 1230490 (E.D.N.Y. Mar. 15, 2019) - District.
67. In re Gliner (Gregory Gliner), 133 F.4th 927 (9th Cir. 2025) - 9th Circuit.
68. In re Kiobel, No. 16-cv-7992, 2017 WL 354183 (S.D.N.Y. Jan. 24, 2017) - District.
69. In re Republic of Kazakhstan, 2015 WL 6437466 (N.D. Cal. Oct. 21, 2015) - District.
70. In re Banco Santander (Investment Prot. Action), 2020 WL 4926557 (S.D. Fla. Aug. 21, 2020) - District.
71. In re PJSC Uralkali, 2019 WL 12262019 (M.D. Fla. Jan. 22, 2019) - District.
72. In re Republic of Guinea, 2014 WL 496719 (W.D. Pa. Feb. 6, 2014) - District.
73. In re Application of Ecuador, 2011 WL 736868 (E.D. Va. Feb. 23, 2011) - District.
74. Grupo Mexico SAB de CV v. SAS Asset Recovery, Ltd., 821 F.3d 573 (5th Cir. 2016) - 5th Circuit.
75. In re Commonwealth of Australia, 2017 WL 4875276 (N.D. Cal. Oct. 27, 2017) - District.
76. In re Baxter Int'l Inc., 2004 WL 2158051 (N.D. Ill. Sept. 24, 2004) - District.
77. In re Tovmasyan, 2022 WL 508335 (D. Mass. Feb. 18, 2022) - District.
78. In re Olympus Corp., 2013 WL 3794662 (D.N.J. July 19, 2013) - District.
79. In re Madhya Pradesh v. Getit Infoservices, 2020 WL 7695053 (C.D. Cal. Nov. 30, 2020) - District.
80. In re Blue Sky Litigation (No. 17-mc-80270), 2018 WL 3845893 (N.D. Cal. Aug. 13, 2018) - District.
81. In re Doosan Heavy Indus. & Constr. Co., 2020 WL 1864903 (E.D. Va. Apr. 14, 2020) - District.
82. In re Sierra Leone (Anti-Corruption Comm'n), 2021 WL 287978 (D. Md. Jan. 27, 2021) - District.
83. In re Punjab State Power Corp. Ltd., 2019 WL 12262019 (C.D. Cal. July 23, 2019) - District.
84. In re BNP Paribas Jersey Tr. Corp., 2012 WL 2433214 (S.D.N.Y. June 4, 2012) - District.
85. In re Yokohama Tire Corp., 2023 WL 2514896 (S.D. Iowa Mar. 14, 2023) - District.
86. In re X, 2022 WL 16727112 (D. Mass. Nov. 4, 2022) - District.
87. In re Mazur, 2021 WL 201150 (D. Colo. Jan. 20, 2021) - District.
88. In re Delta Airlines, 2020 WL 1245341 (N.D. Ga. Mar. 16, 2020) - District.
89. In re Zhiyu Pu, 2021 WL 5331444 (W.D. Wash. Nov. 16, 2021) - District.
90. In re Letter of Request from the Crown Prosecution Service, 870 F.2d 686 (D.C. Cir. 1989) - D.C. Circuit.
91. In re Medytox, Inc., 2021 WL 4461589 (C.D. Cal. Sept. 29, 2021) - District.
92. In re Top Matrix Holdings Ltd., 2020 WL 248716 (S.D.N.Y. Jan. 16, 2020) - District.
93. In re Oasis Focus Fund LP, 2022 WL 17669119 (D. Del. Dec. 14, 2022) - District.
94. In re Avalru Pvt. Ltd., 2022 WL 1197036 (S.D. Tex. Apr. 21, 2022) - District.
95. In re Mariani, 2020 WL 1887855 (S.D. Fla. Apr. 16, 2020) - District.
96. In re Enforcement of a subpoena by Lloyd's Register, 2015 WL 5943346 (D. Md. Oct. 9, 2015) - District.
97. In re B&C KB Holding GmbH, 2021 WL 4476693 (E.D. Mo. Sept. 30, 2021) - District.
98. In re Sasol Ltd., 2019 WL 1559422 (E.D. Va. Apr. 10, 2019) - District.
99. In re OOO Promnefstroy, 2009 WL 3335608 (S.D.N.Y. Oct. 15, 2009) - District.
100. In re Qwest Comm. Int'l Inc., 2008 WL 3823918 (W.D.N.C. Aug. 15, 2008) - District.
"""

STOPWORDS = {
    "in",
    "re",
    "application",
    "order",
    "pursuant",
    "for",
    "an",
    "of",
    "the",
    "and",
    "ex",
    "parte",
    "inc",
    "llc",
    "llp",
    "sa",
    "ltd",
    "co",
    "corp",
    "company",
    "limited",
    "holding",
    "holdings",
    "ag",
    "intl",
    "international",
}


def parse_wishlist(raw_text: str) -> List[str]:
    """Extract the case names from the numbered wish-list text."""

    names: List[str] = []
    pattern = re.compile(r"\s*\d+\.\s+([^,]+),")
    for line in raw_text.splitlines():
        match = pattern.match(line)
        if match:
            names.append(match.group(1).strip())
    return names


def _preprocess(text: str | None) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()

    replacements = {
        r"\bint[’']?l\b": "international",
        r"\bnat[’']?l\b": "national",
        r"\bcomm[’']?n\b": "commission",
        r"\bass[’']?n\b": "association",
        r"\bdep[’']?t\b": "department",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("§", " ")
    return text


def normalize(text: str | None) -> str:
    base = _preprocess(text)
    if not base:
        return ""
    return re.sub(r"[^a-z0-9]+", "", base)


def tokenize(text: str | None) -> FrozenSet[str]:
    base = _preprocess(text)
    if not base:
        return frozenset()
    tokens = re.findall(r"[a-z0-9]+", base)
    tokens = [tok for tok in tokens if tok and tok not in STOPWORDS]
    return frozenset(tokens)


def extract_candidate_strings(path: Path) -> Set[str]:
    """Collect all plausible candidate case-name strings from a JSON file."""

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="ignore")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return set()

    candidates: Set[str] = set()

    for key in ("caseNameFull", "caseName", "name", "short_name"):
        value = data.get(key)
        if isinstance(value, str):
            candidates.add(value)

    if "citations" in data and isinstance(data["citations"], list):
        for cite in data["citations"]:
            cite_name = cite.get("case_name") or cite.get("caseName")
            if isinstance(cite_name, str):
                candidates.add(cite_name)

    candidates.add(path.stem.replace("_", " "))

    casebody = data.get("casebody")
    if isinstance(casebody, dict):
        for key in ("caseName", "caseNameFull"):
            value = casebody.get(key)
            if isinstance(value, str):
                candidates.add(value)

    return {name for name in candidates if name}


def build_corpus_index(
    paths: Iterable[Path],
) -> Tuple[Dict[str, List[str]], List[Tuple[FrozenSet[str], str]]]:
    """Generate indices for exact-string and token-based matching."""

    string_index: Dict[str, List[str]] = defaultdict(list)
    token_entries: List[Tuple[FrozenSet[str], str]] = []

    for path in paths:
        rel_path = str(path.relative_to(ROOT))
        for candidate in extract_candidate_strings(path):
            norm = normalize(candidate)
            if norm:
                string_index[norm].append(rel_path)

            tokens = tokenize(candidate)
            if tokens:
                token_entries.append((tokens, rel_path))

    return string_index, token_entries


def find_matches(
    target_norm: str,
    target_tokens: FrozenSet[str],
    string_index: Dict[str, List[str]],
    token_entries: List[Tuple[FrozenSet[str], str]],
) -> List[str]:
    """Locate matches for a target case name."""

    if target_norm in string_index:
        return string_index[target_norm]

    matches: List[str] = []
    for name, files in string_index.items():
        if target_norm in name or name in target_norm:
            matches.extend(files)

    if not matches and target_tokens:
        for tokens, file in token_entries:
            if not tokens:
                continue
            if target_tokens.issubset(tokens) or tokens.issubset(target_tokens):
                matches.append(file)

    seen: Set[str] = set()
    unique: List[str] = []
    for file in matches:
        if file not in seen:
            seen.add(file)
            unique.append(file)
    return unique


def main() -> None:
    wishlist = parse_wishlist(TOP_100_TEXT)
    normalized_targets = {name: normalize(name) for name in wishlist}
    token_targets = {name: tokenize(name) for name in wishlist}

    corpus_files = sorted(CURATED_DIR.glob("*.json"))
    string_index, token_entries = build_corpus_index(corpus_files)

    found: Dict[str, List[str]] = {}
    missing: List[str] = []

    for name, norm in normalized_targets.items():
        matches = find_matches(norm, token_targets[name], string_index, token_entries)
        if matches:
            found[name] = matches
        else:
            missing.append(name)

    print("Top 100 wish-list coverage")
    print("==========================")
    print(f"Total wish-list cases: {len(wishlist)}")
    print(f"Found in corpus:      {len(found)}")
    print(f"Missing from corpus:  {len(missing)}")
    print()

    print("Found cases:")
    for name in wishlist:
        if name in found:
            files = ", ".join(found[name])
            print(f"  - {name}  ->  {files}")

    print()
    print("Missing cases:")
    for name in missing:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
