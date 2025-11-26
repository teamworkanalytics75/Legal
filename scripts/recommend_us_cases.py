"""Recommend supportive U.S. case law based on embedded lawsuit documents."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = Path("reports") / "analysis_outputs"
EMBED_PATH = OUTPUT_DIR / "lawsuit_doc_embeddings.npz"
META_PATH = OUTPUT_DIR / "lawsuit_doc_chunks.json"


CASE_CANDIDATES: List[Dict[str, str]] = [
    {
        "name": "Intel Corp. v. Advanced Micro Devices, Inc., 542 U.S. 241 (2004)",
        "description": (
            "Leading Section 1782 Supreme Court decision expanding access to U.S. discovery for use in foreign "
            "tribunals and outlining the discretionary Intel factors."
        ),
    },
    {
        "name": "Mees v. Buiter, 793 F.3d 291 (2d Cir. 2015)",
        "description": (
            "Second Circuit confirmed a broad reading of the 'for use' requirement under Section 1782 and rejected "
            "any necessity to show foreign discoverability."
        ),
    },
    {
        "name": "In re Application for Judicial Assistance from the Hong Kong Special Administrative Region, 138 F.3d 113 (2d Cir. 1998)",
        "description": (
            "Second Circuit recognized Hong Kong courts as qualifying foreign tribunals and authorized Section 1782 "
            "discovery to aid Hong Kong criminal proceedings."
        ),
    },
    {
        "name": "In re Accent Delight International Ltd., 869 F.3d 121 (2d Cir. 2017)",
        "description": (
            "Section 1782 case emphasizing the statute's low threshold and willingness to assist complex international "
            "business disputes."
        ),
    },
    {
        "name": "In re del Valle Ruiz, 939 F.3d 520 (2d Cir. 2019)",
        "description": (
            "Second Circuit addressed the reach of Section 1782 discovery, clarifying personal jurisdiction over "
            "U.S.-based custodians with foreign evidence."
        ),
    },
    {
        "name": "Servotronics, Inc. v. Boeing Co., 954 F.3d 209 (4th Cir. 2020)",
        "description": (
            "Fourth Circuit opinion granting Section 1782 discovery for use in a foreign arbitration, highlighting "
            "split authority and expansive interpretations."
        ),
    },
    {
        "name": "Republic of Ecuador v. For the Issuance of a Subpoena, 742 F.3d 860 (9th Cir. 2014)",
        "description": (
            "Ninth Circuit decision supporting Section 1782 discovery to aid international litigation involving "
            "state actors and environmental claims."
        ),
    },
    {
        "name": "New York Times Co. v. Sullivan, 376 U.S. 254 (1964)",
        "description": (
            "Landmark defamation case establishing the actual-malice standard for public officials and influential "
            "institutions."
        ),
    },
    {
        "name": "Gertz v. Robert Welch, Inc., 418 U.S. 323 (1974)",
        "description": (
            "Supreme Court decision clarifying defamation standards for private figures and the role of damages."
        ),
    },
    {
        "name": "Hustler Magazine, Inc. v. Falwell, 485 U.S. 46 (1988)",
        "description": (
            "Defamation and intentional-infliction case delineating limits on parody defenses and protecting "
            "reputation against malicious campaigns."
        ),
    },
    {
        "name": "Dongguk University v. Yale University, 734 F.3d 113 (2d Cir. 2013)",
        "description": (
            "Second Circuit defamation case involving elite universities and credential disputes, illustrating "
            "reputational damages arising from academic institutions."
        ),
    },
    {
        "name": "Zhang v. Baidu.com Inc., 10 F. Supp. 3d 433 (S.D.N.Y. 2014)",
        "description": (
            "Southern District of New York decision addressing allegations of Chinese state influence over content "
            "moderation and the First Amendment implications."
        ),
    },
    {
        "name": "The Reporters Committee for Freedom of the Press v. U.S. Department of Justice, 816 F.2d 730 (D.C. Cir. 1987)",
        "description": (
            "Appellate decision balancing reputational interests during investigations—useful for media-focused "
            "defamation strategies."
        ),
    },
]


def load_embeddings() -> np.ndarray:
    if not EMBED_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBED_PATH}")
    data = np.load(EMBED_PATH)
    return data["embeddings"]


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def main() -> None:
    embeddings = load_embeddings()
    avg_vector = embeddings.mean(axis=0, keepdims=True)
    avg_norm = normalize_rows(avg_vector)[0]

    model = SentenceTransformer(EMBED_MODEL, cache_folder=str(Path("models_cache")))
    case_texts = [case["name"] + " — " + case["description"] for case in CASE_CANDIDATES]
    case_embeddings = model.encode(case_texts, convert_to_numpy=True)
    case_embeddings = normalize_rows(case_embeddings)

    scores = case_embeddings @ avg_norm

    ranked = sorted(zip(CASE_CANDIDATES, scores), key=lambda item: float(item[1]), reverse=True)

    results_path = OUTPUT_DIR / "lawsuit_supporting_cases.json"
    out_data = []
    for case, score in ranked:
        entry = {
            "name": case["name"],
            "description": case["description"],
            "similarity": float(score),
        }
        out_data.append(entry)

    results_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

    print("Top supporting cases:")
    for case, score in ranked[:10]:
        print(f"{case['name']} — similarity {score:.3f}")

    print(f"\nSaved full rankings to {results_path}")


if __name__ == "__main__":
    main()
