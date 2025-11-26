#!/usr/bin/env python3
"""
Generate v18 from v17 by applying normalization rules:
- Standardize EventDate (ISO format, remove "Unknown")
- Normalize TruthStatus
- Normalize ActorRole vs Subject using mapping
- Normalize EvidenceType
- Clean Proposition text
- Clean CausalSalienceReason
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

# Input/Output paths
V17_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v17_final.csv")
V18_OUTPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v18_final.csv")
REPORT_PATH = Path("reports/analysis_outputs/v18_normalization_report.md")

# ActorRole mapping from Subject substring to canonical ActorRole
ROLE_MAP = [
    ("Harvard Club of Hong Kong", "Harvard Club"),
    ("Harvard Clubs of Beijing and Shanghai", "Harvard Club"),
    ("Harvard Club of Beijing", "Harvard Club"),
    ("Harvard Club of Shanghai", "Harvard Club"),
    ("Harvard Clubs", "Harvard Club"),
    ("Harvard Alumni Association", "Harvard Alumni Association"),
    ("Harvard Admissions Office", "Harvard Admissions Office"),
    ("Harvard GSS", "Harvard"),
    ("Harvard Global Support Services", "Harvard"),
    ("Harvard Office of the General Counsel", "Harvard OGC"),
    ("Harvard Office of General Counsel", "Harvard OGC"),
    ("Harvard", "Harvard"),
    ("Malcolm Grayson", "Plaintiff"),
    ("Plaintiff", "Plaintiff"),
    ("PRC Authorities", "PRC Authorities"),
    ("StateActor", "StateActor"),
    ("StateMedia", "StateMedia"),
    ("Platform", "Platform"),
    ("Third-Party Publisher", "Third-Party Publisher"),
    ("Court", "Court"),
]


def normalize_date(value: str) -> str:
    """Normalize EventDate to ISO format or empty string."""
    if not value or not value.strip():
        return ""
    
    v = value.strip()
    
    # Handle "Unknown" explicitly
    if v.lower() == "unknown":
        return ""
    
    # Already ISO format (YYYY-MM-DD)
    try:
        datetime.fromisoformat(v)
        return v
    except (ValueError, TypeError):
        pass
    
    # Common patterns like "April 17 2025", "Apr 17 2025", "17 April 2025"
    date_formats = [
        "%B %d %Y",      # April 17 2025
        "%b %d %Y",      # Apr 17 2025
        "%d %B %Y",      # 17 April 2025
        "%d %b %Y",      # 17 Apr 2025
        "%Y-%m-%d",      # 2025-04-17 (already handled above, but keep for safety)
        "%m/%d/%Y",      # 04/17/2025
        "%d/%m/%Y",      # 17/04/2025
    ]
    
    for fmt in date_formats:
        try:
            dt = datetime.strptime(v, fmt)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    
    # If parsing fails, return original (may need manual review)
    return v


def normalize_actor_role(subject: str, current_actor_role: str) -> str:
    """Normalize ActorRole based on Subject using ROLE_MAP."""
    if not subject:
        return current_actor_role or ""
    
    subject_str = str(subject).strip()
    current_role = (current_actor_role or "").strip()
    
    # Find first matching key in Subject
    for key, canonical_role in ROLE_MAP:
        if key in subject_str:
            return canonical_role
    
    # No match found, return current role
    return current_role


def normalize_evidence_type(row: dict) -> str:
    """Normalize EvidenceType based on FactID prefix and Proposition content."""
    current_et = (row.get("evidencetype") or "").strip()
    factid = (row.get("factid") or "").strip()
    proposition = (row.get("proposition") or "").lower()
    
    # If already set and not "Unknown", keep it
    if current_et and current_et.lower() != "unknown":
        return current_et
    
    # Check FactID prefix
    if factid.startswith("KGFACT_"):
        return "Document"
    
    if factid.startswith("CORR_") or factid.startswith("MISSING_"):
        return "Email"
    
    # Check Proposition content
    if "exhibit" in proposition:
        return "Exhibit"
    
    if "writ of summons" in proposition or "statement of claim" in proposition:
        return "HKFiling"
    
    if "§ 1782" in proposition or "misc. case no." in proposition or "district of massachusetts" in proposition or "d. mass." in proposition:
        return "USFiling"
    
    if "court order" in proposition or "order" in proposition and "court" in proposition:
        return "CourtOrder"
    
    # Return current or "Unknown"
    return current_et or "Unknown"


def clean_proposition(text: str) -> str:
    """Clean Proposition text: remove artifacts, normalize whitespace."""
    if not text:
        return ""
    
    t = str(text).strip()
    
    # Remove leading/trailing pipes
    t = t.strip("|").strip()
    
    # Remove stray "Google Drive" mentions (already captured in EvidenceType)
    t = t.replace("Google Drive", "").strip()
    
    # Remove stray "law.justia.com" or similar URL artifacts
    if "law.justia.com" in t:
        # Only remove if it's at the end or isolated
        t = t.replace("law.justia.com", "").strip()
    
    # Normalize multiple spaces to single space
    t = " ".join(t.split())
    
    return t


def normalize_truth_status(row: dict) -> str:
    """Normalize TruthStatus based on EvidenceType and Proposition content."""
    current_ts = (row.get("truthstatus") or "").strip()
    evidencetype = (row.get("evidencetype") or "").strip()
    proposition = (row.get("proposition") or "").lower()
    
    # If explicitly "HostileFalseClaim", keep it
    if "hostilefalseclaim" in current_ts.lower():
        return "HostileFalseClaim"
    
    # Check for pleaded allegations in proposition
    if any(phrase in proposition for phrase in [
        "alleges that",
        "plaintiff alleges",
        "case involves allegations",
        "allegation",
        "alleged",
    ]):
        return "Alleged"
    
    # For official documents (HKFiling, USFiling, CourtOrder, Document, Exhibit)
    # if they're descriptive (not claims), they're True
    if evidencetype in {"HKFiling", "USFiling", "CourtOrder", "Document", "Exhibit"}:
        # If it's describing what the document says (not a claim), it's True
        # But if it's a claim about liability, it's Alleged
        if "allegation" not in proposition and "alleges" not in proposition:
            return "True"
    
    # Default: return current or "Alleged"
    return current_ts or "Alleged"


def normalize_causal_reason(text: str) -> str:
    """Clean CausalSalienceReason: remove artifacts, normalize whitespace."""
    if not text:
        return ""
    
    t = str(text).strip()
    
    # Remove stray "Google Drive" mentions
    t = t.replace("Google Drive", "").strip()
    
    # Remove stray "#" markers at end
    t = t.rstrip("#").strip()
    
    # Normalize multiple spaces to single space
    t = " ".join(t.split())
    
    return t


def normalize_causal_salience_score(value: str) -> str:
    """Ensure CausalSalienceScore is a float in [0.0, 1.0]."""
    if not value or not value.strip():
        return ""
    
    try:
        score = float(value)
        # Clamp to [0.0, 1.0]
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return f"{score:.2f}"
    except (ValueError, TypeError):
        # If not a number, return empty string
        return ""


def apply_v18_normalization(input_path: Path, output_path: Path) -> dict:
    """Apply all v18 normalization rules and generate output CSV."""
    stats = {
        "total_facts": 0,
        "dates_normalized": 0,
        "dates_removed_unknown": 0,
        "actor_roles_normalized": 0,
        "evidence_types_normalized": 0,
        "propositions_cleaned": 0,
        "truth_status_normalized": 0,
        "causal_reasons_cleaned": 0,
        "causal_scores_normalized": 0,
    }
    
    facts = []
    
    # Load facts
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            stats["total_facts"] += 1
            original_row = row.copy()
            
            # Normalize EventDate
            original_date = row.get("eventdate", "")
            normalized_date = normalize_date(original_date)
            if normalized_date != original_date:
                if original_date.strip().lower() == "unknown":
                    stats["dates_removed_unknown"] += 1
                else:
                    stats["dates_normalized"] += 1
            row["eventdate"] = normalized_date
            
            # Normalize ActorRole
            original_role = row.get("actorrole", "")
            normalized_role = normalize_actor_role(
                row.get("subject", ""),
                original_role
            )
            if normalized_role != original_role:
                stats["actor_roles_normalized"] += 1
            row["actorrole"] = normalized_role
            
            # Normalize EvidenceType
            original_et = row.get("evidencetype", "")
            normalized_et = normalize_evidence_type(row)
            if normalized_et != original_et:
                stats["evidence_types_normalized"] += 1
            row["evidencetype"] = normalized_et
            
            # Clean Proposition
            original_prop = row.get("proposition", "")
            cleaned_prop = clean_proposition(original_prop)
            if cleaned_prop != original_prop:
                stats["propositions_cleaned"] += 1
            row["proposition"] = cleaned_prop
            
            # Normalize TruthStatus
            original_ts = row.get("truthstatus", "")
            normalized_ts = normalize_truth_status(row)
            if normalized_ts != original_ts:
                stats["truth_status_normalized"] += 1
            row["truthstatus"] = normalized_ts
            
            # Clean CausalSalienceReason
            original_reason = row.get("causal_salience_reason", "")
            cleaned_reason = normalize_causal_reason(original_reason)
            if cleaned_reason != original_reason:
                stats["causal_reasons_cleaned"] += 1
            row["causal_salience_reason"] = cleaned_reason
            
            # Normalize CausalSalienceScore
            original_score = row.get("causal_salience_score", "")
            normalized_score = normalize_causal_salience_score(original_score)
            if normalized_score != original_score:
                stats["causal_scores_normalized"] += 1
            row["causal_salience_score"] = normalized_score
            
            facts.append(row)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        if facts:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(facts)
    
    return stats


def write_report(stats: dict, report_path: Path):
    """Write normalization report."""
    report = f"""# V18 Normalization Report

## Summary

Applied v18 normalization rules to v17 dataset.

## Statistics

- **Total facts processed**: {stats['total_facts']}
- **EventDates normalized**: {stats['dates_normalized']}
- **EventDates removed (Unknown)**: {stats['dates_removed_unknown']}
- **ActorRoles normalized**: {stats['actor_roles_normalized']}
- **EvidenceTypes normalized**: {stats['evidence_types_normalized']}
- **Propositions cleaned**: {stats['propositions_cleaned']}
- **TruthStatus normalized**: {stats['truth_status_normalized']}
- **CausalSalienceReasons cleaned**: {stats['causal_reasons_cleaned']}
- **CausalSalienceScores normalized**: {stats['causal_scores_normalized']}

## Normalization Rules Applied

### 1. EventDate Normalization
- Converted "Unknown" to empty string
- Parsed common date formats to ISO (YYYY-MM-DD)
- Preserved existing ISO dates

### 2. ActorRole Normalization
- Applied ROLE_MAP based on Subject field
- Standardized Harvard variants (Harvard Club, Harvard OGC, etc.)
- Standardized Plaintiff references

### 3. EvidenceType Normalization
- Inferred from FactID prefix (KGFACT_ → Document, CORR_/MISSING_ → Email)
- Inferred from Proposition content (Exhibit, HKFiling, USFiling, CourtOrder)

### 4. Proposition Cleaning
- Removed leading/trailing pipes
- Removed "Google Drive" artifacts
- Normalized whitespace

### 5. TruthStatus Normalization
- Detected pleaded allegations
- Set official documents to "True" when descriptive
- Preserved "HostileFalseClaim" markers

### 6. CausalSalienceReason Cleaning
- Removed "Google Drive" artifacts
- Removed stray "#" markers
- Normalized whitespace

### 7. CausalSalienceScore Normalization
- Clamped scores to [0.0, 1.0] range
- Formatted as 2-decimal floats

## Output Files

- **V18 CSV**: `case_law_data/top_1000_facts_for_chatgpt_v18_final.csv`
- **Report**: `reports/analysis_outputs/v18_normalization_report.md`

## Next Steps

1. Review sample facts (e.g., Facts 150-180) for sanity check
2. Generate v18 top 100 facts
3. Generate v18 causation nodes
4. Create v18 human-readable TXT file
"""
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)


def main():
    """Main execution."""
    print("="*80)
    print("V18 NORMALIZATION - APPLYING RULES TO V17")
    print("="*80)
    print()
    
    if not V17_INPUT.exists():
        print(f"❌ Error: Input file not found: {V17_INPUT}")
        return
    
    print(f"📂 Input: {V17_INPUT}")
    print(f"📂 Output: {V18_OUTPUT}")
    print()
    
    print("Applying normalization rules...")
    stats = apply_v18_normalization(V17_INPUT, V18_OUTPUT)
    
    print("Writing report...")
    write_report(stats, REPORT_PATH)
    
    print()
    print("="*80)
    print("V18 NORMALIZATION COMPLETE")
    print("="*80)
    print()
    print(f"✅ Processed {stats['total_facts']} facts")
    print(f"✅ Normalized {stats['dates_normalized']} dates")
    print(f"✅ Removed {stats['dates_removed_unknown']} 'Unknown' dates")
    print(f"✅ Normalized {stats['actor_roles_normalized']} ActorRoles")
    print(f"✅ Normalized {stats['evidence_types_normalized']} EvidenceTypes")
    print(f"✅ Cleaned {stats['propositions_cleaned']} Propositions")
    print(f"✅ Normalized {stats['truth_status_normalized']} TruthStatus")
    print(f"✅ Cleaned {stats['causal_reasons_cleaned']} CausalSalienceReasons")
    print(f"✅ Normalized {stats['causal_scores_normalized']} CausalSalienceScores")
    print()
    print(f"📄 Output: {V18_OUTPUT}")
    print(f"📄 Report: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()

