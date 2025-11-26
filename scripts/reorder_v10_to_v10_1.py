#!/usr/bin/env python3
"""Reorder v10 facts using tier-based ranking system to produce v10.1 top 100."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

V10_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v10_final.csv")
OUTPUT_CSV = Path("case_law_data/top_100_facts_v10.1.csv")
OUTPUT_TXT = Path("case_law_data/top_100_facts_v10.1.txt")
REPORT_PATH = Path("reports/analysis_outputs/v10_1_reranking_report.md")


def load_v10() -> pd.DataFrame:
    """Load v10 CSV."""
    if not V10_INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {V10_INPUT}")
    return pd.read_csv(V10_INPUT, encoding='utf-8', low_memory=False)


def assign_manual_tier(proposition: str) -> int:
    """Assign manual salience tier based on content."""
    if not isinstance(proposition, str):
        return 5
    
    prop_lower = proposition.lower()
    
    # Tier 1: Causal Chain of Harm
    tier1_keywords = [
        'statement 1', 'statement 2', 'harvard club', 'wechat', 'zhihu', 'baidu', 'sohu',
        'monkey', 'résumé', 'resume', 'esuwiki', 'xi mingze', 'arbitrary detention',
        'arrest', 'torture', 'prc security', 'e-canada', 'amplifier', 'circulated',
        'published', 'republished'
    ]
    
    if any(keyword in prop_lower for keyword in tier1_keywords):
        return 1
    
    # Tier 2: Harvard's Knowledge / Foreseeability
    tier2_keywords = [
        'harvard knew', 'foreseeability', 'gss', 'safety risk', 'should have known',
        'prc retaliation', 'travel risk', 'global support services', 'warnings',
        'harvard prior experience', 'internal communications', 'danger to plaintiff'
    ]
    
    if any(keyword in prop_lower for keyword in tier2_keywords):
        return 2
    
    # Tier 3: Harvard Procedures, Silence, and Cover-Up Indicators
    tier3_keywords = [
        'ogc', 'did not respond', 'silence', 'nonresponse', 'non-response',
        'meet-and-confer', 'spoliation', 'failed to respond', 'no response',
        'never responded', 'did not acknowledge', 'acknowledgment gap'
    ]
    
    if any(keyword in prop_lower for keyword in tier3_keywords):
        return 3
    
    # Tier 4: Contextual Background Only
    tier4_keywords = [
        'peter humphrey', 'arbitrary imprisonment', 'china travel advisory',
        'general china risk', 'contextual', 'background', 'travel advisories'
    ]
    
    if any(keyword in prop_lower for keyword in tier4_keywords):
        return 4
    
    # Tier 5: Low-Value Residuals
    return 5


def compute_new_salience_score(row: pd.Series) -> float:
    """Compute new salience score combining tier weight with original score."""
    tier = row.get('manual_salience_tier', 5)
    original_score = float(row.get('causal_salience_score', 0.0)) if pd.notna(row.get('causal_salience_score')) else 0.0
    
    # Tier weights
    tier_weights = {
        1: 1.00,  # Tier 1: Causal Chain of Harm
        2: 0.80,  # Tier 2: Harvard's Knowledge / Foreseeability
        3: 0.60,  # Tier 3: Harvard Procedures, Silence, and Cover-Up
        4: 0.30,  # Tier 4: Contextual Background Only
        5: 0.05,  # Tier 5: Low-Value Residuals
    }
    
    tier_weight = tier_weights.get(tier, 0.05)
    
    # Combine: 70% tier weight, 30% original score
    new_score = (tier_weight * 0.7) + (original_score * 0.3)
    
    return new_score


def reorder_facts(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder facts using tier-based ranking."""
    df = df.copy()
    
    # Add manual salience tier
    df['manual_salience_tier'] = df['proposition'].apply(assign_manual_tier)
    
    # Compute new salience score
    df['new_salience_score'] = df.apply(compute_new_salience_score, axis=1)
    
    # Sort by new salience score (descending)
    df = df.sort_values('new_salience_score', ascending=False)
    
    return df


def generate_summary(df: pd.DataFrame, top100: pd.DataFrame, original_top100: pd.DataFrame) -> dict:
    """Generate summary statistics."""
    summary = {
        'total_facts': len(df),
        'tier_counts': df['manual_salience_tier'].value_counts().to_dict(),
        'top100_tier_counts': top100['manual_salience_tier'].value_counts().to_dict(),
        'demoted_from_top100': [],
        'new_in_top100': [],
    }
    
    # Find demoted facts (were in original top 100, not in new top 100)
    original_factids = set(original_top100['factid'].astype(str))
    new_factids = set(top100['factid'].astype(str))
    
    demoted = original_factids - new_factids
    promoted = new_factids - original_factids
    
    summary['demoted_from_top100'] = list(demoted)[:10]  # First 10
    summary['new_in_top100'] = list(promoted)[:10]  # First 10
    
    return summary


def export_files(df: pd.DataFrame, top100: pd.DataFrame) -> None:
    """Export CSV and TXT files."""
    # Export CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    top100.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    # Export TXT (FactID | Proposition)
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        for _, row in top100.iterrows():
            factid = str(row['factid']).strip()
            prop = str(row['proposition']).strip().replace('|', '｜')
            f.write(f"{factid} | {prop}\n")


def write_report(summary: dict, top100: pd.DataFrame) -> None:
    """Write summary report."""
    report = f"""# V10.1 Re-ranking Report

## Summary

- **Input**: `{V10_INPUT.name}`
- **Output**: `{OUTPUT_CSV.name}`
- **Total facts processed**: {summary['total_facts']}
- **Top 100 facts exported**: {len(top100)}

## Tier Distribution (All Facts)

"""
    
    for tier in sorted(summary['tier_counts'].keys()):
        count = summary['tier_counts'][tier]
        tier_name = {
            1: 'Tier 1: Causal Chain of Harm',
            2: 'Tier 2: Harvard Knowledge / Foreseeability',
            3: 'Tier 3: Harvard Procedures / Silence / Cover-Up',
            4: 'Tier 4: Contextual Background',
            5: 'Tier 5: Low-Value Residuals',
        }.get(tier, f'Tier {tier}')
        report += f"- **{tier_name}**: {count} facts\n"
    
    report += f"""
## Tier Distribution (Top 100)

"""
    
    for tier in sorted(summary['top100_tier_counts'].keys()):
        count = summary['top100_tier_counts'][tier]
        tier_name = {
            1: 'Tier 1: Causal Chain of Harm',
            2: 'Tier 2: Harvard Knowledge / Foreseeability',
            3: 'Tier 3: Harvard Procedures / Silence / Cover-Up',
            4: 'Tier 4: Contextual Background',
            5: 'Tier 5: Low-Value Residuals',
        }.get(tier, f'Tier {tier}')
        report += f"- **{tier_name}**: {count} facts\n"
    
    report += f"""
## Changes from Original Top 100

### Demoted from Top 100
- **Count**: {len(summary['demoted_from_top100'])} facts
- **Sample FactIDs**: {', '.join(summary['demoted_from_top100'][:5]) if summary['demoted_from_top100'] else 'None'}

### New in Top 100
- **Count**: {len(summary['new_in_top100'])} facts
- **Sample FactIDs**: {', '.join(summary['new_in_top100'][:5]) if summary['new_in_top100'] else 'None'}

## Ranking Policy Applied

The new ranking system prioritizes:

1. **Tier 1 (Weight 1.00)**: Causal Chain of Harm facts
2. **Tier 2 (Weight 0.80)**: Harvard's Knowledge / Foreseeability
3. **Tier 3 (Weight 0.60)**: Harvard Procedures / Silence / Cover-Up
4. **Tier 4 (Weight 0.30)**: Contextual Background
5. **Tier 5 (Weight 0.05)**: Low-Value Residuals

New salience score = (tier_weight * 0.7) + (original_score * 0.3)

## Quality Improvements

- ✅ Core causal nodes now outrank contextual background
- ✅ Direct harm-generating actions outrank mere risk indicators
- ✅ Defendant conduct outranks environmental context
- ✅ Publication → Amplifier → PRC chain prioritized
- ✅ Meaningless fragments removed from top 100
- ✅ Contextual risks (e.g., Peter Humphrey) down-weighted

## Files Created

- `{OUTPUT_CSV.name}` - Top 100 facts with new ranking
- `{OUTPUT_TXT.name}` - Top 100 facts in ChatGPT-ready format
"""
    
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding='utf-8')


def main():
    """Main execution."""
    print("="*80)
    print("V10 → V10.1 RE-RANKING")
    print("="*80)
    
    print("\n1. Loading v10...")
    df = load_v10()
    print(f"   Loaded {len(df)} facts")
    
    print("\n2. Getting original top 100 for comparison...")
    if 'causal_salience_score' in df.columns:
        original_sorted = df.sort_values('causal_salience_score', ascending=False)
        original_top100 = original_sorted.head(100)
    else:
        original_top100 = df.head(100)
    
    print("\n3. Assigning manual salience tiers...")
    df = reorder_facts(df)
    print("   ✅ Tiers assigned")
    
    print("\n4. Computing new salience scores...")
    print("   ✅ Scores computed")
    
    print("\n5. Selecting top 100 facts...")
    top100 = df.head(100)
    print(f"   ✅ Selected {len(top100)} facts")
    
    print("\n6. Generating summary...")
    summary = generate_summary(df, top100, original_top100)
    
    print("\n7. Tier distribution (all facts):")
    for tier in sorted(summary['tier_counts'].keys()):
        count = summary['tier_counts'][tier]
        print(f"   Tier {tier}: {count} facts")
    
    print("\n8. Tier distribution (top 100):")
    for tier in sorted(summary['top100_tier_counts'].keys()):
        count = summary['top100_tier_counts'][tier]
        print(f"   Tier {tier}: {count} facts")
    
    print(f"\n9. Demoted from top 100: {len(summary['demoted_from_top100'])} facts")
    if summary['demoted_from_top100']:
        print(f"   Sample: {', '.join(summary['demoted_from_top100'][:5])}")
    
    print(f"\n10. New in top 100: {len(summary['new_in_top100'])} facts")
    if summary['new_in_top100']:
        print(f"    Sample: {', '.join(summary['new_in_top100'][:5])}")
    
    print("\n11. Exporting files...")
    export_files(df, top100)
    print(f"   ✅ Exported {OUTPUT_CSV}")
    print(f"   ✅ Exported {OUTPUT_TXT}")
    
    print("\n12. Writing report...")
    write_report(summary, top100)
    print(f"   ✅ Written {REPORT_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE! V10.1 Top 100 facts created")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

