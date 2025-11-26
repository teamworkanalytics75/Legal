#!/usr/bin/env python3
"""
Factual Audit Script for v18 Dataset
Detects:
- Missing required elements (EventDate, Subject, ActorRole)
- Duplicate facts
- Contradictory facts
- Incomplete propositions
- Improper TruthStatus assignments
- Wrong ActorRole assignments
- Missing causation prerequisites
"""

import csv
import re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

V18_INPUT = Path("case_law_data/top_1000_facts_for_chatgpt_v18_final.csv")
OUTPUT_ANOMALIES_CSV = Path("case_law_data/v18_factual_anomalies.csv")
OUTPUT_ANOMALIES_TXT = Path("case_law_data/v18_factual_anomalies.txt")
OUTPUT_REPORT = Path("reports/analysis_outputs/v18_factual_audit_report.md")

# Required fields that should not be empty
REQUIRED_FIELDS = {
    'factid': 'FactID',
    'proposition': 'Proposition',
    'subject': 'Subject',
    'actorrole': 'ActorRole',
}

# Fields that should ideally be filled
RECOMMENDED_FIELDS = {
    'eventdate': 'EventDate',
    'evidencetype': 'EvidenceType',
    'truthstatus': 'TruthStatus',
}

# TruthStatus values that should be validated
VALID_TRUTH_STATUS = {'True', 'Alleged', 'HostileFalseClaim'}

# ActorRole values that should match Subject
ACTORROLE_SUBJECT_MISMATCH_PATTERNS = [
    # If Subject contains "Harvard" but ActorRole doesn't
    (r'.*[Hh]arvard.*', r'^(?!.*[Hh]arvard).*$'),
    # If Subject contains "PRC" but ActorRole doesn't
    (r'.*PRC.*', r'^(?!.*PRC).*$'),
    # If Subject contains "Plaintiff" but ActorRole doesn't
    (r'.*[Pp]laintiff.*|.*[Gg]rayson.*', r'^(?!.*[Pp]laintiff).*$'),
]


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Lowercase, remove punctuation, normalize whitespace
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text


def similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts."""
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()


def detect_missing_required_fields(fact: dict) -> list[str]:
    """Detect missing required fields."""
    issues = []
    for field, label in REQUIRED_FIELDS.items():
        value = str(fact.get(field, '')).strip()
        if not value or value.lower() in ('nan', 'none', 'unknown'):
            issues.append(f"Missing {label}")
    return issues


def detect_missing_recommended_fields(fact: dict) -> list[str]:
    """Detect missing recommended fields."""
    issues = []
    for field, label in RECOMMENDED_FIELDS.items():
        value = str(fact.get(field, '')).strip()
        if not value or value.lower() in ('nan', 'none', 'unknown'):
            issues.append(f"Missing {label}")
    return issues


def detect_incomplete_proposition(fact: dict) -> list[str]:
    """Detect incomplete or problematic propositions."""
    issues = []
    prop = str(fact.get('proposition', '')).strip()
    
    if not prop:
        return ["Empty proposition"]
    
    # Check for incomplete sentences (ending with incomplete words)
    if prop.endswith(('...', '…', '--', '—')):
        issues.append("Proposition ends with ellipsis/dash (may be incomplete)")
    
    # Check for very short propositions
    if len(prop) < 20:
        issues.append(f"Very short proposition ({len(prop)} chars)")
    
    # Check for missing verb (basic heuristic)
    verbs = ['is', 'was', 'were', 'has', 'have', 'had', 'did', 'does', 'do',
             'published', 'sent', 'received', 'filed', 'stated', 'claimed',
             'warned', 'arrested', 'tortured', 'opened', 'created', 'posted']
    prop_lower = prop.lower()
    if not any(verb in prop_lower for verb in verbs):
        issues.append("Proposition may be missing a verb")
    
    # Check for trailing fragments
    if prop.endswith((' with', ' to', ' for', ' by', ' in', ' at', ' of', ' and', ' or')):
        issues.append("Proposition ends with hanging preposition/conjunction")
    
    return issues


def detect_truth_status_issues(fact: dict) -> list[str]:
    """Detect issues with TruthStatus assignment."""
    issues = []
    truth_status = str(fact.get('truthstatus', '')).strip()
    proposition = str(fact.get('proposition', '')).lower()
    evidencetype = str(fact.get('evidencetype', '')).strip()
    
    # Check if TruthStatus is valid
    if truth_status and truth_status not in VALID_TRUTH_STATUS:
        issues.append(f"Invalid TruthStatus: {truth_status}")
    
    # Check for mismatches
    # If EvidenceType is HKFiling/USFiling/CourtOrder/Document/Exhibit
    # and proposition describes the document (not a claim), should be "True"
    if evidencetype in {'HKFiling', 'USFiling', 'CourtOrder', 'Document', 'Exhibit'}:
        if 'allegation' not in proposition and 'alleges' not in proposition:
            if truth_status == 'Alleged':
                issues.append("Official document fact marked as 'Alleged' but appears descriptive")
    
    # If proposition explicitly states "alleges" or "allegation", should be "Alleged"
    if ('alleges' in proposition or 'allegation' in proposition) and truth_status != 'Alleged':
        issues.append("Proposition contains 'alleges/allegation' but TruthStatus is not 'Alleged'")
    
    # If proposition describes hostile/false content, should be "HostileFalseClaim"
    hostile_keywords = ['hostile', 'false', 'fake', 'misrepresentation', 'defamatory']
    if any(kw in proposition for kw in hostile_keywords) and truth_status != 'HostileFalseClaim':
        if truth_status == 'True':
            issues.append("Proposition describes hostile/false content but marked as 'True'")
    
    return issues


def detect_actorrole_issues(fact: dict) -> list[str]:
    """Detect issues with ActorRole assignment."""
    issues = []
    subject = str(fact.get('subject', '')).strip()
    actorrole = str(fact.get('actorrole', '')).strip()
    proposition = str(fact.get('proposition', '')).lower()
    
    if not subject or not actorrole:
        return issues
    
    # Check for obvious mismatches
    subject_lower = subject.lower()
    actorrole_lower = actorrole.lower()
    
    # Harvard mismatches
    if 'harvard' in subject_lower and 'harvard' not in actorrole_lower:
        # Exception: if ActorRole is "Harvard Club" or "Harvard OGC", that's fine
        if 'harvard club' not in actorrole_lower and 'harvard ogc' not in actorrole_lower:
            issues.append(f"Subject contains 'Harvard' but ActorRole '{actorrole}' doesn't match")
    
    # PRC mismatches
    if 'prc' in subject_lower or 'china' in subject_lower:
        if 'prc' not in actorrole_lower and 'state' not in actorrole_lower:
            issues.append(f"Subject contains 'PRC/China' but ActorRole '{actorrole}' doesn't match")
    
    # Plaintiff mismatches
    if 'plaintiff' in subject_lower or 'grayson' in subject_lower:
        if 'plaintiff' not in actorrole_lower:
            issues.append(f"Subject contains 'Plaintiff/Grayson' but ActorRole '{actorrole}' doesn't match")
    
    # Check if proposition describes action by Subject but ActorRole is different
    # This is a heuristic - may need manual review
    if subject and actorrole:
        # If proposition says "Harvard did X" but ActorRole is not Harvard-related
        if 'harvard' in proposition and 'harvard' not in actorrole_lower:
            if 'harvard' in subject_lower:
                issues.append("Proposition describes Harvard action but ActorRole doesn't match Subject")
    
    return issues


def detect_duplicate_facts(facts: list[dict]) -> list[dict]:
    """Detect duplicate or near-duplicate facts."""
    duplicates = []
    seen = {}
    
    for i, fact1 in enumerate(facts):
        prop1 = normalize_text(str(fact1.get('proposition', '')))
        factid1 = str(fact1.get('factid', ''))
        
        if not prop1:
            continue
        
        # Check against all previously seen facts
        for factid2, (prop2, idx2) in seen.items():
            similarity = similarity_score(prop1, prop2)
            
            # High similarity threshold for duplicates
            if similarity > 0.85:
                duplicates.append({
                    'factid1': factid1,
                    'factid2': factid2,
                    'similarity': similarity,
                    'proposition1': str(fact1.get('proposition', '')),
                    'proposition2': str(facts[idx2].get('proposition', '')),
                })
        
        seen[factid1] = (prop1, i)
    
    return duplicates


def detect_contradictory_facts(facts: list[dict]) -> list[dict]:
    """Detect potentially contradictory facts."""
    contradictions = []
    
    # Group facts by subject/actor
    by_subject = defaultdict(list)
    for fact in facts:
        subject = str(fact.get('subject', '')).strip()
        if subject:
            by_subject[subject].append(fact)
    
    # Check for contradictions within same subject
    for subject, subject_facts in by_subject.items():
        if len(subject_facts) < 2:
            continue
        
        # Check for date contradictions
        dates = {}
        for fact in subject_facts:
            factid = str(fact.get('factid', ''))
            eventdate = str(fact.get('eventdate', '')).strip()
            prop = str(fact.get('proposition', '')).lower()
            
            # Extract date mentions from proposition
            date_pattern = r'\b(20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})\b'
            prop_dates = re.findall(date_pattern, prop)
            
            if eventdate:
                dates[factid] = {'eventdate': eventdate, 'prop_dates': prop_dates}
        
        # Check for conflicting dates for same event
        # This is a simplified check - may need refinement
        for factid1, date_info1 in dates.items():
            for factid2, date_info2 in dates.items():
                if factid1 >= factid2:
                    continue
                
                # If EventDates differ significantly and propositions are similar
                if date_info1['eventdate'] and date_info2['eventdate']:
                    if date_info1['eventdate'] != date_info2['eventdate']:
                        fact1 = next(f for f in subject_facts if str(f.get('factid')) == factid1)
                        fact2 = next(f for f in subject_facts if str(f.get('factid')) == factid2)
                        prop1 = str(fact1.get('proposition', ''))
                        prop2 = str(fact2.get('proposition', ''))
                        
                        # Check if they describe the same event
                        similarity = similarity_score(prop1, prop2)
                        if similarity > 0.6:  # Similar propositions
                            contradictions.append({
                                'factid1': factid1,
                                'factid2': factid2,
                                'issue': f"Conflicting dates: {date_info1['eventdate']} vs {date_info2['eventdate']}",
                                'proposition1': prop1,
                                'proposition2': prop2,
                            })
    
    return contradictions


def detect_missing_causation_prerequisites(facts: list[dict]) -> list[dict]:
    """Detect missing causation prerequisites."""
    issues = []
    
    # Facts that reference other facts or events
    referenced_events = set()
    fact_ids = {str(f.get('factid', '')) for f in facts}
    
    for fact in facts:
        prop = str(fact.get('proposition', '')).lower()
        factid = str(fact.get('factid', ''))
        
        # Check for references to Statement 1, Statement 2
        if 'statement 1' in prop or 'statement 2' in prop:
            # Look for corresponding fact
            statement_facts = [f for f in facts if 'statement 1' in str(f.get('proposition', '')).lower() or 'statement 2' in str(f.get('proposition', '')).lower()]
            if not statement_facts:
                issues.append({
                    'factid': factid,
                    'issue': "References Statement 1/2 but no corresponding fact found",
                    'proposition': str(fact.get('proposition', '')),
                })
        
        # Check for references to EsuWiki
        if 'esuwiki' in prop:
            esuwiki_facts = [f for f in facts if 'esuwiki' in str(f.get('proposition', '')).lower()]
            if len(esuwiki_facts) < 3:  # Should have multiple EsuWiki facts
                issues.append({
                    'factid': factid,
                    'issue': "References EsuWiki but EsuWiki timeline may be incomplete",
                    'proposition': str(fact.get('proposition', '')),
                })
        
        # Check for references to OGC non-response
        if 'ogc' in prop and ('non-response' in prop or 'did not respond' in prop or 'silence' in prop):
            ogc_facts = [f for f in facts if 'ogc' in str(f.get('proposition', '')).lower() and ('non-response' in str(f.get('proposition', '')).lower() or 'did not respond' in str(f.get('proposition', '')).lower())]
            if len(ogc_facts) < 3:  # Should have multiple OGC non-response facts
                issues.append({
                    'factid': factid,
                    'issue': "References OGC non-response but OGC timeline may be incomplete",
                    'proposition': str(fact.get('proposition', '')),
                })
    
    return issues


def main():
    """Run factual audit."""
    print("="*80)
    print("V18 FACTUAL AUDIT")
    print("="*80)
    print()
    
    # Load facts
    print("1. Loading v18 facts...")
    facts = []
    with open(V18_INPUT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            facts.append(row)
    print(f"   Loaded {len(facts)} facts")
    
    # Collect all anomalies
    all_anomalies = []
    
    print("\n2. Checking for missing required fields...")
    missing_required_count = 0
    for fact in facts:
        issues = detect_missing_required_fields(fact)
        if issues:
            missing_required_count += len(issues)
            all_anomalies.append({
                'factid': str(fact.get('factid', '')),
                'category': 'Missing Required Field',
                'issues': '; '.join(issues),
                'proposition': str(fact.get('proposition', ''))[:200],
            })
    print(f"   Found {missing_required_count} missing required field issues")
    
    print("\n3. Checking for missing recommended fields...")
    missing_recommended_count = 0
    for fact in facts:
        issues = detect_missing_recommended_fields(fact)
        if issues:
            missing_recommended_count += len(issues)
            all_anomalies.append({
                'factid': str(fact.get('factid', '')),
                'category': 'Missing Recommended Field',
                'issues': '; '.join(issues),
                'proposition': str(fact.get('proposition', ''))[:200],
            })
    print(f"   Found {missing_recommended_count} missing recommended field issues")
    
    print("\n4. Checking for incomplete propositions...")
    incomplete_count = 0
    for fact in facts:
        issues = detect_incomplete_proposition(fact)
        if issues:
            incomplete_count += len(issues)
            all_anomalies.append({
                'factid': str(fact.get('factid', '')),
                'category': 'Incomplete Proposition',
                'issues': '; '.join(issues),
                'proposition': str(fact.get('proposition', ''))[:200],
            })
    print(f"   Found {incomplete_count} incomplete proposition issues")
    
    print("\n5. Checking TruthStatus assignments...")
    truth_status_count = 0
    for fact in facts:
        issues = detect_truth_status_issues(fact)
        if issues:
            truth_status_count += len(issues)
            all_anomalies.append({
                'factid': str(fact.get('factid', '')),
                'category': 'TruthStatus Issue',
                'issues': '; '.join(issues),
                'proposition': str(fact.get('proposition', ''))[:200],
            })
    print(f"   Found {truth_status_count} TruthStatus issues")
    
    print("\n6. Checking ActorRole assignments...")
    actorrole_count = 0
    for fact in facts:
        issues = detect_actorrole_issues(fact)
        if issues:
            actorrole_count += len(issues)
            all_anomalies.append({
                'factid': str(fact.get('factid', '')),
                'category': 'ActorRole Issue',
                'issues': '; '.join(issues),
                'proposition': str(fact.get('proposition', ''))[:200],
            })
    print(f"   Found {actorrole_count} ActorRole issues")
    
    print("\n7. Detecting duplicate facts...")
    duplicates = detect_duplicate_facts(facts)
    for dup in duplicates:
        all_anomalies.append({
            'factid': f"{dup['factid1']} vs {dup['factid2']}",
            'category': 'Duplicate Fact',
            'issues': f"Similarity: {dup['similarity']:.2%}",
            'proposition': f"1: {dup['proposition1'][:100]} | 2: {dup['proposition2'][:100]}",
        })
    print(f"   Found {len(duplicates)} duplicate pairs")
    
    print("\n8. Detecting contradictory facts...")
    contradictions = detect_contradictory_facts(facts)
    for cont in contradictions:
        all_anomalies.append({
            'factid': f"{cont['factid1']} vs {cont['factid2']}",
            'category': 'Contradiction',
            'issues': cont['issue'],
            'proposition': f"1: {cont['proposition1'][:100]} | 2: {cont['proposition2'][:100]}",
        })
    print(f"   Found {len(contradictions)} contradiction pairs")
    
    print("\n9. Checking for missing causation prerequisites...")
    missing_prereqs = detect_missing_causation_prerequisites(facts)
    for mp in missing_prereqs:
        all_anomalies.append({
            'factid': str(mp.get('factid', '')),
            'category': 'Missing Prerequisite',
            'issues': mp['issue'],
            'proposition': str(mp.get('proposition', ''))[:200],
        })
    print(f"   Found {len(missing_prereqs)} missing prerequisite issues")
    
    # Export anomalies
    print("\n10. Exporting anomalies...")
    if all_anomalies:
        with open(OUTPUT_ANOMALIES_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['factid', 'category', 'issues', 'proposition'])
            writer.writeheader()
            writer.writerows(all_anomalies)
        print(f"   ✅ Exported {OUTPUT_ANOMALIES_CSV}")
        
        # Export TXT
        lines = []
        lines.append("="*80)
        lines.append("V18 FACTUAL ANOMALIES")
        lines.append("="*80)
        lines.append("")
        lines.append(f"Total anomalies found: {len(all_anomalies)}")
        lines.append("")
        
        # Group by category
        by_category = defaultdict(list)
        for anomaly in all_anomalies:
            by_category[anomaly['category']].append(anomaly)
        
        for category, items in sorted(by_category.items()):
            lines.append(f"{category} ({len(items)} issues)")
            lines.append("-"*80)
            for item in items[:20]:  # Limit to first 20 per category
                lines.append(f"FactID: {item['factid']}")
                lines.append(f"Issue: {item['issues']}")
                lines.append(f"Proposition: {item['proposition']}")
                lines.append("")
            if len(items) > 20:
                lines.append(f"... and {len(items) - 20} more")
            lines.append("")
        
        OUTPUT_ANOMALIES_TXT.write_text("\n".join(lines), encoding='utf-8')
        print(f"   ✅ Exported {OUTPUT_ANOMALIES_TXT}")
    else:
        print("   ✅ No anomalies found!")
    
    # Generate report
    print("\n11. Generating report...")
    report = f"""# V18 Factual Audit Report

## Summary

- **Total facts audited**: {len(facts)}
- **Total anomalies found**: {len(all_anomalies)}

## Anomaly Breakdown

- **Missing Required Fields**: {missing_required_count} issues
- **Missing Recommended Fields**: {missing_recommended_count} issues
- **Incomplete Propositions**: {incomplete_count} issues
- **TruthStatus Issues**: {truth_status_count} issues
- **ActorRole Issues**: {actorrole_count} issues
- **Duplicate Facts**: {len(duplicates)} pairs
- **Contradictions**: {len(contradictions)} pairs
- **Missing Prerequisites**: {len(missing_prereqs)} issues

## Files Generated

- **Anomalies CSV**: `case_law_data/v18_factual_anomalies.csv`
- **Anomalies TXT**: `case_law_data/v18_factual_anomalies.txt`
- **Report**: `reports/analysis_outputs/v18_factual_audit_report.md`

## Next Steps

1. Review anomalies CSV/TXT
2. Fix high-priority issues (missing required fields, contradictions)
3. Verify TruthStatus and ActorRole assignments
4. Resolve duplicate facts
5. Fill missing EventDates where possible
6. Complete incomplete propositions

## Notes

This audit detects structural and logical issues. It does NOT verify facts against source documents.
For source validation, see Option B in the v18 completion plan.
"""
    
    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT.write_text(report, encoding='utf-8')
    print(f"   ✅ Written {OUTPUT_REPORT}")
    
    print()
    print("="*80)
    print("FACTUAL AUDIT COMPLETE")
    print("="*80)
    print()
    print(f"✅ Found {len(all_anomalies)} total anomalies across {len(facts)} facts")
    print()
    print("Review the anomalies files to identify issues that need fixing.")
    print()


if __name__ == "__main__":
    main()

