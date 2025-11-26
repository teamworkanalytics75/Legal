#!/usr/bin/env python3
"""Identify words from lawsuit documents that align with CatBoost success signals."""

import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Set

# CatBoost Success Signals (POSITIVE - use these)
SUCCESS_SIGNALS = {
    # Motion language (50% success vs 33.3% without)
    'motion_to_seal': ['motion to seal', 'motion for seal', 'file under seal', 'under seal'],
    'protective_order': ['protective order', 'protective orders'],
    'impound': ['impound', 'impounded', 'impounding'],
    'pseudonym': ['pseudonym', 'proceed under pseudonym', 'motion to proceed pseudonym'],

    # US Citizen Endangerment (83.3% success - highest topic)
    'us_citizen': ['us citizen', 'united states citizen', 'american citizen'],
    'endangerment_implicit': ['safety', 'harm', 'threat', 'risk', 'danger', 'bodily harm',
                              'physical harm', 'immediate harm', 'serious harm'],

    # National Security (45.3% success)
    'national_security': ['national security', 'national security interest', 'security interest'],

    # Privacy/Safety framing (use carefully - implicit better)
    'privacy_safety': ['privacy', 'safety', 'privacy interest', 'safety concern'],

    # Balancing framework language (use framework, not phrase)
    'balance_weigh': ['weigh', 'balance', 'balance of equities', 'weighing', 'balanced'],

    # Thoroughness signals (word count is #1 predictor)
    'comprehensive': ['comprehensive', 'thorough', 'detailed', 'complete', 'extensive'],
}

# CatBoost Failure Signals (NEGATIVE - avoid these)
FAILURE_SIGNALS = {
    # Public Interest (0% success - STRONG AVOID)
    'public_interest': ['public interest', 'public\'s interest', 'public access',
                       'public right', 'public disclosure'],

    # Balancing Test explicit phrase (0% success - STRONG AVOID)
    'balancing_test': ['balancing test', 'applying the balancing test', 'requires a balancing test'],

    # First Amendment (25% vs 42.4% - AVOID)
    'first_amendment': ['first amendment', 'first amendment right', 'free speech',
                       'freedom of speech', 'constitutional right'],

    # Standard of Review (-23.5pp - AVOID)
    'standard_of_review': ['standard of review', 'review standard'],

    # Narrowly Tailored (25% vs 39% - AVOID)
    'narrowly_tailored': ['narrowly tailored', 'narrow tailoring'],

    # Endangerment explicit (25% vs 43.5% - use implicitly)
    'endangerment_explicit': ['endangerment', 'reckless endangerment'],
}

def extract_phrases(text: str, phrase_list: List[str]) -> List[str]:
    """Extract matching phrases from text."""
    text_lower = text.lower()
    found = []
    for phrase in phrase_list:
        if phrase.lower() in text_lower:
            found.append(phrase)
    return found

def analyze_documents(doc_paths: List[Path]) -> Dict[str, any]:
    """Analyze documents for CatBoost success/failure signals."""
    results = {
        'success_signals': {},
        'failure_signals': {},
        'all_text': ''
    }

    for doc_path in doc_paths:
        if not doc_path.exists():
            print(f"⚠️  Warning: {doc_path} not found, skipping...")
            continue

        print(f"📄 Analyzing: {doc_path.name}")
        try:
            text = doc_path.read_text(encoding='utf-8', errors='ignore')
            results['all_text'] += text + "\n\n"
        except Exception as e:
            print(f"   ❌ Error reading {doc_path}: {e}")

    # Check for success signals
    for signal_type, phrases in SUCCESS_SIGNALS.items():
        found = extract_phrases(results['all_text'], phrases)
        if found:
            results['success_signals'][signal_type] = found

    # Check for failure signals
    for signal_type, phrases in FAILURE_SIGNALS.items():
        found = extract_phrases(results['all_text'], phrases)
        if found:
            results['failure_signals'][signal_type] = found

    return results

def extract_success_words(text: str) -> Counter:
    """Extract words that are part of success signal phrases."""
    text_lower = text.lower()
    success_words = []

    # Words that are directly part of success signals
    success_word_list = {
        # Motion language (50% success)
        'motion', 'seal', 'sealed', 'sealing', 'protective', 'order', 'orders',
        'impound', 'impounded', 'impounding', 'pseudonym', 'pseudonyms',

        # US Citizen Endangerment (83.3% success)
        'citizen', 'citizens', 'american',

        # Endangerment implicit (good - use these)
        'safety', 'harm', 'threat', 'threats', 'risk', 'risks', 'danger',
        'bodily', 'physical', 'immediate', 'serious',

        # National Security (45.3% success)
        'national', 'security',

        # Privacy/Safety
        'privacy', 'private',

        # Balancing framework (use these words, not "balancing test")
        'weigh', 'weighing', 'balance', 'balanced', 'equities',

        # Comprehensive/thorough (word count is #1 predictor)
        'comprehensive', 'thorough', 'detailed', 'complete', 'extensive',
    }

    # Extract all words from text
    all_words = re.findall(r'\b[a-z]{4,}\b', text_lower)

    # Count only success signal words
    for word in all_words:
        if word in success_word_list:
            success_words.append(word)

    return Counter(success_words)

def main():
    """Main analysis function."""
    project_root = Path(__file__).parent

    # Define document paths
    documents = [
        project_root / "hk_statement_text.txt",
        project_root / "case_law_data" / "tmp_corpus" / "3 Emails to Harvard_s OGC in 2025.txt",
    ]

    print("🔍 Analyzing lawsuit documents for CatBoost success signals...\n")

    # Analyze documents
    results = analyze_documents(documents)

    # Extract success signal words
    success_word_counts = extract_success_words(results['all_text'])
    top_20 = success_word_counts.most_common(20)

    # Map words to their success signal categories
    word_to_signal = {
        'motion': 'Motion Language (50% success)',
        'seal': 'Motion Language (50% success)',
        'sealed': 'Motion Language (50% success)',
        'sealing': 'Motion Language (50% success)',
        'protective': 'Motion Language (50% success)',
        'order': 'Motion Language (50% success)',
        'orders': 'Motion Language (50% success)',
        'impound': 'Motion Language (50% success)',
        'impounded': 'Motion Language (50% success)',
        'impounding': 'Motion Language (50% success)',
        'pseudonym': 'Motion Language (50% success)',
        'pseudonyms': 'Motion Language (50% success)',
        'citizen': 'US Citizen Endangerment (83.3% success)',
        'citizens': 'US Citizen Endangerment (83.3% success)',
        'american': 'US Citizen Endangerment (83.3% success)',
        'safety': 'Endangerment Implicit (Good)',
        'harm': 'Endangerment Implicit (Good)',
        'threat': 'Endangerment Implicit (Good)',
        'threats': 'Endangerment Implicit (Good)',
        'risk': 'Endangerment Implicit (Good)',
        'risks': 'Endangerment Implicit (Good)',
        'danger': 'Endangerment Implicit (Good)',
        'bodily': 'Endangerment Implicit (Good)',
        'physical': 'Endangerment Implicit (Good)',
        'immediate': 'Endangerment Implicit (Good)',
        'serious': 'Endangerment Implicit (Good)',
        'national': 'National Security (45.3% success)',
        'security': 'National Security (45.3% success)',
        'privacy': 'Privacy/Safety Framing',
        'private': 'Privacy/Safety Framing',
        'weigh': 'Balancing Framework (Good)',
        'weighing': 'Balancing Framework (Good)',
        'balance': 'Balancing Framework (Good)',
        'balanced': 'Balancing Framework (Good)',
        'equities': 'Balancing Framework (Good)',
        'comprehensive': 'Thoroughness (Word count #1 predictor)',
        'thorough': 'Thoroughness (Word count #1 predictor)',
        'detailed': 'Thoroughness (Word count #1 predictor)',
        'complete': 'Thoroughness (Word count #1 predictor)',
        'extensive': 'Thoroughness (Word count #1 predictor)',
    }

    print("\n" + "="*80)
    print("📊 TOP 20 WORDS ALIGNED WITH CATBOOST SUCCESS SIGNALS")
    print("="*80)
    print("\nThese are words from your documents that CatBoost identifies as")
    print("predictive of successful motions to seal/pseudonym.\n")

    print(f"{'Rank':<6} {'Word':<20} {'Count':<10} {'CatBoost Signal':<50}")
    print("-" * 90)

    for rank, (word, count) in enumerate(top_20, 1):
        signal = word_to_signal.get(word, 'Success Signal Word')
        print(f"{rank:<6} {word:<20} {count:<10} {signal:<50}")

    print("\n" + "="*80)
    print("✅ SUCCESS SIGNALS FOUND IN YOUR DOCUMENTS")
    print("="*80)

    for signal_type, phrases in results['success_signals'].items():
        print(f"\n🎯 {signal_type.replace('_', ' ').title()}:")
        for phrase in phrases[:5]:  # Show first 5
            print(f"   ✅ '{phrase}'")

    print("\n" + "="*80)
    print("❌ FAILURE SIGNALS FOUND (SHOULD BE REMOVED/MODIFIED)")
    print("="*80)

    if results['failure_signals']:
        for signal_type, phrases in results['failure_signals'].items():
            print(f"\n⚠️  {signal_type.replace('_', ' ').title()}:")
            for phrase in phrases:
                print(f"   ❌ '{phrase}' - Consider removing or reframing")
    else:
        print("\n✅ No failure signals found! Your documents avoid the problematic phrases.")

    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)

    # Check for missing success signals
    missing_signals = []
    if 'motion_to_seal' not in results['success_signals']:
        missing_signals.append("Add 'motion to seal' or 'file under seal' language")
    if 'protective_order' not in results['success_signals']:
        missing_signals.append("Consider adding 'protective order' language")
    if 'us_citizen' in results['success_signals']:
        print("\n✅ STRONG: You mention US citizen - this aligns with 83.3% success topic!")
    if 'national_security' in results['success_signals']:
        print("\n✅ GOOD: National security framing aligns with 45.3% success rate")
    if 'endangerment_implicit' in results['success_signals']:
        print("\n✅ GOOD: You use implicit endangerment language (safety, harm, threat)")

    if missing_signals:
        print("\n📝 Consider adding:")
        for signal in missing_signals:
            print(f"   - {signal}")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()

