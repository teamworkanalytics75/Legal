#!/usr/bin/env python3
"""Analyze top 20 most important words from lawsuit documents."""

import re
from collections import Counter
from pathlib import Path
from typing import List, Dict

# Common legal stop words to filter out
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
    'might', 'must', 'can', 'this', 'that', 'these', 'those', 'it', 'its', 'they',
    'them', 'their', 'there', 'then', 'than', 'when', 'where', 'what', 'which',
    'who', 'whom', 'whose', 'why', 'how', 'all', 'each', 'every', 'some', 'any',
    'no', 'not', 'only', 'just', 'also', 'more', 'most', 'very', 'much', 'many',
    'such', 'so', 'too', 'well', 'now', 'here', 'there', 'up', 'down', 'out',
    'off', 'over', 'under', 'again', 'further', 'then', 'once', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among',
    'within', 'without', 'against', 'along', 'around', 'behind', 'beyond', 'near',
    'since', 'until', 'while', 'if', 'unless', 'because', 'although', 'though',
    'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore', 'additionally',
    'defendant', 'defendants', 'plaintiff', 'plaintiffs', 'court', 'courts', 'case',
    'cases', 'claim', 'claims', 'action', 'actions', 'motion', 'motions', 'order',
    'orders', 'document', 'documents', 'section', 'sections', 'paragraph', 'paragraphs',
    'said', 'shall', 'pursuant', 'herein', 'hereof', 'thereof', 'therein', 'whereof',
    'his', 'her', 'him', 'she', 'he', 'you', 'your', 'yours', 'me', 'my', 'mine',
    'we', 'our', 'ours', 'us', 'they', 'them', 'their', 'theirs', 'its', 'it',
    'end', 'part', 'parts', 'one', 'two', 'three', 'four', 'five', 'first', 'second',
    'third', 'last', 'next', 'previous', 'other', 'others', 'another', 'same', 'different'
}

def clean_word(word: str) -> str:
    """Clean and normalize a word."""
    # Remove punctuation and convert to lowercase
    word = re.sub(r'[^\w\s]', '', word.lower())
    # Remove numbers
    word = re.sub(r'\d+', '', word)
    return word.strip()

def extract_meaningful_words(text: str, min_length: int = 4) -> List[str]:
    """Extract meaningful words from text, filtering stop words."""
    # First, extract bigrams for important phrases
    text_lower = text.lower()

    # Replace common phrases with combined terms
    phrase_replacements = {
        'hong kong': 'hong_kong',
        'hong kong special administrative region': 'hong_kong',
        'harvard club': 'harvard_club',
        'harvard clubs': 'harvard_club',
        'harvard alumni': 'harvard_alumni',
        'statement of claim': 'statement_of_claim',
        'office of general counsel': 'ogc',
        'general counsel': 'ogc',
        'xi mingze': 'xi_mingze',
        'weiqi zhang': 'weiqi_zhang',
        'malcolm grayson': 'malcolm_grayson',
        'blue oak': 'blue_oak',
        'blue oak education': 'blue_oak_education',
        'prc': 'china',
        'people\'s republic of china': 'china',
        'people republic of china': 'china',
    }

    for phrase, replacement in phrase_replacements.items():
        text_lower = text_lower.replace(phrase, replacement)

    words = re.findall(r'\b\w+\b', text_lower)
    meaningful = []
    for word in words:
        cleaned = clean_word(word)
        if (len(cleaned) >= min_length and
            cleaned not in STOP_WORDS and
            not cleaned.isdigit()):
            meaningful.append(cleaned)
    return meaningful

def analyze_documents(doc_paths: List[Path]) -> Counter:
    """Analyze word frequency across multiple documents."""
    all_words = []

    for doc_path in doc_paths:
        if not doc_path.exists():
            print(f"⚠️  Warning: {doc_path} not found, skipping...")
            continue

        print(f"📄 Analyzing: {doc_path.name}")
        try:
            text = doc_path.read_text(encoding='utf-8', errors='ignore')
            words = extract_meaningful_words(text)
            all_words.extend(words)
            print(f"   Extracted {len(words)} meaningful words")
        except Exception as e:
            print(f"   ❌ Error reading {doc_path}: {e}")

    return Counter(all_words)

def main():
    """Main analysis function."""
    project_root = Path(__file__).parent

    # Define document paths
    documents = [
        project_root / "hk_statement_text.txt",
        project_root / "case_law_data" / "tmp_corpus" / "3 Emails to Harvard_s OGC in 2025.txt",
    ]

    print("🔍 Analyzing lawsuit documents for top keywords...\n")

    # Analyze documents
    word_counts = analyze_documents(documents)

    if not word_counts:
        print("❌ No words found. Check document paths.")
        return

    # Get top 20 words
    top_20 = word_counts.most_common(20)

    print("\n" + "="*80)
    print("📊 TOP 20 MOST IMPORTANT WORDS DESCRIBING YOUR LAWSUIT")
    print("="*80)
    print(f"\nTotal unique meaningful words: {len(word_counts)}")
    print(f"Total word occurrences: {sum(word_counts.values())}\n")

    print(f"{'Rank':<6} {'Word':<25} {'Count':<10} {'% of Total':<15}")
    print("-" * 80)

    total_words = sum(word_counts.values())
    for rank, (word, count) in enumerate(top_20, 1):
        percentage = (count / total_words * 100) if total_words > 0 else 0
        print(f"{rank:<6} {word:<25} {count:<10} {percentage:>6.2f}%")

    print("\n" + "="*80)
    print("💡 KEY INSIGHTS:")
    print("="*80)

    # Group by themes - check all words, not just top 20
    all_top_words = [w for w, _ in word_counts.most_common(100)]

    harvard_terms = [w for w, _ in top_20 if 'harvard' in w]
    legal_terms = [w for w, _ in top_20 if any(term in w for term in ['defam', 'claim', 'alleg', 'court', 'jurisdict', 'tort', 'neglig', 'damage', 'particular'])]
    china_terms = [w for w, _ in top_20 if any(term in w for term in ['china', 'chinese', 'prc', 'beijing', 'shanghai', 'hong', 'kong'])]
    safety_terms = [w for w in all_top_words if any(term in w for term in ['safety', 'harm', 'danger', 'risk', 'threat', 'retali', 'persecut', 'dox', 'incit', 'conspir'])]

    # Find important terms that might not be in top 20
    important_terms = {}
    for term in ['defamation', 'retaliation', 'safety', 'danger', 'harm', 'threat', 'persecution',
                 'doxing', 'incitement', 'conspiracy', 'reckless', 'endangerment', 'torture',
                 'arrest', 'political', 'privacy', 'seal', 'pseudonym', 'ogc', 'weiqi', 'mingze']:
        if term in word_counts:
            important_terms[term] = word_counts[term]

    if harvard_terms:
        print(f"\n🏛️  Harvard-related: {', '.join(harvard_terms)}")
    if legal_terms:
        print(f"⚖️  Legal terms: {', '.join(legal_terms)}")
    if china_terms:
        print(f"🇨🇳 China/PRC-related: {', '.join(china_terms)}")
    if safety_terms:
        print(f"🛡️  Safety/harm-related: {', '.join(safety_terms[:10])}")  # Show top 10

    if important_terms:
        print(f"\n⚖️  Important legal/safety terms (may not be in top 20):")
        sorted_important = sorted(important_terms.items(), key=lambda x: x[1], reverse=True)
        for term, count in sorted_important[:15]:
            print(f"   - {term}: {count} occurrences")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()

