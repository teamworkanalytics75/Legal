#!/usr/bin/env python3
"""
Generate Summary Report from PDF Analysis Results
"""

import json

def generate_summary_report():
    """Generate a comprehensive summary report."""

    # Load analysis results
    with open("data/case_law/simple_pdf_analysis.json", 'r') as f:
        results = json.load(f)

    print("🎯 1782 PDF ANALYSIS SUMMARY REPORT")
    print("="*60)
    print(f"📄 Total Cases Analyzed: {len(results)}")

    # Text Statistics
    text_lengths = [r['text_length'] for r in results]
    word_counts = [r['word_count'] for r in results]
    page_counts = [r['page_count'] for r in results]

    print(f"\n📊 Document Statistics:")
    print(f"  Average Text Length: {sum(text_lengths)/len(text_lengths):.0f} characters")
    print(f"  Average Word Count: {sum(word_counts)/len(word_counts):.0f} words")
    print(f"  Average Page Count: {sum(page_counts)/len(page_counts):.1f} pages")
    print(f"  Text Range: {min(text_lengths):,} - {max(text_lengths):,} characters")

    # Outcome Distribution
    outcomes = [r['outcome']['outcome'] for r in results]
    outcome_counts = {}
    for outcome in outcomes:
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    print(f"\n📈 Outcome Distribution:")
    for outcome, count in outcome_counts.items():
        percentage = (count / len(results)) * 100
        print(f"  {outcome.upper()}: {count} cases ({percentage:.1f}%)")

    # Intel Factor Analysis
    print(f"\n🔍 Intel Factor Detection Rates:")
    factor_stats = {}
    for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
        detected = sum(1 for r in results if r['intel_factors'].get(factor, False))
        factor_stats[factor] = {
            'detected': detected,
            'total': len(results),
            'rate': detected / len(results)
        }
        print(f"  {factor}: {detected}/{len(results)} ({detected/len(results):.1%})")

    # Success Patterns
    successful_cases = [r for r in results if r['outcome']['outcome'] == 'granted']
    denied_cases = [r for r in results if r['outcome']['outcome'] == 'denied']

    print(f"\n🎯 Success vs Failure Patterns:")
    print(f"  Successful Cases: {len(successful_cases)}")
    print(f"  Denied Cases: {len(denied_cases)}")

    if successful_cases and denied_cases:
        print(f"\n  Intel Factors in SUCCESSFUL Cases:")
        for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
            success_rate = sum(1 for r in successful_cases if r['intel_factors'].get(factor, False)) / len(successful_cases)
            print(f"    {factor}: {success_rate:.1%}")

        print(f"\n  Intel Factors in DENIED Cases:")
        for factor in ['factor_1', 'factor_2', 'factor_3', 'factor_4']:
            denied_rate = sum(1 for r in denied_cases if r['intel_factors'].get(factor, False)) / len(denied_cases)
            print(f"    {factor}: {denied_rate:.1%}")

    # Citation Analysis
    all_citations = []
    for r in results:
        for citation in r['citations']:
            all_citations.append(citation['pattern'])

    citation_counts = {}
    for citation in all_citations:
        citation_counts[citation] = citation_counts.get(citation, 0) + 1

    if citation_counts:
        print(f"\n📚 Citation Patterns:")
        sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)
        for citation, count in sorted_citations:
            print(f"  {citation}: {count} mentions")

    # Section 1782 Mentions
    section_1782_total = sum(r['section_1782_mentions'] for r in results)
    print(f"\n⚖️ Section 1782 Mentions:")
    print(f"  Total Mentions: {section_1782_total}")
    print(f"  Average per Case: {section_1782_total/len(results):.1f}")

    # Key Insights
    print(f"\n💡 KEY INSIGHTS:")

    # Most successful case
    if successful_cases:
        most_successful = max(successful_cases, key=lambda x: x['outcome']['granted_count'])
        print(f"  🏆 Most Successful Case: {most_successful['case_name']}")
        print(f"     - Granted mentions: {most_successful['outcome']['granted_count']}")
        print(f"     - Intel factors: {sum(most_successful['intel_factors'].values())}/4")
        print(f"     - Section 1782 mentions: {most_successful['section_1782_mentions']}")

    # Factor 4 correlation
    factor_4_cases = [r for r in results if r['intel_factors'].get('factor_4', False)]
    factor_4_success_rate = sum(1 for r in factor_4_cases if r['outcome']['outcome'] == 'granted') / len(factor_4_cases) if factor_4_cases else 0
    print(f"  📊 Factor 4 (Unduly Intrusive) Success Rate: {factor_4_success_rate:.1%}")

    # Intel Corp citation correlation
    intel_cited_cases = [r for r in results if any(c['pattern'] == 'Intel\\s+Corp' for c in r['citations'])]
    intel_success_rate = sum(1 for r in intel_cited_cases if r['outcome']['outcome'] == 'granted') / len(intel_cited_cases) if intel_cited_cases else 0
    print(f"  📊 Intel Corp Citation Success Rate: {intel_success_rate:.1%}")

    # Text length correlation
    successful_text_lengths = [r['text_length'] for r in successful_cases]
    denied_text_lengths = [r['text_length'] for r in denied_cases]
    if successful_text_lengths and denied_text_lengths:
        avg_success_length = sum(successful_text_lengths) / len(successful_text_lengths)
        avg_denied_length = sum(denied_text_lengths) / len(denied_text_lengths)
        print(f"  📊 Average Text Length - Success: {avg_success_length:.0f}, Denied: {avg_denied_length:.0f}")

    print(f"\n✅ Analysis Complete!")
    print(f"📁 Results saved in: data/case_law/simple_pdf_analysis.json")

if __name__ == "__main__":
    generate_summary_report()
