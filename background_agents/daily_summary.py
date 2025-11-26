"""Generate daily summary of agent activities.

Optionally also generate the cross-repo Activity Digest that all agents can
consume (see scripts/recent_activity_digest.py).
"""

import json
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def count_files_today(directory: Path, extension: str) -> int:
    """Count files created today in a directory."""
    if not directory.exists():
        return 0

    today = datetime.now().date()
    count = 0

    for file in directory.glob(f"**/*{extension}"):
        if file.stat().st_mtime:
            file_date = datetime.fromtimestamp(file.stat().st_mtime).date()
            if file_date == today:
                count += 1

    return count


def get_latest_insight(directory: Path, extension: str) -> str:
    """Get a snippet from the latest file."""
    if not directory.exists():
        return "No insights yet"

    files = sorted(directory.glob(f"**/*{extension}"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not files:
        return "No insights yet"

    try:
        with open(files[0]) as f:
            if extension == '.json':
                data = json.load(f)
                # Try to extract a meaningful snippet
                if 'summary' in data:
                    return data['summary'][:200] + "..."
                elif 'analysis' in data:
                    return data['analysis'][:200] + "..."
                else:
                    return "Analysis complete"
            else:
                content = f.read()
                lines = content.split('\n')
                # Find first substantial line
                for line in lines:
                    if len(line.strip()) > 50:
                        return line.strip()[:200] + "..."
                return "Analysis complete"
    except:
        return "Analysis complete"


def _generate_activity_digest(days: int = 7) -> None:
    """Run the recent activity digest script to update shared reports."""
    md_out = Path("reports/analysis_outputs/activity_digest.md")
    json_out = Path("reports/analysis_outputs/activity_digest.json")
    md_out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable or "python3",
        "scripts/recent_activity_digest.py",
        "--days", str(days),
        "--output-md", str(md_out),
        "--output-json", str(json_out),
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"🗂  Activity digest updated: {md_out}")
    except Exception as e:
        print(f"⚠️  Failed to generate activity digest: {e}")


def main():
    """Generate daily summary."""
    print("\n" + "="*70)
    print("📊 Background Agent Daily Summary")
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    print("="*70 + "\n")

    outputs_dir = Path("background_agents/outputs")

    if not outputs_dir.exists():
        print("❌ No outputs found yet\n")
        print("The background agents may not have started yet.")
        print("Run: python background_agents/start_agents.py\n")
        return

    # Document Analysis
    doc_dir = outputs_dir / "document_analysis"
    doc_count = count_files_today(doc_dir, ".json")
    print(f"📄 Documents Processed: {doc_count}")
    if doc_count > 0:
        print(f"   Latest: {get_latest_insight(doc_dir, '.json')}")
    print()

    # Research Insights
    research_dir = outputs_dir / "research"
    summary_count = count_files_today(research_dir / "case_summaries", ".md")
    principles_count = count_files_today(research_dir / "legal_principles", ".md")
    trends_count = count_files_today(research_dir / "trend_analysis", ".md")

    total_research = summary_count + principles_count + trends_count
    print(f"🔍 Research Insights Generated: {total_research}")
    if summary_count > 0:
        print(f"   - Case Summaries: {summary_count}")
    if principles_count > 0:
        print(f"   - Legal Principles: {principles_count}")
    if trends_count > 0:
        print(f"   - Trend Analyses: {trends_count}")
    print()

    # Citation Networks
    network_dir = outputs_dir / "networks"
    network_count = count_files_today(network_dir, ".json")
    print(f"🔗 Citation Networks Built: {network_count}")
    if network_count > 0:
        print(f"   Latest: {get_latest_insight(network_dir, '.json')}")
    print()

    # Pattern Detection
    pattern_dir = outputs_dir / "patterns"
    pattern_count = count_files_today(pattern_dir, ".json")
    print(f"🔍 Pattern Analyses: {pattern_count}")
    if pattern_count > 0:
        print(f"   Latest: {get_latest_insight(pattern_dir, '.json')}")
    print()

    # Settlements
    settlement_dir = outputs_dir / "settlements"
    settlement_count = count_files_today(settlement_dir, ".json")
    print(f"💰 Settlement Optimizations: {settlement_count}")
    if settlement_count > 0:
        print(f"   Latest: {get_latest_insight(settlement_dir, '.json')}")
    print()

    # Summary
    total_outputs = doc_count + total_research + network_count + pattern_count + settlement_count

    print("="*70)
    print(f"\n📈 Total Outputs Today: {total_outputs}")

    if total_outputs == 0:
        print("\n💡 Tip: The agents may have just started. Check back in an hour!")
    elif total_outputs < 10:
        print("\n🟡 Agents are warming up. More insights coming soon!")
    else:
        print("\n✅ Agents are working hard! Great progress today!")

    print("\n📂 View details: python background_agents/view_insights.py")
    print("📊 Check status: python background_agents/status.py")
    print()

    # Auto-generate shared activity digest for all agents (last 7 days)
    try:
        _generate_activity_digest(days=7)
    except Exception:
        pass


if __name__ == "__main__":
    main()
