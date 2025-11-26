"""Generate a PDF rendition of the Hong Kong case analysis JSON report."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

from fpdf import FPDF


def write_wrapped(pdf: FPDF, text: str, size: int = 12, style: str = "", wrap: int = 100) -> None:
    """Write wrapped text to the PDF."""
    pdf.set_font("Arial", style, size)
    for line in text.splitlines():
        if not line.strip():
            pdf.ln(6)
            continue
        for chunk in textwrap.wrap(line, width=wrap):
            pdf.cell(0, 6, chunk, ln=True)


def add_heading(pdf: FPDF, title: str) -> None:
    """Add a section heading."""
    pdf.set_font("Arial", "B", 14)
    pdf.ln(4)
    pdf.cell(0, 8, title, ln=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "ml_system" / "models" / "hk_case_analysis_results.json"
    output_path = base_dir / "reports" / "analysis_outputs" / "hk_case_analysis_results.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Hong Kong Case Analysis", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Model: {data.get('model_name', 'sentence-transformers/all-mpnet-base-v2')}", ln=True)

    add_heading(pdf, "Case Summary")
    write_wrapped(pdf, data.get("case_summary", ""))

    add_heading(pdf, "Legal Issues")
    for idx, issue in enumerate(data.get("legal_issues", []), start=1):
        write_wrapped(pdf, f"{idx}. {issue}")

    add_heading(pdf, "Jurisdiction Analysis")
    write_wrapped(pdf, data.get("jurisdiction_analysis", ""))

    add_heading(pdf, "Similar US Cases")
    for case, score in data.get("similar_us_cases", []):
        write_wrapped(pdf, f"- {case} (Similarity: {score:.2f})")

    defamation = data.get("defamation_analysis", {})
    add_heading(pdf, "Defamation Analysis")
    write_wrapped(
        pdf,
        "\n".join(
            [
                f"Case Type: {defamation.get('case_type', '')}",
                f"Jurisdiction: {defamation.get('jurisdiction', '')}",
                f"Legal Basis: {defamation.get('legal_basis', '')}",
            ]
        ),
    )

    cross_border = data.get("cross_border_analysis", {})
    add_heading(pdf, "Cross-Border Analysis")
    if isinstance(cross_border, dict):
        write_wrapped(pdf, json.dumps(cross_border, indent=2))

    recommendations = data.get("recommendations", [])
    add_heading(pdf, "Recommendations")
    for idx, rec in enumerate(recommendations, start=1):
        write_wrapped(pdf, f"{idx}. {rec}")

    pdf.output(output_path.as_posix())
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    main()
