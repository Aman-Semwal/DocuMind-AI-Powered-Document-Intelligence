from __future__ import annotations

from io import BytesIO

from openpyxl import load_workbook

from frontend.components.evaluation_export import build_evaluation_xlsx


def test_evaluation_export_contains_pdf_metrics_and_question_results():
    workbook_data = build_evaluation_xlsx(
        {
            "questions": 1,
            "groundedness": 1.0,
            "support_ratio": 0.9,
            "citation_coverage": 1.0,
            "unsupported_claims": 0,
        },
        {"basis": "Uploaded PDFs", "documents": ["Policy.pdf"]},
        [
            {
                "id": "Q1",
                "question": "Example question?",
                "grounded": True,
                "support_ratio": 0.9,
                "unsupported_claims": 0,
                "unverified_number_count": 0,
                "unverified_entity_count": 0,
                "citations": ["Policy.pdf p.1"],
                "confidence": "high",
                "latency_ms": 12.5,
            }
        ],
    )

    workbook = load_workbook(BytesIO(workbook_data), data_only=True)

    assert workbook.sheetnames == ["Evaluation Summary", "Per-question Results"]
    summary = workbook["Evaluation Summary"]
    assert summary["A1"].value == "Metric"
    assert summary["A4"].value == "Verifier support ratio"
    assert summary["B4"].value == 0.9
    assert summary["A9"].value == "Basis"
    assert summary["B9"].value == "Uploaded PDFs"

    questions = workbook["Per-question Results"]
    assert questions["A2"].value == "Q1"
    assert questions["B2"].value == "Example question?"
    assert questions["D2"].value == 0.9
    assert questions["H2"].value == "Policy.pdf p.1"
    assert questions.max_column == 10


def test_evaluation_export_neutralizes_formula_like_question_text():
    workbook_data = build_evaluation_xlsx(
        {"questions": 1},
        None,
        [{"id": "Q1", "question": '=HYPERLINK("https://example.test")'}],
    )

    workbook = load_workbook(BytesIO(workbook_data), data_only=False)

    assert workbook["Per-question Results"]["B2"].value.startswith("'=")
    assert workbook["Per-question Results"]["B2"].data_type == "s"
    summary = workbook["Evaluation Summary"]
    assert "Coherence" not in str([cell.value for row in summary.iter_rows() for cell in row])
    assert "Hallucination" not in str([cell.value for row in summary.iter_rows() for cell in row])
