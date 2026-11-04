"""Build a formatted Excel workbook from PDF-attributed runtime evaluations."""

from __future__ import annotations

from io import BytesIO
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

METRIC_LABELS = {
    "questions": "Verified answers",
    "groundedness": "Groundedness",
    "support_ratio": "Verifier support ratio",
    "citation_coverage": "Answers with citations",
    "unsupported_claims": "Unsupported claims",
}
PERCENT_METRICS = {
    "groundedness",
    "support_ratio",
    "citation_coverage",
    "pass_rate",
    "answer_relevance",
    "context_relevance",
    "citation_accuracy",
    "hallucination_rate",
    "coherence",
    "recall_at_5",
    "recall_at_10",
}
HEADER_FILL = PatternFill("solid", fgColor="176B4D")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def _format_sheet(sheet: Any) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 30
    for column in sheet.columns:
        letter = get_column_letter(column[0].column)
        values = [str(cell.value or "") for cell in column]
        width = min(max(max((len(value) for value in values), default=0) + 2, 12), 58)
        sheet.column_dimensions[letter].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def _excel_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value[:32767]
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def build_evaluation_xlsx(
    summary: dict[str, Any],
    meta: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> bytes:
    workbook = Workbook()
    overview = workbook.active
    overview.title = "Evaluation Summary"
    overview.append(["Metric", "Value", ""])
    for key, label in METRIC_LABELS.items():
        if key not in summary:
            continue
        value = summary[key]
        display_value = value
        if key in PERCENT_METRICS and isinstance(value, (int, float)):
            display_value = float(value)
        overview.append([label, display_value, ""])
        result_cell = overview.cell(overview.max_row, 2)
        if key in PERCENT_METRICS and isinstance(value, (int, float)):
            result_cell.number_format = "0.0%"

    metric_end = overview.max_row
    overview.append([])
    overview.append(["PDF evaluation basis", "", ""])
    overview.cell(overview.max_row, 1).fill = HEADER_FILL
    overview.cell(overview.max_row, 1).font = HEADER_FONT
    overview.cell(overview.max_row, 2).fill = HEADER_FILL
    overview.cell(overview.max_row, 2).font = HEADER_FONT
    for key, value in (meta or {}).items():
        overview.append([_excel_text(key.replace("_", " ").title()), _excel_text(str(value)), ""])
    _format_sheet(overview)
    overview.auto_filter.ref = f"A1:C{metric_end}"
    for cell in overview[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
    for row_index in range(2, overview.max_row + 1):
        if overview.cell(row_index, 1).value == "PDF evaluation basis":
            for cell in overview[row_index][:2]:
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT

    questions = workbook.create_sheet("Per-question Results")
    question_headers = [
        "Question ID",
        "Question",
        "Grounded",
        "Verifier support ratio",
        "Unsupported claims",
        "Unverified number count",
        "Unverified entity count",
        "Citations",
        "Confidence",
        "Latency (ms)",
    ]
    questions.append(question_headers)
    for row in rows:
        questions.append(
            [
                _excel_text(row.get("id")),
                _excel_text(row.get("question")),
                row.get("grounded"),
                row.get("support_ratio"),
                row.get("unsupported_claims"),
                row.get("unverified_number_count"),
                row.get("unverified_entity_count"),
                _excel_text("; ".join(row.get("citations", []))),
                _excel_text(row.get("confidence")),
                row.get("latency_ms"),
            ]
        )
        questions.cell(questions.max_row, 4).number_format = "0.0%"
        questions.cell(questions.max_row, 10).number_format = "0.0"
    _format_sheet(questions)

    output = BytesIO()
    workbook.save(output)
    return output.getvalue()
