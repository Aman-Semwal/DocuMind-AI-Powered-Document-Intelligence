"""Insights tab: PDF-attributed evaluation quality, latency, questions and feedback."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from frontend.components.evaluation_export import PERCENT_METRICS, build_evaluation_xlsx
from frontend.services.api_client import ApiClient, ApiError

STAGES = [
    ("embed_ms", "Embed"),
    ("vector_ms", "Vector search"),
    ("bm25_ms", "Keyword search"),
    ("rerank_ms", "Rerank"),
    ("llm_ms", "LLM"),
]
METRIC_GROUPS = [
    (
        "Verified answers from uploaded PDFs",
        [
            ("groundedness", "Groundedness"),
            ("support_ratio", "Verifier support ratio"),
            ("citation_coverage", "Answers with citations"),
            ("unsupported_claims", "Unsupported claims"),
        ],
    ),
]


def latency_frame(stage_avg: dict[str, float]) -> pd.DataFrame:
    rows = [{"Stage": label, "Average (ms)": stage_avg.get(key, 0.0)} for key, label in STAGES]
    return pd.DataFrame(rows).set_index("Stage")


def _metric(value: object, key: str) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "n/a"
    if key in PERCENT_METRICS:
        return f"{float(value) * 100:.0f}%"
    if key in {"unsupported_claims", "questions"}:
        return str(int(value))
    if key.startswith("latency_"):
        return f"{float(value):.0f} ms"
    return f"{float(value):.2f}"


def _render_evaluation(data: dict[str, Any]) -> None:
    evaluation = data.get("evaluation")
    if not evaluation:
        return

    meta = data.get("evaluation_meta") or {}
    st.caption(
        f"Calculated from {evaluation.get('questions', 0)} verified answers using: "
        + ", ".join(meta.get("documents", []))
    )
    for group_title, metrics in METRIC_GROUPS:
        st.markdown(f"###### {group_title}")
        columns = st.columns(4)
        for column, (key, label) in zip(columns, metrics, strict=False):
            column.metric(label, _metric(evaluation.get(key), key))

    rows = data.get("evaluation_rows") or []
    if rows:
        with st.expander("How to read these evaluation scores"):
            st.markdown(
                "- **Groundedness** is the answer verifier's pass rate on questions searched against your uploaded PDFs.\n"
                "- **Verifier support ratio** measures sentence-level support found in the cited passages.\n"
                "- **Answers with citations** means the response included document citations; it is not a citation-accuracy score.\n"
                "- **Unsupported claims** are answer sentences the verifier could not support. No benchmark or generated test questions are included."
            )
        st.markdown("###### Question-level quality")
        table_rows = [
            {
                "Question ID": row.get("id", ""),
                "Question": row.get("question", ""),
                "Grounded": row.get("grounded"),
                "Support ratio": _metric(row.get("support_ratio"), "support_ratio"),
                "Unsupported claims": row.get("unsupported_claims"),
                "Unverified numbers": row.get("unverified_number_count"),
                "Unverified entities": row.get("unverified_entity_count"),
                "Citations": "; ".join(row.get("citations", [])),
                "Confidence": row.get("confidence"),
                "Latency (ms)": row.get("latency_ms"),
            }
            for row in rows
        ]
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")

    workbook = build_evaluation_xlsx(evaluation, meta, rows)
    st.download_button(
        "Download evaluation report (.xlsx)",
        data=workbook,
        file_name="documind-evaluation-report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_evaluation_xlsx",
        help="Excel workbook with a metric summary, run details, and per-question results.",
    )


def _render_benchmark(benchmark: dict[str, Any], title: str = "Evaluation set") -> None:
    summary = benchmark.get("summary") or {}
    meta = benchmark.get("meta") or {}
    rows = benchmark.get("rows") or []
    st.markdown(f"##### {title}")
    st.caption(
        f"{summary.get('questions', len(rows))} fixed questions · "
        f"{meta.get('timestamp', 'Run eval/run_eval.py to refresh')} · "
        f"{meta.get('llm', 'model not recorded')}"
    )
    metrics = [
        ("pass_rate", "Pass rate", "pass_rate"),
        ("groundedness", "Groundedness", "groundedness"),
        ("answer_relevance", "Answer relevance", "answer_relevance"),
        ("context_relevance", "Context relevance", "context_relevance"),
        ("citation_accuracy", "Citation accuracy", "citation_accuracy"),
        ("citation_coverage", "Citation coverage", "citation_coverage"),
        ("hallucination_rate", "Hallucination rate", "hallucination_rate"),
        ("coherence", "Coherence", "coherence"),
        ("recall_at_5", "Recall@5", "recall_at_5"),
        ("recall_at_10", "Recall@10", "recall_at_10"),
        ("mrr", "MRR", "mrr"),
        ("latency_p50_ms", "Latency p50", "latency_p50_ms"),
        ("latency_p95_ms", "Latency p95", "latency_p95_ms"),
    ]
    for start in range(0, len(metrics), 5):
        cols = st.columns(5)
        for col, (key, label, fmt) in zip(cols, metrics[start : start + 5], strict=False):
            col.metric(label, _metric(summary.get(key), fmt))
    if rows:
        st.caption("Relevance values are deterministic token-overlap diagnostics. Coherence is shown only when an LLM judge was run.")
        table = [
            {
                "ID": row.get("id"),
                "Type": row.get("category"),
                "Question": row.get("question"),
                "Pass": row.get("passed"),
                "Grounded": row.get("grounded"),
                "Answer rel.": _metric(row.get("answer_relevance"), "answer_relevance"),
                "Context rel.": _metric(row.get("context_relevance"), "context_relevance"),
                "Citation accurate": row.get("citation_accuracy", row.get("citation_ok")),
                "Hallucination": row.get("hallucination"),
                "Similarity": _metric(row.get("retrieval_similarity"), "retrieval_similarity"),
                "Coherence": _metric(row.get("coherence_score"), "coherence"),
                "Coherence rationale": row.get("coherence_rationale"),
                "Latency (ms)": row.get("latency_ms"),
            }
            for row in rows
        ]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
def render_insights_tab(client: ApiClient, documents: list[dict[str, Any]]) -> None:
    completed = [doc for doc in documents if doc.get("status") == "completed"]
    total_pages = sum(int(doc.get("pages") or 0) for doc in completed)
    total_chunks = sum(int(doc.get("chunks") or 0) for doc in completed)
    st.markdown("### Your document insights")
    overview = st.columns(3)
    overview[0].metric("PDFs in library", len(documents))
    overview[1].metric("Pages indexed", total_pages)
    overview[2].metric("Passages indexed", total_chunks)

    try:
        data: dict[str, Any] = client.metrics()
    except ApiError as exc:
        st.warning(f"Insights unavailable: {exc.message}")
        return
    if not data["recent"] and not data.get("benchmark") and not data.get("uploaded_benchmark"):
        if completed:
            st.info("Your PDFs are indexed. Ask your first question in Chat to generate groundedness, citation, and latency insights.")
        else:
            st.info("Insights will be ready when PDF processing finishes and you ask a question in Chat.")
        return
    if st.button("Refresh", key="refresh_insights"):
        st.rerun()

    if data.get("evaluation"):
        st.markdown("##### Evaluation quality")
        _render_evaluation(data)

    if data.get("benchmark"):
        _render_benchmark(data["benchmark"])

    if data.get("uploaded_benchmark"):
        st.markdown("#### Evaluation benchmark for your two PDFs")
        _render_benchmark(data["uploaded_benchmark"], "Two-PDF benchmark results")

    if data["stage_avg_ms"] or data["latency_ms"].get("p50", 0) > 0:
        st.markdown("##### Runtime latency")
        st.bar_chart(latency_frame(data["stage_avg_ms"]), horizontal=True)
        lat = data["latency_ms"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Median total (p50)", f"{lat.get('p50', 0):.0f} ms")
        c2.metric("p95 total", f"{lat.get('p95', 0):.0f} ms")
        c3.metric("Answered", f"{data.get('answered_rate', 0) * 100:.0f}%")

    st.markdown("##### Feedback and cache")
    fb, cache = data["feedback"], data["cache"]
    f1, f2, f3 = st.columns(3)
    f1.metric("👍 helpful", fb.get("up", 0))
    f2.metric("👎 not helpful", fb.get("down", 0))
    lookups = cache.get("hits", 0) + cache.get("misses", 0)
    f3.metric("Cache hit rate", f"{100 * cache.get('hits', 0) / lookups:.0f}%" if lookups else "n/a")

    st.markdown("##### Recent questions")
    if data["recent"]:
        rows = [
            {
                "Question": r["question"],
                "Answered": "yes" if r["answered"] else "no",
                "Confidence": r.get("confidence") or "-",
                "Total (ms)": round(r["timings"].get("total_ms", 0), 1),
            }
            for r in data["recent"]
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
