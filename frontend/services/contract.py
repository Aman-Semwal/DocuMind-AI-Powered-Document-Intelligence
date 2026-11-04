"""The response fields and endpoints the UI relies on.

`frontend/tests/test_contract.py` checks them against the backend's OpenAPI schema, so a
backend change that would break the UI fails the build instead of the demo.
"""

REQUIRED_SCHEMA_FIELDS: dict[str, set[str]] = {
    "AskResponse": {
        "question_id",
        "answer",
        "answered",
        "confidence",
        "citations",
        "sources",
        "timings",
        "cached",
        "model",
    },
    "SourceOut": {"number", "document_id", "document", "page", "section", "snippet", "score"},
    "VerificationOut": {"ok", "support_ratio", "unsupported_sentences", "unverified_numbers", "unverified_entities"},
    "DocumentOut": {
        "document_id",
        "filename",
        "status",
        "stage",
        "pages",
        "chunks",
        "created_at",
        "error",
        "resumable",
        "warnings",
        "progress",
        "summary",
        "suggested_questions",
    },
    "Progress": {"pages_processed", "total_pages", "percent", "eta_seconds"},
    "ReadyResponse": {"ready", "llm", "embedder", "llm_model", "embed_model", "documents", "chunks"},
    "MetricsResponse": {
        "recent",
        "latency_ms",
        "stage_avg_ms",
        "answered_rate",
        "feedback",
        "cache",
        "evaluation",
        "evaluation_meta",
        "evaluation_rows",
    },
    "FeedbackRequest": {"question_id", "rating"},
    "AskRequest": {"question", "document_ids", "top_k", "session_id", "deep_mode"},
}

REQUIRED_ENDPOINTS: set[tuple[str, str]] = {
    ("get", "/ready"),
    ("get", "/documents"),
    ("post", "/documents/upload"),
    ("delete", "/documents/{doc_id}"),
    ("post", "/documents/{doc_id}/resume"),
    ("get", "/documents/{doc_id}/pages/{page}/preview"),
    ("post", "/ask"),
    ("post", "/ask/stream"),
    ("post", "/feedback"),
    ("get", "/metrics/recent"),
}
