"""Session-state helpers and configuration. Only IDs and small dictionaries live in session_state."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from frontend.services.api_client import ApiClient

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

DEFAULTS: dict[str, Any] = {
    "messages": [],
    "streaming": False,
    "pending": None,
    "doc_filter_pending": None,
    "doc_filter": [],
    "top_k": 5,
    "show_timings": True,
    "deep": False,
    "uploader_nonce": 0,
    "upload_log": [],
    "last_question": None,
}


def new_session_id() -> str:
    return uuid.uuid4().hex[:16]


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = new_session_id()


def clear_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["pending"] = None
    st.session_state["streaming"] = False
    st.session_state["session_id"] = new_session_id()  # backend conversation memory starts fresh


def _secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file configured
        return None
    return str(value) if value else None


def read_config() -> tuple[str, str]:
    """API base URL and key. The key is read from env or st.secrets - never from code or the URL."""
    base = os.getenv("API_BASE_URL") or _secret("API_BASE_URL") or "http://localhost:8000"
    key = os.getenv("API_KEY") or _secret("API_KEY") or ""
    return base, key


def build_client() -> ApiClient:
    base, key = read_config()
    return ApiClient(base, key)


@st.cache_resource(show_spinner=False)
def get_client() -> ApiClient:
    """One long-lived client (connection pooling) shared by every rerun."""
    return build_client()


def export_markdown(messages: list[dict[str, Any]]) -> str:
    """Render the conversation as Markdown, including the cited sources."""
    lines = [f"# Document Q&A export ({datetime.now():%Y-%m-%d %H:%M})", ""]
    for msg in messages:
        if msg["role"] == "user":
            lines += [f"**Q:** {msg['content']}", ""]
            continue
        lines += [f"**A:** {msg.get('content') or msg.get('error') or ''}", ""]
        result = msg.get("result") or {}
        for s in result.get("sources", []):
            page = f", page {s['page']}" if s.get("page") else ""
            lines.append(f"- {s['document']}{page}" + (f" ({s['section']})" if s.get("section") else ""))
        if result.get("sources"):
            lines.append("")
    return "\n".join(lines)


def stage_label(doc: dict[str, Any]) -> str:
    """Friendly status for the upload/progress panel."""
    status, stage = doc["status"], doc.get("stage", "")
    if status == "failed":
        return "Interrupted" if doc.get("resumable") else "Failed"
    if status == "completed":
        return "Ready"
    return {
        "queued": "Uploaded",
        "extracting": "Extracting text",
        "embedding": "Embedding",
        "summarizing": "Summarising",
    }.get(stage, "Processing")
