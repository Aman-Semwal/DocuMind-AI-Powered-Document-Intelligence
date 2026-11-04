"""Sidebar: brand, backend status lights, document library, filter and settings."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from frontend.services.api_client import ApiClient, ApiError, ApiOffline
from frontend.services.state import MAX_UPLOAD_MB, stage_label


@st.cache_data(ttl=3, show_spinner=False)
def fetch_documents(_client: ApiClient) -> list[dict[str, Any]]:
    return _client.list_documents()


@st.cache_data(ttl=5, show_spinner=False)
def fetch_ready(_client: ApiClient) -> dict[str, Any]:
    return _client.ready()


def status_line(label: str, state: str) -> str:
    cls = {"up": "up", "ok": "up", "down": "down"}.get(state, "na")
    return f'<div class="status-line"><span class="dot {cls}"></span>{escape(label)}</div>'


def render_status(client: ApiClient) -> dict[str, Any] | None:
    """Show API / Ollama lights. Returns the /ready payload, or None when the API is offline."""
    try:
        ready = fetch_ready(client)
    except ApiError:
        st.markdown(status_line("API offline", "down"), unsafe_allow_html=True)
        return None
    st.markdown(status_line("API online", "up"), unsafe_allow_html=True)
    fake = str(ready.get("llm_model", "")).startswith("fake")
    ollama = "n/a" if fake else ("up" if ready.get("llm") else "down")
    label = {"up": "Ollama running", "down": "Ollama not reachable", "n/a": "Offline demo model"}[ollama]
    st.markdown(status_line(label, ollama), unsafe_allow_html=True)
    st.caption(f"Model: {ready.get('llm_model', '?')}")
    return ready


def chip(doc: dict[str, Any]) -> str:
    cls = {"completed": "ok", "failed": "bad"}.get(doc["status"], "warn")
    return f'<span class="chip {cls}">{escape(stage_label(doc))}</span>'


def handle_uploads(client: ApiClient) -> None:
    ss = st.session_state
    files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{ss['uploader_nonce']}",
        help=f"PDF only, up to {MAX_UPLOAD_MB} MB. Duplicates and scanned PDFs are rejected with a clear message.",
    )
    if not files:
        return
    log: list[tuple[str, str]] = []
    bar = st.progress(0.0, text="Uploading…")
    for i, f in enumerate(files):
        data = f.getvalue()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            log.append(("error", f"{f.name}: larger than {MAX_UPLOAD_MB} MB."))
        else:
            try:
                doc = client.upload(f.name, data)
                log.append(("ok", f"{doc['filename']} uploaded as {doc['document_id']}."))
            except ApiError as exc:
                log.append(("error", f"{f.name}: {exc.message}"))
        bar.progress((i + 1) / len(files), text=f"Uploaded {i + 1} of {len(files)}")
    bar.empty()
    ss["upload_log"] = log
    ss["uploader_nonce"] += 1  # clears the uploader widget
    fetch_documents.clear()
    st.rerun()


def render_upload_log() -> None:
    for kind, text in st.session_state["upload_log"]:
        (st.success if kind == "ok" else st.error)(text, icon="✅" if kind == "ok" else "⚠️")


def _document_list(client: ApiClient) -> None:
    try:
        docs = fetch_documents(client)
    except ApiError:
        st.caption("Cannot load documents.")
        return
    if not docs:
        st.caption("No documents yet. Upload a PDF to get started.")
    for d in docs:
        with st.container(border=True):
            filename = escape(d['filename'])
            st.markdown(f"**{filename}**", unsafe_allow_html=True)
            chunks = d.get('chunks', 0)
            chunk_info = f" · {chunks:,} chunks" if chunks and d["status"] == "completed" else ""
            short_id = d['document_id'][:12] + "…"
            st.markdown(
                chip(d) + f'<span class="muted" title="{escape(d["document_id"])}">{escape(short_id)}{escape(chunk_info)}</span>',
                unsafe_allow_html=True
            )
            if d["status"] in ("pending", "processing"):
                pr = d["progress"]
                eta_s = int(pr.get("eta_seconds") or 0)
                eta = f" · ~{eta_s}s left" if eta_s > 0 else ""
                pct = min(1.0, pr["percent"] / 100)
                label = f"{pr['pages_processed']}/{pr['total_pages']} pages{eta}"
                st.progress(pct, text=label)
            if d["status"] == "failed":
                st.caption(d.get("error") or "Processing failed.")
                if d.get("resumable") and st.button("Resume", key=f"resume_{d['document_id']}"):
                    _safe(lambda doc_id=d["document_id"]: client.resume_document(doc_id))  # type: ignore[misc]
            with st.popover("Delete", use_container_width=False):
                st.caption("This removes the file, its vectors and its record. This cannot be undone.")
                confirmed = st.button("⚠ Confirm delete", key=f"del_{d['document_id']}", type="primary")
                if confirmed and _safe(lambda doc_id=d["document_id"]: client.delete_document(doc_id)):  # type: ignore[misc]
                    st.session_state["doc_filter_pending"] = [
                        x for x in st.session_state["doc_filter"] if x != d["document_id"]
                    ]
                    fetch_documents.clear()
                    st.rerun()


def _safe(action: Any) -> bool:
    try:
        action()
        return True
    except ApiError as exc:
        st.toast(exc.message)
        return False


def render_sidebar(client: ApiClient) -> list[dict[str, Any]]:
    """Draw the sidebar and return the current document list."""
    ss = st.session_state
    with st.sidebar:
        st.markdown(
            '<div class="brand"><span class="brand-logo">📄</span><span class="brand-title">DocuMind</span></div>'
            '<div class="brand-tag">Private answers from your PDFs, with citations.</div>',
            unsafe_allow_html=True,
        )
        ready = render_status(client)
        if ready is None:
            st.error("The backend API is offline.")
            if st.button("Retry connection"):
                fetch_ready.clear()
                fetch_documents.clear()
                st.rerun()
            return []

        st.divider()
        st.markdown("##### Documents")
        handle_uploads(client)
        render_upload_log()
        try:
            docs = fetch_documents(client)
        except ApiOffline:
            docs = []
        busy = any(d["status"] in ("pending", "processing") for d in docs)
        fragment = st.fragment(run_every="2s" if busy else None)(_document_list)
        fragment(client)

        ready_docs = {d["document_id"]: d["filename"] for d in docs if d["status"] == "completed"}
        pending_filter = ss.pop("doc_filter_pending", None)
        if pending_filter is not None:
            ss["doc_filter"] = pending_filter
        ss["doc_filter"] = [x for x in ss["doc_filter"] if x in ready_docs]
        st.multiselect(
            "Ask only these documents (empty = all)",
            options=list(ready_docs),
            key="doc_filter",
            format_func=lambda i: f"{i} · {ready_docs[i]}",
        )
        st.divider()
        st.markdown("##### ⚙ Settings")
        st.slider(
            "Passages used (top-k)", 1, 10, key="top_k",
            help="Number of document passages used as context. Higher = more context but slower."
        )
        st.toggle("Show timing breakdown", key="show_timings", help="Display per-stage latency after each answer.")
        st.toggle(
            "Deep mode (slower, larger model)", key="deep",
            help="Routes questions to the 7B model for harder reasoning tasks. Slower but more capable."
        )
    return docs
