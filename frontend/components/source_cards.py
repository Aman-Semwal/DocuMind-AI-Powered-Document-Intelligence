"""Source cards: document, page, matched snippet and a rendered image of the cited page."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from frontend.services.api_client import ApiClient, ApiError


@st.cache_data(ttl=600, show_spinner=False, max_entries=64)
def fetch_preview(_client: ApiClient, doc_id: str, page: int) -> bytes | None:
    """Page image (cached so reruns do not hit the API again). None if unavailable."""
    try:
        return _client.preview(doc_id, page)
    except ApiError:
        return None


def confidence_badge(confidence: str) -> str:
    label = {
        "high": "High confidence",
        "medium": "Medium confidence",
        "low": "Low confidence",
    }.get(confidence)
    if label is None:
        return ""
    return f'<span class="badge {escape(confidence)}" title="{escape(label)}">{escape(label)}</span>'


def _relevance_bar(score: float) -> str:
    """Render a compact relevance bar from 0-1 as a percentage."""
    pct = min(100, max(0, round(score * 100)))
    color = "#3fb950" if pct >= 65 else ("#e3b341" if pct >= 40 else "#ff7b72")
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px">'
        f'<div style="flex:1;height:4px;border-radius:2px;background:rgba(48,65,100,.4)">'
        f'<div style="width:{pct}%;height:100%;border-radius:2px;background:{color};transition:width .4s ease"></div>'
        f'</div>'
        f'<span style="color:{color};font-size:.7rem;font-weight:600;min-width:32px">{pct}%</span>'
        f'</div>'
    )


def render_sources(client: ApiClient, sources: list[dict[str, Any]], key: str) -> None:
    if not sources:
        return
    n = len(sources)
    st.markdown(
        f'<div class="muted" style="margin-top:.5rem">📎 {n} source{"s" if n != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    for src in sources:
        page = f" · page {src['page']}" if src.get("page") else ""
        title = f"[{src['number']}] {src['document']}{page}".strip()
        if src.get("section"):
            title += f" · {src['section'][:60]}"
        with st.expander(title, expanded=False):
            text_col, img_col = st.columns([3, 2])
            with text_col:
                st.caption("Matched passage")
                st.text(src["snippet"])  # plain text: document content is never rendered as HTML
                st.markdown(_relevance_bar(src["score"]), unsafe_allow_html=True)
            with img_col:
                image = fetch_preview(client, src["document_id"], int(src["page"])) if src.get("page") else None
                if image:
                    st.image(image, caption=f"Page {src['page']}", use_container_width=True)
                else:
                    st.caption("Page preview unavailable.")

