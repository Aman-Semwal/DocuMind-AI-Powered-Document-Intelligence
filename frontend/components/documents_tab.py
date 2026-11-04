"""Documents tab: one card per document with summary and suggested questions."""

from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

import streamlit as st

from frontend.components.sidebar import chip


def render_documents_tab(docs: list[dict[str, Any]]) -> None:
    if not docs:
        st.info("No documents yet. Upload PDFs in the sidebar.")
        return
    for d in docs:
        with st.container(border=True):
            st.markdown(f"#### {escape(d['filename'])}", unsafe_allow_html=True)
            uploaded = datetime.fromtimestamp(d["created_at"]).strftime("%Y-%m-%d %H:%M")
            st.markdown(
                chip(d)
                + f'<span class="chip">{d["pages"]} pages</span>'
                + f'<span class="chip">{d["chunks"]} chunks</span>'
                + f'<span class="chip">{escape(uploaded)}</span>',
                unsafe_allow_html=True,
            )
            if d.get("summary"):
                st.write(d["summary"])
            elif d["status"] == "completed":
                st.caption("Summary is being generated…")
            for w in d.get("warnings", []):
                st.caption(f"⚠ {w}")
            if d["status"] == "failed":
                st.error(d.get("error") or "Processing failed.")
            if d["status"] == "completed" and d.get("suggested_questions"):
                st.caption("Suggested questions")
                for j, q in enumerate(d["suggested_questions"]):
                    if st.button(q, key=f"sq_{d['document_id']}_{j}"):
                        st.session_state["pending"], st.session_state["streaming"] = q, True
                        st.session_state["doc_filter_pending"] = [d["document_id"]]
                        st.toast("Question sent. Open the Chat tab to see the answer.")
                        st.rerun()
