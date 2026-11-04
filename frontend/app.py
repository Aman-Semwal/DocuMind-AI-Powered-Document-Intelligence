"""Streamlit entry point:  streamlit run frontend/app.py

The UI talks to the backend only through its REST API (see frontend/services/api_client.py).
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `streamlit run frontend/app.py` from any directory
    sys.path.insert(0, str(ROOT))

from frontend.components.about_tab import render_about_tab  # noqa: E402
from frontend.components.chat import render_chat  # noqa: E402
from frontend.components.documents_tab import render_documents_tab  # noqa: E402
from frontend.components.insights_tab import render_insights_tab  # noqa: E402
from frontend.components.sidebar import render_sidebar  # noqa: E402
from frontend.components.styles import inject_css  # noqa: E402
from frontend.services.state import get_client, init_state  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="DocuMind — Document Q&A", page_icon="📄", layout="wide")
    inject_css()
    init_state()
    client = get_client()
    docs = render_sidebar(client)
    st.markdown(
        '<section class="workspace-hero">'
        '<div class="workspace-kicker"><span class="kicker-dot"></span> YOUR PRIVATE KNOWLEDGE SPACE</div>'
        "<h1>Answers, grounded in your documents.</h1>"
        "<p>Ask naturally. Every answer comes with a source you can verify.</p>"
        '<span class="hero-private">🔒&nbsp; Local &amp; private</span>'
        "</section>",
        unsafe_allow_html=True,
    )
    has_uploaded_pdfs = any(doc.get("status") != "deleted" for doc in docs)
    tab_labels = ["💬 Chat", "📚 Documents"]
    if has_uploaded_pdfs:
        tab_labels.append("📊 Insights")
    tab_labels.append("ℹ️ About")
    tabs = iter(st.tabs(tab_labels))
    chat_tab, docs_tab = next(tabs), next(tabs)
    with chat_tab:
        render_chat(client, docs)
    with docs_tab:
        render_documents_tab(docs)
    if has_uploaded_pdfs:
        with next(tabs):
            render_insights_tab(client, docs)
    about_tab = next(tabs)
    with about_tab:
        render_about_tab()


main()
