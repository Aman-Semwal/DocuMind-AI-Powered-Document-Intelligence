"""About tab: architecture diagram, tech stack and privacy note."""

from pathlib import Path

import streamlit as st

DIAGRAM = Path(__file__).resolve().parents[2] / "docs" / "architecture.png"

STACK = [
    ("API", "FastAPI, Pydantic v2, Uvicorn"),
    ("PDF", "PyMuPDF (page-aware, tables, headings)"),
    ("Retrieval", "Chroma (HNSW, cosine) + BM25, Reciprocal Rank Fusion, cross-encoder reranker, MMR diversity"),
    ("Models", "Ollama: qwen2.5:3b-instruct and nomic-embed-text"),
    ("Storage", "SQLite (WAL) for the registry and embedding cache"),
    ("Quality", "pytest, hypothesis, mypy strict, ruff, bandit, pip-audit, evaluation suite"),
]


def render_about_tab() -> None:
    st.markdown("### How it works")
    if DIAGRAM.exists():
        st.image(str(DIAGRAM), width="stretch")
    else:
        st.caption("See docs/architecture.png in the project.")
    st.markdown("### Tech stack")
    for name, detail in STACK:
        st.markdown(f"- **{name}**: {detail}")
    st.markdown("### Privacy")
    st.success(
        "Everything runs locally on your GPU. Your documents, questions and answers never leave this machine, "
        "and nothing is sent to a cloud service.",
        icon="🔒",
    )
    st.caption("Answers come only from your documents. When the evidence is weak, the app says so instead of guessing.")
