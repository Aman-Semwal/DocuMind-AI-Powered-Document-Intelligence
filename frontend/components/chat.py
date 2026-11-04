"""Chat tab: streamed answers, confidence badge, source cards, timings, feedback, export."""

from __future__ import annotations

from collections.abc import Iterator
from html import escape
from typing import Any

import streamlit as st

from frontend.components.source_cards import confidence_badge, render_sources
from frontend.services.api_client import ApiClient, ApiError
from frontend.services.state import clear_chat, export_markdown

NOT_FOUND_TITLE = "Not found in your documents"
STAGE_LABELS = {
    "cache_ms": "cache",
    "rewrite_ms": "rewrite",
    "embed_ms": "embed",
    "vector_ms": "vector search",
    "bm25_ms": "keyword search",
    "rerank_ms": "rerank",
    "expand_ms": "context expand",
    "first_token_ms": "first token",
    "llm_ms": "LLM generate",
    "verify_ms": "verify",
    "total_ms": "total",
}

# Stages to always show in the inline summary (the quick chips)
_INLINE_STAGES = ("first_token_ms", "llm_ms", "total_ms")


def fmt_ms(ms: float) -> str:
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def timing_chips(timings: dict[str, float], cached: bool) -> str:
    chips = []
    if cached:
        chips.append('<span class="chip info">⚡ cached</span>')
    for key in _INLINE_STAGES:
        if key in timings:
            chips.append(f'<span class="chip">{escape(STAGE_LABELS[key])} {escape(fmt_ms(timings[key]))}</span>')
    return "".join(chips)


def token_stream(events: Iterator[Any], holder: dict[str, Any]) -> Iterator[str]:
    """Adapt the API event stream for st.write_stream and capture the final payload."""
    for ev in events:
        if ev.event == "token":
            yield str(ev.data.get("t", ""))
        elif ev.event == "final":
            holder["result"] = ev.data


def _send_feedback(client: ApiClient, idx: int, rating: int) -> None:
    msg = st.session_state["messages"][idx]
    try:
        client.feedback(msg["result"]["question_id"], rating)
        msg["feedback"] = rating
        st.toast("Thanks for the feedback! 🙏")
    except ApiError as exc:
        st.toast(f"Could not save feedback: {exc.message}")


def render_assistant(client: ApiClient, msg: dict[str, Any], idx: int, show_timings: bool) -> None:
    if msg.get("error"):
        st.error(msg["error"], icon="⚠️")
        if idx == len(st.session_state["messages"]) - 1 and st.button("Retry", key=f"retry_{idx}"):
            st.session_state["messages"].pop()  # drop the failed answer, ask the same question again
            st.session_state["pending"] = msg.get("question")
            st.session_state["streaming"] = True
            st.rerun()
        return
    result = msg["result"]
    if not result["answered"]:
        st.markdown(
            f'<div class="notfound"><b>{NOT_FOUND_TITLE}</b><br>{escape(result["answer"])}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(msg["content"])  # model text: Markdown only, never raw HTML
        badge = confidence_badge(result["confidence"])
        chips = timing_chips(result["timings"], result.get("cached", False))
        if badge or chips:
            st.markdown(badge + chips, unsafe_allow_html=True)
        for w in result.get("warnings", []):
            st.caption(f"⚠ {w}")
        render_sources(client, result["sources"], key=f"src_{idx}")

    if show_timings and result.get("timings"):
        with st.expander("⏱ Timing breakdown", expanded=False):
            rows = {STAGE_LABELS.get(k, k): fmt_ms(v) for k, v in result["timings"].items() if k in STAGE_LABELS}
            st.table(rows)

    # Feedback + copy row
    cols = st.columns([1, 1, 1, 6])
    given = msg.get("feedback")
    feedback_disabled = given is not None
    up_label = "👍" if given != 1 else "👍 ✓"
    down_label = "👎" if given != -1 else "👎 ✓"
    cols[0].button(up_label, key=f"up_{idx}", disabled=feedback_disabled, on_click=_send_feedback, args=(client, idx, 1))
    cols[1].button(down_label, key=f"down_{idx}", disabled=feedback_disabled, on_click=_send_feedback, args=(client, idx, -1))
    with cols[2].popover("📋 Copy"):
        st.code(msg["content"], language=None, wrap_lines=True)


def suggestions(docs: list[dict[str, Any]], selected: list[str]) -> list[str]:
    out: list[str] = []
    for d in docs:
        if d["status"] == "completed" and (not selected or d["document_id"] in selected):
            out.extend(d.get("suggested_questions", [])[:2])
    return out[:6]  # show up to 6 suggestion chips


def _stream_answer(client: ApiClient, question: str) -> None:
    ss = st.session_state
    holder: dict[str, Any] = {}
    with st.chat_message("assistant"):
        try:
            events = client.ask_stream(
                question,
                document_ids=ss["doc_filter"],
                top_k=ss["top_k"],
                session_id=ss["session_id"],
                deep=ss["deep"],
            )
            with st.spinner("🔍 Searching and reasoning…"):
                text = st.write_stream(token_stream(events, holder))
            result = holder.get("result")
            if result is None:
                raise ApiError(502, "incomplete", "The answer ended unexpectedly. Please retry.")
            ss["messages"].append(
                {
                    "role": "assistant",
                    "content": result["answer"] or str(text),
                    "result": result,
                    "error": None,
                    "feedback": None,
                    "question": question,
                }
            )
        except ApiError as exc:
            ss["messages"].append(
                {"role": "assistant", "content": "", "result": {"answered": False, "answer": exc.message, "timings": {}, "sources": [], "confidence": "none", "warnings": [], "cached": False}, "error": exc.message, "question": question}
            )
    ss["pending"] = None
    ss["streaming"] = False
    st.rerun()


def render_chat(client: ApiClient, docs: list[dict[str, Any]]) -> None:
    ss = st.session_state
    ready_docs = [d for d in docs if d["status"] == "completed"]

    top = st.columns([5, 1, 1])
    top[0].markdown("### 💬 Ask your documents")
    if top[1].button("Clear chat", disabled=not ss["messages"] or ss["streaming"], help="Clear chat history"):
        clear_chat()
        st.rerun()
    top[2].download_button(
        "Export",
        data=export_markdown(ss["messages"]),
        file_name="documind-chat.md",
        mime="text/markdown",
        disabled=not ss["messages"],
        help="Download the conversation as a Markdown file",
    )

    for i, msg in enumerate(ss["messages"]):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_assistant(client, msg, i, ss["show_timings"])

    if not ss["messages"] and not ss["pending"]:
        if not ready_docs:
            st.info("⬅ Upload a PDF in the sidebar to get started. Everything stays on your machine.", icon="📄")
        else:
            suggs = suggestions(docs, ss["doc_filter"])
            if suggs:
                st.caption("✨ Try one of these suggestions:")
                # Arrange suggestions in a 2- or 3-column grid for a cleaner look
                cols = st.columns(min(3, len(suggs)))
                for j, q in enumerate(suggs):
                    with cols[j % len(cols)]:
                        if st.button(q, key=f"chip_{j}", use_container_width=True):
                            ss["pending"], ss["streaming"] = q, True
                            st.rerun()

    typed = st.chat_input(
        "Ask a question about your documents…" if ready_docs else "Upload a document first…",
        disabled=ss["streaming"] or not ready_docs,
    )
    if typed:
        ss["pending"], ss["streaming"] = typed, True
        st.rerun()
    if ss["pending"] and ss["streaming"]:
        question = ss["pending"]
        ss["last_question"] = question
        ss["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        _stream_answer(client, question)

