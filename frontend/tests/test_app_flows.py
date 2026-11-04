"""Streamlit AppTest flows against the real backend (fake models)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from eval.sample_docs import build_sample_pdfs
from frontend.services.api_client import ApiClient

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def make_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30)


def upload_ready(api: ApiClient, tmp_path: Path) -> list[dict]:
    pdfs = build_sample_pdfs(tmp_path / "pdfs")
    ids = [api.upload(p.name, p.read_bytes())["document_id"] for p in pdfs.values()]
    deadline = time.time() + 60
    while time.time() < deadline:
        docs = api.list_documents()
        if all(d["status"] == "completed" for d in docs) and len(docs) == len(ids):
            return docs
        time.sleep(0.1)
    raise AssertionError("documents did not finish")


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


def test_empty_state_prompts_upload(ui_client):
    at = make_app().run()
    assert not at.exception
    assert any("Upload a PDF" in i.value for i in at.info)
    assert at.chat_input[0].disabled  # nothing to ask yet
    assert [t.label for t in at.tabs] == ["💬 Chat", "📚 Documents", "ℹ️ About"]
    assert not any(m.label in {"Pass rate", "Recall@5"} for m in at.metric)


def test_ask_question_shows_answer_and_sources(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    assert not at.chat_input[0].disabled
    at.chat_input[0].set_value("How many days of annual leave do employees get?").run()
    assert not at.exception, at.exception
    text = all_text(at)
    assert "24" in text
    assert any("Northwind_Leave_Policy.pdf" in (e.label or "") for e in at.expander)
    assert [m.avatar or m.name for m in at.chat_message] == ["user", "assistant"]


def test_out_of_document_question_shows_not_found(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    at.chat_input[0].set_value("What is the capital of France?").run()
    assert "Not found in your documents" in all_text(at)


def test_clear_chat_empties_conversation(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    at.chat_input[0].set_value("How long is maternity leave?").run()
    assert len(at.chat_message) == 2
    next(b for b in at.button if b.label == "Clear chat").click().run()
    assert len(at.chat_message) == 0


def test_suggested_question_chips_are_offered(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    assert any(b.key and b.key.startswith("chip_") for b in at.button)


def test_document_suggestion_does_not_mutate_widget_state_after_creation(ui_client, tmp_path):
    docs = upload_ready(ui_client, tmp_path)
    at = make_app().run()
    suggestion = next(b for b in at.button if b.key == f"sq_{docs[0]['document_id']}_0")

    at = suggestion.click().run()

    assert not at.exception, at.exception
    assert at.session_state["doc_filter"] == [docs[0]["document_id"]]


def test_documents_and_insights_tabs_render(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    at.chat_input[0].set_value("What is the hotel limit in London?").run()
    assert not at.exception
    text = all_text(at)
    assert "Northwind_Leave_Policy.pdf" in text and "pages" in text
    assert len(at.metric) >= 3  # insights metrics


def test_uploaded_pdfs_show_insights_before_first_question(ui_client, tmp_path, monkeypatch):
    from app.config import Settings

    results_dir = tmp_path / "empty-eval"
    results_dir.mkdir()
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: results_dir))
    docs = upload_ready(ui_client, tmp_path)
    at = make_app().run()

    assert not at.exception
    assert "Your document insights" in all_text(at)
    assert "Ask your first question in Chat" in all_text(at)
    assert len(at.metric) >= 3
    assert any(metric.label == "PDFs in library" and metric.value == str(len(docs)) for metric in at.metric)


def test_uploaded_pdf_benchmark_has_separate_coherence_section(ui_client, tmp_path, monkeypatch):
    upload_ready(ui_client, tmp_path)
    benchmark = {
        "summary": {"questions": 20, "coherence": 0.5},
        "meta": {"timestamp": "2026-11-03", "llm": "qwen2.5:3b-instruct"},
        "rows": [{"id": "L2", "question": "How much leave?", "coherence_score": 0.5, "coherence_rationale": "Clear."}],
    }
    monkeypatch.setattr(
        ui_client,
        "metrics",
        lambda: {
            "recent": [],
            "benchmark": None,
            "uploaded_benchmark": benchmark,
            "evaluation": None,
            "stage_avg_ms": {},
            "latency_ms": {"p50": 0, "p95": 0},
            "answered_rate": 0,
            "feedback": {"up": 0, "down": 0},
            "cache": {"hits": 0, "misses": 0},
        },
    )

    at = make_app().run()

    assert not at.exception, at.exception
    assert "Evaluation benchmark for your two PDFs" in all_text(at)
    rendered_metrics = [(metric.label, metric.value) for metric in at.metric]
    assert any(label == "Coherence" for label, _ in rendered_metrics), rendered_metrics


def test_feedback_button_is_stored(ui_client, tmp_path):
    upload_ready(ui_client, tmp_path)
    at = make_app().run()
    at.chat_input[0].set_value("How many sick days per year?").run()
    next(b for b in at.button if b.key == "up_1").click().run()
    assert ui_client.metrics()["feedback"]["up"] == 1


def test_offline_banner_and_retry(monkeypatch, ui_client):
    import frontend.services.state as state

    dead = ApiClient("http://127.0.0.1:9", "k", retries=0)
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    monkeypatch.setattr(state, "build_client", lambda: dead)
    at = make_app().run()
    assert not at.exception
    assert any("offline" in e.value.lower() for e in at.error)
    assert any(b.label == "Retry connection" for b in at.button)


def test_api_error_during_answer_shows_message_not_traceback(ui_client, tmp_path, monkeypatch):
    upload_ready(ui_client, tmp_path)
    from app.core.exceptions import OllamaUnavailable

    def boom(*a, **k):
        raise OllamaUnavailable()

    ui_client._client.app.state.container.llm.stream = boom  # type: ignore[attr-defined]
    ui_client._client.app.state.container.llm.chat = boom  # type: ignore[attr-defined]
    at = make_app().run()
    at.chat_input[0].set_value("How long is maternity leave?").run()
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "Ollama" in errors and "Traceback" not in errors
    assert any(b.label == "Retry" for b in at.button)


@pytest.mark.browser
def test_playwright_flow_placeholder():
    pytest.skip("Run frontend/tests/test_e2e_playwright.py against a live UI (see README).")
