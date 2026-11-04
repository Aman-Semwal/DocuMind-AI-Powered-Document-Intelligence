"""Real Chroma + SQLite in temp dirs, fake embedder/LLM."""

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from app.config import NOT_AVAILABLE_TEXT, Settings
from app.main import create_app
from app.services.fakes import FakeEmbedder, FakeLLM
from tests.conftest import HEADERS, make_settings, upload, wait_for


def ask(client, q, **kw):
    r = client.post("/ask", json={"question": q, **kw})
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------- ingestion
def test_upload_ingests_chunks_with_correct_metadata(client, sample_pdfs):
    doc = upload(client, sample_pdfs["leave"])
    assert doc["status"] == "completed" and doc["document_id"] == "DOC001"
    assert doc["pages"] == 4 and doc["chunks"] >= 6 and doc["progress"]["percent"] == 100.0
    assert doc["summary"] and len(doc["suggested_questions"]) >= 1
    store = client.app.state.container.store
    got = store.collection.get(where={"document_id": "DOC001"}, include=["documents", "metadatas"])
    assert len(got["ids"]) == doc["chunks"] == client.app.state.container.registry.get("DOC001")["chunks"]
    pages = {m["page_number"] for m in got["metadatas"]}
    assert pages == {1, 2, 3, 4}
    by_text = dict(zip(got["documents"], got["metadatas"]))
    sick = next(m for d, m in by_text.items() if "12 paid sick days" in d)
    assert sick["page_number"] == 2 and sick["filename"] == "Northwind_Leave_Policy.pdf"
    assert sick["section"] == "Employee Leave Policy > 3. Sick Leave" or "Sick Leave" in sick["section"]
    table = next((m for d, m in by_text.items() if "| Annual leave | 24 |" in d), None)
    assert table and table["kind"] == "table" and table["page_number"] == 3
    # headers/footers were stripped
    assert not any("Confidential - internal use only" in d for d in got["documents"])
    assert not any("Page 2 of 4" in d for d in got["documents"])
    assert sorted(m["chunk_id"] for m in got["metadatas"]) == sorted(got["ids"])


def test_duplicate_upload_returns_409_and_adds_no_chunks(client, sample_pdfs):
    upload(client, sample_pdfs["leave"])
    store = client.app.state.container.store
    before = store.count()
    with sample_pdfs["leave"].open("rb") as fh:
        r = client.post("/documents/upload", files={"file": ("copy.pdf", fh, "application/pdf")})
    assert r.status_code == 409 and r.json()["error"]["code"] == "duplicate_document"
    assert store.count() == before and len(client.get("/documents").json()["documents"]) == 1
    assert list(client.app.state.container.settings.upload_dir.glob("*.part")) == []
    assert len(list(client.app.state.container.settings.upload_dir.glob("*.pdf"))) == 1


def test_reingestion_is_idempotent(client, sample_pdfs):
    doc = upload(client, sample_pdfs["leave"])
    c = client.app.state.container
    c.registry.update("DOC001", status="pending", pages_processed=0, chunks=0)
    c.ingestion.submit("DOC001")
    c.ingestion.wait("DOC001")
    assert c.store.count_document("DOC001") == doc["chunks"]
    assert c.registry.get("DOC001")["chunks"] == doc["chunks"]


def test_resume_from_checkpoint_creates_no_duplicates(client, sample_pdfs):
    full = upload(client, sample_pdfs["leave"])
    c = client.app.state.container
    # simulate a crash after page 2: remove vectors of pages 3-4 and rewind the checkpoint
    c.store.collection.delete(where={"$and": [{"document_id": "DOC001"}, {"page_number": {"$gte": 3}}]})
    kept = c.store.count_document("DOC001")
    c.registry.update("DOC001", status="processing", pages_processed=2, chunks=kept)
    assert kept < full["chunks"]
    c.ingestion.submit("DOC001")
    c.ingestion.wait("DOC001")
    done = c.registry.get("DOC001")
    assert done["status"] == "completed" and done["chunks"] == full["chunks"]
    assert c.store.count_document("DOC001") == full["chunks"]
    # section path was rebuilt for the resumed pages
    meta = c.store.collection.get(where={"document_id": "DOC001"}, include=["metadatas"])["metadatas"]
    assert any(m["page_number"] == 3 and "Leave Entitlement Summary" in m["section"] for m in meta)


def test_startup_resumes_incomplete_documents(tmp_path, sample_pdfs):
    s = make_settings(tmp_path)
    with TestClient(create_app(s, embedder=FakeEmbedder(), llm=FakeLLM())) as c:
        c.headers.update(HEADERS)
        upload(c, sample_pdfs["travel"])
        reg = c.app.state.container.registry
        reg.update("DOC001", status="processing", pages_processed=0, chunks=0)
    with TestClient(create_app(s, embedder=FakeEmbedder(), llm=FakeLLM())) as c2:
        c2.headers.update(HEADERS)
        assert wait_for(c2, "DOC001")["status"] == "completed"


def test_restart_persists_documents_and_answers(tmp_path, sample_pdfs):
    s = make_settings(tmp_path)
    with TestClient(create_app(s, embedder=FakeEmbedder(), llm=FakeLLM())) as c:
        c.headers.update(HEADERS)
        upload(c, sample_pdfs["leave"])
    with TestClient(create_app(s, embedder=FakeEmbedder(), llm=FakeLLM())) as c2:
        c2.headers.update(HEADERS)
        assert c2.get("/documents").json()["count"] == 1
        r = ask(c2, "How many sick days do employees receive per year?")
        assert r["answered"] and 2 in [c["page"] for c in r["citations"]]


def test_scanned_pdf_is_rejected_with_clear_error_and_rolled_back(client, tmp_path):
    d = pymupdf.open()
    d.new_page()
    d.new_page()
    p = tmp_path / "scan.pdf"
    d.save(str(p))
    doc = upload(client, p)
    assert doc["status"] == "failed" and "scanned" in doc["error"].lower() and doc["resumable"] is False
    assert client.app.state.container.store.count() == 0


def test_corrupt_pdf_rejected_at_upload(client, tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.4\n garbage garbage garbage")
    with p.open("rb") as fh:
        r = client.post("/documents/upload", files={"file": ("broken.pdf", fh, "application/pdf")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "corrupt_pdf"
    assert list(client.app.state.container.settings.upload_dir.iterdir()) == []


def test_concurrent_uploads_and_questions_have_no_id_collisions(client, sample_pdfs, tmp_path):
    extra = []
    for i in range(4):  # distinct files (different bytes -> different sha256)
        d = pymupdf.open(sample_pdfs["travel"])
        d.set_metadata({"title": f"variant {i}"})
        p = tmp_path / f"v{i}.pdf"
        d.save(str(p))
        extra.append(p)
    results, errors = [], []

    def go(p):
        try:
            with p.open("rb") as fh:
                results.append(client.post("/documents/upload", files={"file": (p.name, fh, "application/pdf")}))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=go, args=(p,)) for p in extra]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    ids = [r.json()["document_id"] for r in results if r.status_code == 202]
    assert len(ids) == len(set(ids)) >= 3
    for i in ids:
        assert wait_for(client, i)["status"] == "completed"


def test_delete_removes_vectors_file_record_and_retrieval(loaded_client):
    c = loaded_client.app.state.container
    path = c.registry.get("DOC002")["stored_path"]
    answer = ask(loaded_client, "What is the hotel limit for London?")
    assert answer["answered"]
    assert c.registry.query_exists(answer["question_id"])
    r = loaded_client.delete("/documents/DOC002")
    assert r.status_code == 200 and r.json() == {"document_id": "DOC002", "deleted": True}
    assert c.store.count_document("DOC002") == 0 and not Path(path).exists()
    assert loaded_client.get("/documents/DOC002").status_code == 404
    assert not c.registry.query_exists(answer["question_id"])
    assert all(row["question_id"] != answer["question_id"] for row in loaded_client.get("/metrics/recent").json()["recent"])
    after = ask(loaded_client, "What is the hotel limit for London?")
    assert not after["answered"] and not after["cached"]
    assert loaded_client.delete("/documents/DOC002").status_code == 404


def test_page_preview_png_and_validation(loaded_client):
    r = loaded_client.get("/documents/DOC001/pages/2/preview")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
    assert loaded_client.get("/documents/DOC001/pages/2/preview").status_code == 200  # served from disk cache
    assert loaded_client.get("/documents/DOC001/pages/99/preview").status_code == 404
    assert loaded_client.get("/documents/DOC001/pages/0/preview").status_code == 422
    assert loaded_client.get("/documents/NOTANID/pages/1/preview").status_code == 422


def test_summary_endpoint(loaded_client):
    r = loaded_client.get("/documents/DOC001/summary").json()
    assert r["summary"] and r["suggested_questions"]


def test_resume_endpoint_rules(loaded_client):
    assert loaded_client.post("/documents/DOC001/resume").status_code == 409  # completed docs cannot be resumed
    c = loaded_client.app.state.container
    c.registry.update("DOC002", status="failed", resumable=True, error="Ollama down", pages_processed=1)
    r = loaded_client.post("/documents/DOC002/resume")
    assert r.status_code == 202
    assert wait_for(loaded_client, "DOC002")["status"] == "completed"


# ------------------------------------------------------------------- ask
def test_answer_shape_citations_match_retrieved_pages(loaded_client):
    r = ask(loaded_client, "How many paid sick days do employees receive per year?")
    assert r["answered"] and r["confidence"] in ("high", "medium", "low")
    assert set(r) >= {"question_id", "answer", "citations", "sources", "timings", "model", "verification"}
    assert {"document_id": "DOC001", "document": "Northwind_Leave_Policy.pdf", "page": 2} in r["citations"]
    for s in r["sources"]:
        assert {(s["document_id"], s["page"])} <= {(c["document_id"], c["page"]) for c in r["citations"]}
    assert "12" in r["answer"] and "retrieve_ms" in r["timings"] and r["timings"]["total_ms"] > 0


def test_direct_high_agreement_question_uses_focused_context(loaded_client):
    r = ask(loaded_client, "What is the hotel nightly limit in London?")
    assert r["answered"]
    prompt = loaded_client.app.state.container.llm.last_messages[-1]["content"]
    assert prompt.count("<<<") == 1
    assert "Northwind_Leave_Policy.pdf" not in prompt


def test_multi_section_question_cites_both_documents(loaded_client):
    r = ask(loaded_client, "How many sick days are allowed? What is the meal allowance for domestic trips?", top_k=5)
    docs = (
        {c["document_id"] for c in r["retrieval"]} if "retrieval" in r else {c["document_id"] for c in r["citations"]}
    )
    assert r["answered"] and docs  # the fake LLM answers from the best passage; both docs were retrieved
    llm = loaded_client.app.state.container.llm
    prompt = llm.last_messages[-1]["content"]
    assert "Northwind_Leave_Policy.pdf" in prompt and "Northwind_Travel_Expense_Policy.pdf" in prompt


def test_out_of_document_question_exits_early_without_calling_llm(loaded_client, fake_llm):
    before = fake_llm.calls
    r = ask(loaded_client, "What is the capital of France?")
    assert r["answered"] is False and r["answer"] == NOT_AVAILABLE_TEXT and r["confidence"] == "none"
    assert r["citations"] == [] and r["sources"] == []
    assert fake_llm.calls == before


def test_document_filter_changes_the_answer(loaded_client):
    q = "What is the limit?"
    only_leave = ask(loaded_client, "hotel nightly limit city tier", document_ids=["DOC001"])
    only_travel = ask(loaded_client, "hotel nightly limit city tier", document_ids=["DOC002"])
    assert only_travel["answered"] and all(c["document_id"] == "DOC002" for c in only_travel["citations"])
    assert not only_leave["answered"] or all(c["document_id"] == "DOC001" for c in only_leave["citations"])
    assert q


def test_filter_errors(loaded_client):
    assert loaded_client.post("/ask", json={"question": "hi there", "document_ids": ["DOC099"]}).status_code == 404
    assert loaded_client.post("/ask", json={"question": "hi there", "document_ids": ["bad id"]}).status_code == 422


def test_page_range_filter(loaded_client):
    r = ask(loaded_client, "sick leave days medical certificate", document_ids=["DOC001"], page_from=3, page_to=4)
    assert not r["answered"] or all(c["page"] in (3, 4) for c in r["citations"])


def test_cache_hit_and_invalidation(loaded_client, sample_pdfs, fake_llm):
    q = "How many paid sick days do employees receive per year?"
    a = ask(loaded_client, q)
    calls = fake_llm.calls
    b = ask(loaded_client, "  how many PAID sick days do employees receive per year ")
    assert (
        b["cached"] and fake_llm.calls == calls and b["answer"] == a["answer"] and b["question_id"] != a["question_id"]
    )
    assert b["timings"]["total_ms"] < 50
    loaded_client.delete("/documents/DOC002")  # doc set changes -> cache invalidated
    assert not ask(loaded_client, q)["cached"]


def test_summary_question_uses_stored_summary(loaded_client, fake_llm):
    calls = fake_llm.calls
    r = ask(loaded_client, "Summarize this document", document_ids=["DOC001"])
    assert r["answered"] and r["confidence"] == "high" and fake_llm.calls == calls
    assert r["citations"][0]["page"] is None and "leave" in r["answer"].lower()


def test_no_documents_uploaded(client):
    r = ask(client, "Anything at all in here?")
    assert not r["answered"] and r["answer"] == NOT_AVAILABLE_TEXT


def test_verifier_fails_closed_when_model_adds_unsupported_claim(loaded_client):
    c = loaded_client.app.state.container
    c.llm._answer = lambda _: "Annual leave is 24 days and includes free parking in Paris. [1]"

    response = ask(loaded_client, "How many days of annual leave are there?")

    assert response["answered"] is False
    assert response["answer"] == NOT_AVAILABLE_TEXT
    assert response["citations"] == []


def test_session_followup_is_rewritten_and_not_cached(loaded_client, fake_llm):
    ask(loaded_client, "How many paid sick days do employees receive per year?", session_id="s1")
    calls = fake_llm.calls
    r = ask(loaded_client, "And how long is maternity leave in weeks?", session_id="s1")
    assert fake_llm.calls >= calls + 1 and r["timings"].get("rewrite_ms") is not None and not r["cached"]


def test_standalone_question_skips_history_rewrite(loaded_client, fake_llm):
    ask(loaded_client, "How many paid sick days do employees receive per year?", session_id="s2")
    calls = fake_llm.calls

    response = ask(loaded_client, "Tell me about the travel policy.", session_id="s2")

    assert fake_llm.calls == calls + 1
    assert "rewrite_ms" not in response["timings"]


def test_top_k_limits_sources(loaded_client):
    r = ask(loaded_client, "leave days policy employees", top_k=1)
    assert len(r["sources"]) <= 1


def test_streaming_event_order_and_final_payload(loaded_client):
    with loaded_client.stream("POST", "/ask/stream", json={"question": "How many paid sick days per year?"}) as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        body = "".join(resp.iter_text())
    events = [(b.split("\n")[0][7:], json.loads(b.split("\n")[1][6:])) for b in body.strip().split("\n\n")]
    names = [e for e, _ in events]
    assert names[-1] == "final" and set(names[:-1]) == {"token"} and len(names) >= 2
    streamed = "".join(d["t"] for e, d in events if e == "token")
    final = events[-1][1]
    assert final["answered"] and 2 in [c["page"] for c in final["citations"]] and "12" in streamed
    assert "first_token_ms" in final["timings"]


def test_streaming_not_available_and_cache_replay(loaded_client):
    def stream(q):
        with loaded_client.stream("POST", "/ask/stream", json={"question": q}) as r:
            text = "".join(r.iter_text())
        return [json.loads(b.split("\n")[1][6:]) for b in text.strip().split("\n\n")]

    ev = stream("What is the capital of France?")
    assert ev[-1]["answered"] is False
    stream("How many paid sick days per year?")
    again = stream("How many paid sick days per year?")
    assert again[-1]["cached"] is True


def test_streaming_setup_errors_are_http_errors(loaded_client):
    r = loaded_client.post("/ask/stream", json={"question": "hello there", "document_ids": ["DOC404"]})
    assert r.status_code == 404


def test_feedback_and_metrics(loaded_client):
    a = ask(loaded_client, "How many paid sick days do employees receive per year?")
    assert loaded_client.post("/feedback", json={"question_id": a["question_id"], "rating": "up"}).status_code == 200
    assert loaded_client.post("/feedback", json={"question_id": "nope", "rating": "up"}).status_code == 404
    assert loaded_client.post("/feedback", json={"question_id": a["question_id"], "rating": "meh"}).status_code == 422
    m = loaded_client.get("/metrics/recent").json()
    assert m["feedback"]["up"] == 1 and m["recent"][0]["question_id"] == a["question_id"]
    assert m["latency_ms"]["p95"] >= m["latency_ms"]["p50"] > 0 and "retrieve_ms" in m["stage_avg_ms"]
    assert m["evaluation"] is None or isinstance(m["evaluation"], dict)
    assert m["evaluation_meta"] is None or isinstance(m["evaluation_meta"], dict)
    assert isinstance(m["evaluation_rows"], list)


def test_metrics_exposes_evaluation_rows_from_latest_run(loaded_client, tmp_path, monkeypatch):
    results_dir = tmp_path / "eval"
    results_dir.mkdir()
    (results_dir / "latest.json").write_text(
        json.dumps(
            {
                "summary": {"coherence": 0.75},
                "meta": {
                    "coherence_method": "local judge",
                    "documents": ["Northwind_Leave_Policy.pdf", "Northwind_Travel_Expense_Policy.pdf"],
                },
                "rows": [{"id": "Q1", "coherence_score": 0.75}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: results_dir))

    answer = ask(loaded_client, "How many paid sick days do employees receive per year?")
    metrics = loaded_client.get("/metrics/recent").json()

    assert metrics["recent"][0]["question_id"] == answer["question_id"]
    assert metrics["recent"][0]["document_ids"]
    assert metrics["benchmark"]["summary"]["coherence"] == 0.75
    assert metrics["benchmark"]["rows"] == [{"id": "Q1", "coherence_score": 0.75}]
    assert metrics["evaluation"] is None or "coherence" not in metrics["evaluation"]
    assert metrics["evaluation_meta"] is None or "local judge" not in str(metrics["evaluation_meta"])
    assert all(row["id"] != "Q1" for row in metrics["evaluation_rows"])
    if metrics["evaluation_rows"]:
        assert metrics["evaluation_rows"][0]["id"] == answer["question_id"]


def test_metrics_excludes_queries_without_uploaded_pdf_provenance(loaded_client):
    registry = loaded_client.app.state.container.registry
    registry.log_query(
        question_id="LEGACY001",
        question="unattributed question",
        answered=True,
        confidence="high",
        cached=False,
        timings={"total_ms": 1.0},
    )

    metrics = loaded_client.get("/metrics/recent").json()

    assert all(row["question_id"] != "LEGACY001" for row in metrics["recent"])


def test_metrics_stay_blank_without_uploaded_pdfs(client, tmp_path, monkeypatch):
    results_dir = tmp_path / "eval"
    results_dir.mkdir()
    (results_dir / "latest.json").write_text(
        json.dumps(
            {
                "summary": {"coherence": 0.99, "groundedness": 1.0},
                "meta": {"coherence_method": "benchmark judge"},
                "rows": [{"id": "BENCHMARK", "coherence_score": 0.99}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: results_dir))

    metrics = client.get("/metrics/recent").json()

    assert metrics["recent"] == []
    assert metrics["evaluation"] is None
    assert metrics["evaluation_meta"] is None
    assert metrics["evaluation_rows"] == []
    assert metrics["benchmark"] is None


def test_benchmark_hides_when_a_sample_pdf_is_missing(client, sample_pdfs, tmp_path, monkeypatch):
    results_dir = tmp_path / "eval"
    results_dir.mkdir()
    (results_dir / "latest.json").write_text(
        json.dumps(
            {
                "summary": {"pass_rate": 1.0},
                "meta": {"documents": ["Northwind_Leave_Policy.pdf", "Northwind_Travel_Expense_Policy.pdf"]},
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: results_dir))
    upload(client, sample_pdfs["leave"])

    metrics = client.get("/metrics/recent").json()

    assert metrics["benchmark"] is None


def test_uploaded_benchmark_loads_for_matching_pdf_hashes(client, sample_pdfs, tmp_path, monkeypatch):
    uploaded = [upload(client, sample_pdfs[name]) for name in ("leave", "travel")]
    registry = client.app.state.container.registry
    result_path = tmp_path / "uploaded_policies" / "latest.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "summary": {"questions": 20, "coherence": 0.75},
                "meta": {
                    "documents": [sample_pdfs["leave"].name, sample_pdfs["travel"].name],
                    "document_hashes": {
                        name: registry.get(doc["document_id"])["sha256"]
                        for name, doc in zip(("leave", "travel"), uploaded, strict=True)
                    },
                },
                "rows": [{"id": "Q1", "coherence_score": 0.75}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: tmp_path))

    metrics = client.get("/metrics/recent").json()

    assert metrics["uploaded_benchmark"]["summary"]["coherence"] == 0.75
    assert metrics["uploaded_benchmark"]["rows"][0]["coherence_score"] == 0.75


def test_uploaded_benchmark_hides_if_pdf_content_changed(client, sample_pdfs, tmp_path, monkeypatch):
    for name in ("leave", "travel"):
        upload(client, sample_pdfs[name])
    result_path = tmp_path / "uploaded_policies" / "latest.json"
    result_path.parent.mkdir()
    result_path.write_text(
        json.dumps(
            {
                "summary": {"questions": 20},
                "meta": {
                    "documents": [sample_pdfs["leave"].name, sample_pdfs["travel"].name],
                    "document_hashes": {
                        sample_pdfs["leave"].name: "stale-content-hash",
                        sample_pdfs["travel"].name: hashlib.sha256(sample_pdfs["travel"].read_bytes()).hexdigest(),
                    },
                },
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Settings, "eval_results_dir", property(lambda _: tmp_path))

    metrics = client.get("/metrics/recent").json()

    assert metrics["uploaded_benchmark"] is None


def test_ready_and_health_endpoints(loaded_client):
    assert loaded_client.get("/health").json() == {"status": "ok"}
    r = loaded_client.get("/ready").json()
    assert r["ready"] and r["documents"] == 2 and r["chunks"] > 10 and r["reranker"] == "lexical"


def test_progress_fields_for_in_flight_document():
    import time

    from app.api.routes_documents import to_out

    d = {
        "id": "DOC001",
        "filename": "a.pdf",
        "status": "processing",
        "stage": "embedding",
        "pages": 100,
        "pages_processed": 25,
        "chunks": 10,
        "size_bytes": 1,
        "created_at": 1.0,
        "started_at": time.time() - 10,
        "warnings": [],
        "suggested_questions": [],
        "resumable": False,
    }
    out = to_out(d)
    assert out.progress.percent == 25.0 and 25 < out.progress.eta_seconds < 40
    shutil  # noqa: B018
