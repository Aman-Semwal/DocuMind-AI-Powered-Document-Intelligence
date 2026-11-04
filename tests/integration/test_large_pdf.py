"""Large-document behaviour on synthetic PDFs with known facts (deterministic)."""

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import OllamaUnavailable
from app.main import create_app
from app.services import pdf_extractor
from app.services.fakes import FakeEmbedder, FakeLLM
from eval.stress import build_stress_pdf
from tests.conftest import HEADERS, make_settings, upload, wait_for

PAGES = 150


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    return build_stress_pdf(tmp_path_factory.mktemp("big") / "agreement.pdf", PAGES)


@pytest.fixture
def big_client(tmp_path, big):
    app = create_app(make_settings(tmp_path), embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        yield c


def test_ingests_fast_with_progress_and_all_pages(big_client, big):
    path, _ = big
    t0 = time.time()
    doc = upload(big_client, path, timeout=120)
    secs = time.time() - t0
    assert doc["status"] == "completed" and doc["pages"] == PAGES and doc["chunks"] >= PAGES
    assert PAGES / secs >= 5, f"ingestion too slow: {PAGES / secs:.1f} pages/s"
    assert doc["progress"] == {"pages_processed": PAGES, "total_pages": PAGES, "percent": 100.0, "eta_seconds": None}


def test_needle_in_haystack_start_middle_end(big_client, big):
    path, needles = big
    upload(big_client, path, timeout=120)
    for n in needles:
        r = big_client.post("/ask", json={"question": n.question}).json()
        assert r["answered"], n.question
        assert n.answer in r["answer"], (n.question, r["answer"])
        assert n.page in [c["page"] for c in r["citations"]], (n.page, r["citations"])


def test_out_of_document_questions_on_large_pdf_are_refused(big_client, big):
    upload(big_client, big[0], timeout=120)
    for q in [
        "What is the capital of France?",
        "Who won the football world cup in 2018?",
        "How do I bake sourdough bread?",
    ]:
        assert big_client.post("/ask", json={"question": q}).json()["answered"] is False


def test_retrieval_latency_stays_small(big_client, big):
    upload(big_client, big[0], timeout=120)
    times = []
    for i, n in enumerate(big[1] * 4):
        r = big_client.post("/ask", json={"question": f"{n.question} (variant {i})"}).json()
        times.append(r["timings"]["retrieve_ms"])
    times.sort()
    assert times[int(len(times) * 0.95) - 1] < 150


def test_resume_after_embedder_failure_has_no_duplicates(tmp_path, big):
    path, _ = big
    baseline = create_app(make_settings(tmp_path / "base"), embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(baseline) as c0:
        c0.headers.update(HEADERS)
        expected = upload(c0, path, timeout=120)["chunks"]

    class Flaky(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.fail_after = 2

        def embed_documents(self, texts):
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise OllamaUnavailable()
            return super().embed_documents(texts)

    emb = Flaky()
    with TestClient(create_app(make_settings(tmp_path / "flaky"), embedder=emb, llm=FakeLLM())) as c:
        c.headers.update(HEADERS)
        doc = upload(c, path, timeout=120)
        assert doc["status"] == "failed" and doc["resumable"] is True
        partial = c.app.state.container.store.count_document(doc["document_id"])
        assert 0 < partial < expected and 0 < doc["progress"]["pages_processed"] < PAGES
        emb.fail_after = None  # Ollama is back
        assert c.post(f"/documents/{doc['document_id']}/resume").status_code == 202
        done = wait_for(c, doc["document_id"], 120)
        assert done["status"] == "completed" and done["chunks"] == expected
        assert c.app.state.container.store.count_document(doc["document_id"]) == expected


def test_corrupt_page_in_the_middle_is_skipped_rest_processed(big_client, big, monkeypatch):
    real = pdf_extractor._extract_page

    def flaky(page, body, tables):
        if page.number + 1 == 70:
            raise RuntimeError("damaged page")
        return real(page, body, tables)

    monkeypatch.setattr(pdf_extractor, "_extract_page", flaky)
    doc = upload(big_client, big[0], timeout=120)
    assert doc["status"] == "completed" and doc["chunks"] >= PAGES - 5
    meta = big_client.app.state.container.store.collection.get(include=["metadatas"])["metadatas"]
    pages = {m["page_number"] for m in meta}
    assert 70 not in pages and 69 in pages and 71 in pages


def test_several_large_uploads_concurrently(tmp_path):
    paths = [build_stress_pdf(tmp_path / f"b{i}.pdf", 40, seed=i)[0] for i in range(3)]
    app = create_app(make_settings(tmp_path / "d"), embedder=FakeEmbedder(), llm=FakeLLM())
    with TestClient(app) as c:
        c.headers.update(HEADERS)
        out, lock = [], threading.Lock()

        def go(p: Path):
            with p.open("rb") as fh:
                r = c.post("/documents/upload", files={"file": (p.name, fh, "application/pdf")})
            with lock:
                out.append(r.json()["document_id"])

        ts = [threading.Thread(target=go, args=(p,)) for p in paths]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert len(set(out)) == 3
        for i in out:
            d = wait_for(c, i, 120)
            assert d["status"] == "completed" and c.app.state.container.store.count_document(i) == d["chunks"]
