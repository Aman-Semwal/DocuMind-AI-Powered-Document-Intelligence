"""Embedder, LLM client, registry, vector store, BM25 and PDF extractor."""

import asyncio
import json
import sqlite3
import threading

import httpx
import pymupdf
import pytest
import respx

from app.config import Settings
from app.core.exceptions import (
    CorruptPdf,
    DuplicateDocument,
    EmbeddingMismatch,
    EncryptedPdf,
    LLMTimeout,
    NotFound,
    OllamaUnavailable,
    TooManyPages,
)
from app.db.registry import EmbeddingCache, Registry
from app.services.chunker import Chunk
from app.services.embedder import CachedEmbedder, OllamaEmbedder, cache_key
from app.services.fakes import FakeEmbedder, FakeLLM
from app.services.llm_client import OllamaLLM
from app.services.pdf_extractor import (
    edge_lines,
    inspect_pdf,
    iter_pages,
    normalize_edge,
    render_page_png,
    repeated_boilerplate,
)
from app.services.vector_store import BM25Index, ChromaStore

BASE = "http://ollama.test"


# ------------------------------------------------------------------ embedder
@respx.mock
def test_ollama_embedder_batches_and_prefixes():
    route = respx.post(f"{BASE}/api/embed").mock(
        side_effect=lambda req: httpx.Response(
            200, json={"embeddings": [[1.0, 0.0]] * len(json.loads(req.content)["input"])}
        )
    )
    e = OllamaEmbedder(BASE, "nomic-embed-text", batch_size=2, retries=0)
    out = e.embed_documents(["a", "b", "c"])
    assert len(out) == 3 and route.call_count == 2
    assert json.loads(route.calls[0].request.content)["input"] == ["search_document: a", "search_document: b"]
    e.embed_query("q")
    assert json.loads(route.calls[-1].request.content)["input"] == ["search_query: q"]
    e.close()


@respx.mock
def test_ollama_embedder_errors():
    e = OllamaEmbedder(BASE, "other-model", retries=0)
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(404))
    with pytest.raises(OllamaUnavailable, match="ollama pull"):
        e.embed_query("x")
    respx.post(f"{BASE}/api/embed").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(OllamaUnavailable):
        e.embed_query("x")
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(200, json={"embeddings": []}))
    with pytest.raises(OllamaUnavailable):
        e.embed_query("x")


@respx.mock
def test_ollama_embedder_ping():
    e = OllamaEmbedder(BASE, "m")
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={}))
    assert e.ping()
    respx.get(f"{BASE}/api/tags").mock(side_effect=httpx.ConnectError("x"))
    assert not e.ping()


def test_cached_embedder_never_reembeds_identical_text(tmp_path):
    inner = FakeEmbedder()
    ce = CachedEmbedder(inner, EmbeddingCache(tmp_path / "c.sqlite"))
    first = ce.embed_documents(["alpha beta", "gamma", "alpha beta"])
    assert inner.calls == 1 and ce.misses == 2
    second = ce.embed_documents(["gamma", "alpha beta"])
    assert inner.calls == 1 and ce.hits >= 2
    assert second[0] == pytest.approx(first[1], abs=1e-6)
    assert cache_key("a", "m1") != cache_key("a", "m2")
    assert ce.embed_query("alpha") == inner.embed_query("alpha") and ce.ping()
    ce.close()
    ce.cache.put_many({})


# ---------------------------------------------------------------------- LLM
def llm_settings(**kw):
    return Settings(_env_file=None, ollama_base_url=BASE, llm_retries=0, **kw)


MSGS = [{"role": "user", "content": "hi"}]


@respx.mock
def test_llm_chat_payload_and_response():
    route = respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": " hello "}})
    )

    async def go():
        llm = OllamaLLM(llm_settings())
        out = await llm.chat(MSGS)
        await llm.aclose()
        return out

    assert asyncio.run(go()) == "hello"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen2.5:3b-instruct" and body["stream"] is False
    assert body["options"] == {"temperature": 0.1, "seed": 42, "num_ctx": 4096, "num_predict": 256}
    assert body["keep_alive"] == "30m"


@respx.mock
def test_llm_deep_mode_uses_big_model():
    route = respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, json={"message": {"content": "x"}}))
    asyncio.run(OllamaLLM(llm_settings()).chat(MSGS, deep=True, max_tokens=10))
    body = json.loads(route.calls[0].request.content)
    assert body["model"].startswith("qwen2.5:7b") and body["options"]["num_predict"] == 10


@respx.mock
def test_llm_deep_mode_falls_back_when_big_model_is_missing():
    route = respx.post(f"{BASE}/api/chat").mock(
        side_effect=[httpx.Response(404), httpx.Response(200, json={"message": {"content": "fallback"}})]
    )

    async def go():
        llm = OllamaLLM(llm_settings())
        return await llm.chat(MSGS, deep=True)

    assert asyncio.run(go()) == "fallback"
    models = [json.loads(call.request.content)["model"] for call in route.calls]
    assert models == ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5:3b-instruct"]


@respx.mock
def test_llm_stream_yields_tokens():
    lines = "\n".join(
        json.dumps(x)
        for x in [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
    )
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, content=lines.encode()))

    async def go():
        llm = OllamaLLM(llm_settings())
        return [t async for t in llm.stream(MSGS)]

    assert asyncio.run(go()) == ["Hel", "lo"]


@respx.mock
def test_llm_deep_stream_falls_back_when_big_model_is_missing():
    tokens = json.dumps({"message": {"content": "answer"}, "done": False}) + "\n"
    tokens += json.dumps({"message": {"content": ""}, "done": True}) + "\n"
    route = respx.post(f"{BASE}/api/chat").mock(
        side_effect=[httpx.Response(404), httpx.Response(200, content=tokens.encode())]
    )

    async def go():
        llm = OllamaLLM(llm_settings())
        return [piece async for piece in llm.stream(MSGS, deep=True)]

    assert asyncio.run(go()) == ["answer"]
    models = [json.loads(call.request.content)["model"] for call in route.calls]
    assert models == ["qwen2.5:7b-instruct-q4_K_M", "qwen2.5:3b-instruct"]


@respx.mock
@pytest.mark.parametrize(
    "mock,exc",
    [
        (httpx.Response(404), OllamaUnavailable),
        (httpx.Response(500), OllamaUnavailable),
        (httpx.ConnectError("refused"), OllamaUnavailable),
        (httpx.ReadTimeout("slow"), LLMTimeout),
    ],
)
def test_llm_error_mapping(mock, exc):
    respx.post(f"{BASE}/api/chat").mock(
        **({"side_effect": mock} if isinstance(mock, Exception) else {"return_value": mock})
    )

    async def go():
        llm = OllamaLLM(llm_settings())
        with pytest.raises(exc):
            await llm.chat(MSGS)
        with pytest.raises(exc):
            _ = [t async for t in llm.stream(MSGS)]

    asyncio.run(go())
    with pytest.raises(exc):
        OllamaLLM(llm_settings()).chat_sync(MSGS)


@respx.mock
def test_llm_stream_error_line_and_retry_on_connect():
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, content=b'{"error": "boom"}'))

    async def go():
        with pytest.raises(OllamaUnavailable):
            _ = [t async for t in OllamaLLM(llm_settings()).stream(MSGS)]

    asyncio.run(go())
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("first fails")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    respx.post(f"{BASE}/api/chat").mock(side_effect=flaky)
    s = Settings(_env_file=None, ollama_base_url=BASE, llm_retries=2)
    assert asyncio.run(OllamaLLM(s).chat(MSGS)) == "ok" and calls["n"] == 2


@respx.mock
def test_llm_ping_and_warm_up():
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={}))
    route = respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, json={"message": {"content": ""}}))

    async def go():
        llm = OllamaLLM(llm_settings())
        assert await llm.ping()
        await llm.warm_up()
        respx.get(f"{BASE}/api/tags").mock(side_effect=httpx.ConnectError("x"))
        assert not await llm.ping()
        respx.post(f"{BASE}/api/chat").mock(side_effect=httpx.ConnectError("x"))
        await llm.warm_up()  # must not raise

    asyncio.run(go())
    assert json.loads(route.calls[0].request.content)["options"]["num_predict"] == 1


def test_fake_llm_behaviours():
    llm = FakeLLM()
    out = llm.chat_sync(
        [
            {"role": "system", "content": "TASK=qa"},
            {
                "role": "user",
                "content": "[1] (source: a.pdf, page 1)\n<<<\nThe limit is 250 dollars per night.\n>>>\n\nQuestion: What is the limit per night?\nAnswer:",
            },
        ]
    )
    assert out.endswith("[1]") and "250" in out
    assert "not available" in llm.chat_sync(
        [{"role": "system", "content": "TASK=qa"}, {"role": "user", "content": "Question: x"}]
    )

    async def go():
        return [
            t
            async for t in llm.stream(
                [{"role": "system", "content": "TASK=qa"}, {"role": "user", "content": "Question: x"}]
            )
        ]

    assert "".join(asyncio.run(go())).startswith("Information not available in the uploaded documents.")


# ------------------------------------------------------------------ registry
def doc(reg, n, **kw):
    return reg.create_document(
        sha256=f"hash{n}", filename=f"f{n}.pdf", stored_path=f"/tmp/f{n}.pdf", size_bytes=10, pages=2, **kw
    )


def test_registry_sequential_ids_and_duplicates(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    assert [doc(reg, i)["id"] for i in range(3)] == ["DOC001", "DOC002", "DOC003"]
    with pytest.raises(DuplicateDocument) as ei:
        doc(reg, 1)
    assert ei.value.existing_id == "DOC002"
    assert len(reg.list_documents()) == 3
    reg.delete("DOC003")
    assert doc(reg, 9)["id"] == "DOC003" or doc(reg, 10)  # sequence continues safely


def test_registry_concurrent_creates_have_unique_ids(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    ids, errors = [], []

    def work(i):
        try:
            ids.append(doc(reg, i)["id"])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(25)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and len(set(ids)) == 25


def test_registry_update_status_and_json_fields(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    d = doc(reg, 1)
    assert d["status"] == "pending" and d["warnings"] == [] and d["resumable"] is False
    reg.update(d["id"], status="completed", suggested_questions=["a?"], resumable=True, summary="s")
    reg.add_warning(d["id"], "w1")
    got = reg.get(d["id"])
    assert got["suggested_questions"] == ["a?"] and got["warnings"] == ["w1"] and got["resumable"] is True
    assert reg.completed_ids() == ["DOC001"] and reg.count_active() == 0 and reg.incomplete() == []
    assert reg.exists("DOC001") and not reg.exists("DOC404")
    with pytest.raises(NotFound):
        reg.get("DOC404")
    reg.update("DOC001")  # no fields: no-op


def test_registry_version_queries_feedback(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    assert reg.docset_version() == 0 and reg.bump_version() == 1
    qid = reg.new_question_id()
    reg.log_query(
        question_id=qid, question="q?", answered=True, confidence="high", cached=False, timings={"total_ms": 5}
    )
    assert reg.query_exists(qid) and not reg.query_exists("nope")
    reg.set_feedback(qid, "up", None)
    reg.set_feedback(qid, "down", "wrong")  # latest rating wins
    assert reg.feedback_counts() == {"up": 0, "down": 1}
    rows = reg.recent_queries()
    assert rows[0]["rating"] == "down" and rows[0]["timings"] == {"total_ms": 5}


def test_registry_deletes_document_insights_and_feedback(tmp_path):
    reg = Registry(tmp_path / "r.sqlite")
    removed_qid = reg.new_question_id()
    retained_qid = reg.new_question_id()
    reg.log_query(
        question_id=removed_qid,
        question="document specific?",
        answered=True,
        confidence="high",
        cached=False,
        timings={"total_ms": 5},
        document_ids=["DOC001"],
    )
    reg.log_query(
        question_id=retained_qid,
        question="other document?",
        answered=True,
        confidence="high",
        cached=False,
        timings={"total_ms": 6},
        document_ids=["DOC002"],
    )
    reg.set_feedback(removed_qid, "up", "helpful")
    reg.set_feedback(retained_qid, "down", "not helpful")

    assert reg.delete_document_insights("DOC001") == 1
    assert not reg.query_exists(removed_qid)
    assert reg.query_exists(retained_qid)
    assert [row["question_id"] for row in reg.recent_queries()] == [retained_qid]
    assert reg.feedback_counts() == {"up": 0, "down": 1}


def test_registry_migrates_query_provenance_columns(tmp_path):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE query_log (question_id TEXT PRIMARY KEY, ts REAL NOT NULL, question TEXT NOT NULL,"
            " answered INTEGER NOT NULL, confidence TEXT NOT NULL, cached INTEGER NOT NULL DEFAULT 0,"
            " timings TEXT NOT NULL DEFAULT '{}')"
        )
        connection.execute("INSERT INTO query_log VALUES ('old', 1, 'old question', 1, 'high', 0, '{}')")

    rows = Registry(path).recent_queries()

    assert rows[0]["question_id"] == "old"
    assert rows[0]["document_ids"] == []
    assert rows[0]["verification"] is None
    assert rows[0]["citations"] == []


def test_registry_persists_across_reopen(tmp_path):
    doc(Registry(tmp_path / "r.sqlite"), 1)
    assert Registry(tmp_path / "r.sqlite").get("DOC001")["filename"] == "f1.pdf"


def test_embedding_cache_roundtrip(tmp_path):
    c = EmbeddingCache(tmp_path / "e.sqlite")
    c.put_many({"k1": [0.5, 0.25], "k2": [1.0, 2.0]})
    got = c.get_many(["k1", "k2", "missing"])
    assert got["k1"] == [0.5, 0.25] and "missing" not in got


# --------------------------------------------------------------- vector store
def mk_chunks(n=4, doc_id="DOC001"):
    return [
        Chunk(f"{doc_id}-p{i + 1:04d}-c000", doc_id, "a.pdf", i + 1, 0, f"text about topic {i} unique{i}")
        for i in range(n)
    ]


def test_chroma_store_roundtrip_and_persistence(tmp_path):
    emb = FakeEmbedder()
    store = ChromaStore(tmp_path / "chroma", emb.model_name)
    chunks = mk_chunks()
    store.upsert(chunks, emb.embed_documents([c.stored_text for c in chunks]))
    store.upsert(chunks, emb.embed_documents([c.stored_text for c in chunks]))  # idempotent
    assert store.count() == 4 and store.count_document("DOC001") == 4
    assert store.count_document("DOC001", max_page=2) == 2
    hits = store.query(emb.embed_query("topic 2 unique2"), 2, {"document_id": "DOC001"})
    assert hits[0].chunk_id == "DOC001-p0003-c000" and 0 < hits[0].score <= 1.0 + 1e-6
    assert hits[0].metadata["page_number"] == 3
    assert store.query(emb.embed_query("x"), 3, {"document_id": "NOPE"}) == []
    assert set(store.get_many(["DOC001-p0001-c000", "zzz"])) == {"DOC001-p0001-c000"}
    assert store.get_many([]) == {}
    assert len(list(store.iter_all(batch=3))) == 4
    assert store.heartbeat()
    # restart persistence
    again = ChromaStore(tmp_path / "chroma", emb.model_name)
    assert again.count() == 4
    again.delete_document("DOC001")
    assert again.count() == 0 and again.query([0.1] * 384, 3, None) == []


def test_chroma_store_refuses_other_embedding_model(tmp_path):
    emb = FakeEmbedder()
    store = ChromaStore(tmp_path / "chroma", "model-a")
    c = mk_chunks(1)
    store.upsert(c, emb.embed_documents([c[0].stored_text]))
    reopened = ChromaStore(tmp_path / "chroma", "model-b")
    with pytest.raises(EmbeddingMismatch):
        reopened.assert_model("model-b")
    reopened.assert_model("model-a")


# ----------------------------------------------------------------------- BM25
def test_bm25_search_filters_and_removal():
    idx = BM25Index()
    idx.add_chunks(mk_chunks(3))
    idx.add_chunks(mk_chunks(1, "DOC002"))
    assert len(idx) == 4
    hits = idx.search("unique1", 5)
    assert hits[0].chunk_id == "DOC001-p0002-c000"
    assert all(h.metadata["document_id"] == "DOC002" for h in idx.search("topic", 5, {"DOC002"}))
    assert [h.metadata["page_number"] for h in idx.search("topic", 5, {"DOC001"}, page_from=3)] == [3]
    assert idx.search("", 3) == [] and idx.search("zzzz", 3) == []
    idx.remove_document("DOC001")
    assert len(idx) == 1 and idx.search("unique1", 3) == []
    idx.add("x", "re-add same id", {"document_id": "D"})
    idx.add("x", "replaced text", {"document_id": "D"})
    assert len(idx) == 2 and idx.search("replaced", 1)[0].chunk_id == "x"


def test_bm25_rebuild_from_store(tmp_path):
    emb = FakeEmbedder()
    store = ChromaStore(tmp_path / "c", emb.model_name)
    cs = mk_chunks(3)
    store.upsert(cs, emb.embed_documents([c.stored_text for c in cs]))
    idx = BM25Index()
    assert idx.rebuild_from(store) == 3 and idx.search("unique2", 1)


# ------------------------------------------------------------------ extractor
def test_inspect_and_iterate_sample(sample_pdfs):
    info = inspect_pdf(sample_pdfs["leave"], 100)
    assert info.pages == 4 and info.body_font_size == 10.5
    pages = list(iter_pages(sample_pdfs["leave"], info.body_font_size))
    assert [p.page_number for p in pages] == [1, 2, 3, 4]
    heads = [b.text for b in pages[0].blocks if b.kind == "heading"]
    assert "2. Annual Leave" in heads and "Employee Leave Policy" in heads
    tables = [b for p in pages for b in p.blocks if b.kind == "table"]
    assert len(tables) == 1 and "| Annual leave | 24 |" in tables[0].text and "Leave type" in tables[0].text
    only = list(iter_pages(sample_pdfs["leave"], 10.5, start=2, end=3))
    assert [p.page_number for p in only] == [2, 3]
    notables = list(iter_pages(sample_pdfs["leave"], 10.5, 3, 3, tables=False))
    assert all(b.kind != "table" for b in notables[0].blocks)


def test_boilerplate_detection(sample_pdfs):
    edges, total, with_text = edge_lines(sample_pdfs["leave"])
    bp = repeated_boilerplate(edges)
    assert "page # of #" in bp and "confidential - internal use only" in bp
    assert total > 1000 and with_text == 4
    assert repeated_boilerplate([["a"], ["a"]]) == set()  # too few pages
    assert normalize_edge("Page 12 of 30") == "page # of #"


def test_render_preview_png(sample_pdfs):
    png = render_page_png(sample_pdfs["leave"], 1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000
    with pytest.raises(IndexError):
        render_page_png(sample_pdfs["leave"], 99)


def test_corrupt_encrypted_and_long_pdfs(tmp_path, sample_pdfs):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not really a pdf at all")
    with pytest.raises(CorruptPdf):
        inspect_pdf(bad, 100)
    enc = tmp_path / "enc.pdf"
    d = pymupdf.open(sample_pdfs["leave"])
    d.save(str(enc), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    with pytest.raises(EncryptedPdf):
        inspect_pdf(enc, 100)
    with pytest.raises(TooManyPages):
        inspect_pdf(sample_pdfs["leave"], 2)


def test_scanned_like_pdf_has_no_text(tmp_path):
    d = pymupdf.open()
    d.new_page()
    p = tmp_path / "blank.pdf"
    d.save(str(p))
    _, total, with_text = edge_lines(p)
    assert total == 0 and with_text == 0
