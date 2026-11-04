"""Retriever logic with a fake vector store (no Chroma, no network)."""

import pytest

from app.config import Settings
from app.core.exceptions import EmbeddingMismatch
from app.services.chunker import Chunk
from app.services.fakes import FakeEmbedder
from app.services.reranker import LexicalReranker, NoopReranker
from app.services.retriever import Candidate, Retriever, mmr_select
from app.services.vector_store import BM25Index, Hit, build_where

DOCS = [
    ("DOC001", "leave.pdf", 1, "Annual leave is 24 paid days per calendar year for every employee."),
    ("DOC001", "leave.pdf", 2, "Sick leave is 12 days and needs a medical certificate after 2 days."),
    ("DOC001", "leave.pdf", 3, "Maternity leave is 26 weeks paid at full base salary."),
    ("DOC002", "travel.pdf", 1, "Hotel limit in Tier 1 cities such as London is 250 USD per night."),
    ("DOC002", "travel.pdf", 2, "Meal allowance is 60 USD per day for domestic trips."),
]


class FakeStore:
    def __init__(self, emb, model="fake-hash-embed"):
        self.stored_model = model
        self.emb = emb
        self.rows = []
        for i, (doc, name, page, text) in enumerate(DOCS):
            c = Chunk(f"{doc}-p{page:04d}-c000", doc, name, page, 0, text)
            self.rows.append((c, emb.embed_documents([c.stored_text])[0]))

    def assert_model(self, model):
        if self.stored_model != model:
            raise EmbeddingMismatch()

    def query(self, vec, n, where):
        out = []
        for c, v in self.rows:
            if where:
                clause = where.get("document_id") or {}
                ids = clause.get("$in") if isinstance(clause, dict) else [clause]
                if ids and c.document_id not in ids:
                    continue
            sim = sum(a * b for a, b in zip(vec, v))
            out.append(Hit(c.chunk_id, c.stored_text, c.metadata(), sim))
        return sorted(out, key=lambda h: h.score, reverse=True)[:n]

    def get_many(self, ids):
        return {c.chunk_id: Hit(c.chunk_id, c.stored_text, c.metadata(), 0) for c, _ in self.rows if c.chunk_id in ids}


def make(min_similarity=0.2, reranker=None, **kw):
    s = Settings(_env_file=None, min_similarity=min_similarity, **kw)
    emb = FakeEmbedder()
    store = FakeStore(emb)
    bm25 = BM25Index()
    bm25.add_chunks([c for c, _ in store.rows])
    return Retriever(s, emb, store, bm25, reranker or LexicalReranker()), store


def test_relevant_question_passes_and_ranks_right_chunk_first():
    r, _ = make()
    res = r.retrieve("How many paid days of annual leave?")
    assert res.passed and res.candidates[0].page == 1 and res.candidates[0].document_id == "DOC001"
    assert res.best_vector_sim > 0.2 and res.agreement


def test_out_of_document_question_fails_threshold():
    r, _ = make()
    res = r.retrieve("What is the capital of France?")
    assert not res.passed


def test_top_k_is_respected():
    r, _ = make()
    assert len(r.retrieve("leave days", final_k=2).candidates) <= 2
    assert len(r.retrieve("leave days", final_k=5).candidates) <= 5


def test_document_filter():
    r, _ = make()
    res = r.retrieve("limit per night", document_ids=["DOC001"])
    assert all(c.document_id == "DOC001" for c in res.candidates)
    assert r.retrieve("anything", document_ids=[]).candidates == []


def test_page_range_filter_on_bm25():
    r, _ = make()
    res = r.retrieve("leave days", document_ids=["DOC001"], page_from=3, page_to=3)
    bm25_only = [c for c in res.candidates if c.in_bm25]
    assert all(c.page == 3 for c in bm25_only)


def test_subqueries_merge_evidence():
    r, _ = make()
    res = r.retrieve(
        "maternity leave and hotel limit?",
        ["How long is maternity leave?", "What is the hotel limit in London?"],
        final_k=5,
    )
    pages = {(c.document_id, c.page) for c in res.candidates}
    assert ("DOC001", 3) in pages and ("DOC002", 1) in pages


def test_embedding_model_mismatch_refused():
    r, store = make()
    store.stored_model = "other-model"
    with pytest.raises(EmbeddingMismatch):
        r.retrieve("anything")


def test_noop_reranker_keeps_rrf_order():
    r, _ = make(reranker=NoopReranker())
    res = r.retrieve("annual leave paid days calendar year")
    assert res.candidates and res.candidates[0].page == 1


def test_cross_encoder_gate_uses_rerank_score():
    class Fake:
        name = "cross_encoder"
        is_cross_encoder = True

        def __init__(self, v):
            self.v = v

        def score(self, q, texts):
            return [self.v] * len(texts)

    low, _ = make(reranker=Fake(0.01))
    high, _ = make(reranker=Fake(0.9))
    assert not low.retrieve("annual leave days").passed
    assert high.retrieve("annual leave days").passed


def test_neighbors_lookup():
    r, _ = make()
    cand = Candidate("DOC001-p0001-c000", "DOC001", "leave.pdf", 1, 0, "", "text", "x")
    assert r.neighbors(cand) == ("", "")


def cand(i, text):
    return Candidate(f"c{i}", "D", "f", 1, i, "", "text", text)


def test_mmr_drops_near_duplicates_and_keeps_diversity():
    a = cand(1, "annual leave is 24 days per year for employees")
    dup = cand(2, "annual leave is 24 days per year for employees")
    other = cand(3, "hotel limit is 250 dollars per night in london")
    out = mmr_select([a, dup, other], 3, 0.7)
    assert [c.chunk_id for c in out] == ["c1", "c3"]
    assert mmr_select([a], 3, 0.7) == [a]
    assert mmr_select([], 3, 0.7) == []


def test_build_where():
    assert build_where(None, None, None) is None
    assert build_where(["DOC001"], None, None) == {"document_id": "DOC001"}
    assert build_where(["A", "B"], None, None) == {"document_id": {"$in": ["A", "B"]}}
    w = build_where(["A"], 2, 4)
    assert w == {"$and": [{"document_id": "A"}, {"page_number": {"$gte": 2}}, {"page_number": {"$lte": 4}}]}
    assert build_where(None, 3, None) == {"page_number": {"$gte": 3}}


def test_lexical_reranker_scores():
    lr = LexicalReranker()
    s = lr.score("annual leave days", ["annual leave is 24 days", "hotel prices in london", ""])
    assert s[0] > 0.8 and s[1] == 0 and s[2] == 0
    assert lr.score("the of", ["anything"]) == [0.0]
    assert lr.score("annual leave", ["annual leave"])[0] > 0.95
