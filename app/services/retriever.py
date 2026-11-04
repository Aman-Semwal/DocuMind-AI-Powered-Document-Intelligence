"""Hybrid retrieval: vector top-N + BM25 top-N -> Reciprocal Rank Fusion -> rerank -> MMR."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.core.timing import StageTimer
from app.services.chunker import strip_section_prefix
from app.services.embedder import Embedder
from app.services.reranker import LexicalReranker, Reranker
from app.services.textutil import stemmed_set
from app.services.vector_store import BM25Index, ChromaStore, Hit, build_where

RRF_K = 60


@dataclass
class Candidate:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    idx: int
    section: str
    kind: str
    text: str  # stored text (includes section prefix)
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf: float = 0.0
    rerank: float = 0.0
    lexical: float = 0.0
    in_vector: bool = False
    in_bm25: bool = False

    @property
    def body(self) -> str:
        return strip_section_prefix(self.text)


@dataclass
class RetrievalResult:
    candidates: list[Candidate] = field(default_factory=list)
    best_vector_sim: float = 0.0
    best_rerank: float = 0.0
    best_lexical: float = 0.0
    passed: bool = False
    agreement: bool = False
    timings: dict[str, float] = field(default_factory=dict)


def _from_hit(h: Hit) -> Candidate:
    m = h.metadata
    return Candidate(
        chunk_id=h.chunk_id,
        document_id=str(m.get("document_id", "")),
        filename=str(m.get("filename", "")),
        page=int(m.get("page_number", 0)),
        idx=int(m.get("idx", 0)),
        section=str(m.get("section", "")),
        kind=str(m.get("kind", "text")),
        text=h.text,
    )


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mmr_select(cands: list[Candidate], k: int, lam: float) -> list[Candidate]:
    """Maximal marginal relevance using token-set Jaccard as the redundancy measure.

    `cands` must already be ordered by relevance (best first). Near duplicates (>0.85) are dropped.
    Optimized: O(n·k) instead of O(n²·k) by tracking max similarity incrementally.
    """
    if len(cands) <= 1:
        return cands[:k]
    sets = {c.chunk_id: stemmed_set(c.body) for c in cands}
    n = len(cands)
    rel = {c.chunk_id: 1.0 - i / n for i, c in enumerate(cands)}  # rank-based relevance
    chosen: list[Candidate] = [cands[0]]
    # Track max similarity to any chosen item, updated incrementally
    max_sim: dict[str, float] = {}
    for c in cands[1:]:
        max_sim[c.chunk_id] = _jaccard(sets[c.chunk_id], sets[cands[0].chunk_id])

    remaining = list(cands[1:])
    while remaining and len(chosen) < k:
        last_chosen = chosen[-1]
        last_set = sets[last_chosen.chunk_id]
        # Update max similarities with the last chosen item
        for c in remaining:
            sim_to_last = _jaccard(sets[c.chunk_id], last_set)
            if sim_to_last > max_sim.get(c.chunk_id, 0.0):
                max_sim[c.chunk_id] = sim_to_last

        best: Candidate | None = None
        best_score = -1e9
        best_idx = -1
        for idx, c in enumerate(remaining):
            sim = max_sim.get(c.chunk_id, 0.0)
            if sim > 0.85:
                continue
            score = lam * rel[c.chunk_id] - (1 - lam) * sim
            if score > best_score:
                best, best_score, best_idx = c, score, idx
        if best is None:
            break
        chosen.append(best)
        remaining.pop(best_idx)
    return chosen


class Retriever:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: ChromaStore,
        bm25: BM25Index,
        reranker: Reranker,
    ) -> None:
        self.s = settings
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self._lexical = LexicalReranker()

    def retrieve(
        self,
        question: str,
        subqueries: list[str] | None = None,
        *,
        document_ids: list[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
        final_k: int | None = None,
    ) -> RetrievalResult:
        timer = StageTimer()
        final_k = final_k or self.s.final_k
        queries = list(dict.fromkeys([question, *(subqueries or [])]))
        # Roadmap 2.2: the question must be embedded with the model used at ingestion.
        self.store.assert_model(self.embedder.model_name)
        where = build_where(document_ids, page_from, page_to)
        id_set = set(document_ids) if document_ids is not None else None
        if document_ids is not None and not document_ids:
            return RetrievalResult(timings=timer.as_dict())

        cands: dict[str, Candidate] = {}
        for q in queries:
            with timer.stage("embed_ms"):
                vec = self.embedder.embed_query(q)
            with timer.stage("vector_ms"):
                v_hits = self.store.query(vec, self.s.candidates_k, where)
            with timer.stage("bm25_ms"):
                b_hits = (
                    self.bm25.search(q, self.s.candidates_k, id_set, page_from, page_to)
                    if self.s.hybrid_enabled
                    else []
                )
            with timer.stage("fusion_ms"):
                for rank, h in enumerate(v_hits):
                    c = cands.setdefault(h.chunk_id, _from_hit(h))
                    c.in_vector = True
                    c.vector_score = max(c.vector_score or -1.0, h.score)
                    c.rrf += 1.0 / (RRF_K + rank + 1)
                for rank, h in enumerate(b_hits):
                    c = cands.setdefault(h.chunk_id, _from_hit(h))
                    c.in_bm25 = True
                    c.bm25_score = max(c.bm25_score or 0.0, h.score)
                    c.rrf += 1.0 / (RRF_K + rank + 1)

        if not cands:
            return RetrievalResult(timings=timer.as_dict())

        pool = sorted(cands.values(), key=lambda c: c.rrf, reverse=True)[: self.s.candidates_k]

        with timer.stage("rerank_ms"):
            bodies = [c.body for c in pool]
            lexical = self._lexical.score(question, bodies)
            for c, lx in zip(pool, lexical, strict=True):
                c.lexical = lx
            if self.reranker.is_cross_encoder:
                for c, sc in zip(pool, self.reranker.score(question, bodies), strict=True):
                    c.rerank = sc
                ranked = sorted(pool, key=lambda c: (c.rerank, c.rrf), reverse=True)
            else:
                max_rrf = max(c.rrf for c in pool) or 1.0
                for c in pool:
                    c.rerank = c.lexical
                if self.reranker.name == "none":
                    ranked = pool
                else:
                    ranked = sorted(pool, key=lambda c: 0.6 * c.lexical + 0.4 * (c.rrf / max_rrf), reverse=True)

        with timer.stage("mmr_ms"):
            final = mmr_select(ranked[: max(final_k * 3, final_k)], final_k, self.s.mmr_lambda)

        best_vec = max((c.vector_score for c in pool if c.vector_score is not None), default=0.0)
        best_rerank = max((c.rerank for c in pool), default=0.0)
        best_lex = max((c.lexical for c in pool), default=0.0)
        if self.reranker.is_cross_encoder:
            passed = best_rerank >= self.s.rerank_min_score
        else:
            passed = best_vec >= self.s.min_similarity or best_lex >= 0.75
        agreement = bool(final) and final[0].in_vector and final[0].in_bm25

        return RetrievalResult(
            candidates=final,
            best_vector_sim=round(best_vec, 4),
            best_rerank=round(best_rerank, 4),
            best_lexical=round(best_lex, 4),
            passed=passed,
            agreement=agreement,
            timings=timer.as_dict(),
        )

    def neighbors(self, cand: Candidate) -> tuple[str, str]:
        """Previous and next chunk text on the *same page* (parent/neighbour context)."""
        ids = [
            f"{cand.document_id}-p{cand.page:04d}-c{cand.idx - 1:03d}",
            f"{cand.document_id}-p{cand.page:04d}-c{cand.idx + 1:03d}",
        ]
        got = self.store.get_many([ids[0], ids[1]] if cand.idx > 0 else [ids[1]])
        prev = strip_section_prefix(got[ids[0]].text) if ids[0] in got else ""
        nxt = strip_section_prefix(got[ids[1]].text) if ids[1] in got else ""
        return prev, nxt
