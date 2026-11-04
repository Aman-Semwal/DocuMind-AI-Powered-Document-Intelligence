"""The question-answering pipeline (plain and streaming share the same steps)."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.config import NOT_AVAILABLE_TEXT, Settings
from app.core.exceptions import Conflict, NotFound
from app.core.logging import get_logger
from app.core.security import sanitize_question
from app.core.timing import StageTimer
from app.db.registry import Registry
from app.schemas.models import AskResponse, CitationOut, SourceOut, VerificationOut
from app.services.cache import LRUCache, SessionStore
from app.services.citation import (
    build_citations,
    is_not_available,
    strip_invalid_markers,
)
from app.services.llm_client import LLMClient
from app.services.prompt_builder import (
    Passage,
    build_messages,
    build_rewrite_messages,
    fit_to_budget,
)
from app.services.query_planner import decompose, is_summary_question
from app.services.retriever import RetrievalResult, Retriever
from app.services.verifier import Verification, confidence_label, verify_answer

log = get_logger("rag")

MAX_EXPANDED_CHARS = 1800
NO_DOCS_TEXT = NOT_AVAILABLE_TEXT
_FOLLOW_UP_PREFIX = re.compile(r"^\s*(and|also|then|what about|how about|why is that)\b", re.I)
_FOLLOW_UP_REFERENCE = re.compile(r"\b(it|they|them|this|that|these|those|same|former|latter|above|previous)\b", re.I)


def needs_history_rewrite(question: str) -> bool:
    """Rewrite likely follow-ups; standalone questions need no extra model call."""
    words = question.split()
    return bool(_FOLLOW_UP_PREFIX.search(question) or _FOLLOW_UP_REFERENCE.search(question) or len(words) <= 3)


@dataclass
class AskParams:
    question: str
    top_k: int = 5
    document_ids: list[str] | None = None
    page_from: int | None = None
    page_to: int | None = None
    session_id: str | None = None
    deep_mode: bool = False


@dataclass
class Prepared:
    question_id: str
    question: str  # sanitised original
    effective_question: str
    params: AskParams
    timer: StageTimer
    cache_key: tuple[Any, ...] | None = None
    early: AskResponse | None = None
    messages: list[dict[str, str]] = field(default_factory=list)
    passages: list[Passage] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None
    document_ids: list[str] = field(default_factory=list)


class RagPipeline:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        retriever: Retriever,
        llm: LLMClient,
        cache: LRUCache,
        sessions: SessionStore,
    ) -> None:
        self.s = settings
        self.registry = registry
        self.retriever = retriever
        self.llm = llm
        self.cache = cache
        self.sessions = sessions

    # ----------------------------------------------------------------- helpers
    def _response(
        self,
        prep: Prepared,
        *,
        answer: str,
        answered: bool,
        confidence: str,
        citations: list[CitationOut] | None = None,
        sources: list[SourceOut] | None = None,
        verification: Verification | None = None,
        cached: bool = False,
    ) -> AskResponse:
        return AskResponse(
            question_id=prep.question_id,
            answer=answer,
            answered=answered,
            confidence=confidence,  # type: ignore[arg-type]
            citations=citations or [],
            sources=sources or [],
            verification=(
                VerificationOut(
                    ok=verification.ok,
                    support_ratio=verification.support_ratio,
                    unsupported_sentences=verification.unsupported_sentences,
                    unverified_numbers=verification.unverified_numbers,
                    unverified_entities=verification.unverified_entities,
                )
                if verification
                else None
            ),
            timings=prep.timer.as_dict(),
            cached=cached,
            model=self.llm.model_name,
        )

    def _not_available(self, prep: Prepared, text: str = NOT_AVAILABLE_TEXT) -> AskResponse:
        return self._response(prep, answer=text, answered=False, confidence="none")

    def _scope(self, params: AskParams) -> tuple[list[str], list[str] | None]:
        """Return (document ids to search, explicit filter or None)."""
        completed = self.registry.completed_ids()
        if not params.document_ids:
            return completed, None
        wanted = list(dict.fromkeys(params.document_ids))
        for d in wanted:
            if not self.registry.exists(d):
                raise NotFound(f"Document {d} was not found.")
        ready = [d for d in wanted if d in completed]
        if not ready:
            raise Conflict("The selected documents are not ready yet.")
        return ready, ready

    def _summary_answer(self, prep: Prepared, scope: list[str]) -> AskResponse | None:
        docs = [self.registry.get(d) for d in scope]
        docs = [d for d in docs if d.get("summary")]
        if not docs:
            return None
        if len(docs) == 1:
            text = docs[0]["summary"]
        else:
            text = "\n\n".join(f"**{d['filename']}**: {d['summary']}" for d in docs)
        cites = [CitationOut(document_id=d["id"], document=d["filename"], page=None) for d in docs]
        sources = [
            SourceOut(
                number=i + 1,
                document_id=d["id"],
                document=d["filename"],
                page=None,
                snippet=d["summary"][:400],
                score=1.0,
            )
            for i, d in enumerate(docs)
        ]
        return self._response(prep, answer=text, answered=True, confidence="high", citations=cites, sources=sources)

    def _build_passages(self, result: RetrievalResult, top_k: int) -> tuple[list[Passage], dict[int, float]]:
        passages: list[Passage] = []
        scores: dict[int, float] = {}
        cross = self.retriever.reranker.is_cross_encoder
        for n, c in enumerate(result.candidates[:top_k], start=1):
            body = c.body
            if self.s.neighbor_expansion and len(body) < MAX_EXPANDED_CHARS // 2:
                prev, nxt = self.retriever.neighbors(c)
                expanded = "\n".join(x for x in (prev, body, nxt) if x)
                body = expanded if len(expanded) <= MAX_EXPANDED_CHARS else body
            passages.append(Passage(n, c.document_id, c.filename, c.page, c.section, body, c.chunk_id))
            scores[n] = round(c.rerank if cross else max(c.vector_score or 0.0, c.lexical), 3)
        return fit_to_budget(passages, self.s.context_token_budget), scores

    # ----------------------------------------------------------------- prepare
    async def _prepare(self, params: AskParams) -> Prepared:
        timer = StageTimer()
        question = sanitize_question(params.question)
        prep = Prepared(
            question_id=self.registry.new_question_id(),
            question=question,
            effective_question=question,
            params=params,
            timer=timer,
        )
        scope, explicit = self._scope(params)
        prep.document_ids = scope
        if not scope:
            prep.early = self._not_available(prep, NO_DOCS_TEXT)
            return prep

        with timer.stage("cache_ms"):
            key = LRUCache.make_key(
                question,
                self.registry.docset_version(),
                params.top_k,
                explicit,
                params.deep_mode,
                params.page_from,
                params.page_to,
            )
            prep.cache_key = key
            hit = self.cache.get(key) if not self.sessions.history(params.session_id) else None
        if hit is not None:
            t = timer.as_dict()
            resp: AskResponse = hit.model_copy(update={"question_id": prep.question_id, "cached": True, "timings": t})
            prep.early = resp
            return prep

        if is_summary_question(question) and not params.page_from:
            summary = self._summary_answer(prep, scope)
            if summary is not None:
                prep.early = summary
                return prep

        history = self.sessions.history(params.session_id)
        if history and needs_history_rewrite(question):
            with timer.stage("rewrite_ms"):
                try:
                    rewritten = await self.llm.chat(build_rewrite_messages(history, question), max_tokens=64)
                    rewritten = sanitize_question(rewritten.strip().strip('"'))
                    if 3 <= len(rewritten) <= 400:
                        prep.effective_question = rewritten
                except Exception as exc:
                    log.warning("rewrite_failed", error=type(exc).__name__)

        subqueries = decompose(prep.effective_question, self.s.max_subqueries)
        with timer.stage("retrieve_ms"):
            result = await asyncio.to_thread(
                self.retriever.retrieve,
                prep.effective_question,
                subqueries,
                document_ids=scope,
                page_from=params.page_from,
                page_to=params.page_to,
                final_k=params.top_k,
            )
        timer.merge(result.timings)
        prep.retrieval = result

        if not result.passed or not result.candidates:
            prep.early = self._not_available(prep)  # early exit: no LLM call
            return prep
        has_and = any(term.strip(".,?!;:") == "and" for term in prep.effective_question.casefold().split())
        if not subqueries and not has_and and result.agreement and result.best_lexical >= 0.6:
            # A strong direct match is clearer without unrelated passages competing for attention.
            # Use a copy to avoid mutating the shared result object.
            result = RetrievalResult(
                candidates=result.candidates[:1],
                best_vector_sim=result.best_vector_sim,
                best_rerank=result.best_rerank,
                best_lexical=result.best_lexical,
                passed=result.passed,
                agreement=result.agreement,
                timings=result.timings,
            )
            prep.retrieval = result

        with timer.stage("expand_ms"):
            passages, scores = await asyncio.to_thread(self._build_passages, result, params.top_k)
        if not passages:
            prep.early = self._not_available(prep)
            return prep
        prep.passages, prep.scores = passages, scores
        prep.messages = build_messages(prep.effective_question, passages)
        return prep

    # ---------------------------------------------------------------- finalize
    def _finalize(self, prep: Prepared, raw_answer: str) -> AskResponse:
        timer = prep.timer
        passages = prep.passages
        valid = {p.number for p in passages}
        answer = strip_invalid_markers(raw_answer.strip(), valid)

        if not answer or is_not_available(answer):
            resp = self._not_available(prep)
            self._remember(prep, resp)
            return resp

        citations, used = build_citations(answer, passages)
        verification: Verification | None = None
        if self.s.verify_enabled:
            with timer.stage("verify_ms"):
                context_passages = used or passages
                verification = verify_answer(
                    answer,
                    [f"{p.section}\n{p.text}" for p in context_passages],
                    question=prep.effective_question,
                )
            if not verification.ok:
                resp = self._not_available(prep)
                self._remember(prep, resp)
                return resp

        r = prep.retrieval
        if r is None:  # only reachable if _finalize is called without a retrieval step
            raise RuntimeError("retrieval result missing")
        cross = self.retriever.reranker.is_cross_encoder
        evidence = r.best_rerank if cross else r.best_vector_sim
        conf = confidence_label(
            evidence=evidence,
            high_threshold=0.6 if cross else self.s.evidence_high,
            low_threshold=0.35 if cross else self.s.min_similarity,
            agreement=r.agreement,
            verification=verification,
        )
        sources = [
            SourceOut(
                number=p.number,
                document_id=p.document_id,
                document=p.filename,
                page=p.page,
                section=p.section,
                snippet=p.text[:400],
                score=prep.scores.get(p.number, 0.0),
            )
            for p in used
        ]
        resp = self._response(
            prep,
            answer=answer,
            answered=True,
            confidence=conf,
            citations=[CitationOut(document_id=c.document_id, document=c.document, page=c.page) for c in citations],
            sources=sources,
            verification=verification,
        )
        if prep.cache_key is not None and not self.sessions.history(prep.params.session_id):
            self.cache.put(prep.cache_key, resp)
        self._remember(prep, resp)
        return resp

    def _remember(self, prep: Prepared, resp: AskResponse) -> None:
        self.sessions.add(prep.params.session_id, prep.question, resp.answer)
        self._log(prep, resp)

    def _log(self, prep: Prepared, resp: AskResponse) -> None:
        try:
            document_ids = sorted({source.document_id for source in resp.sources} or set(prep.document_ids))
            verification = (
                {
                    "ok": resp.verification.ok,
                    "support_ratio": resp.verification.support_ratio,
                    "unsupported_claims": len(resp.verification.unsupported_sentences),
                    "unverified_number_count": len(resp.verification.unverified_numbers),
                    "unverified_entity_count": len(resp.verification.unverified_entities),
                }
                if resp.verification
                else None
            )
            self.registry.log_query(
                question_id=resp.question_id,
                question=prep.question,
                answered=resp.answered,
                confidence=resp.confidence,
                cached=resp.cached,
                timings=resp.timings,
                document_ids=document_ids,
                verification=verification,
                citations=[
                    f"{citation.document} p.{citation.page}" if citation.page is not None else citation.document
                    for citation in resp.citations
                ],
            )
        except Exception as exc:  # logging must never break an answer
            log.warning("query_log_failed", error=type(exc).__name__)
        # per-stage timings only: never the question or any document text
        log.info("ask", answered=resp.answered, confidence=resp.confidence, cached=resp.cached, **resp.timings)

    # --------------------------------------------------------------------- API
    async def ask(self, params: AskParams) -> AskResponse:
        prep = await self._prepare(params)
        if prep.early is not None:
            resp = prep.early
            resp = resp.model_copy(update={"timings": {**resp.timings, "total_ms": prep.timer.total_ms()}})
            if not resp.cached:
                self._remember(prep, resp)
            else:
                self._log(prep, resp)
            return resp
        with prep.timer.stage("llm_ms"):
            raw = await self.llm.chat(prep.messages, deep=params.deep_mode)
        return self._finalize(prep, raw)

    async def ask_stream(self, params: AskParams) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        prep = await self._prepare(params)
        if prep.early is not None:
            resp = prep.early.model_copy(update={"timings": {**prep.early.timings, "total_ms": prep.timer.total_ms()}})
            yield "token", {"t": resp.answer}
            if resp.cached:
                self._log(prep, resp)
            else:
                self._remember(prep, resp)
            yield "final", resp.model_dump()
            return

        parts: list[str] = []
        t0 = time.perf_counter()
        first = True
        async for piece in self.llm.stream(prep.messages, deep=params.deep_mode):
            if first:
                prep.timer.add("first_token_ms", (time.perf_counter() - t0) * 1000)
                first = False
            parts.append(piece)
            yield "token", {"t": piece}
        prep.timer.add("llm_ms", (time.perf_counter() - t0) * 1000)
        resp = self._finalize(prep, "".join(parts))
        yield "final", resp.model_dump()
