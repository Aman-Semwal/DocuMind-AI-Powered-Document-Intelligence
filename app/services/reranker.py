"""Rerankers. Scores are normalised to 0..1.

- CrossEncoderReranker: bge-reranker-base on CPU (needs `pip install sentence-transformers`).
- LexicalReranker: dependency-free fallback based on query-term coverage.
"""

from __future__ import annotations

import itertools
import math
import threading
from typing import Protocol

from app.config import Settings
from app.core.logging import get_logger
from app.services.textutil import content_tokens, stem

log = get_logger("reranker")


class Reranker(Protocol):
    name: str
    is_cross_encoder: bool

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class NoopReranker:
    name = "none"
    is_cross_encoder = False

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


class LexicalReranker:
    """Fraction of distinct query terms (stemmed) found in the passage, plus a phrase bonus."""

    name = "lexical"
    is_cross_encoder = False

    def score(self, query: str, texts: list[str]) -> list[float]:
        q_terms = [stem(t) for t in dict.fromkeys(content_tokens(query))]
        if not q_terms:
            return [0.0] * len(texts)
        bigrams = {f"{a} {b}" for a, b in itertools.pairwise(q_terms)}
        out = []
        for text in texts:
            stems = [stem(t) for t in content_tokens(text)]
            present = set(stems)
            coverage = sum(1 for t in q_terms if t in present) / len(q_terms)
            joined = " ".join(stems)
            bonus = 0.15 * (sum(1 for bg in bigrams if bg in joined) / len(bigrams)) if bigrams else 0.0
            out.append(min(1.0, coverage * 0.85 + bonus))
        return out


class CrossEncoderReranker:
    name = "cross_encoder"
    is_cross_encoder = True

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self) -> object:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder  # lazy: heavy import

                self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
            return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        raw = model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)  # type: ignore[attr-defined]
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]

    def warm_up(self) -> None:
        self.score("warm up", ["warm up"])


def build_reranker(settings: Settings) -> Reranker:
    if not settings.rerank_enabled or settings.rerank_backend == "none":
        return NoopReranker()
    backend = settings.rerank_backend
    if backend in ("auto", "cross_encoder"):
        try:
            import sentence_transformers  # noqa: F401

            return CrossEncoderReranker(settings.rerank_model)
        except ImportError:
            if backend == "cross_encoder":
                raise
            log.warning("cross_encoder_unavailable", hint="pip install sentence-transformers")
    return LexicalReranker()
