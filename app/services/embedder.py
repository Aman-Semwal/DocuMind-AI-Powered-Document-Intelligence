"""Embedder interface, Ollama implementation and an embedding cache wrapper."""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

import httpx

from app.core.exceptions import OllamaUnavailable
from app.db.registry import EmbeddingCache


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def ping(self) -> bool: ...

    def close(self) -> None: ...


class OllamaEmbedder:
    """Calls Ollama /api/embed in batches. Runs on CPU if OLLAMA is configured that way."""

    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int = 64,
        keep_alive: str = "30m",
        timeout_s: float = 60.0,
        retries: int = 2,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.keep_alive = keep_alive
        self.retries = retries
        self._nomic = "nomic" in model.lower()
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
        )

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model_name, "input": texts, "keep_alive": self.keep_alive}
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._client.post("/api/embed", json=payload)
                if resp.status_code == 404:
                    raise OllamaUnavailable(
                        f"Embedding model '{self.model_name}' is not installed. Run: ollama pull {self.model_name}"
                    )
                resp.raise_for_status()
                embeddings = resp.json()["embeddings"]
                if len(embeddings) != len(texts):
                    raise OllamaUnavailable("Embedding service returned an unexpected result.")
                return [list(map(float, e)) for e in embeddings]
            except OllamaUnavailable:
                raise
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.HTTPStatusError) as exc:
                last = exc
                time.sleep(0.4 * (2**attempt))
        raise OllamaUnavailable() from last

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "search_document: " if self._nomic else ""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [prefix + t for t in texts[i : i + self.batch_size]]
            out.extend(self._post(batch))
        return out

    def embed_query(self, text: str) -> list[float]:
        prefix = "search_query: " if self._nomic else ""
        return self._post([prefix + text])[0]

    def ping(self) -> bool:
        try:
            return self._client.get("/api/tags").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


def cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


class CachedEmbedder:
    """Wraps any Embedder: identical chunk texts are looked up instead of re-embedded."""

    def __init__(self, inner: Embedder, cache: EmbeddingCache) -> None:
        self.inner = inner
        self.cache = cache
        self.model_name = inner.model_name
        self.hits = 0
        self.misses = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [cache_key(t, self.model_name) for t in texts]
        cached = self.cache.get_many(list(dict.fromkeys(keys)))
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in cached:
                missing[k] = t
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            vectors = self.inner.embed_documents(list(missing.values()))
            fresh = dict(zip(missing.keys(), vectors, strict=True))
            self.cache.put_many(fresh)
            cached.update(fresh)
        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)

    def ping(self) -> bool:
        return self.inner.ping()

    def close(self) -> None:
        self.inner.close()
