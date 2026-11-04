"""Chroma vector store (persistent, cosine HNSW) and an in-memory BM25 keyword index."""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from app.core.exceptions import EmbeddingMismatch
from app.services.chunker import Chunk
from app.services.textutil import tokenize


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity (1 - distance) or BM25 score


def build_where(document_ids: list[str] | None, page_from: int | None, page_to: int | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if document_ids is not None:
        clauses.append(
            {"document_id": document_ids[0]} if len(document_ids) == 1 else {"document_id": {"$in": document_ids}}
        )
    if page_from is not None:
        clauses.append({"page_number": {"$gte": page_from}})
    if page_to is not None:
        clauses.append({"page_number": {"$lte": page_to}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaStore:
    COLLECTION = "documents"

    def __init__(
        self,
        path: Path,
        embed_model: str,
        *,
        hnsw_m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=chromadb.Settings(anonymized_telemetry=False))
        self.embed_model = embed_model
        self.collection = self.client.get_or_create_collection(
            self.COLLECTION,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "max_neighbors": hnsw_m,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                }
            },
            metadata={"embed_model": embed_model},
            embedding_function=None,
        )
        self._max_batch = min(self.client.get_max_batch_size(), 2000)
        self._lock = threading.Lock()
        self.stored_model: str | None = (self.collection.metadata or {}).get("embed_model")
        if self.collection.count() == 0 and self.stored_model != embed_model:
            # empty collection: adopt the current model name
            self.collection.modify(metadata={"embed_model": embed_model})
            self.stored_model = embed_model

    def assert_model(self, model: str) -> None:
        """Roadmap 2.2: refuse queries made with a different embedding model."""
        if self.stored_model and self.stored_model != model:
            raise EmbeddingMismatch()

    # ------------------------------------------------------------------ writes
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        with self._lock:
            for i in range(0, len(chunks), self._max_batch):
                part = chunks[i : i + self._max_batch]
                self.collection.upsert(
                    ids=[c.chunk_id for c in part],
                    documents=[c.stored_text for c in part],
                    embeddings=embeddings[i : i + self._max_batch],  # type: ignore[arg-type]
                    metadatas=[c.metadata() for c in part],
                )

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self.collection.delete(where={"document_id": document_id})

    # ------------------------------------------------------------------- reads
    def query(self, embedding: list[float], n: int, where: dict[str, Any] | None) -> list[Hit]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[embedding],  # type: ignore[arg-type]
            n_results=min(n, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = (res["documents"] or [[]])[0]
        metas = (res["metadatas"] or [[]])[0]
        dists = (res["distances"] or [[]])[0]
        return [Hit(i, d, dict(m), 1.0 - float(dist)) for i, d, m, dist in zip(ids, docs, metas, dists, strict=True)]

    def get_many(self, ids: list[str]) -> dict[str, Hit]:
        if not ids:
            return {}
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        docs = res["documents"] or []
        metas = res["metadatas"] or []
        return {i: Hit(i, d, dict(m), 0.0) for i, d, m in zip(res["ids"], docs, metas, strict=True)}

    def count(self) -> int:
        return int(self.collection.count())

    def count_document(self, document_id: str, max_page: int | None = None) -> int:
        where = build_where([document_id], None, max_page)
        return len(self.collection.get(where=where, include=[])["ids"])

    def iter_all(self, batch: int = 5000) -> Any:
        offset = 0
        while True:
            res = self.collection.get(limit=batch, offset=offset, include=["documents", "metadatas"])
            ids = res["ids"]
            if not ids:
                return
            yield from zip(ids, res["documents"] or [], res["metadatas"] or [], strict=True)
            offset += len(ids)

    def heartbeat(self) -> bool:
        try:
            self.client.heartbeat()
            return True
        except Exception:
            return False


class BM25Index:
    """Small in-memory BM25 (Okapi) with incremental add/remove. Rebuilt from Chroma at startup."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._docs: dict[str, tuple[int, dict[str, Any], str]] = {}  # id -> (len, meta, text)
        self._total_len = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, chunk_id: str, text: str, metadata: dict[str, Any]) -> None:
        tokens = tokenize(text)
        with self._lock:
            if chunk_id in self._docs:
                self._remove_unlocked(chunk_id)
            for term, tf in Counter(tokens).items():
                self._postings[term][chunk_id] = tf
            self._docs[chunk_id] = (len(tokens), metadata, text)
            self._total_len += len(tokens)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            self.add(c.chunk_id, c.stored_text, c.metadata())

    def _remove_unlocked(self, chunk_id: str) -> None:
        length, _, text = self._docs.pop(chunk_id)
        self._total_len -= length
        for term in set(tokenize(text)):
            plist = self._postings.get(term)
            if plist is not None:
                plist.pop(chunk_id, None)
                if not plist:
                    del self._postings[term]

    def remove_document(self, document_id: str) -> None:
        with self._lock:
            ids = [i for i, (_, m, _) in self._docs.items() if m.get("document_id") == document_id]
            for i in ids:
                self._remove_unlocked(i)

    def search(
        self,
        query: str,
        n: int,
        document_ids: set[str] | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))
        with self._lock:
            n_docs = len(self._docs)
            if not n_docs or not terms:
                return []
            avg_len = self._total_len / n_docs
            scores: dict[str, float] = defaultdict(float)
            for term in terms:
                plist = self._postings.get(term)
                if not plist:
                    continue
                idf = math.log(1 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
                for cid, tf in plist.items():
                    length, meta, _ = self._docs[cid]
                    if document_ids is not None and meta.get("document_id") not in document_ids:
                        continue
                    page = int(meta.get("page_number", 0))
                    if (page_from and page < page_from) or (page_to and page > page_to):
                        continue
                    denom = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                    scores[cid] += idf * tf * (self.k1 + 1) / denom
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
            return [Hit(cid, self._docs[cid][2], dict(self._docs[cid][1]), sc) for cid, sc in top]

    def rebuild_from(self, store: ChromaStore) -> int:
        count = 0
        for cid, text, meta in store.iter_all():
            self.add(cid, text, dict(meta))
            count += 1
        return count
