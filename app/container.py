"""Wires every service together. Tests can inject fake embedder / LLM."""

from __future__ import annotations

from dataclasses import dataclass

from app.api.ratelimit import RateLimiter
from app.config import Settings
from app.db.registry import EmbeddingCache, Registry
from app.services.cache import LRUCache, SessionStore
from app.services.embedder import CachedEmbedder, Embedder, OllamaEmbedder
from app.services.fakes import FakeEmbedder, FakeLLM
from app.services.ingestion import IngestionManager
from app.services.llm_client import LLMClient, OllamaLLM
from app.services.rag_pipeline import RagPipeline
from app.services.reranker import Reranker, build_reranker
from app.services.retriever import Retriever
from app.services.vector_store import BM25Index, ChromaStore


@dataclass
class Container:
    settings: Settings
    registry: Registry
    embedder: Embedder
    llm: LLMClient
    store: ChromaStore
    bm25: BM25Index
    reranker: Reranker
    retriever: Retriever
    cache: LRUCache
    sessions: SessionStore
    pipeline: RagPipeline
    ingestion: IngestionManager
    limiter: RateLimiter


def build_container(settings: Settings, embedder: Embedder | None = None, llm: LLMClient | None = None) -> Container:
    settings.ensure_dirs()
    registry = Registry(settings.sqlite_path)

    if embedder is None:
        if settings.embedder_backend == "fake":
            base: Embedder = FakeEmbedder()
        else:
            base = OllamaEmbedder(
                settings.ollama_base_url,
                settings.embed_model,
                batch_size=settings.embed_batch_size,
                keep_alive=settings.ollama_keep_alive,
            )
        embedder = CachedEmbedder(base, EmbeddingCache(settings.embed_cache_path))
    if llm is None:
        llm = FakeLLM() if settings.llm_backend == "fake" else OllamaLLM(settings)

    store = ChromaStore(
        settings.chroma_dir,
        embedder.model_name,
        hnsw_m=settings.hnsw_m,
        ef_construction=settings.hnsw_ef_construction,
        ef_search=settings.hnsw_ef_search,
    )
    bm25 = BM25Index()
    bm25.rebuild_from(store)
    reranker = build_reranker(settings)
    retriever = Retriever(settings, embedder, store, bm25, reranker)
    cache = LRUCache(settings.cache_size)
    sessions = SessionStore(settings.session_history_turns)
    pipeline = RagPipeline(settings, registry, retriever, llm, cache, sessions)

    def on_change() -> None:
        cache.clear()

    ingestion = IngestionManager(settings, registry, embedder, store, bm25, llm, on_change)
    return Container(
        settings=settings,
        registry=registry,
        embedder=embedder,
        llm=llm,
        store=store,
        bm25=bm25,
        reranker=reranker,
        retriever=retriever,
        cache=cache,
        sessions=sessions,
        pipeline=pipeline,
        ingestion=ingestion,
        limiter=RateLimiter(),
    )
