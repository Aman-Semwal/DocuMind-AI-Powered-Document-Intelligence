"""Application settings. Every value comes from environment variables or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # --- general -----------------------------------------------------------
    app_env: Literal["dev", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True
    data_dir: Path = Path("data")

    # --- security ----------------------------------------------------------
    api_keys: str = ""  # comma separated list, e.g. "key-one,key-two"
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"
    rate_limit_ask_per_min: int = 30
    rate_limit_upload_per_min: int = 10
    rate_limit_default_per_min: int = 120
    max_question_chars: int = 1000

    # --- backends ----------------------------------------------------------
    llm_backend: Literal["ollama", "fake"] = "ollama"
    embedder_backend: Literal["ollama", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct"
    deep_llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    embed_model: str = "nomic-embed-text"
    ollama_keep_alive: str = "30m"
    llm_timeout_s: float = 90.0
    llm_retries: int = 2
    llm_concurrency: int = 1
    llm_temperature: float = 0.1
    llm_seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 256

    # --- ingestion ---------------------------------------------------------
    max_file_mb: int = 100
    max_pages: int = 1000
    min_chars_per_page: int = 15  # below this on average the PDF is treated as scanned
    ingest_workers: int = 2
    max_queued_ingestions: int = 8
    extract_workers: int = 1  # >1 enables a process pool for page extraction
    extract_window: int = 40  # pages extracted/embedded per window
    tables_enabled: bool = True
    chunk_size: int = 700
    chunk_overlap: int = 100
    embed_batch_size: int = 64
    summaries_enabled: bool = True

    # --- vector index (HNSW) -----------------------------------------------
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

    # --- retrieval ---------------------------------------------------------
    hybrid_enabled: bool = True  # vector + BM25 fusion (False = vector search only)
    candidates_k: int = 30
    final_k: int = 5
    min_similarity: float = 0.60  # best cosine similarity needed to attempt an answer
    rerank_backend: Literal["auto", "cross_encoder", "lexical", "none"] = "auto"
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_enabled: bool = True
    rerank_min_score: float = 0.2  # sigmoid score gate when a cross-encoder is active
    mmr_lambda: float = 0.7
    neighbor_expansion: bool = True
    context_token_budget: int = 3000
    max_subqueries: int = 3
    evidence_high: float = 0.62  # similarity at or above which confidence can be "high"

    # --- answer safeguards -------------------------------------------------
    verify_enabled: bool = True
    cache_size: int = 256
    session_history_turns: int = 8

    # ----------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    # ----------------------------------------------------------------------
    @property
    def api_key_list(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def preview_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "registry.sqlite"

    @property
    def embed_cache_path(self) -> Path:
        return self.data_dir / "embed_cache.sqlite"

    @property
    def eval_results_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "eval" / "results"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.upload_dir, self.chroma_dir, self.preview_dir):
            p.mkdir(parents=True, exist_ok=True)


NOT_AVAILABLE_TEXT = "Information not available in the uploaded documents."


@lru_cache
def get_settings() -> Settings:
    return Settings()
