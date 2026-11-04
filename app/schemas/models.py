"""Pydantic request/response models (also drive the Swagger docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Confidence = Literal["high", "medium", "low", "none"]
DocStatus = Literal["pending", "processing", "completed", "failed"]


class ErrorBody(BaseModel):
    code: str = Field(examples=["duplicate_document"])
    message: str = Field(examples=["This file was already uploaded as DOC001."])
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class Progress(BaseModel):
    pages_processed: int
    total_pages: int
    percent: float
    eta_seconds: float | None = None


class DocumentOut(BaseModel):
    document_id: str = Field(examples=["DOC001"])
    filename: str = Field(examples=["HR_Leave_Policy.pdf"])
    status: DocStatus
    stage: str = Field(description="uploaded, queued, extracting, embedding, summarizing, ready, failed")
    pages: int
    chunks: int
    size_bytes: int
    created_at: float
    error: str | None = None
    resumable: bool = False
    warnings: list[str] = []
    progress: Progress
    summary: str | None = None
    suggested_questions: list[str] = []


class DocumentList(BaseModel):
    documents: list[DocumentOut]
    count: int


class DeleteResult(BaseModel):
    document_id: str
    deleted: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["How many days of annual leave do I get?"])
    top_k: int = Field(default=5, ge=1, le=10, description="Passages sent to the LLM")
    document_ids: list[str] | None = Field(default=None, max_length=50, description="Restrict to these documents")
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_\-]+$")
    deep_mode: bool = Field(default=False, description="Use the larger (slower) model")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v

    @field_validator("document_ids")
    @classmethod
    def _ids(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for d in v:
                if not (d.startswith("DOC") and d[3:].isdigit() and len(d) <= 12):
                    raise ValueError("document ids look like DOC001")
        return v


class CitationOut(BaseModel):
    document_id: str
    document: str
    page: int | None = None


class SourceOut(BaseModel):
    number: int
    document_id: str
    document: str
    page: int | None
    section: str = ""
    snippet: str
    score: float = Field(description="Retrieval relevance 0..1")


class VerificationOut(BaseModel):
    ok: bool
    support_ratio: float
    unsupported_sentences: list[str] = []
    unverified_numbers: list[str] = []
    unverified_entities: list[str] = []


class AskResponse(BaseModel):
    question_id: str
    answer: str
    answered: bool = Field(description="False when the documents do not contain the answer")
    confidence: Confidence
    citations: list[CitationOut]
    sources: list[SourceOut]
    verification: VerificationOut | None = None
    timings: dict[str, float]
    cached: bool = False
    model: str


class FeedbackRequest(BaseModel):
    question_id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_\-]+$")
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    question_id: str
    rating: str


class RecentQuery(BaseModel):
    question_id: str
    ts: float
    question: str
    document_ids: list[str] = Field(default_factory=list)
    answered: bool
    confidence: str
    cached: bool
    rating: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    recent: list[RecentQuery]
    latency_ms: dict[str, float]
    stage_avg_ms: dict[str, float]
    answered_rate: float
    feedback: dict[str, int]
    evaluation: dict[str, object] | None = None
    evaluation_meta: dict[str, object] | None = None
    evaluation_rows: list[dict[str, object]] = Field(default_factory=list)
    cache: dict[str, int]
    benchmark: dict[str, object] | None = None
    uploaded_benchmark: dict[str, object] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    api: bool = True
    llm: bool
    embedder: bool
    vector_store: bool
    llm_model: str
    embed_model: str
    reranker: str
    documents: int
    chunks: int
