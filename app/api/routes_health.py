"""Liveness, readiness, feedback and metrics endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_container, rate_limit, require_api_key
from app.core.exceptions import NotFound
from app.core.timing import percentile
from app.schemas.models import (
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    ReadyResponse,
    RecentQuery,
)

router = APIRouter()
ERRORS: dict[int | str, dict[str, Any]] = {401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}}


def _load_benchmark(path: Path, completed_docs: list[dict[str, Any]]) -> dict[str, object] | None:
    """Load only a benchmark whose source PDFs are present and unchanged."""
    completed_by_name = {str(doc["filename"]): doc for doc in completed_docs}
    if not completed_by_name:
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        not isinstance(loaded, dict)
        or not isinstance(loaded.get("summary"), dict)
        or not isinstance(loaded.get("rows"), list)
    ):
        return None
    metadata = loaded.get("meta", {})
    if not isinstance(metadata, dict):
        return None
    benchmark_docs = set(metadata.get("documents", []))
    if not benchmark_docs or not benchmark_docs.issubset(completed_by_name):
        return None
    hashes = metadata.get("document_hashes", {})
    if isinstance(hashes, dict):
        for name in benchmark_docs:
            expected_hash = hashes.get(name)
            if expected_hash and str(completed_by_name[name].get("sha256")) != expected_hash:
                return None
    return loaded


@router.get("/health", response_model=HealthResponse, tags=["health"], summary="Liveness probe")
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse, tags=["health"], summary="Readiness: LLM, embedder, vector store")
async def ready(request: Request) -> ReadyResponse:
    c = get_container(request)
    llm_ok = await c.llm.ping()
    emb_ok = c.embedder.ping()
    vs_ok = c.store.heartbeat()
    return ReadyResponse(
        ready=llm_ok and emb_ok and vs_ok,
        llm=llm_ok,
        embedder=emb_ok,
        vector_store=vs_ok,
        llm_model=c.llm.model_name,
        embed_model=c.embedder.model_name,
        reranker=c.reranker.name,
        documents=len(c.registry.completed_ids()),
        chunks=c.store.count(),
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["metrics"],
    responses=ERRORS,
    summary="Thumbs up / down for an answer",
    dependencies=[Depends(require_api_key), Depends(rate_limit("default"))],
)
def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    c = get_container(request)
    if not c.registry.query_exists(body.question_id):
        raise NotFound("Unknown question_id.")
    c.registry.set_feedback(body.question_id, body.rating, body.comment)
    return FeedbackResponse(question_id=body.question_id, rating=body.rating)


@router.get(
    "/metrics/recent",
    response_model=MetricsResponse,
    tags=["metrics"],
    responses=ERRORS,
    summary="PDF-attributed recent questions, runtime metrics and verifier evaluation",
    dependencies=[Depends(require_api_key), Depends(rate_limit("default"))],
)
def recent_metrics(request: Request, limit: int = 25) -> MetricsResponse:
    c = get_container(request)
    completed_docs = [doc for doc in c.registry.list_documents() if doc["status"] == "completed"]
    benchmark = _load_benchmark(c.settings.eval_results_dir / "latest.json", completed_docs)
    uploaded_benchmark = _load_benchmark(
        c.settings.eval_results_dir / "uploaded_policies" / "latest.json", completed_docs
    )
    limit = max(1, min(limit, 200))
    completed_ids = {doc["id"] for doc in completed_docs}
    recent_rows = [
        row
        for row in c.registry.recent_queries(200)
        if row["document_ids"] and set(row["document_ids"]).issubset(completed_ids)
    ]
    fresh = [r for r in recent_rows if not r["cached"]]
    totals = [r["timings"].get("total_ms", 0.0) for r in fresh if r["answered"]]
    stage_sums: dict[str, list[float]] = {}
    for r in fresh:
        for k, v in r["timings"].items():
            if k != "total_ms":
                stage_sums.setdefault(k, []).append(float(v))
    verified_rows = [row for row in recent_rows if row["answered"] and isinstance(row["verification"], dict)]
    evaluation: dict[str, object] | None = None
    evaluation_meta: dict[str, object] | None = None
    evaluation_rows: list[dict[str, object]] = []
    if verified_rows:
        verified_count = len(verified_rows)
        grounded_count = sum(bool(row["verification"]["ok"]) for row in verified_rows)
        citation_count = sum(bool(row["citations"]) for row in verified_rows)
        used_document_ids = {document_id for row in verified_rows for document_id in row["document_ids"]}
        evaluation = {
            "questions": verified_count,
            "groundedness": round(grounded_count / verified_count, 3),
            "support_ratio": round(
                sum(float(row["verification"]["support_ratio"]) for row in verified_rows) / verified_count, 3
            ),
            "unsupported_claims": sum(int(row["verification"]["unsupported_claims"]) for row in verified_rows),
            "citation_coverage": round(citation_count / verified_count, 3),
        }
        evaluation_meta = {
            "basis": "Verified answers to questions searched against completed uploaded PDFs",
            "documents": sorted(doc["filename"] for doc in completed_docs if doc["id"] in used_document_ids),
        }
        evaluation_rows = [
            {
                "id": row["question_id"],
                "question": row["question"],
                "answered": row["answered"],
                "grounded": row["verification"]["ok"],
                "support_ratio": row["verification"]["support_ratio"],
                "unsupported_claims": row["verification"]["unsupported_claims"],
                "unverified_number_count": row["verification"]["unverified_number_count"],
                "unverified_entity_count": row["verification"]["unverified_entity_count"],
                "citations": row["citations"],
                "confidence": row["confidence"],
                "latency_ms": row["timings"].get("total_ms"),
            }
            for row in verified_rows
        ]
    return MetricsResponse(
        recent=[RecentQuery(**{k: r[k] for k in RecentQuery.model_fields if k in r}) for r in recent_rows[:limit]],
        latency_ms={"p50": round(percentile(totals, 50), 1), "p95": round(percentile(totals, 95), 1)},
        stage_avg_ms={k: round(sum(v) / len(v), 1) for k, v in stage_sums.items()},
        answered_rate=(
            round(sum(1 for r in recent_rows if r["answered"]) / len(recent_rows), 3) if recent_rows else 0.0
        ),
        feedback=c.registry.feedback_counts([row["question_id"] for row in recent_rows]),
        evaluation=evaluation,
        evaluation_meta=evaluation_meta,
        evaluation_rows=evaluation_rows,
        cache={
            "hits": sum(row["cached"] for row in recent_rows),
            "misses": sum(not row["cached"] for row in recent_rows),
        },
        benchmark=benchmark,
        uploaded_benchmark=uploaded_benchmark,
    )
