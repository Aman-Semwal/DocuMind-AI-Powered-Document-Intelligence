"""Question answering: plain JSON and Server-Sent Events streaming."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_container, rate_limit, require_api_key
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.models import AskRequest, AskResponse, ErrorResponse
from app.services.rag_pipeline import AskParams

log = get_logger("ask")
router = APIRouter(tags=["ask"], dependencies=[Depends(require_api_key), Depends(rate_limit("ask"))])

ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def _params(body: AskRequest) -> AskParams:
    return AskParams(
        question=body.question,
        top_k=body.top_k,
        document_ids=body.document_ids,
        page_from=body.page_from,
        page_to=body.page_to,
        session_id=body.session_id,
        deep_mode=body.deep_mode,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses=ERRORS,
    summary="Ask a question about the uploaded documents",
)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    return await get_container(request).pipeline.ask(_params(body))


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/ask/stream",
    responses={**ERRORS, 200: {"content": {"text/event-stream": {}}}},
    summary="Ask with a streamed answer (Server-Sent Events: token*, then final)",
)
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    gen = get_container(request).pipeline.ask_stream(_params(body))
    # Pull the first event before replying so that setup errors (404, 409, 503...) are real HTTP errors.
    first = await gen.__anext__()

    async def events() -> AsyncIterator[str]:
        try:
            yield _sse(*first)
            async for event, data in gen:
                yield _sse(event, data)
        except AppError as exc:
            yield _sse("error", {"code": exc.code, "message": exc.message})
        except Exception:
            log.error("stream_failed", exc_info=True)
            yield _sse("error", {"code": "internal_error", "message": "Something went wrong on the server."})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
