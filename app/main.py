"""FastAPI application factory: middleware, error handlers, lifespan warm-up."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import routes_ask, routes_documents, routes_health
from app.config import Settings, get_settings
from app.container import Container, build_container
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.services.embedder import Embedder
from app.services.llm_client import LLMClient

log = get_logger("main")

DESCRIPTION = """
Ask questions about your own PDFs. Answers are grounded in the documents and carry
**citations (document + page)**. Everything runs locally with Ollama.

Click **Authorize** and paste your `X-API-Key` to try the endpoints below.
"""


def _error(status: int, code: str, message: str, request: Request) -> JSONResponse:
    rid = getattr(request.state, "request_id", None)
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": rid}})


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = build_container(settings, embedder, llm)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c: Container = app.state.container
        if not settings.api_key_list:
            log.warning("no_api_keys_configured", hint="Set API_KEYS in .env; all requests get 401")
        resumed = c.ingestion.resume_incomplete()
        warm = asyncio.create_task(_warm_up(c))
        log.info("startup", resumed_ingestions=resumed, chunks=c.store.count(), reranker=c.reranker.name)
        yield
        warm.cancel()
        c.ingestion.shutdown()
        await c.llm.aclose()
        c.embedder.close()

    app = FastAPI(
        title="Document Q&A Assistant",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.container = container

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
        expose_headers=["X-Request-ID"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID", "")[:64] or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=rid)
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            response.headers.setdefault("Cache-Control", "no-store")
        if request.url.path not in ("/health",):
            log.info(
                "request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            log.warning("app_error", code=exc.code, status=exc.status_code)
        return _error(exc.status_code, exc.code, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # loc + message only: never echo the submitted values back
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}" for e in exc.errors()[:5]
        )
        return _error(422, "validation_error", problems or "Invalid request.", request)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error(exc.status_code, codes.get(exc.status_code, "http_error"), str(exc.detail), request)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong on the server.", request)

    app.include_router(routes_health.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_ask.router)
    return app


async def _warm_up(c: Container) -> None:
    """Load models so the first real request is fast (roadmap section 7.1)."""
    try:
        await asyncio.to_thread(c.embedder.embed_query, "warm up")
        await c.llm.warm_up()
        warm = getattr(c.reranker, "warm_up", None)
        if warm is not None:
            await asyncio.to_thread(warm)
        log.info("warm_up_done")
    except Exception as exc:
        log.warning("warm_up_failed", error=type(exc).__name__)


def get_app() -> FastAPI:
    """Factory used by uvicorn: `uvicorn app.main:get_app --factory`."""
    return create_app()
