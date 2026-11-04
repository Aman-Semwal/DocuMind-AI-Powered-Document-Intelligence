"""Document endpoints: upload, list, get, delete, resume, page preview, summary."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from app.api.deps import get_container, rate_limit, require_api_key
from app.container import Container
from app.core.exceptions import (
    Conflict,
    DuplicateDocument,
    FileTooLarge,
    NotFound,
    RateLimited,
    UnsupportedMediaType,
)
from app.core.logging import get_logger
from app.core.security import looks_like_pdf, sanitize_filename, stored_filename
from app.schemas.models import DeleteResult, DocumentList, DocumentOut, ErrorResponse, Progress
from app.services.ingestion import remove_files
from app.services.pdf_extractor import inspect_pdf, render_page_png

log = get_logger("documents")
router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])

DocId = Annotated[str, PathParam(pattern=r"^DOC\d{3,6}$", examples=["DOC001"])]
ALLOWED_MIME = {"application/pdf", "application/x-pdf"}
CHUNK = 1024 * 1024
ERRORS: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    409: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    429: {"model": ErrorResponse},
}


def to_out(doc: dict[str, Any]) -> DocumentOut:
    total = int(doc["pages"])
    done = int(doc["pages_processed"])
    eta: float | None = None
    if doc["status"] == "processing" and doc.get("started_at") and done > 0 and total > done:
        elapsed = max(time.time() - float(doc["started_at"]), 0.001)
        eta = round((total - done) / (done / elapsed), 1)
    pct = 100.0 if doc["status"] == "completed" else (round(100 * done / total, 1) if total else 0.0)
    return DocumentOut(
        document_id=doc["id"],
        filename=doc["filename"],
        status=doc["status"],
        stage=doc["stage"],
        pages=total,
        chunks=int(doc["chunks"]),
        size_bytes=int(doc["size_bytes"]),
        created_at=float(doc["created_at"]),
        error=doc.get("error"),
        resumable=bool(doc.get("resumable")),
        warnings=list(doc.get("warnings", [])),
        progress=Progress(pages_processed=done, total_pages=total, percent=pct, eta_seconds=eta),
        summary=doc.get("summary"),
        suggested_questions=list(doc.get("suggested_questions", [])),
    )


@router.post(
    "/documents/upload",
    response_model=DocumentOut,
    status_code=202,
    responses=ERRORS,
    summary="Upload a PDF (processed in the background)",
    dependencies=[Depends(rate_limit("upload"))],
)
async def upload_document(request: Request, file: Annotated[UploadFile, File(description="A PDF file")]) -> DocumentOut:
    c: Container = get_container(request)
    s = c.settings

    if c.registry.count_active() >= s.max_queued_ingestions:
        raise RateLimited("The ingestion queue is full. Try again in a minute.")

    display_name = sanitize_filename(file.filename)
    if not display_name.lower().endswith(".pdf"):
        raise UnsupportedMediaType("Only files with a .pdf extension are accepted.")
    if (file.content_type or "").lower() not in ALLOWED_MIME:
        raise UnsupportedMediaType("The content type must be application/pdf.")

    s.upload_dir.mkdir(parents=True, exist_ok=True)
    final_path = s.upload_dir / stored_filename()  # server-generated; client path ignored
    part_path = final_path.with_suffix(".part")
    sha = hashlib.sha256()
    size = 0
    first = True
    try:
        with part_path.open("wb") as out:
            while chunk := await file.read(CHUNK):
                if first:
                    first = False
                    if not looks_like_pdf(chunk[:1024]):
                        raise UnsupportedMediaType("The file content is not a PDF.")
                size += len(chunk)
                if size > s.max_file_bytes:
                    raise FileTooLarge(f"The file exceeds the {s.max_file_mb} MB limit.")
                sha.update(chunk)
                out.write(chunk)
        if size == 0:
            raise UnsupportedMediaType("The file is empty.")
        # raises 422 for corrupt / encrypted / too long files
        pages = (await asyncio.to_thread(inspect_pdf, part_path, s.max_pages)).pages
        part_path.rename(final_path)
        try:
            doc = c.registry.create_document(
                sha256=sha.hexdigest(),
                filename=display_name,
                stored_path=str(final_path),
                size_bytes=size,
                pages=pages,
            )
        except DuplicateDocument:
            final_path.unlink(missing_ok=True)
            raise
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    c.ingestion.submit(doc["id"])
    log.info("upload_accepted", doc=doc["id"], size=size, pages=pages)
    return to_out(c.registry.get(doc["id"]))


@router.get("/documents", response_model=DocumentList, summary="List documents", responses=ERRORS)
def list_documents(request: Request, _: None = Depends(rate_limit("default"))) -> DocumentList:
    c = get_container(request)
    docs = [to_out(d) for d in c.registry.list_documents()]
    return DocumentList(documents=docs, count=len(docs))


@router.get("/documents/{doc_id}", response_model=DocumentOut, summary="Document status and progress", responses=ERRORS)
def get_document(doc_id: DocId, request: Request, _: None = Depends(rate_limit("default"))) -> DocumentOut:
    return to_out(get_container(request).registry.get(doc_id))


@router.post(
    "/documents/{doc_id}/resume",
    response_model=DocumentOut,
    status_code=202,
    responses=ERRORS,
    summary="Resume an interrupted ingestion from its last checkpoint",
)
def resume_document(doc_id: DocId, request: Request, _: None = Depends(rate_limit("default"))) -> DocumentOut:
    c = get_container(request)
    doc = c.registry.get(doc_id)
    if doc["status"] != "failed" or not doc["resumable"]:
        raise Conflict("Only failed, resumable documents can be resumed.")
    if c.ingestion.is_running(doc_id):
        raise Conflict("This document is already being processed.")
    c.registry.update(doc_id, status="pending", stage="queued", error=None)
    c.ingestion.submit(doc_id)
    return to_out(c.registry.get(doc_id))


@router.delete(
    "/documents/{doc_id}",
    response_model=DeleteResult,
    responses=ERRORS,
    summary="Delete a document (vectors, file and record)",
)
def delete_document(doc_id: DocId, request: Request, _: None = Depends(rate_limit("default"))) -> DeleteResult:
    c = get_container(request)
    doc = c.registry.get(doc_id)
    if c.ingestion.is_running(doc_id):
        raise Conflict("The document is still being processed. Try again when it finishes.")
    c.store.delete_document(doc_id)
    c.bm25.remove_document(doc_id)
    remove_files(c.settings, doc)
    c.registry.delete_document_insights(doc_id)
    c.registry.delete(doc_id)
    c.registry.bump_version()
    c.cache.clear()
    log.info("document_deleted", doc=doc_id)
    return DeleteResult(document_id=doc_id)


@router.get("/documents/{doc_id}/summary", summary="Auto-generated summary and suggested questions", responses=ERRORS)
def get_summary(doc_id: DocId, request: Request, _: None = Depends(rate_limit("default"))) -> dict[str, Any]:
    doc = get_container(request).registry.get(doc_id)
    return {
        "document_id": doc_id,
        "summary": doc.get("summary"),
        "suggested_questions": doc.get("suggested_questions", []),
    }


@router.get(
    "/documents/{doc_id}/pages/{page}/preview",
    summary="PNG preview of one page (used to show the cited page)",
    responses={**ERRORS, 200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def page_preview(
    doc_id: DocId,
    page: Annotated[int, PathParam(ge=1, le=100_000)],
    request: Request,
    _: None = Depends(rate_limit("default")),
) -> Response:
    c = get_container(request)
    doc = c.registry.get(doc_id)
    if page > int(doc["pages"]):
        raise NotFound(f"Page {page} does not exist in {doc_id}.")
    cache_file: Path = c.settings.preview_dir / doc_id / f"{page}.png"
    if cache_file.exists():
        return FileResponse(cache_file, media_type="image/png")
    try:
        png = await asyncio.to_thread(render_page_png, Path(doc["stored_path"]), page)
    except FileNotFoundError as exc:
        raise NotFound("The stored file is missing.") from exc
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(png)
    return Response(png, media_type="image/png")
