"""Background ingestion: streaming, checkpointed, resumable. Memory stays flat per page."""

from __future__ import annotations

import hashlib
import shutil
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from app.config import Settings
from app.core.exceptions import (
    AppError,
    OllamaUnavailable,
    ScannedPdf,
    UnprocessableDocument,
)
from app.core.logging import get_logger
from app.db.registry import Registry
from app.services.chunker import Chunk, Chunker, SectionTracker
from app.services.embedder import Embedder
from app.services.llm_client import LLMClient
from app.services.pdf_extractor import (
    PageContent,
    edge_lines,
    extract_range,
    inspect_pdf,
    iter_pages,
    normalize_edge,
    repeated_boilerplate,
)
from app.services.summarizer import GROUP_CHARS, MAX_GROUPS, summarize_document
from app.services.text_cleaner import clean_text
from app.services.vector_store import BM25Index, ChromaStore

log = get_logger("ingestion")


class IngestionManager:
    def __init__(
        self,
        settings: Settings,
        registry: Registry,
        embedder: Embedder,
        store: ChromaStore,
        bm25: BM25Index,
        llm: LLMClient,
        on_change: Callable[[], None],
    ) -> None:
        self.s = settings
        self.registry = registry
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.llm = llm
        self.on_change = on_change
        self.chunker = Chunker(settings.chunk_size, settings.chunk_overlap)
        self._pool = ThreadPoolExecutor(max_workers=max(1, settings.ingest_workers), thread_name_prefix="ingest")
        self._proc_pool: ProcessPoolExecutor | None = None
        self._futures: dict[str, object] = {}
        self._lock = threading.Lock()

    # --------------------------------------------------------------- public API
    def submit(self, doc_id: str) -> None:
        with self._lock:
            self._futures[doc_id] = self._pool.submit(self._run_safe, doc_id)

    def resume_incomplete(self) -> int:
        """Called at startup: re-queue documents interrupted by a crash or restart."""
        docs = self.registry.incomplete()
        for d in docs:
            self.submit(d["id"])
        return len(docs)

    def wait(self, doc_id: str, timeout: float = 120.0) -> None:
        """Block until a document finishes (used by tests and scripts)."""
        fut = self._futures.get(doc_id)
        if fut is not None:
            fut.result(timeout=timeout)  # type: ignore[attr-defined]

    def is_running(self, doc_id: str) -> bool:
        fut = self._futures.get(doc_id)
        return bool(fut is not None and not fut.done())  # type: ignore[attr-defined]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self._proc_pool:
            self._proc_pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ worker
    def _run_safe(self, doc_id: str) -> None:
        try:
            self._run(doc_id)
        except UnprocessableDocument as exc:
            log.warning("ingest_rejected", doc=doc_id, code=exc.code)
            self._rollback(doc_id)
            self.registry.update(doc_id, status="failed", stage="failed", error=exc.message, resumable=False)
        except OllamaUnavailable as exc:
            # keep the checkpoint: the user can resume once Ollama is back
            log.warning("ingest_paused", doc=doc_id, code=exc.code)
            self.registry.update(doc_id, status="failed", stage="failed", error=exc.message, resumable=True)
        except AppError as exc:
            self._rollback(doc_id)
            self.registry.update(doc_id, status="failed", stage="failed", error=exc.message, resumable=False)
        except Exception as exc:
            log.error("ingest_crashed", doc=doc_id, error=type(exc).__name__, exc_info=True)
            self.registry.update(
                doc_id, status="failed", stage="failed", error="Unexpected error while processing.", resumable=True
            )
        finally:
            self.on_change()

    def _rollback(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        self.bm25.remove_document(doc_id)
        self.registry.update(doc_id, pages_processed=0, chunks=0)

    def _pages(self, path: Path, body: float, start: int, end: int) -> list[PageContent]:
        """Extract a window of pages, optionally across a process pool."""
        if self.s.extract_workers <= 1 or end - start < 8:
            return list(iter_pages(path, body, start, end, self.s.tables_enabled))
        if self._proc_pool is None:
            self._proc_pool = ProcessPoolExecutor(max_workers=self.s.extract_workers)
        step = max(1, (end - start + 1) // self.s.extract_workers)
        futs = [
            self._proc_pool.submit(extract_range, str(path), body, a, min(a + step - 1, end), self.s.tables_enabled)
            for a in range(start, end + 1, step)
        ]
        out: list[PageContent] = []
        for f in futs:
            out.extend(f.result())
        return out

    def _run(self, doc_id: str) -> None:
        doc = self.registry.get(doc_id)
        path = Path(doc["stored_path"])
        t0 = time.time()
        self.registry.update(doc_id, status="processing", stage="extracting", error=None, started_at=t0)

        info = inspect_pdf(path, self.s.max_pages)
        total = info.pages
        self.registry.update(doc_id, pages=total)

        # ---- pass 1: boilerplate + scanned detection (small strings only)
        edges, total_chars, pages_with_text = edge_lines(path)
        if total_chars / max(1, total) < self.s.min_chars_per_page and pages_with_text == 0:
            raise ScannedPdf()
        boilerplate = repeated_boilerplate(edges)
        del edges

        # ---- pass 2: stream windows of pages
        start_page = int(doc["pages_processed"]) + 1
        tracker = SectionTracker()
        chunk_count = self.store.count_document(doc_id, max_page=start_page - 1) if start_page > 1 else 0
        if start_page > 1:  # rebuild the heading path for pages already stored
            for pg in iter_pages(path, info.body_font_size, 1, start_page - 1, False):
                for b in pg.blocks:
                    if b.kind == "heading":
                        tracker.update(b.level, b.text)

        seen_hashes: set[str] = set()
        samples: list[str] = []
        group_size = max(1, total // MAX_GROUPS)
        buffer: list[Chunk] = []
        last_buffered_page = start_page - 1
        window = max(1, self.s.extract_window)

        for w_start in range(start_page, total + 1, window):
            w_end = min(w_start + window - 1, total)
            self.registry.update(doc_id, stage="extracting")
            for page in self._pages(path, info.body_font_size, w_start, w_end):
                last_buffered_page = page.page_number
                page.blocks = [
                    b for b in page.blocks if not (b.kind == "heading" and normalize_edge(b.text) in boilerplate)
                ]
                if not page.blocks:
                    continue  # blank page or running header only
                for block in page.blocks:
                    if block.kind != "heading":
                        block.text = clean_text(block.text, boilerplate) if block.kind == "para" else block.text
                text_sig = hashlib.blake2b(" ".join(b.text for b in page.blocks).lower().encode()).hexdigest()
                if text_sig in seen_hashes:
                    continue  # near-duplicate / repeated boilerplate page
                seen_hashes.add(text_sig)
                body_text = "\n".join(b.text for b in page.blocks if b.kind == "para")
                if (page.page_number - 1) % group_size == 0 and len(samples) < MAX_GROUPS:
                    samples.append(body_text[:GROUP_CHARS])
                buffer.extend(
                    self.chunker.chunk_page(page, document_id=doc_id, filename=doc["filename"], tracker=tracker)
                )
                if len(buffer) >= self.s.embed_batch_size:
                    chunk_count += self._flush(doc_id, buffer, last_buffered_page, chunk_count)
                    buffer = []
            if buffer:
                chunk_count += self._flush(doc_id, buffer, last_buffered_page, chunk_count)
                buffer = []
            self.registry.update(doc_id, pages_processed=w_end, chunks=chunk_count)

        if chunk_count == 0:
            raise ScannedPdf("No text chunks could be created from this PDF.")

        # ---- summary + suggested questions (non-fatal)
        summary = ""
        questions: list[str] = []
        if self.s.summaries_enabled:
            self.registry.update(doc_id, stage="summarizing")
            summary, questions = summarize_document(self.llm, doc["filename"], samples)

        self.registry.update(
            doc_id,
            status="completed",
            stage="ready",
            pages_processed=total,
            chunks=chunk_count,
            summary=summary or None,
            suggested_questions=questions,
            embed_model=self.embedder.model_name,
            resumable=False,
            error=None,
        )
        self.registry.bump_version()
        log.info("ingest_done", doc=doc_id, pages=total, chunks=chunk_count, seconds=round(time.time() - t0, 2))

    def _flush(self, doc_id: str, chunks: list[Chunk], last_page: int, chunks_so_far: int) -> int:
        """Embed + store a batch, then checkpoint. All pages <= last_page are now fully stored."""
        self.registry.update(doc_id, stage="embedding")
        vectors = self.embedder.embed_documents([c.stored_text for c in chunks])
        self.store.upsert(chunks, vectors)  # deterministic IDs -> idempotent
        self.bm25.add_chunks(chunks)
        self.registry.update(doc_id, pages_processed=last_page, chunks=chunks_so_far + len(chunks), stage="extracting")
        return len(chunks)


def remove_files(settings: Settings, doc: dict[str, object]) -> None:
    """Delete the stored PDF and cached previews for a document."""
    Path(str(doc["stored_path"])).unlink(missing_ok=True)
    shutil.rmtree(settings.preview_dir / str(doc["id"]), ignore_errors=True)
