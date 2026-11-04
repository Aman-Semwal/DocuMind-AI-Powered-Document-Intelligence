"""Structure-aware chunking. Chunks never cross page boundaries.

Each chunk carries the section path (e.g. "Leave Policy > Maternity Leave"), which is
prepended to the stored text so the chunk keeps its context when embedded and retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.pdf_extractor import PageContent

MIN_CHUNK_CHARS = 20
_PREFIX = re.compile(r"^\[[^\]\n]{1,200}\]\n")


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    idx: int  # index of the chunk inside its page
    text: str  # raw chunk text (no section prefix)
    section: str = ""
    kind: str = "text"

    @property
    def stored_text(self) -> str:
        """Text that is embedded, indexed and shown to the LLM."""
        return f"[{self.section}]\n{self.text}" if self.section else self.text

    def metadata(self) -> dict[str, str | int]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "idx": self.idx,
            "section": self.section,
            "kind": self.kind,
            "chunk_id": self.chunk_id,
        }


def strip_section_prefix(text: str) -> str:
    return _PREFIX.sub("", text, count=1)


def make_chunk_id(document_id: str, page: int, idx: int) -> str:
    """Deterministic ID: re-ingesting the same page produces the same IDs (idempotent)."""
    return f"{document_id}-p{page:04d}-c{idx:03d}"


@dataclass
class SectionTracker:
    """Keeps the current heading path across pages."""

    stack: list[tuple[int, str]] = field(default_factory=list)

    def update(self, level: int, title: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title[:80]))

    @property
    def path(self) -> str:
        return " > ".join(t for _, t in self.stack)


class Chunker:
    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            length_function=len,
        )

    def _split_table(self, table_md: str) -> list[str]:
        """Keep each row with its column labels so similar values cannot merge together."""
        lines = table_md.split("\n")
        header, rows = lines[:2], lines[2:]
        if not rows:
            return [table_md]
        return ["\n".join([*header, row]) for row in rows]

    def chunk_page(
        self,
        page: PageContent,
        *,
        document_id: str,
        filename: str,
        tracker: SectionTracker,
    ) -> list[Chunk]:
        """Chunk one cleaned page. `page.blocks` hold cleaned text; the tracker is updated."""
        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_section = tracker.path

        def flush() -> None:
            if not buffer:
                return
            text = "\n\n".join(buffer)
            buffer.clear()
            for piece in self.splitter.split_text(text):
                piece = piece.strip()
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(self._make(document_id, filename, page, len(chunks), piece, buffer_section, "text"))

        for block in page.blocks:
            if not block.text.strip():
                continue
            if block.kind == "heading":
                flush()
                tracker.update(block.level, block.text)
                buffer_section = tracker.path
                continue
            if block.kind == "table":
                flush()
                for part in self._split_table(block.text):
                    chunks.append(self._make(document_id, filename, page, len(chunks), part, tracker.path, "table"))
                buffer_section = tracker.path
                continue
            if not buffer:
                buffer_section = tracker.path
            buffer.append(block.text)
        flush()
        return chunks

    @staticmethod
    def _make(
        document_id: str, filename: str, page: PageContent, idx: int, text: str, section: str, kind: str
    ) -> Chunk:
        return Chunk(
            chunk_id=make_chunk_id(document_id, page.page_number, idx),
            document_id=document_id,
            filename=filename,
            page_number=page.page_number,
            idx=idx,
            text=text,
            section=section,
            kind=kind,
        )
