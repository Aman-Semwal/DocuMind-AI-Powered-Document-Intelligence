"""PDF extraction with PyMuPDF. Pages are streamed one at a time so memory stays flat."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pymupdf

from app.core.exceptions import CorruptPdf, EncryptedPdf, TooManyPages

BlockKind = Literal["heading", "para", "table"]


@dataclass
class Block:
    text: str
    kind: BlockKind = "para"
    level: int = 0  # headings only: 1 (largest) .. 3


@dataclass
class PageContent:
    page_number: int  # 1-based
    blocks: list[Block] = field(default_factory=list)

    @property
    def raw_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


@dataclass
class PdfInfo:
    pages: int
    body_font_size: float


def _open(path: Path) -> pymupdf.Document:
    try:
        doc = pymupdf.open(path)
    except Exception:  # pymupdf raises several types for broken files
        doc = None
    if doc is None:
        raise CorruptPdf()
    if doc.needs_pass or doc.is_encrypted:
        doc.close()
        raise EncryptedPdf()
    return doc


def inspect_pdf(path: Path, max_pages: int) -> PdfInfo:
    """Validate that the file opens, is not encrypted and is within the page limit."""
    doc = _open(path)
    try:
        n = doc.page_count
        if n == 0:
            raise CorruptPdf("The PDF has no pages.")
        if n > max_pages:
            raise TooManyPages(f"The PDF has {n} pages; the limit is {max_pages}.")
        return PdfInfo(pages=n, body_font_size=_estimate_body_size(doc))
    finally:
        doc.close()


def _estimate_body_size(doc: pymupdf.Document) -> float:
    """Most common font size (weighted by characters) over up to 12 sampled pages."""
    n = doc.page_count
    sample = sorted({int(i * (n - 1) / max(1, min(12, n) - 1)) for i in range(min(12, n))})
    sizes: Counter[float] = Counter()
    for i in sample:
        try:
            data = doc[i].get_text("dict")
        except Exception:  # noqa: S112 - sampling only; an unreadable page is skipped
            continue
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    if not sizes:
        return 11.0
    return sizes.most_common(1)[0][0]


def _is_bold(span: dict) -> bool:  # type: ignore[type-arg]
    return bool(span.get("flags", 0) & 16) or "bold" in str(span.get("font", "")).lower()


def _table_to_markdown(rows: list[list[str]]) -> str:
    clean = [[(c or "").replace("\n", " ").replace("|", "/").strip() for c in r] for r in rows]
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    lines = ["| " + " | ".join(clean[0]) + " |", "|" + "|".join(["---"] * width) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(lines)


def _extract_page(page: pymupdf.Page, body_size: float, tables_enabled: bool) -> PageContent:
    items: list[tuple[float, float, Block]] = []
    table_boxes: list[pymupdf.Rect] = []

    # find_tables() is the slowest call, so only run it on pages that draw ruling lines / cell boxes
    if tables_enabled and len(page.get_drawings()) >= 4:
        try:
            found = page.find_tables()
            for t in found.tables:
                # read each cell with get_text(clip=...): t.extract() can drop the spaces between words
                rows = [
                    [" ".join(page.get_text("text", clip=pymupdf.Rect(c)).split()) if c else "" for c in r.cells]
                    for r in t.rows
                ]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    rect = pymupdf.Rect(t.bbox)
                    table_boxes.append(rect)
                    items.append((rect.y0, rect.x0, Block(_table_to_markdown(rows), "table")))
        except Exception:  # table detection is best-effort
            table_boxes = []
            items = []

    data = page.get_text("dict", sort=True)
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        x0, y0, x1, y1 = block["bbox"]
        centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
        if any(centre in r for r in table_boxes):
            continue
        lines: list[str] = []
        max_size = 0.0
        bold_chars = total_chars = 0
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(s["text"] for s in spans).rstrip()
            if not text.strip():
                continue
            lines.append(text)
            for s in spans:
                n = len(s["text"].strip())
                total_chars += n
                if _is_bold(s):
                    bold_chars += n
                max_size = max(max_size, s["size"])
        if not lines:
            continue
        text = "\n".join(lines)
        bold = total_chars > 0 and bold_chars / total_chars > 0.8
        short = len(lines) <= 2 and len(text) <= 120 and not text.rstrip().endswith((".", ",", ";"))
        level = 0
        if short and max_size >= body_size * 1.5:
            level = 1
        elif short and max_size >= body_size * 1.25:
            level = 2
        elif short and (max_size >= body_size * 1.1 or (bold and max_size >= body_size * 0.98)):
            level = 3
        kind: BlockKind = "heading" if level else "para"
        items.append((y0, x0, Block(" ".join(text.split()) if level else text, kind, level)))

    items.sort(key=lambda t: (round(t[0], 0), t[1]))
    return PageContent(page_number=page.number + 1, blocks=[b for _, _, b in items])


def iter_pages(
    path: Path, body_size: float, start: int = 1, end: int | None = None, tables: bool = True
) -> Iterator[PageContent]:
    """Yield pages one by one. A corrupt page yields an empty PageContent instead of raising."""
    doc = _open(path)
    try:
        last = min(end or doc.page_count, doc.page_count)
        for n in range(start, last + 1):
            try:
                yield _extract_page(doc[n - 1], body_size, tables)
            except Exception:
                yield PageContent(page_number=n, blocks=[])
    finally:
        doc.close()


def extract_range(path: str, body_size: float, start: int, end: int, tables: bool) -> list[PageContent]:
    """Picklable helper for the optional process pool."""
    return list(iter_pages(Path(path), body_size, start, end, tables))


def edge_lines(path: Path, n_edge: int = 3) -> tuple[list[list[str]], int, int]:
    """Pass 1: first/last lines of every page (for header/footer detection) + text stats.

    Returns (edge_lines_per_page, total_chars, pages_with_text). Only small strings are kept.
    """
    doc = _open(path)
    edges: list[list[str]] = []
    total = with_text = 0
    try:
        for page in doc:
            try:
                lines = [ln.strip() for ln in page.get_text("text", sort=True).splitlines() if ln.strip()]
            except Exception:
                lines = []
            chars = sum(len(ln) for ln in lines)
            total += chars
            with_text += 1 if chars >= 15 else 0
            segments = [seg for ln in lines for seg in re.split(r"\s{3,}", ln) if seg.strip()]
            edges.append(segments[:n_edge] + segments[-n_edge:] if segments else [])
    finally:
        doc.close()
    return edges, total, with_text


_NUM = re.compile(r"\d+")


def normalize_edge(line: str) -> str:
    return _NUM.sub("#", " ".join(line.lower().split()))


def repeated_boilerplate(edges: list[list[str]], threshold: float = 0.5) -> set[str]:
    """Edge lines that repeat on more than `threshold` of pages (headers, footers)."""
    n_pages = len(edges)
    if n_pages < 3:
        return set()
    counts: Counter[str] = Counter()
    for page_edges in edges:
        counts.update({normalize_edge(ln) for ln in page_edges})
    return {line for line, c in counts.items() if c / n_pages > threshold and len(line) > 1}


def median_text_len(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def render_page_png(path: Path, page_number: int, zoom: float = 1.6) -> bytes:
    doc = _open(path)
    try:
        if not 1 <= page_number <= doc.page_count:
            raise IndexError(page_number)
        pix = doc[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        return bytes(pix.tobytes("png"))
    finally:
        doc.close()
