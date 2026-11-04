"""Citations are built from retrieved chunk metadata, never from model-written text."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import NOT_AVAILABLE_TEXT
from app.services.prompt_builder import Passage

_MARKER = re.compile(r"\[(\d{1,2})\]")
_NOT_AVAILABLE_HINTS = (
    "not available in the uploaded documents",
    "not available in the document",
    "does not contain",
    "do not contain",
    "no information about",
    "cannot find",
    "can't find",
    "not mentioned in",
)


@dataclass(frozen=True)
class Citation:
    document_id: str
    document: str
    page: int | None


def cited_numbers(answer: str, valid: set[int]) -> list[int]:
    """Passage numbers the model cited, in order of first use, ignoring invalid numbers."""
    seen: list[int] = []
    for m in _MARKER.finditer(answer):
        n = int(m.group(1))
        if n in valid and n not in seen:
            seen.append(n)
    return seen


def strip_invalid_markers(answer: str, valid: set[int]) -> str:
    return _MARKER.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", answer)


def is_not_available(answer: str) -> bool:
    low = answer.lower().strip()
    if low.startswith(NOT_AVAILABLE_TEXT.lower()[:40]):
        return True
    return len(low) < 220 and any(h in low for h in _NOT_AVAILABLE_HINTS)


def build_citations(
    answer: str, passages: list[Passage], *, fallback_top: int = 2
) -> tuple[list[Citation], list[Passage]]:
    """De-duplicated (document, page) citations + the passages they come from.

    If the model cited nothing, the top passages that were sent are used.
    """
    by_number = {p.number: p for p in passages}
    nums = cited_numbers(answer, set(by_number))
    used = [by_number[n] for n in nums] or passages[:fallback_top]
    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for p in used:
        key = (p.document_id, p.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(Citation(p.document_id, p.filename, p.page))
    citations.sort(key=lambda c: (c.document_id, c.page or 0))
    return citations, used
