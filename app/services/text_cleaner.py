"""Text cleaning: control characters, whitespace, hyphenation, headers/footers."""

from __future__ import annotations

import re
import unicodedata

from app.services.pdf_extractor import normalize_edge

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​‌‍﻿]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\s*)([a-z])")
_HYPHEN_CAPITAL = re.compile(r"(\w)-\n\s*([A-Z])")
_LIST_START = re.compile(r"^\s*(?:[-*•●▪]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")
_SPACES = re.compile(r"[ \t\u00a0]+")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


_WIDE_GAP = re.compile(r"\s{3,}")


def _drop_boilerplate(line: str, boilerplate: set[str]) -> str:
    """Remove header/footer text. Lines are also checked in segments split on wide gaps."""
    if not line.strip() or (normalize_edge(line) not in boilerplate and not _WIDE_GAP.search(line)):
        return line
    if normalize_edge(line) in boilerplate:
        return ""
    return "  ".join(seg for seg in _WIDE_GAP.split(line) if normalize_edge(seg) not in boilerplate)


def clean_text(text: str, boilerplate: set[str] | None = None) -> str:
    """Return cleaned paragraph text. Paragraphs are separated by one blank line."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if boilerplate:
        text = "\n".join(_drop_boilerplate(ln, boilerplate) for ln in text.split("\n"))

    text = _HYPHEN_BREAK.sub(r"\1\3", text)  # repair "inter-\nnational" -> "international"
    text = _HYPHEN_CAPITAL.sub(r"\1-\2", text)  # "Anglo-\nSaxon" keeps its real hyphen

    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        lines = [_SPACES.sub(" ", ln).strip() for ln in para.split("\n")]
        lines = [ln for ln in lines if ln]
        if not lines:
            continue
        merged: list[str] = [lines[0]]
        for ln in lines[1:]:
            if _LIST_START.match(ln):
                merged.append(ln)  # keep list items on their own line
            else:
                merged[-1] = f"{merged[-1]} {ln}"
        paragraphs.append("\n".join(merged))
    return "\n\n".join(paragraphs).strip()
