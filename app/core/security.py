"""Small security helpers: key comparison, filename sanitising, text sanitising."""

from __future__ import annotations

import hmac
import re
import unicodedata
import uuid

PDF_MAGIC = b"%PDF-"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._\- ]+")


def verify_api_key(provided: str | None, valid_keys: list[str]) -> bool:
    """Constant-time comparison against every configured key (no early exit)."""
    if not provided or not valid_keys:
        return False
    ok = False
    for key in valid_keys:
        ok |= hmac.compare_digest(provided.encode(), key.encode())
    return ok


def sanitize_filename(name: str | None, default: str = "document.pdf") -> str:
    """Return a display-safe filename. The client path is never used on disk."""
    if not name:
        return default
    name = name.replace("\\", "/").split("/")[-1]  # drop any directory part
    name = unicodedata.normalize("NFKC", name)
    name = _CONTROL.sub("", name)
    name = _UNSAFE_NAME.sub("_", name).strip(" .")
    name = re.sub(r"_{2,}", "_", name)
    if not name or set(name) <= {"_", "."}:
        return default
    return name[:120]


def stored_filename() -> str:
    """Server-generated filename used on disk."""
    return f"{uuid.uuid4().hex}.pdf"


def strip_control_chars(text: str) -> str:
    return _CONTROL.sub("", text)


def sanitize_question(text: str) -> str:
    text = strip_control_chars(text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_pdf(head: bytes) -> bool:
    return head.lstrip(b"\x00\r\n\t ")[:5] == PDF_MAGIC or head[:5] == PDF_MAGIC
