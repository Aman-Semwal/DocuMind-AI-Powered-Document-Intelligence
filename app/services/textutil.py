"""Shared tokenisation helpers (BM25, lexical reranker, verification, fakes)."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_/][a-z0-9]+)*")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "any",
        "also",
        "all",
        "not",
        "no",
    ]
)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def stem(token: str) -> str:
    """Very light stemming so 'leaves', 'leave' and 'leaving' match. Good enough for matching."""
    t = token
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        t = t[:-1]
    for suffix in ("ing", "ed"):
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            t = t[: -len(suffix)]
            break
    if t.endswith("e") and len(t) > 4:
        t = t[:-1]
    return t


def stemmed_set(text: str) -> set[str]:
    return {stem(t) for t in content_tokens(text)}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)
