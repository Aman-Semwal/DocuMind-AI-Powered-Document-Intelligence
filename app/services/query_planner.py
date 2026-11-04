"""Query planning: summary detection and rule-based decomposition of multi-part questions."""

from __future__ import annotations

import re

_SUMMARY = re.compile(
    r"^\s*(please\s+)?(can you\s+)?(summari[sz]e|give me (a|an) (summary|overview)|"
    r"what (is|are) (this|the) (document|documents|file|pdf|policy|policies) (about|for)|"
    r"what are the main (topics|points|themes)|tl;?dr|overview of (this|the))",
    re.I,
)
_SPLIT = re.compile(r"\?\s+(?=\S)|;\s+|\s+as well as\s+|\s+and also\s+|\s+and then\s+", re.I)
_SECOND_QUESTION = re.compile(r"\b(and|also)\s+(what|how|when|who|where|which|is|are|can|does|do)\b", re.I)


def is_summary_question(question: str) -> bool:
    return bool(_SUMMARY.search(question))


def decompose(question: str, max_parts: int = 3) -> list[str]:
    """Split a compound question into sub-questions using simple rules.

    Returns [] when the question is a single question (the caller then uses it as is).
    """
    q = question.strip()
    parts = [p.strip(" ?.") for p in _SPLIT.split(q) if p and p.strip(" ?.")]
    if len(parts) < 2:
        m = _SECOND_QUESTION.search(q)
        if m and m.start() > 15:
            parts = [q[: m.start()].strip(" ?,."), q[m.start() + len(m.group(1)) :].strip(" ?,.")]
    parts = [p for p in parts if len(p.split()) >= 3]
    if len(parts) < 2:
        return []
    return [p + "?" for p in parts[:max_parts]]
