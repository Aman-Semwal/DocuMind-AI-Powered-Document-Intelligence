"""Answer verification: sentence support, number/date/entity check, confidence score."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.services.textutil import STOPWORDS, content_tokens, stem

_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_MARKER = re.compile(r"\[\d{1,2}\]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_ENTITY = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-zA-Z]{2,})\b")
_CONDITION = re.compile(r"\b(?:trip|expense|cost|amount|price|purchase|invoice)\b[^.!?]{0,48}", re.IGNORECASE)


@dataclass
class Verification:
    ok: bool = True
    support_ratio: float = 1.0
    unsupported_sentences: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)
    unverified_entities: list[str] = field(default_factory=list)


def _norm_number(n: str) -> str:
    return n.replace(",", "").rstrip(".")


def verify_answer(
    answer: str, context_texts: list[str], *, min_support: float = 0.5, question: str = ""
) -> Verification:
    """Check every sentence, number and named entity of `answer` against `context_texts`."""
    clean = _MARKER.sub("", answer).strip()
    if not clean:
        return Verification(ok=False, support_ratio=0.0)
    context = "\n".join(context_texts)
    ctx_stems = {stem(t) for t in content_tokens(context)}
    ctx_lower = context.lower()
    ctx_numbers = {_norm_number(n) for n in _NUMBER.findall(context)}
    question_numbers = {
        _norm_number(number)
        for clause in _CONDITION.findall(question)
        for number in _NUMBER.findall(clause)
    }

    unsupported: list[str] = []
    sentences = [s.strip() for s in _SENTENCE.split(clean) if s.strip()]
    for s in sentences:
        toks = [stem(t) for t in content_tokens(s)]
        if len(toks) < 3:
            continue
        support = sum(1 for t in toks if t in ctx_stems) / len(toks)
        if support < min_support:
            unsupported.append(s)

    bad_numbers = sorted(
        {n for n in map(_norm_number, _NUMBER.findall(clean)) if n and n not in ctx_numbers and n not in question_numbers}
    )
    entities = {e for e in _ENTITY.findall(clean) if e.lower() not in STOPWORDS and e.lower() != "information"}
    bad_entities = sorted(e for e in entities if e.lower() not in ctx_lower)

    ratio = 1.0 - (len(unsupported) / len(sentences) if sentences else 0.0)
    return Verification(
        ok=not unsupported and not bad_numbers and not bad_entities,
        support_ratio=round(ratio, 3),
        unsupported_sentences=unsupported,
        unverified_numbers=bad_numbers,
        unverified_entities=bad_entities,
    )


def confidence_label(
    *,
    evidence: float,
    high_threshold: float,
    low_threshold: float,
    agreement: bool,
    verification: Verification | None,
) -> str:
    """Combine retrieval strength, method agreement and verification into high/medium/low."""
    if verification is not None and (verification.unverified_numbers or verification.support_ratio < 0.5):
        return "low"
    verified = verification is None or verification.ok
    if evidence >= high_threshold and agreement and verified:
        return "high"
    if evidence < low_threshold or not verified:
        return "low" if evidence < low_threshold else "medium"
    return "medium"
