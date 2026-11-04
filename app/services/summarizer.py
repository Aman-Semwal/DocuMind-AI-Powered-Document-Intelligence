"""Hierarchical document summary + suggested questions, generated once after ingestion."""

from __future__ import annotations

import re

from app.core.logging import get_logger
from app.services.llm_client import LLMClient
from app.services.prompt_builder import build_summary_messages

log = get_logger("summarizer")

MAX_GROUPS = 6
GROUP_CHARS = 1800


def parse_summary(raw: str) -> tuple[str, list[str]]:
    summary_part, _, q_part = raw.partition("QUESTIONS:")
    summary = re.sub(r"^\s*SUMMARY:\s*", "", summary_part, flags=re.I).strip()
    questions = [
        ln.strip().lstrip("-*0123456789. ").strip() for ln in q_part.splitlines() if ln.strip().lstrip("-*0123456789. ")
    ]
    questions = [q if q.endswith("?") else q + "?" for q in questions if len(q) > 8][:3]
    return summary, questions


def summarize_document(llm: LLMClient, filename: str, samples: list[str]) -> tuple[str, list[str]]:
    """`samples` holds the opening text of evenly spaced page groups (collected during ingestion).

    Small documents use a single call. Larger ones summarise each group, then reduce.
    """
    samples = [s for s in samples if s.strip()][:MAX_GROUPS]
    if not samples:
        return "", []
    try:
        if len(samples) == 1:
            return parse_summary(
                llm.chat_sync(build_summary_messages(filename, samples[0][:GROUP_CHARS]), max_tokens=200)
            )
        partials: list[str] = []
        for s in samples:
            raw = llm.chat_sync(build_summary_messages(filename, s[:GROUP_CHARS]), max_tokens=120)
            part, _ = parse_summary(raw)
            if part:
                partials.append(part)
        combined = "\n".join(partials)
        return parse_summary(
            llm.chat_sync(build_summary_messages(filename, combined[: GROUP_CHARS * 2]), max_tokens=220)
        )
    except Exception as exc:  # summaries are a nice-to-have; never fail ingestion for them
        log.warning("summary_failed", error=type(exc).__name__)
        return "", []
