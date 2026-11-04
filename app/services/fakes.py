"""Deterministic offline backends used by tests, CI and demos (EMBEDDER_BACKEND=fake, LLM_BACKEND=fake).

FakeEmbedder is a hashed bag-of-words model: texts that share words are similar,
unrelated texts are near-orthogonal. FakeLLM answers extractively from the first passage.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import AsyncIterator

from app.config import NOT_AVAILABLE_TEXT
from app.services.textutil import content_tokens, stem


class FakeEmbedder:
    def __init__(self, dim: int = 384, model_name: str = "fake-hash-embed") -> None:
        self.dim = dim
        self.model_name = model_name
        self.calls = 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in content_tokens(text):
            t = stem(tok)
            h = int.from_bytes(hashlib.blake2b(t.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


_PASSAGE = re.compile(r"^\[(\d+)\] \(source:[^\n]*\)\n<<<\n(.*?)\n>>>", re.S | re.M)
_QUESTION = re.compile(r"^Question: (.*)$", re.M)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


class FakeLLM:
    model_name = "fake-llm"

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    def _answer(self, messages: list[dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "TASK=rewrite" in system:
            m = re.search(r"Follow-up question: (.*)$", user, re.M)
            return m.group(1) if m else user
        if "TASK=summarize" in system:
            body = re.sub(r"\s+", " ", user.split("TEXT:", 1)[-1]).strip()
            sentences = re.split(r"(?<=[.!?])\s+", body)
            summary = " ".join(sentences[:2])[:400]
            first = content_tokens(sentences[0] if sentences else "")[:4]
            topic = " ".join(first) or "this document"
            return f"SUMMARY: {summary}\nQUESTIONS:\n- What does the document say about {topic}?\n- Who is this policy for?\n- What are the key rules?"
        # TASK=qa
        q = _QUESTION.search(user)
        question = q.group(1) if q else ""
        passages = [(int(n), t) for n, t in _PASSAGE.findall(user)]
        if not passages:
            return NOT_AVAILABLE_TEXT
        q_terms = {stem(t) for t in content_tokens(question)}
        scored: list[tuple[float, int, str]] = []
        for num, text in passages:
            for sent in _SENT.split(re.sub(r"^\[[^\]]*\]\n", "", text)):
                s_terms = {stem(t) for t in content_tokens(sent)}
                if not s_terms:
                    continue
                score = len(q_terms & s_terms) / (len(q_terms) or 1)
                if score >= 0.3:
                    scored.append((score, num, sent.strip()))
        if not scored:
            return NOT_AVAILABLE_TEXT
        unique: dict[str, tuple[float, int, str]] = {}
        for item in scored:
            unique.setdefault(item[2], item)
        top = sorted(unique.values(), key=lambda t: t[0], reverse=True)[:3]
        return " ".join(f"{sent} [{num}]" for _, num, sent in sorted(top, key=lambda t: t[1]))

    async def chat(self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None) -> str:
        self.calls += 1
        self.last_messages = messages
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return self._answer(messages)

    async def stream(
        self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        self.calls += 1
        self.last_messages = messages
        text = self._answer(messages)
        for word in re.findall(r"\S+\s*", text):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            yield word

    def chat_sync(self, messages: list[dict[str, str]], *, deep: bool = False, max_tokens: int | None = None) -> str:
        self.calls += 1
        self.last_messages = messages
        return self._answer(messages)

    async def warm_up(self) -> None:
        return None

    async def ping(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
