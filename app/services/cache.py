"""In-memory LRU answer cache (thread-safe) and per-session chat history."""

from __future__ import annotations

import re
import threading
from collections import OrderedDict, deque
from typing import Any


def normalize_question(q: str) -> str:
    return re.sub(r"[^\w\s]", "", q.lower()).strip()


class LRUCache:
    """Keys include the document-set version, so changing documents invalidates old answers."""

    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._data: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        question: str,
        version: int,
        top_k: int,
        document_ids: list[str] | None,
        deep: bool,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> tuple[Any, ...]:
        docs = tuple(sorted(document_ids)) if document_ids is not None else None
        return (normalize_question(question), version, top_k, docs, deep, page_from, page_to)

    def get(self, key: tuple[Any, ...]) -> Any | None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: tuple[Any, ...], value: Any) -> None:
        if self.capacity <= 0:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class SessionStore:
    """Conversation memory: last N (question, answer) turns per session id."""

    def __init__(self, turns: int = 6, max_sessions: int = 500) -> None:
        self.turns = turns
        self.max_sessions = max_sessions
        self._data: OrderedDict[str, deque[tuple[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str | None) -> list[tuple[str, str]]:
        if not session_id:
            return []
        with self._lock:
            return list(self._data.get(session_id, []))

    def add(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id:
            return
        with self._lock:
            dq = self._data.setdefault(session_id, deque(maxlen=self.turns))
            dq.append((question, answer))
            self._data.move_to_end(session_id)
            while len(self._data) > self.max_sessions:
                self._data.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)
