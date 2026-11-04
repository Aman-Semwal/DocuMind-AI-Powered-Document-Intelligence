"""Tiny in-memory sliding-window rate limiter (per API key and per client IP)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, window_s: float = 60.0) -> None:
        self.window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, bucket: str, limit: int) -> bool:
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[bucket]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            if len(self._hits) > 10_000:  # bound memory
                for k in [k for k, v in self._hits.items() if not v][:1000]:
                    del self._hits[k]
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()
