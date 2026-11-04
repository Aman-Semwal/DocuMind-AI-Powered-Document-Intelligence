"""Per-stage timing collector used for logs, API responses and the Insights tab."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, (time.perf_counter() - t0) * 1000)

    def add(self, name: str, ms: float) -> None:
        self.stages[name] = round(self.stages.get(name, 0.0) + ms, 2)

    def merge(self, other: dict[str, float]) -> None:
        for k, v in other.items():
            self.add(k, v)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)

    def as_dict(self) -> dict[str, float]:
        out = dict(self.stages)
        out["total_ms"] = self.total_ms()
        return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
