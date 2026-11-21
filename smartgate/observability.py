"""Event hooks and in-process metrics. Bridge to OpenTelemetry/Prometheus/Datadog via hooks."""
from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List

logger = logging.getLogger("smartgate")

EVENTS = (
    "request.start", "request.complete", "request.error",
    "cache.hit", "cache.miss", "cache.store",
    "route.decided", "provider.retry", "provider.error", "fallback",
    "budget.downgrade", "budget.exceeded", "policy.redacted", "policy.blocked",
)

Hook = Callable[[str, Dict[str, Any]], None]


class EventBus:
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Hook]] = defaultdict(list)

    def on(self, event: str, hook: Hook) -> None:
        """Subscribe to one event, or '*' for all."""
        if event != "*" and event not in EVENTS:
            raise ValueError(f"Unknown event '{event}'. Known: {EVENTS}")
        self._hooks[event].append(hook)

    def emit(self, event: str, payload: Dict[str, Any]) -> None:
        for hook in self._hooks.get(event, []) + self._hooks.get("*", []):
            try:
                hook(event, payload)
            except Exception:  # a broken hook must never break a request
                logger.exception("hook for %s failed", event)


class Metrics:
    def __init__(self, window: int = 2000) -> None:
        self.counters: Dict[str, int] = defaultdict(int)
        self.latencies = deque(maxlen=window)
        self._lock = threading.Lock()

    def incr(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.counters[key] += n

    def observe_latency(self, ms: float) -> None:
        with self._lock:
            self.latencies.append(ms)

    def _pct(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        return round(values[min(len(values) - 1, int(p * len(values)))], 2)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lat = list(self.latencies)
            counters = dict(self.counters)
        return {"counters": counters,
                "latency_ms": {"p50": self._pct(lat, 0.50), "p95": self._pct(lat, 0.95),
                               "p99": self._pct(lat, 0.99)}}
