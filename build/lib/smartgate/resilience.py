"""Fallback mechanisms: retries with backoff, circuit breakers."""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, TypeVar

from .errors import ProviderError

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 3  # per model, including the first try
    base_delay: float = 0.25
    max_delay: float = 8.0
    jitter: float = 0.2

    def delay(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None:
            return min(self.max_delay, retry_after)
        d = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return d * (1 + random.uniform(-self.jitter, self.jitter))

    def run(self, fn: Callable[[], T], sleep: Callable[[float], None] = time.sleep,
            on_retry: Optional[Callable[[int, ProviderError], None]] = None) -> T:
        attempt = 0
        while True:
            attempt += 1
            try:
                return fn()
            except ProviderError as exc:
                if not exc.retryable or attempt >= self.max_attempts:
                    raise
                if on_retry:
                    on_retry(attempt, exc)
                sleep(self.delay(attempt, exc.retry_after))


class CircuitBreaker:
    """closed -> (N failures) -> open -> (cooldown) -> half_open -> success => closed."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.clock = clock
        self.failures = 0
        self.opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.recovery_timeout:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.failure_threshold:
                self.opened_at = self.clock()


class BreakerRegistry:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> CircuitBreaker:
        with self._lock:
            if key not in self._breakers:
                self._breakers[key] = CircuitBreaker(self.failure_threshold, self.recovery_timeout)
            return self._breakers[key]

    def states(self) -> Dict[str, str]:
        return {k: b.state for k, b in self._breakers.items()}
