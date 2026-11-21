"""Deterministic offline provider for tests, demos and CI."""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from ..analysis import estimate_tokens
from ..errors import ProviderError
from ..types import GatewayRequest, ModelSpec, ProviderResponse, Usage
from .base import Provider


class MockProvider(Provider):
    """Echo-style provider with scriptable failures.

    `fail_models={"model-x": "rate_limit"}` makes model-x always fail;
    `fail_times={"model-x": 2}` makes it fail twice, then succeed.
    """

    def __init__(self, name: str = "mock", latency: float = 0.0,
                 fail_models: Optional[Dict[str, str]] = None,
                 fail_times: Optional[Dict[str, int]] = None) -> None:
        self.name = name
        self.latency = latency
        self.fail_models = dict(fail_models or {})
        self.fail_times = dict(fail_times or {})
        self.calls: List[str] = []
        self._lock = threading.Lock()

    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        with self._lock:
            self.calls.append(model.name)
            remaining = self.fail_times.get(model.name, 0)
            if remaining:
                self.fail_times[model.name] = remaining - 1
        if self.latency:
            time.sleep(self.latency)
        kind = self.fail_models.get(model.name) or ("server" if remaining else None)
        if kind:
            raise ProviderError(f"mock {kind}", kind=kind,
                                retryable=kind in ("rate_limit", "server", "timeout"),
                                fallback=kind != "bad_request")
        text = f"[{model.name}] answer to: {request.last_user_message[:200]}"
        return ProviderResponse(
            text=text, model=model.name,
            usage=Usage(estimate_tokens(request.full_text), estimate_tokens(text)),
            stop_reason="end_turn",
        )
