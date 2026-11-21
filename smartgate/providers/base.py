"""Provider interface. Implement `complete` to add any LLM vendor or in-house model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from ..types import GatewayRequest, ModelSpec, ProviderResponse


class Provider(ABC):
    name: str = "provider"

    @abstractmethod
    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        """Run one completion. Raise `ProviderError` on failure.

        Set `retryable=True` for transient failures (rate limits, 5xx, timeouts)
        and `fallback=False` for errors another model cannot fix (bad request).
        """


class CallableProvider(Provider):
    """Adapt a plain function: fn(model_spec, request) -> ProviderResponse | str."""

    def __init__(self, name: str, fn: Callable[[ModelSpec, GatewayRequest], Any]) -> None:
        self.name = name
        self.fn = fn

    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        from ..analysis import estimate_tokens
        from ..types import Usage

        out = self.fn(model, request)
        if isinstance(out, ProviderResponse):
            return out
        text = str(out)
        return ProviderResponse(
            text=text, model=model.name,
            usage=Usage(estimate_tokens(request.full_text), estimate_tokens(text)),
            stop_reason="end_turn",
        )


ProviderMap = Dict[str, Provider]
