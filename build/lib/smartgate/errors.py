"""Exception hierarchy."""
from __future__ import annotations

from typing import List, Optional


class GatewayError(Exception):
    """Base class for all SDK errors."""


class ConfigError(GatewayError):
    pass


class ProviderError(GatewayError):
    """Raised by providers. `retryable` drives retry; `fallback` drives failover."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        fallback: bool = True,
        kind: str = "error",
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.fallback = fallback
        self.kind = kind  # "rate_limit" | "timeout" | "server" | "refusal" | "bad_request" | "auth" | "error"
        self.status_code = status_code
        self.retry_after = retry_after


class BudgetExceededError(GatewayError):
    def __init__(self, tenant_id: str, spent: float, limit: float, estimate: float) -> None:
        super().__init__(
            f"Budget exceeded for tenant '{tenant_id}': spent ${spent:.4f} of "
            f"${limit:.4f}, request estimate ${estimate:.4f}"
        )
        self.tenant_id = tenant_id
        self.spent = spent
        self.limit = limit
        self.estimate = estimate


class PolicyViolationError(GatewayError):
    """Raised when a prompt violates a configured policy (e.g. PII blocked)."""

    def __init__(self, message: str, findings: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.findings = findings or []


class AllModelsFailedError(GatewayError):
    def __init__(self, attempts: list) -> None:
        summary = "; ".join(f"{a.model}: {a.error}" for a in attempts) or "no candidates"
        super().__init__(f"All models failed: {summary}")
        self.attempts = attempts
