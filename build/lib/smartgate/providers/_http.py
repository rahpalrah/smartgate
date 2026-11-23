"""Minimal HTTP+JSON client on the standard library (urllib), shared by the built-in providers."""
from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from ..errors import ProviderError


def parse_retry_after(headers: Any) -> Optional[float]:
    try:
        return float(headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def error_from_status(status: int, message: str, retry_after: Optional[float] = None) -> ProviderError:
    """Map an HTTP status to a ProviderError that tells the gateway whether to retry or fail over."""
    msg = f"api error {status}: {message}"[:500]
    if status in (400, 413, 422):  # the request itself is wrong; another model cannot fix it
        return ProviderError(msg, fallback=False, kind="bad_request", status_code=status)
    if status in (401, 403):  # credentials for this provider are broken
        return ProviderError(msg, kind="auth", status_code=status)
    if status == 404:
        return ProviderError(msg, kind="not_found", status_code=status)
    if status == 429:
        return ProviderError(msg, retryable=True, kind="rate_limit", status_code=status, retry_after=retry_after)
    if status in (408, 409) or status >= 500:  # includes 529 "overloaded"
        return ProviderError(msg, retryable=True, kind="timeout" if status == 408 else "server",
                             status_code=status, retry_after=retry_after)
    return ProviderError(msg, kind="error", status_code=status)


def post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:  # must precede URLError (its base class)
        try:
            detail = json.loads(exc.read() or b"{}")
            message = (detail.get("error") or {}).get("message") if isinstance(detail.get("error"), dict) \
                else str(detail.get("error") or detail)
        except (ValueError, AttributeError):
            message = exc.reason if isinstance(exc.reason, str) else "http error"
        raise error_from_status(exc.code, message or "http error", parse_retry_after(exc.headers)) from None
    except (socket.timeout, TimeoutError):
        raise ProviderError("timeout", retryable=True, kind="timeout") from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise ProviderError("timeout", retryable=True, kind="timeout") from None
        raise ProviderError(f"connection error: {exc.reason}", retryable=True, kind="connection") from None
    except OSError as exc:  # connection reset, broken pipe, ...
        raise ProviderError(f"connection error: {exc}", retryable=True, kind="connection") from None
    try:
        parsed = json.loads(raw)
    except ValueError:
        raise ProviderError("provider returned invalid JSON", retryable=True, kind="server") from None
    if not isinstance(parsed, dict):
        raise ProviderError("provider returned an unexpected response", retryable=True, kind="server")
    return parsed
