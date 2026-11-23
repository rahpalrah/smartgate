"""Claude provider. Talks to the Anthropic Messages API directly over HTTPS (standard library only).

No SDK is required. Credentials come from the constructor or the environment:
  ANTHROPIC_API_KEY (sent as x-api-key)  or  ANTHROPIC_AUTH_TOKEN (sent as a Bearer token)
  ANTHROPIC_BASE_URL (optional; default https://api.anthropic.com)

Retries and failover are handled by the gateway, so this provider makes exactly one request.
Per-model extra request fields come from `ModelSpec.params`, e.g. {"output_config": {"effort": "low"}}.

Already using an Anthropic SDK client (including its Bedrock / Vertex / Foundry variants)?
Pass it as `client=` and it is used instead of HTTP; the SDK is never imported by smartgate.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..errors import ProviderError
from ..types import GatewayRequest, ModelSpec, ProviderResponse, Usage
from ._http import error_from_status, parse_retry_after, post_json
from .base import Provider

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


def map_sdk_exception(exc: Exception) -> Optional[ProviderError]:
    """Translate an exception raised by an SDK-style client without importing the SDK."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        headers = getattr(getattr(exc, "response", None), "headers", None)
        return error_from_status(status, str(getattr(exc, "message", exc)), parse_retry_after(headers))
    name = type(exc).__name__
    if "Timeout" in name:
        return ProviderError("timeout", retryable=True, kind="timeout")
    if "Connection" in name:
        return ProviderError("connection error", retryable=True, kind="connection")
    return None


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 120.0,
                 client: Any = None, anthropic_version: str = ANTHROPIC_VERSION,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.client = client
        self.anthropic_version = anthropic_version
        self.extra_headers = dict(extra_headers or {})

    # ------------------------------------------------------------------ request
    def _payload(self, model: ModelSpec, request: GatewayRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(
            model=model.name,
            max_tokens=min(request.max_tokens, model.max_output_tokens),
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            **model.params,
        )
        if request.system:
            payload["system"] = request.system
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"anthropic-version": self.anthropic_version, **self.extra_headers}
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if key:
            headers["x-api-key"] = key
        elif token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            raise ProviderError("no credentials: set ANTHROPIC_API_KEY (or pass api_key=)", fallback=False,
                                kind="auth")
        return headers

    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        payload = self._payload(model, request)
        if self.client is not None:
            return self._complete_with_client(payload)
        base = (self.base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        data = post_json(f"{base}/v1/messages", self._headers(), payload, self.timeout)
        text = "".join(b.get("text", "") for b in data.get("content") or []
                       if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        return self._finish(text, data.get("model") or model.name, int(usage.get("input_tokens", 0)),
                            int(usage.get("output_tokens", 0)), data.get("stop_reason"),
                            (data.get("stop_details") or {}).get("category"), data)

    def _complete_with_client(self, payload: Dict[str, Any]) -> ProviderResponse:
        try:
            response = self.client.messages.create(**payload)
        except ProviderError:
            raise
        except Exception as exc:
            mapped = map_sdk_exception(exc)
            if mapped is None:
                raise
            raise mapped from None
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        category = getattr(getattr(response, "stop_details", None), "category", None)
        return self._finish(text, response.model, response.usage.input_tokens, response.usage.output_tokens,
                            response.stop_reason, category, response)

    @staticmethod
    def _finish(text, model_name, input_tokens, output_tokens, stop_reason, category, raw) -> ProviderResponse:
        if stop_reason == "refusal":
            # Not retryable on the same model; a different model may answer.
            raise ProviderError(f"refusal ({category})", kind="refusal", fallback=True)
        return ProviderResponse(text=text, model=model_name, usage=Usage(input_tokens, output_tokens),
                                stop_reason=stop_reason, raw=raw)
