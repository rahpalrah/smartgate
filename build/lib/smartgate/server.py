"""Language-agnostic HTTP server (stdlib only). Any system that speaks HTTP+JSON can use it.

Endpoints
  POST /v1/generate               {"messages":[...]|"prompt":"...", "system", "max_tokens", "tier", "model", ...}
  POST /v1/analyze                same body, dry run (no model call)
  GET  /v1/stats                  metrics, cache hit rate, spend, circuit breakers
  GET  /v1/costs[?tenant_id=x]    spend report
  POST /v1/cache/invalidate       {"tenant_id": "x"} (omit to clear everything — admin key required)
  GET  /health
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .errors import (AllModelsFailedError, BudgetExceededError, ConfigError, GatewayError,
                     PolicyViolationError, ProviderError)
from .gateway import Gateway
from .types import GatewayRequest, Message

logger = logging.getLogger("smartgate.server")
MAX_BODY = 5 * 1024 * 1024
REQUEST_FIELDS = {"system", "max_tokens", "model", "tier", "use_cache", "metadata", "tags"}


def _status_for(exc: Exception) -> Tuple[int, str]:
    if isinstance(exc, BudgetExceededError):
        return 402, "budget_exceeded"
    if isinstance(exc, PolicyViolationError):
        return 422, "policy_violation"
    if isinstance(exc, AllModelsFailedError):
        return 503, "all_models_failed"
    if isinstance(exc, ProviderError):
        return 502, exc.kind
    if isinstance(exc, (ConfigError, ValueError, TypeError, KeyError)):
        return 400, "invalid_request"
    return 500, "internal_error"


def make_handler(gateway: Gateway, api_keys: Optional[Dict[str, str]] = None,
                 admin_keys: Optional[set] = None):
    """api_keys maps API key -> tenant_id. When set, the tenant comes from the key,
    never from the body, so tenants cannot spend each other's budgets."""
    api_keys = api_keys or {}
    admin_keys = admin_keys or set()

    class Handler(BaseHTTPRequestHandler):
        server_version = "smartgate/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        # ------------------------------------------------------------ helpers
        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _key(self) -> Optional[str]:
            auth = self.headers.get("Authorization", "")
            if auth.lower().startswith("bearer "):
                return auth[7:].strip()
            return self.headers.get("X-API-Key")

        def _tenant(self, body: Dict[str, Any]) -> Optional[str]:
            if not api_keys:
                return body.get("tenant_id", "default")
            return api_keys.get(self._key() or "")

        def _body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def _build_request(self, body: Dict[str, Any], tenant: str) -> GatewayRequest:
            if "messages" in body:
                messages = [Message.coerce(m) for m in body["messages"]]
            elif "prompt" in body:
                messages = [Message("user", str(body["prompt"]))]
            else:
                raise ValueError("provide 'messages' or 'prompt'")
            kwargs = {k: v for k, v in body.items() if k in REQUEST_FIELDS}
            return GatewayRequest(messages=messages, tenant_id=tenant, **kwargs)

        # ------------------------------------------------------------- routes
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path == "/health":
                return self._send(200, {"status": "ok"})
            if api_keys and self._tenant({}) is None:
                return self._send(401, {"error": "unauthorized"})
            if url.path == "/v1/stats":
                return self._send(200, gateway.stats())
            if url.path == "/v1/costs":
                tenant = self._tenant({}) if api_keys else (parse_qs(url.query).get("tenant_id") or [None])[0]
                return self._send(200, gateway.costs.report(tenant))
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                tenant = self._tenant(body)
                if tenant is None:
                    return self._send(401, {"error": "unauthorized"})
                if path == "/v1/generate":
                    return self._send(200, gateway.generate(self._build_request(body, tenant)).to_dict())
                if path == "/v1/analyze":
                    return self._send(200, gateway.analyze(self._build_request(body, tenant)))
                if path == "/v1/cache/invalidate":
                    target = body.get("tenant_id") if not api_keys else tenant
                    if target is None and api_keys and self._key() not in admin_keys:
                        return self._send(403, {"error": "admin key required"})
                    if api_keys and self._key() in admin_keys:
                        target = body.get("tenant_id")
                    return self._send(200, {"removed": gateway.cache.invalidate(target)})
                return self._send(404, {"error": "not_found"})
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid_json"})
            except (GatewayError, ValueError, TypeError, KeyError) as exc:
                status, code = _status_for(exc)
                return self._send(status, {"error": code, "message": str(exc)})
            except Exception as exc:  # pragma: no cover
                logger.exception("unhandled error")
                return self._send(500, {"error": "internal_error", "message": str(exc)})

    return Handler


def serve(gateway: Gateway, host: str = "127.0.0.1", port: int = 8080,
          api_keys: Optional[Dict[str, str]] = None, admin_keys: Optional[set] = None) -> ThreadingHTTPServer:
    class _Server(ThreadingHTTPServer):
        request_queue_size = 1024  # listen backlog; the stdlib default of 5 drops bursts
        daemon_threads = True

    return _Server((host, port), make_handler(gateway, api_keys, admin_keys))
