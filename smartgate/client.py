"""Thin Python client for a remote smartgate server (stdlib only)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class GatewayHTTPError(Exception):
    def __init__(self, status: int, payload: Dict[str, Any]) -> None:
        super().__init__(f"HTTP {status}: {payload}")
        self.status = status
        self.payload = payload


class GatewayClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", api_key: Optional[str] = None,
                 timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _call(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise GatewayHTTPError(e.code, json.loads(e.read() or b"{}")) from None

    def generate(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None,
                 **kwargs: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = dict(kwargs)
        if messages is not None:
            body["messages"] = messages
        else:
            body["prompt"] = prompt
        return self._call("POST", "/v1/generate", body)

    def analyze(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return self._call("POST", "/v1/analyze", {"prompt": prompt, **kwargs})

    def stats(self) -> Dict[str, Any]:
        return self._call("GET", "/v1/stats")

    def costs(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return self._call("GET", "/v1/costs" + (f"?tenant_id={tenant_id}" if tenant_id else ""))

    def invalidate_cache(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return self._call("POST", "/v1/cache/invalidate", {"tenant_id": tenant_id} if tenant_id else {})
