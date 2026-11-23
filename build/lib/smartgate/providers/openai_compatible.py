"""Provider for any server that speaks the common `/chat/completions` JSON protocol.

Use it for self-hosted and local models with no extra packages: Ollama, vLLM, LM Studio, the
llama.cpp server, a company-internal gateway... Standard library only.

    OpenAICompatibleProvider("http://localhost:11434/v1")               # Ollama
    OpenAICompatibleProvider("http://llm.internal:8000/v1", api_key="...")  # vLLM behind a key
"""
from __future__ import annotations

from typing import Dict, Optional

from ..types import GatewayRequest, ModelSpec, ProviderResponse, Usage
from ._http import post_json
from .base import Provider


class OpenAICompatibleProvider(Provider):
    name = "openai_compatible"

    def __init__(self, base_url: str = "http://localhost:11434/v1", api_key: Optional[str] = None,
                 timeout: float = 120.0, extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})

    def complete(self, model: ModelSpec, request: GatewayRequest) -> ProviderResponse:
        messages = ([{"role": "system", "content": request.system}] if request.system else []) + \
                   [{"role": m.role, "content": m.content} for m in request.messages]
        payload = dict(model=model.name, messages=messages,
                       max_tokens=min(request.max_tokens, model.max_output_tokens), **model.params)
        headers = dict(self.extra_headers)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = post_json(f"{self.base_url}/chat/completions", headers, payload, self.timeout)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        return ProviderResponse(
            text=text, model=data.get("model") or model.name,
            usage=Usage(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))),
            stop_reason={"stop": "end_turn", "length": "max_tokens"}.get(choice.get("finish_reason"),
                                                                       choice.get("finish_reason")),
            raw=data)
