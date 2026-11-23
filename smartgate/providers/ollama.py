"""Ollama (https://ollama.com) as a fully local, free model backend. Standard library only.

Ollama is a separate program you run on your own machine; smartgate talks to it over local HTTP
(its `/v1/chat/completions` endpoint), so nothing is added to smartgate's dependencies.

    ollama pull qwen2.5:1.5b qwen2.5:3b qwen2.5:7b        # once, in a terminal
    python -m smartgate --ollama ask "How do I reset my password?"

    from smartgate import gateway_from_ollama
    gw = gateway_from_ollama()                             # small / medium / large = Qwen2.5 1.5b / 3b / 7b
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import urlparse

from ..errors import ConfigError
from ..types import ModelSpec
from ._http import get_json
from .openai_compatible import OpenAICompatibleProvider

DEFAULT_HOST = "http://localhost:11434"

# (small, medium, large) Ollama model tags. Qwen2.5 sizes: 0.5b 1.5b 3b 7b 14b 32b 72b.
QWEN_PROFILES: Dict[str, tuple] = {
    "light": ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"),      # about 4 GB of RAM; weakest answers
    "standard": ("qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"),     # about 8 GB of RAM; sensible default
    "strong": ("qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"),        # about 16 GB of RAM
}
# Claude list prices (USD per million tokens, in / out) used only when shadow_prices=True
SHADOW_PRICES = (("claude-haiku-5-5", 0.10, 0.50), ("claude-sonnet-5-5", 2.00, 10.00), ("claude-opus-5-5", 4.00, 20.00))
TIERS = ("small", "medium", "large")


def normalize_host(host: Optional[str] = None) -> str:
    """Resolve the Ollama address: argument, else the OLLAMA_HOST variable, else localhost:11434.

    Accepts the forms Ollama itself accepts ("host", "host:port", "http://host:port") with Ollama's own
    port defaults. A server bind address such as 0.0.0.0 is turned into 127.0.0.1 for connecting.
    """
    raw = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).strip()
    explicit_scheme = "://" in raw
    if not explicit_scheme:
        raw = "http://" + raw
    parsed = urlparse(raw)
    hostname = parsed.hostname or "127.0.0.1"
    if hostname in ("0.0.0.0", "::"):
        hostname = "127.0.0.1"
    # Same defaults as Ollama: no scheme -> port 11434; an explicit scheme with no port -> 80 / 443.
    port = parsed.port or ((80 if parsed.scheme == "http" else 443) if explicit_scheme else 11434)
    if ":" in hostname:
        hostname = f"[{hostname}]"
    return f"{parsed.scheme}://{hostname}:{port}"


def _tag(name: str) -> str:
    """Ollama treats a bare name as :latest."""
    return name if ":" in name else name + ":latest"


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"

    def __init__(self, host: Optional[str] = None, timeout: float = 300.0, api_key: Optional[str] = None,
                 extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.host = normalize_host(host)
        # Generous default timeout: the first request after idle loads the model into memory.
        super().__init__(self.host + "/v1", api_key=api_key, timeout=timeout, extra_headers=extra_headers)

    def installed_models(self, timeout: float = 5.0) -> List[str]:
        """Names of the models the server has. Raises ProviderError(kind="connection") if it is not running."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            data = get_json(f"{self.host}/v1/models", headers, timeout)
            names = [m.get("id") for m in data.get("data") or [] if isinstance(m, dict)]
        except Exception as first:
            if getattr(first, "kind", None) == "connection":
                raise
            data = get_json(f"{self.host}/api/tags", headers, timeout)  # older Ollama versions
            names = [m.get("name") for m in data.get("models") or [] if isinstance(m, dict)]
        return sorted(n for n in names if n)

    def missing_models(self, models: Iterable[Union[str, ModelSpec]], timeout: float = 5.0) -> List[str]:
        have = {_tag(n) for n in self.installed_models(timeout)}
        wanted = [m.name if isinstance(m, ModelSpec) else m for m in models]
        return [w for w in wanted if _tag(w) not in have]


def qwen_models(profile: str = "standard", models: Optional[Iterable[str]] = None, context_tokens: int = 4096,
                max_output_tokens: int = 2048, shadow_prices: bool = False, provider: str = "ollama") -> List[ModelSpec]:
    """Small / medium / large model specs backed by Qwen models on Ollama.

    profile          "light" | "standard" | "strong" (see QWEN_PROFILES), or pass `models=` with three tags.
    context_tokens   the context window you run Ollama with. Ollama's default is small (4096 in current
                     versions); raise it with OLLAMA_CONTEXT_LENGTH and pass the same number here, or long
                     prompts are silently cut off by Ollama.
    shadow_prices    price the local models at Claude Haiku / Sonnet / Opus list prices, so cost reports and
                     budgets show what the same traffic would cost on Claude. Off = every call costs $0.
    """
    names = tuple(models) if models is not None else QWEN_PROFILES.get(profile)
    if not names or len(names) != 3:
        raise ConfigError(f"profile must be one of {sorted(QWEN_PROFILES)}, or give exactly 3 model tags")
    out = []
    for tier, name, (_, price_in, price_out) in zip(TIERS, names, SHADOW_PRICES):
        out.append(ModelSpec(name, provider, tier, price_in if shadow_prices else 0.0,
                             price_out if shadow_prices else 0.0, max_context_tokens=context_tokens,
                             max_output_tokens=max_output_tokens, capabilities=["text"]))
    return out


def gateway_from_ollama(profile: str = "standard", host: Optional[str] = None, preset: str = "general",
                        models: Optional[Iterable[str]] = None, shadow_prices: bool = False,
                        context_tokens: int = 4096, timeout: float = 300.0, check: bool = False,
                        verify_with: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None,
                        **gateway_kwargs: Any):
    """Build a Gateway whose three tiers are Qwen models served by Ollama.

    `preset` is any industry preset ("customer_support", "healthcare", ...); `overrides` is a config
    dict merged on top (cache, budgets, rules, policy...). With `check=True` it first verifies the
    server is reachable and has the models, and says which `ollama pull` commands to run if not.

    `verify_with="medium"` (or "large") makes that Qwen model double-check every semantic cache hit
    ("is one answer correct for both questions?") before it is served. Do not use "small" with the
    0.5b model: on a 12-pair test it was right only about 2 times in 3, while 1.5b and 3b were right
    on all 12. Costs one short local call per candidate hit.
    """
    from ..config import gateway_from_config
    provider = OllamaProvider(host, timeout=timeout)
    specs = qwen_models(profile, models, context_tokens, shadow_prices=shadow_prices)
    if check:
        try:
            missing = provider.missing_models(specs)
        except Exception as exc:
            raise ConfigError(f"cannot reach Ollama at {provider.host} ({exc}). Is it running? "
                              f"Start it with `ollama serve`.") from None
        if missing:
            raise ConfigError("Ollama is missing models: " + ", ".join(missing) + ". Run: "
                              + "; ".join(f"ollama pull {m}" for m in missing))
    if verify_with:
        if verify_with not in TIERS:
            raise ConfigError(f"verify_with must be one of {TIERS}")
        from ..cache.verify import MemoizedVerifier, provider_verifier
        judge = next(m for m in specs if m.tier == verify_with)
        gateway_kwargs.setdefault("verifier", MemoizedVerifier(provider_verifier(provider, judge)))
    config = dict(overrides or {})
    config["preset"] = preset
    config["models"] = [asdict(s) for s in specs]
    return gateway_from_config(config, providers={"ollama": provider}, **gateway_kwargs)
