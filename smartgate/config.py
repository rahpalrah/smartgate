"""Declarative configuration (dict / JSON / TOML) and industry presets."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional

from .analysis import PromptAnalyzer
from .cache import CacheConfig, InMemoryVectorStore, SemanticCache, SQLiteVectorStore
from .cost import Budget, CostController
from .errors import ConfigError
from .gateway import Gateway, PolicyConfig, ResilienceConfig
from .providers.base import Provider
from .resilience import RetryPolicy
from .routing import RoutingConfig, RoutingRule
from .types import ModelSpec

# Default catalog: Claude models, list prices in USD per million tokens.
DEFAULT_MODELS: List[Dict[str, Any]] = [
    {"name": "claude-haiku-5-5", "provider": "anthropic", "tier": "small",
     "input_cost_per_mtok": 0.10, "output_cost_per_mtok": 0.50,
     "max_context_tokens": 1_000_000, "max_output_tokens": 128_000,
     "capabilities": ["text", "vision", "tools"]},
    {"name": "claude-sonnet-5-5", "provider": "anthropic", "tier": "medium",
     "input_cost_per_mtok": 2.00, "output_cost_per_mtok": 10.00,
     "max_context_tokens": 1_000_000, "max_output_tokens": 128_000,
     "capabilities": ["text", "vision", "tools", "code"]},
    {"name": "claude-opus-5-5", "provider": "anthropic", "tier": "large",
     "input_cost_per_mtok": 4.00, "output_cost_per_mtok": 20.00,
     "max_context_tokens": 1_000_000, "max_output_tokens": 128_000,
     "capabilities": ["text", "vision", "tools", "code", "reasoning"]},
]

# Industry presets. Each is a partial config merged under the user's config.
PRESETS: Dict[str, Dict[str, Any]] = {
    "general": {},
    "healthcare": {
        # Wrong cached answers are costly: only near-identical questions hit.
        "cache": {"similarity_threshold": 0.97, "ttl_seconds": 6 * 3600, "cache_pii": False},
        "policy": {"pii": "redact"},
        "routing": {"min_tier": "medium"},
        "domain_keywords": {"diagnosis": 0.25, "differential": 0.25, "dosage": 0.3,
                            "contraindication": 0.3, "drug interaction": 0.3, "symptom": 0.1},
        "quality": {"grounding": True, "adaptive_routing": {"min_score": 0.85, "min_samples": 30}},
        "rules": [{"name": "clinical-safety", "keywords": ["dosage", "contraindication",
                                                           "drug interaction", "overdose"],
                   "tier": "large"}],
    },
    "finance": {
        "cache": {"similarity_threshold": 0.95, "ttl_seconds": 900, "cache_time_sensitive": False},
        "policy": {"pii": "redact"},
        "domain_keywords": {"portfolio": 0.15, "risk": 0.1, "valuation": 0.2, "regulation": 0.2,
                            "derivative": 0.2, "compliance": 0.2, "audit": 0.15},
        "quality": {"grounding": True, "adaptive_routing": {"min_score": 0.85, "min_samples": 30}},
        "rules": [{"name": "regulatory", "keywords": ["sec filing", "aml", "kyc", "basel", "compliance"],
                   "min_tier": "medium"}],
    },
    "ecommerce": {
        "cache": {"similarity_threshold": 0.86, "ttl_seconds": 24 * 3600},
        "policy": {"pii": "redact"},
        "routing": {"small_max": 0.45, "medium_max": 0.80},
        "quality": {"grounding": True, "adaptive_routing": {"min_score": 0.8, "min_samples": 30}},
        "rules": [{"name": "catalog-copy", "task_types": ["creative"], "tier": "medium"}],
    },
    "customer_support": {
        "cache": {"similarity_threshold": 0.88, "ttl_seconds": 12 * 3600},
        "policy": {"pii": "redact"},
        "routing": {"small_max": 0.45},
        "quality": {"grounding": True, "adaptive_routing": {"min_score": 0.8, "min_samples": 30}},
        "rules": [{"name": "escalation", "keywords": ["refund", "legal action", "cancel my account",
                                                      "complaint"], "min_tier": "medium"}],
    },
    "legal": {
        "cache": {"similarity_threshold": 0.97, "ttl_seconds": 7 * 24 * 3600},
        "policy": {"pii": "redact"},
        "routing": {"min_tier": "medium"},
        "domain_keywords": {"clause": 0.2, "indemnif": 0.25, "liability": 0.2, "jurisdiction": 0.2,
                            "precedent": 0.25, "statute": 0.2},
    },
    "education": {
        "cache": {"similarity_threshold": 0.88, "ttl_seconds": 7 * 24 * 3600},
        "policy": {"pii": "redact"},
        "rules": [{"name": "math-proofs", "task_types": ["math"], "min_tier": "medium"}],
    },
    "saas_devtools": {
        "cache": {"similarity_threshold": 0.93},
        "routing": {"required_capabilities": {"code": ["code"]}},
        "rules": [{"name": "code-gen", "task_types": ["code"], "min_tier": "medium"}],
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        elif k == "rules" and isinstance(out.get(k), list):
            out[k] = list(out[k]) + list(v)  # user rules appended after preset rules
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON (.json) or TOML (.toml) config file. Both are read by the standard library."""
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        raise ConfigError("YAML is not supported (smartgate has no dependencies). Use a .json or .toml file.")
    if lower.endswith(".toml"):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    with open(path) as f:
        return json.loads(f.read())


def _rule_from_dict(d: Dict[str, Any]) -> RoutingRule:
    keywords = [k.lower() for k in d.get("keywords", [])]
    task_types = set(d.get("task_types", []))
    min_c = d.get("min_complexity")
    tenants = set(d.get("tenants", []))
    tags = set(d.get("tags", []))

    def when(request, analysis) -> bool:
        if tenants and request.tenant_id not in tenants:
            return False
        if tags and not tags.intersection(request.tags):
            return False
        if task_types and analysis.task_type not in task_types:
            return False
        if keywords and not any(k in request.last_user_message.lower() for k in keywords):
            return False
        if min_c is not None and analysis.complexity < min_c:
            return False
        return bool(keywords or task_types or min_c is not None or tenants or tags)

    return RoutingRule(name=d["name"], when=when, tier=d.get("tier"),
                       min_tier=d.get("min_tier"), model=d.get("model"))


def build_providers(names: set, overrides: Optional[Dict[str, Provider]]) -> Dict[str, Provider]:
    providers = dict(overrides or {})
    for name in names - set(providers):
        if name == "anthropic":
            from .providers.anthropic_provider import AnthropicProvider
            providers[name] = AnthropicProvider()
        elif name == "ollama":
            from .providers.ollama import OllamaProvider
            providers[name] = OllamaProvider()
        elif name == "mock":
            from .providers.mock import MockProvider
            providers[name] = MockProvider()
        else:
            raise ConfigError(f"Provider '{name}' must be passed in `providers=`")
    return providers


def gateway_from_config(config: Dict[str, Any], providers: Optional[Dict[str, Provider]] = None,
                        verifier=None, **gateway_kwargs: Any) -> Gateway:
    preset_name = config.get("preset", "general")
    if preset_name not in PRESETS:
        raise ConfigError(f"Unknown preset '{preset_name}'. Available: {sorted(PRESETS)}")
    cfg = deep_merge(PRESETS[preset_name], config)

    model_dicts = cfg.get("models") or DEFAULT_MODELS
    if cfg.get("provider_override"):  # e.g. run the default catalog against "mock"
        model_dicts = [{**m, "provider": cfg["provider_override"]} for m in model_dicts]
    models = [ModelSpec(**m) for m in model_dicts]

    cache_cfg = CacheConfig(**cfg.get("cache", {}))
    store_cfg = cfg.get("cache_store", {"type": "memory"})
    if store_cfg.get("type") == "sqlite":
        store = SQLiteVectorStore(store_cfg.get("path", "smartgate_cache.db"), cache_cfg.max_entries)
    else:
        store = InMemoryVectorStore(cache_cfg.max_entries)

    res = cfg.get("resilience", {})
    resilience = ResilienceConfig(
        retry=RetryPolicy(**res.get("retry", {})),
        max_models_tried=res.get("max_models_tried", 3),
        breaker_failure_threshold=res.get("breaker_failure_threshold", 5),
        breaker_recovery_seconds=res.get("breaker_recovery_seconds", 30.0),
    )
    costs = CostController(
        budgets={t: Budget(**b) for t, b in cfg.get("budgets", {}).items()},
        default_budget=Budget(**cfg["default_budget"]) if cfg.get("default_budget") else None,
    )
    q = cfg.get("quality", {})
    from .quality import GroundingValidator, QualityTracker, chain, default_validator
    validator = chain(default_validator, GroundingValidator(q.get("known_facts")) if q.get("grounding") else None)
    adaptive = q.get("adaptive_routing")
    gateway_kwargs.setdefault("validator", validator)
    if adaptive:
        gateway_kwargs.setdefault("quality_tracker", QualityTracker(**adaptive))
    return Gateway(
        models=models,
        providers=build_providers({m.provider for m in models}, providers),
        cache=SemanticCache(cache_cfg, store, verifier=verifier),
        analyzer=PromptAnalyzer(domain_keywords=cfg.get("domain_keywords")),
        routing=RoutingConfig(**cfg.get("routing", {})),
        rules=[_rule_from_dict(r) for r in cfg.get("rules", [])],
        costs=costs,
        policy=PolicyConfig(**cfg.get("policy", {})),
        resilience=resilience,
        **gateway_kwargs,
    )


def gateway_from_file(path: str, providers: Optional[Dict[str, Provider]] = None) -> Gateway:
    return gateway_from_config(load_config_file(path), providers)


def gateway_from_preset(preset: str, providers: Optional[Dict[str, Provider]] = None,
                        **overrides: Any) -> Gateway:
    return gateway_from_config({"preset": preset, **overrides}, providers)


def gateway_from_env(providers: Optional[Dict[str, Provider]] = None) -> Gateway:
    """SMARTGATE_CONFIG=path.json or SMARTGATE_PRESET=healthcare."""
    path = os.environ.get("SMARTGATE_CONFIG")
    if path:
        return gateway_from_file(path, providers)
    return gateway_from_preset(os.environ.get("SMARTGATE_PRESET", "general"), providers)
