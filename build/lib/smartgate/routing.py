"""Model selection based on prompt complexity, rules, capabilities and budget."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .analysis import PromptAnalysis
from .errors import ConfigError
from .types import TIERS, GatewayRequest, ModelSpec


@dataclass
class RoutingRule:
    """If `when(request, analysis)` is true, route to `tier` and/or `model`.

    Rules are evaluated in order; the first match wins. Use `min_tier` to set a
    floor instead of an exact tier (e.g. "clinical questions never go below medium").
    """

    name: str
    when: Callable[[GatewayRequest, PromptAnalysis], bool]
    tier: Optional[str] = None
    min_tier: Optional[str] = None
    model: Optional[str] = None


@dataclass
class RoutingConfig:
    # complexity < small_max -> small; < medium_max -> medium; else large
    small_max: float = 0.35
    medium_max: float = 0.70
    min_tier: str = "small"  # global floor
    max_tier: str = "large"  # global ceiling
    downgrade_on_budget_pressure: bool = True
    fallback_strategy: str = "same_tier_then_up"  # | "same_tier_then_down" | "any"
    required_capabilities: Dict[str, List[str]] = field(default_factory=dict)  # task -> caps


@dataclass
class RouteDecision:
    primary: ModelSpec
    candidates: List[ModelSpec]  # primary first, then fallbacks in order
    tier: str
    reason: str


def tier_index(tier: str) -> int:
    return TIERS.index(tier)


class ModelRouter:
    def __init__(self, models: List[ModelSpec], config: Optional[RoutingConfig] = None,
                 rules: Optional[List[RoutingRule]] = None) -> None:
        if not models:
            raise ConfigError("ModelRouter needs at least one model")
        self.models = list(models)
        self.config = config or RoutingConfig()
        self.rules = list(rules or [])

    def get(self, name: str) -> Optional[ModelSpec]:
        return next((m for m in self.models if m.name == name), None)

    def add_rule(self, rule: RoutingRule) -> None:
        self.rules.append(rule)

    def tier_for_complexity(self, complexity: float) -> str:
        if complexity < self.config.small_max:
            return "small"
        if complexity < self.config.medium_max:
            return "medium"
        return "large"

    def _clamp(self, tier: str) -> str:
        lo, hi = tier_index(self.config.min_tier), tier_index(self.config.max_tier)
        return TIERS[max(lo, min(hi, tier_index(tier)))]

    def _eligible(self, request: GatewayRequest, analysis: PromptAnalysis) -> List[ModelSpec]:
        need = analysis.estimated_input_tokens + request.max_tokens
        caps = set(self.config.required_capabilities.get(analysis.task_type, []))
        return [m for m in self.models
                if m.enabled and m.max_context_tokens >= need and caps.issubset(m.capabilities)]

    def route(self, request: GatewayRequest, analysis: PromptAnalysis,
              budget_pressure: bool = False, floor_tier: Optional[str] = None) -> RouteDecision:
        eligible = self._eligible(request, analysis)
        if not eligible:
            raise ConfigError("No enabled model fits this request (context window / capabilities)")

        reasons: List[str] = []
        forced_model: Optional[ModelSpec] = None

        if request.model:
            forced_model = self.get(request.model)
            if not forced_model:
                raise ConfigError(f"Unknown model '{request.model}'")
            tier = forced_model.tier
            reasons.append(f"model forced by request ({request.model})")
        elif request.tier:
            tier = request.tier
            reasons.append(f"tier forced by request ({tier})")
        else:
            tier = self.tier_for_complexity(analysis.complexity)
            reasons.append(f"complexity {analysis.complexity:.2f} -> {tier}")
            for rule in self.rules:
                if rule.when(request, analysis):
                    if rule.model and self.get(rule.model):
                        forced_model = self.get(rule.model)
                        tier = forced_model.tier
                    elif rule.tier:
                        tier = rule.tier
                    elif rule.min_tier and tier_index(tier) < tier_index(rule.min_tier):
                        tier = rule.min_tier
                    reasons.append(f"rule '{rule.name}' -> {forced_model.name if forced_model else tier}")
                    break
            if budget_pressure and self.config.downgrade_on_budget_pressure and tier_index(tier) > 0:
                tier = TIERS[tier_index(tier) - 1]
                reasons.append(f"budget pressure -> downgraded to {tier}")
            if floor_tier and tier_index(tier) < tier_index(floor_tier):
                tier = floor_tier
                reasons.append(f"quality floor -> {tier} (cheaper tier below quality bar)")
            clamped = self._clamp(tier)
            if clamped != tier:
                reasons.append(f"clamped to {clamped}")
                tier = clamped

        candidates = self._order_candidates(eligible, tier, forced_model)
        return RouteDecision(primary=candidates[0], candidates=candidates, tier=tier,
                             reason="; ".join(reasons))

    def _order_candidates(self, eligible: List[ModelSpec], tier: str,
                          forced: Optional[ModelSpec]) -> List[ModelSpec]:
        t = tier_index(tier)
        strategy = self.config.fallback_strategy

        def key(m: ModelSpec):
            d = tier_index(m.tier) - t
            if strategy == "same_tier_then_down":
                rank = (0, 0) if d == 0 else ((1, -d) if d < 0 else (2, d))
            elif strategy == "any":
                rank = (abs(d), 0)
            else:  # same_tier_then_up: escalate for quality, then degrade for availability
                rank = (0, 0) if d == 0 else ((1, d) if d > 0 else (2, -d))
            return (*rank, m.cost(1000, 500))

        ordered = sorted(eligible, key=key)
        if forced:
            ordered = [forced] + [m for m in ordered if m.name != forced.name]
        return ordered
