"""The Gateway: analysis -> policy -> cache -> routing -> budget -> provider w/ fallback."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

from .analysis import PromptAnalysis, PromptAnalyzer, redact_pii
from .cache import CacheConfig, SemanticCache
from .cost import Budget, CostController
from .errors import (AllModelsFailedError, BudgetExceededError, ConfigError,
                     PolicyViolationError, ProviderError)
from .observability import EventBus, Metrics
from .providers.base import Provider
from .resilience import BreakerRegistry, RetryPolicy
from .routing import ModelRouter, RoutingConfig, RoutingRule
from .types import Attempt, GatewayRequest, GatewayResponse, Message, Usage


@dataclass
class PolicyConfig:
    pii: str = "allow"  # "allow" | "redact" | "block"
    max_input_tokens: Optional[int] = None
    blocked_terms: List[str] = field(default_factory=list)


@dataclass
class ResilienceConfig:
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_models_tried: int = 3
    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 30.0


class Gateway:
    """Main entry point.

    >>> gw = Gateway(models=[...], providers={"anthropic": AnthropicProvider()})
    >>> gw.generate("Summarize this ticket ...", tenant_id="acme").text
    """

    def __init__(
        self,
        models: List[ModelSpec],
        providers: Dict[str, Provider],
        *,
        cache: Optional[SemanticCache] = None,
        cache_config: Optional[CacheConfig] = None,
        analyzer: Optional[PromptAnalyzer] = None,
        routing: Optional[RoutingConfig] = None,
        rules: Optional[List[RoutingRule]] = None,
        costs: Optional[CostController] = None,
        default_budget: Optional[Budget] = None,
        policy: Optional[PolicyConfig] = None,
        resilience: Optional[ResilienceConfig] = None,
        sleep=time.sleep,
    ) -> None:
        missing = {m.provider for m in models} - set(providers)
        if missing:
            raise ConfigError(f"No provider registered for: {sorted(missing)}")
        self.providers = providers
        self.router = ModelRouter(models, routing, rules)
        self.cache = cache or SemanticCache(cache_config)
        self.analyzer = analyzer or PromptAnalyzer()
        self.costs = costs or CostController(default_budget=default_budget)
        self.policy = policy or PolicyConfig()
        self.resilience = resilience or ResilienceConfig()
        self.breakers = BreakerRegistry(self.resilience.breaker_failure_threshold,
                                        self.resilience.breaker_recovery_seconds)
        self.events = EventBus()
        self.metrics = Metrics()
        self._sleep = sleep

    # ----------------------------------------------------------- public API
    def on(self, event: str, hook) -> "Gateway":
        self.events.on(event, hook)
        return self

    def analyze(self, prompt: Union[str, GatewayRequest], **kwargs: Any) -> Dict[str, Any]:
        """Dry run: analysis + routing decision, no provider call."""
        request = self._to_request(prompt, **kwargs)
        analysis = self.analyzer.analyze(request)
        decision = self.router.route(request, analysis, self.costs.should_downgrade(request.tenant_id))
        return {"analysis": analysis.to_dict(), "tier": decision.tier, "model": decision.primary.name,
                "fallbacks": [m.name for m in decision.candidates[1:]], "reason": decision.reason,
                "estimated_cost_usd": round(self.costs.estimate(
                    decision.primary, analysis.estimated_input_tokens, request.max_tokens), 6)}

    def generate(self, prompt: Union[str, GatewayRequest, List[Any]], **kwargs: Any) -> GatewayResponse:
        request = self._to_request(prompt, **kwargs)
        rid = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        self.metrics.incr("requests")
        self.events.emit("request.start", {"request_id": rid, "tenant_id": request.tenant_id})
        try:
            response = self._generate(request, rid, started)
        except Exception as exc:
            self.metrics.incr("errors")
            self.events.emit("request.error", {"request_id": rid, "error": repr(exc)})
            raise
        self.metrics.observe_latency(response.latency_ms)
        self.events.emit("request.complete", response.to_dict())
        return response

    async def agenerate(self, prompt: Union[str, GatewayRequest, List[Any]], **kwargs: Any) -> GatewayResponse:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def set_budget(self, tenant_id: str, budget: Budget) -> None:
        self.costs.set_budget(tenant_id, budget)

    def stats(self) -> Dict[str, Any]:
        snap = self.metrics.snapshot()
        snap.update({
            "cache": {**self.cache.stats, "hit_rate": round(self.cache.hit_rate, 4),
                      "size": self.cache.store.size()},
            "cost": {"total_usd": round(self.costs.total_cost, 6),
                     "saved_usd": round(self.costs.total_saved, 6)},
            "circuit_breakers": self.breakers.states(),
        })
        return snap

    # ------------------------------------------------------------- internals
    @staticmethod
    def _to_request(prompt: Any, **kwargs: Any) -> GatewayRequest:
        if isinstance(prompt, GatewayRequest):
            return replace(prompt, **kwargs) if kwargs else prompt
        if isinstance(prompt, str):
            return GatewayRequest(messages=[Message("user", prompt)], **kwargs)
        if isinstance(prompt, list):
            return GatewayRequest(messages=prompt, **kwargs)
        raise TypeError("prompt must be a str, list of messages, or GatewayRequest")

    def _apply_policy(self, request: GatewayRequest, analysis: PromptAnalysis, rid: str):
        lower = request.full_text.lower()
        hits = [t for t in self.policy.blocked_terms if t.lower() in lower]
        if hits:
            self.events.emit("policy.blocked", {"request_id": rid, "reason": "blocked_terms"})
            raise PolicyViolationError("Prompt contains blocked terms", hits)
        if self.policy.max_input_tokens and analysis.estimated_input_tokens > self.policy.max_input_tokens:
            raise PolicyViolationError(
                f"Prompt too long ({analysis.estimated_input_tokens} > {self.policy.max_input_tokens} tokens)")
        if analysis.pii and self.policy.pii == "block":
            self.events.emit("policy.blocked", {"request_id": rid, "pii": analysis.pii})
            raise PolicyViolationError("Prompt contains PII", analysis.pii)
        if analysis.pii and self.policy.pii == "redact":
            request = replace(
                request,
                system=redact_pii(request.system) if request.system else None,
                messages=[Message(m.role, redact_pii(m.content)) for m in request.messages],
            )
            redacted = self.analyzer.analyze(request)
            redacted.signals["pii_redacted"] = 1.0
            self.events.emit("policy.redacted", {"request_id": rid, "pii": analysis.pii})
            self.metrics.incr("pii_redacted")
            return request, redacted
        return request, analysis

    def _cacheable(self, request: GatewayRequest, analysis: PromptAnalysis) -> bool:
        cfg = self.cache.config
        if not (cfg.enabled and request.use_cache):
            return False
        if analysis.pii and not cfg.cache_pii:
            return False
        if analysis.time_sensitive and not cfg.cache_time_sensitive:
            return False
        return True

    def _generate(self, request: GatewayRequest, rid: str, started: float) -> GatewayResponse:
        analysis = self.analyzer.analyze(request)
        request, analysis = self._apply_policy(request, analysis, rid)
        partition = request.model or ""

        # 1) semantic cache
        cacheable = self._cacheable(request, analysis)
        if cacheable:
            hit = self.cache.lookup(request, partition)
            if hit:
                e = hit.entry
                self.metrics.incr("cache_hits")
                self.costs.record(request.tenant_id, e.model, 0.0, 0, 0, cache_hit=True, saved=e.cost_usd)
                self.events.emit("cache.hit", {"request_id": rid, "similarity": hit.similarity,
                                               "exact": hit.exact, "model": e.model})
                return GatewayResponse(
                    text=e.response, model=e.model, provider="cache",
                    usage=Usage(),
                    cost_usd=0.0, cache_hit=True, cache_similarity=hit.similarity,
                    saved_usd=e.cost_usd, route_reason="semantic cache hit" if not hit.exact else "exact cache hit",
                    analysis=analysis.to_dict(), latency_ms=_ms(started), request_id=rid)
            self.metrics.incr("cache_misses")
            self.events.emit("cache.miss", {"request_id": rid})

        # 2) routing (+ budget pressure downgrade)
        pressure = self.costs.should_downgrade(request.tenant_id)
        if pressure:
            self.events.emit("budget.downgrade", {"request_id": rid, "tenant_id": request.tenant_id})
        decision = self.router.route(request, analysis, budget_pressure=pressure)
        self.events.emit("route.decided", {"request_id": rid, "model": decision.primary.name,
                                           "tier": decision.tier, "reason": decision.reason})

        # 3) budget: candidates whose worst-case cost the tenant cannot hold are skipped,
        #    so routing degrades to a cheaper model before refusing outright.
        reason = decision.reason
        attempts: List[Attempt] = []
        tried = 0
        budget_skipped: Dict[str, float] = {}
        for model in decision.candidates:
            if tried >= self.resilience.max_models_tried:
                break
            breaker = self.breakers.get(f"{model.provider}:{model.name}")
            if not breaker.allow():
                attempts.append(Attempt(model.name, model.provider, False, "circuit open"))
                continue
            hold = self.costs.worst_case(model, analysis.estimated_input_tokens,
                                         min(request.max_tokens, model.max_output_tokens))
            token = self.costs.reserve(request.tenant_id, hold)
            if token is None:
                budget_skipped[model.name] = hold
                continue
            if budget_skipped and not attempts:
                reason += f"; budget -> {model.name}"
            tried += 1
            provider = self.providers[model.provider]
            t0 = time.perf_counter()
            try:
                result = self.resilience.retry.run(
                    lambda: provider.complete(model, request),
                    sleep=self._sleep,
                    on_retry=lambda n, exc: self.events.emit(
                        "provider.retry", {"request_id": rid, "model": model.name, "attempt": n, "error": str(exc)}),
                )
            except ProviderError as exc:
                self.costs.release(token)
                breaker.failure()
                attempts.append(Attempt(model.name, model.provider, False, str(exc), _ms(t0)))
                self.metrics.incr(f"provider_errors.{exc.kind}")
                self.events.emit("provider.error", {"request_id": rid, "model": model.name,
                                                    "kind": exc.kind, "error": str(exc)})
                if not exc.fallback:
                    raise
                self.events.emit("fallback", {"request_id": rid, "from": model.name})
                continue
            except BaseException:
                self.costs.release(token)
                raise
            breaker.success()
            attempts.append(Attempt(model.name, model.provider, True, None, _ms(t0)))
            if any(not a.ok for a in attempts):
                self.metrics.incr("fallbacks_used")
                reason += f"; fell back to {model.name}"

            cost = model.cost(result.usage.input_tokens, result.usage.output_tokens)
            self.costs.settle(token, cost, model.name, result.usage.input_tokens, result.usage.output_tokens)
            self.metrics.incr(f"model.{model.name}")
            if cacheable and result.stop_reason in (None, "end_turn", "stop_sequence"):
                if self.cache.store_response(request, result.text, model.name, cost,
                                             result.usage.input_tokens, result.usage.output_tokens, partition):
                    self.events.emit("cache.store", {"request_id": rid})
            return GatewayResponse(
                text=result.text, model=model.name, provider=model.provider, usage=result.usage,
                cost_usd=cost, route_reason=reason, analysis=analysis.to_dict(),
                attempts=attempts, latency_ms=_ms(started), request_id=rid)

        if budget_skipped and not any(a.error and a.error != "circuit open" for a in attempts):
            budget = self.costs.budget_for(request.tenant_id)
            self.events.emit("budget.exceeded", {"request_id": rid, "tenant_id": request.tenant_id})
            raise BudgetExceededError(request.tenant_id, self.costs.spent(request.tenant_id),
                                      budget.limit_usd if budget else 0.0, min(budget_skipped.values()))
        raise AllModelsFailedError(attempts)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
