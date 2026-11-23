"""Cost controls: per-tenant budgets, spend ledger, per-request caps."""
from __future__ import annotations

import itertools
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Protocol

from .types import ModelSpec


@dataclass
class Budget:
    limit_usd: float
    period: str = "monthly"  # "daily" | "monthly" | "total"
    hard_limit: bool = True  # False => alert only, never block
    downgrade_at: float = 0.8  # fraction of budget after which routing steps down a tier
    max_cost_per_request: Optional[float] = None

    def __post_init__(self) -> None:
        if self.period not in ("daily", "monthly", "total"):
            raise ValueError("Budget.period must be daily | monthly | total")


def period_key(period: str, now: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(now or time.time(), tz=timezone.utc)
    if period == "daily":
        return dt.strftime("%Y-%m-%d")
    if period == "monthly":
        return dt.strftime("%Y-%m")
    return "total"


class SpendLedger(Protocol):
    """Persist spend wherever you like (Redis, Postgres, DynamoDB...)."""

    def add(self, tenant_id: str, period_key: str, amount: float) -> None: ...
    def get(self, tenant_id: str, period_key: str) -> float: ...


class InMemoryLedger:
    def __init__(self) -> None:
        self._data: Dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def add(self, tenant_id: str, period_key: str, amount: float) -> None:
        with self._lock:
            self._data[(tenant_id, period_key)] += amount

    def get(self, tenant_id: str, period_key: str) -> float:
        return self._data.get((tenant_id, period_key), 0.0)


@dataclass
class CostRecord:
    tenant_id: str
    model: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    saved_usd: float
    timestamp: float


class CostController:
    def __init__(self, budgets: Optional[Dict[str, Budget]] = None,
                 default_budget: Optional[Budget] = None,
                 ledger: Optional[SpendLedger] = None,
                 keep_records: int = 10_000, input_safety_margin: float = 1.25) -> None:
        self.budgets: Dict[str, Budget] = dict(budgets or {})
        self.default_budget = default_budget
        self.ledger: SpendLedger = ledger or InMemoryLedger()
        self.records: List[CostRecord] = []
        self.keep_records = keep_records
        self._lock = threading.RLock()
        self.total_cost = 0.0
        self.total_saved = 0.0
        # Hard limits are enforced by *reserving* the worst-case cost of a call before it
        # runs (atomic check-and-hold), then settling the actual cost afterwards. Without
        # this, N concurrent requests can all pass the check and overspend together.
        self.input_safety_margin = input_safety_margin
        self._reservations: Dict[int, tuple] = {}
        self._reserved: Dict[str, float] = defaultdict(float)
        self._ids = itertools.count(1)

    def set_budget(self, tenant_id: str, budget: Budget) -> None:
        self.budgets[tenant_id] = budget

    def budget_for(self, tenant_id: str) -> Optional[Budget]:
        return self.budgets.get(tenant_id, self.default_budget)

    def spent(self, tenant_id: str) -> float:
        budget = self.budget_for(tenant_id)
        return self.ledger.get(tenant_id, period_key(budget.period if budget else "monthly"))

    def utilization(self, tenant_id: str) -> float:
        budget = self.budget_for(tenant_id)
        if not budget or budget.limit_usd <= 0:
            return 0.0
        return self.spent(tenant_id) / budget.limit_usd

    @staticmethod
    def estimate(model: ModelSpec, input_tokens: int, max_output_tokens: int,
                 expected_output_ratio: float = 0.5) -> float:
        """Expected cost assuming the reply uses ~half of max_tokens."""
        return model.cost(input_tokens, int(max_output_tokens * expected_output_ratio))

    def allows(self, tenant_id: str, estimate: float) -> bool:
        budget = self.budget_for(tenant_id)
        if not budget:
            return True
        if budget.max_cost_per_request is not None and estimate > budget.max_cost_per_request:
            return False
        if not budget.hard_limit:
            return True
        return self.spent(tenant_id) + estimate <= budget.limit_usd

    def worst_case(self, model: ModelSpec, input_tokens: int, max_output_tokens: int) -> float:
        return model.cost(int(input_tokens * self.input_safety_margin) + 1, max_output_tokens)

    def reserved(self, tenant_id: str) -> float:
        return self._reserved.get(tenant_id, 0.0)

    def reserve(self, tenant_id: str, amount: float) -> Optional[int]:
        """Atomically hold `amount` against the tenant's budget. Returns a token, or None
        if the hold would break the hard limit or the per-request cap."""
        budget = self.budget_for(tenant_id)
        with self._lock:
            if budget:
                if budget.max_cost_per_request is not None and amount > budget.max_cost_per_request:
                    return None
                if budget.hard_limit and (self.spent(tenant_id) + self._reserved[tenant_id] + amount
                                          > budget.limit_usd + 1e-12):
                    return None
            token = next(self._ids)
            self._reservations[token] = (tenant_id, amount)
            self._reserved[tenant_id] += amount
            return token

    def release(self, token: int) -> None:
        with self._lock:
            tenant_id, amount = self._reservations.pop(token, (None, 0.0))
            if tenant_id is not None:
                self._reserved[tenant_id] = max(0.0, self._reserved[tenant_id] - amount)

    def settle(self, token: int, actual_cost: float, model: str = "", input_tokens: int = 0,
               output_tokens: int = 0) -> None:
        """Convert a hold into real spend (release + record under one lock)."""
        with self._lock:
            tenant_id, amount = self._reservations.pop(token, (None, 0.0))
            if tenant_id is None:
                return
            self._reserved[tenant_id] = max(0.0, self._reserved[tenant_id] - amount)
            self._record_locked(tenant_id, model, actual_cost, input_tokens, output_tokens, False, 0.0)

    def should_downgrade(self, tenant_id: str) -> bool:
        budget = self.budget_for(tenant_id)
        return bool(budget) and self.utilization(tenant_id) >= budget.downgrade_at

    def record(self, tenant_id: str, model: str, cost: float, input_tokens: int,
               output_tokens: int, cache_hit: bool = False, saved: float = 0.0) -> None:
        with self._lock:
            self._record_locked(tenant_id, model, cost, input_tokens, output_tokens, cache_hit, saved)

    def _record_locked(self, tenant_id: str, model: str, cost: float, input_tokens: int,
                       output_tokens: int, cache_hit: bool, saved: float) -> None:
        budget = self.budget_for(tenant_id)
        if cost:
            self.ledger.add(tenant_id, period_key(budget.period if budget else "monthly"), cost)
        if True:
            self.total_cost += cost
            self.total_saved += saved
            self.records.append(CostRecord(tenant_id, model, cost, input_tokens, output_tokens,
                                           cache_hit, saved, time.time()))
            if len(self.records) > self.keep_records:
                del self.records[: len(self.records) - self.keep_records]

    def report(self, tenant_id: Optional[str] = None) -> Dict:
        with self._lock:
            recs = [r for r in self.records if tenant_id is None or r.tenant_id == tenant_id]
        by_model: Dict[str, Dict[str, float]] = defaultdict(lambda: {"requests": 0, "cost_usd": 0.0})
        for r in recs:
            by_model[r.model]["requests"] += 1
            by_model[r.model]["cost_usd"] += r.cost_usd
        out = {
            "requests": len(recs),
            "cost_usd": round(sum(r.cost_usd for r in recs), 6),
            "saved_usd": round(sum(r.saved_usd for r in recs), 6),
            "cache_hits": sum(1 for r in recs if r.cache_hit),
            "by_model": {k: {"requests": v["requests"], "cost_usd": round(v["cost_usd"], 6)}
                         for k, v in by_model.items()},
        }
        if tenant_id:
            budget = self.budget_for(tenant_id)
            out["budget"] = None if not budget else {
                "limit_usd": budget.limit_usd, "period": budget.period,
                "spent_usd": round(self.spent(tenant_id), 6),
                "utilization": round(self.utilization(tenant_id), 4),
            }
        return out
