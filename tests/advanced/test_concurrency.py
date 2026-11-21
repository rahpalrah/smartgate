"""Concurrency stress: many threads hammering one gateway must keep every invariant."""
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from smartgate import (Budget, BudgetExceededError, Gateway, MockProvider, ResilienceConfig,
                       RetryPolicy)

from ..conftest import MODELS

THREADS = 32


def gateway(provider=None, **kw):
    return Gateway(models=MODELS, providers={"mock": provider or MockProvider(latency=0.001)},
                   resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=2, base_delay=0)),
                   sleep=lambda s: None, **kw)


def test_hard_budget_is_never_overspent_under_contention():
    """32 threads race on one tenant whose budget covers only a fraction of the demand."""
    gw = gateway()
    limit = 0.05
    gw.set_budget("t", Budget(limit_usd=limit, period="total", downgrade_at=2.0))
    prompts = [f"Analyze and compare trade-offs of design option {i}, then propose a step-by-step plan."
               for i in range(2000)]
    ok, blocked, costs = [], [], []
    lock = threading.Lock()

    def run(p):
        try:
            r = gw.generate(p, tenant_id="t", max_tokens=400, use_cache=False)
            with lock:
                ok.append(r); costs.append(r.cost_usd)
        except BudgetExceededError:
            with lock:
                blocked.append(p)

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(run, prompts))

    spent = gw.costs.spent("t")
    assert len(ok) + len(blocked) == len(prompts)
    assert ok and blocked, "test must actually saturate the budget"
    assert spent <= limit + 1e-12, f"overspent: {spent} > {limit}"
    assert abs(spent - sum(costs)) < 1e-9, "ledger must equal the sum of billed responses"
    assert gw.costs.reserved("t") == pytest.approx(0.0), "no leaked reservations"


def test_ledger_and_metrics_consistency_under_load():
    gw = gateway()
    tenants = [f"tenant-{i}" for i in range(8)]
    rnd = random.Random(7)
    pool = [f"How do I configure feature {i}?" for i in range(150)]
    work = [(rnd.choice(tenants), rnd.choice(pool)) for _ in range(4000)]
    results = []
    lock = threading.Lock()

    def run(item):
        t, p = item
        r = gw.generate(p, tenant_id=t)
        with lock:
            results.append((t, r))

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(run, work))

    s = gw.stats()
    hits = sum(1 for _, r in results if r.cache_hit)
    assert s["counters"]["requests"] == len(work)
    assert s["cache"]["hits"] + s["cache"]["misses"] == len(work)
    assert s["cache"]["hits"] == hits
    for t in tenants:
        billed = sum(r.cost_usd for tt, r in results if tt == t)
        assert gw.costs.spent(t) == pytest.approx(billed, abs=1e-9)
    # Each (tenant, prompt) can miss more than once only if two threads raced on the very first
    # request; total provider calls are bounded by distinct keys + concurrency.
    distinct = len(set(work))
    misses = len(work) - hits
    assert distinct <= misses <= distinct + THREADS
    # best achievable hit rate = every repeat of an already-seen pair is served from cache
    assert hits >= len(work) - distinct - THREADS


def test_cache_answers_never_cross_tenants_under_load():
    gw = gateway(provider=_TenantEchoProvider())
    rnd = random.Random(3)
    work = [(f"t{rnd.randint(0, 9)}", f"What is the policy for item {rnd.randint(0, 40)}?") for _ in range(3000)]
    bad = []

    def run(item):
        t, p = item
        r = gw.generate(p, tenant_id=t)
        if not r.text.startswith(f"<{t}>"):
            bad.append((t, r.text))

    with ThreadPoolExecutor(THREADS) as ex:
        list(ex.map(run, work))
    assert bad == []


class _TenantEchoProvider(MockProvider):
    def complete(self, model, request):
        r = super().complete(model, request)
        r.text = f"<{request.tenant_id}> {r.text}"
        return r
