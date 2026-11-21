"""Chaos / fault injection: providers fail randomly; the gateway must stay correct and available."""
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from smartgate import (AllModelsFailedError, Gateway, ProviderError, ProviderResponse,
                       ResilienceConfig, RetryPolicy, Usage)
from smartgate.analysis import estimate_tokens
from smartgate.providers.base import Provider

from ..conftest import MODELS


class ChaosProvider(Provider):
    """Each model has its own failure profile; seeded so runs are reproducible."""

    name = "mock"

    def __init__(self, profile, seed=0):
        self.profile = profile  # model -> {"transient": p, "refusal": p, "fatal": p}
        self.rnd = random.Random(seed)
        self.lock = threading.Lock()
        self.calls = {m: 0 for m in profile}
        self.failures = {m: 0 for m in profile}

    def complete(self, model, request):
        with self.lock:
            self.calls[model.name] += 1
            roll = self.rnd.random()
        p = self.profile[model.name]
        edge = 0.0
        for kind, retryable, fallback in (("transient", True, True), ("refusal", False, True),
                                          ("fatal", False, True)):
            edge += p.get(kind, 0)
            if roll < edge:
                with self.lock:
                    self.failures[model.name] += 1
                raise ProviderError(f"chaos {kind}", retryable=retryable, fallback=fallback,
                                    kind="server" if kind == "transient" else kind)
        text = f"[{model.name}] ok"
        return ProviderResponse(text, model.name, Usage(estimate_tokens(request.full_text), 5), "end_turn")


def make(profile, seed=0, attempts=3):
    prov = ChaosProvider(profile, seed)
    gw = Gateway(models=MODELS, providers={"mock": prov},
                 resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=attempts, base_delay=0),
                                             breaker_failure_threshold=10**9),
                 sleep=lambda s: None)
    return gw, prov


def run_load(gw, n=3000, threads=24):
    ok, failed = [], []
    lock = threading.Lock()

    def one(i):
        try:
            r = gw.generate(f"unique request {i}: classify this ticket", use_cache=False)
            with lock:
                ok.append(r)
        except AllModelsFailedError as e:
            with lock:
                failed.append(e)

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, range(n)))
    return ok, failed


def test_availability_with_30pct_transient_failures_everywhere():
    """Single attempt success is 70%; retries + 3-model fallback should push it to ~100%."""
    gw, prov = make({m.name: {"transient": 0.30} for m in MODELS})
    ok, failed = run_load(gw)
    # P(request fails) = 0.3^(3 attempts * 3 models) ≈ 2e-5
    assert len(failed) <= 2
    assert all(r.attempts[-1].ok for r in ok)
    assert all(not a.ok for r in ok for a in r.attempts[:-1])
    assert sum(prov.calls.values()) > len(ok)  # retries really happened


def test_primary_model_hard_down_traffic_fails_over():
    gw, prov = make({"small-a": {"fatal": 1.0}, "medium-a": {}, "large-a": {}})
    ok, failed = run_load(gw, n=1000)
    assert not failed
    assert {r.model for r in ok} == {"medium-a"}           # next in the same_tier_then_up chain
    assert prov.calls["small-a"] == 1000                     # fatal => no retries wasted
    assert all("fell back" in r.route_reason for r in ok)


def test_refusals_fail_over_without_retry():
    gw, prov = make({"small-a": {"refusal": 1.0}, "medium-a": {"refusal": 1.0}, "large-a": {}})
    ok, failed = run_load(gw, n=500)
    assert not failed and {r.model for r in ok} == {"large-a"}
    assert prov.calls["small-a"] == prov.calls["medium-a"] == 500


def test_total_outage_reports_every_attempt():
    gw, prov = make({m.name: {"fatal": 1.0} for m in MODELS})
    ok, failed = run_load(gw, n=200)
    assert not ok and len(failed) == 200
    assert all(len(e.attempts) == 3 for e in failed)
    assert gw.costs.total_cost == 0                          # failed calls are never billed


def test_circuit_breaker_sheds_load_from_dead_model():
    prov = ChaosProvider({"small-a": {"fatal": 1.0}, "medium-a": {}, "large-a": {}})
    gw = Gateway(models=MODELS, providers={"mock": prov},
                 resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=1),
                                             breaker_failure_threshold=5, breaker_recovery_seconds=3600),
                 sleep=lambda s: None)
    for i in range(1000):
        assert gw.generate(f"req {i}", use_cache=False).model == "medium-a"
    assert prov.calls["small-a"] == 5                        # breaker opened after 5 failures


def test_circuit_breaker_recovers_when_model_comes_back():
    prov = ChaosProvider({"small-a": {"fatal": 1.0}, "medium-a": {}, "large-a": {}})
    gw = Gateway(models=MODELS, providers={"mock": prov},
                 resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=1),
                                             breaker_failure_threshold=3, breaker_recovery_seconds=0.05),
                 sleep=lambda s: None)
    for i in range(10):
        gw.generate(f"a{i}", use_cache=False)
    prov.profile["small-a"] = {}                              # model recovers
    import time
    time.sleep(0.06)
    assert gw.generate("after recovery", use_cache=False).model == "small-a"
    assert gw.stats()["circuit_breakers"]["mock:small-a"] == "closed"


@pytest.mark.parametrize("seed", range(5))
def test_random_mixed_chaos_invariants(seed):
    rnd = random.Random(seed)
    profile = {m.name: {"transient": rnd.uniform(0, .4), "refusal": rnd.uniform(0, .1),
                        "fatal": rnd.uniform(0, .1)} for m in MODELS}
    gw, prov = make(profile, seed=seed)
    ok, failed = run_load(gw, n=1500)
    assert len(ok) + len(failed) == 1500
    assert gw.costs.total_cost == pytest.approx(sum(r.cost_usd for r in ok))
    assert gw.stats()["counters"]["requests"] == 1500
    assert gw.stats()["counters"].get("errors", 0) == len(failed)
    assert len(ok) / 1500 > 0.97
