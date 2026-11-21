"""Property-based tests: invariants that must hold for *any* input (hypothesis fuzzing)."""
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from smartgate import (CacheConfig, GatewayRequest, ModelRouter, PromptAnalyzer, RoutingConfig,
                       SemanticCache, detect_pii, redact_pii)
from smartgate.cost import Budget, CostController
from smartgate.embeddings import HashingEmbedder, cosine
from smartgate.types import TIERS, ModelSpec

from ..conftest import MODELS

import os
SETTINGS = settings(max_examples=int(os.environ.get("SMARTGATE_FUZZ", 300)), deadline=None, suppress_health_check=[HealthCheck.too_slow])
text = st.text(min_size=1, max_size=600)
pii_fragment = st.sampled_from([
    "john.doe@example.com", "555-867-5309", "(212) 555-0100", "123-45-6789",
    "4111 1111 1111 1111", "5500-0000-0000-0004", "10.0.0.12", "GB29NWBK60161331926819", "MRN: 00482913",
])


def req(t, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": t}], **kw)


@SETTINGS
@given(text)
def test_analysis_is_total_and_bounded(t):
    a = PromptAnalyzer().analyze(req(t))
    assert 0.0 <= a.complexity <= 1.0
    assert a.estimated_input_tokens >= 1
    assert a.cacheable == (not a.pii and not a.time_sensitive)


@SETTINGS
@given(st.lists(st.one_of(text, pii_fragment), min_size=1, max_size=8))
def test_redaction_removes_all_detectable_pii(parts):
    t = " ".join(parts)
    assert detect_pii(redact_pii(t)) == []


@SETTINGS
@given(pii_fragment, text)
def test_known_pii_is_always_detected(fragment, noise):
    assert detect_pii(f"{noise} {fragment} {noise}")


@SETTINGS
@given(text, text)
def test_embedding_is_normalized_and_symmetric(a, b):
    e = HashingEmbedder()
    va, vb = e.embed(a), e.embed(b)
    for v in (va, vb):
        n = sum(x * x for x in v)
        assert n == 0 or abs(n - 1) < 1e-9
    assert abs(cosine(va, vb) - cosine(vb, va)) < 1e-12
    assert cosine(va, vb) <= 1 + 1e-9


@SETTINGS
@given(text, text, st.sampled_from(["t1", "t2", "t3"]), st.sampled_from(["t1", "t2", "t3"]))
def test_cache_roundtrip_and_tenant_isolation(prompt, answer, tenant, other):
    c = SemanticCache(CacheConfig(similarity_threshold=0.9))
    c.store_response(req(prompt, tenant_id=tenant), answer, "m")
    hit = c.lookup(req(prompt, tenant_id=tenant))
    assert hit is not None and hit.entry.response == answer
    if other != tenant:
        assert c.lookup(req(prompt, tenant_id=other)) is None


@SETTINGS
@given(text, st.sampled_from(TIERS), st.sampled_from(TIERS), st.booleans(),
       st.integers(min_value=1, max_value=4000))
def test_router_invariants(t, lo, hi, pressure, max_tokens):
    if TIERS.index(lo) > TIERS.index(hi):
        lo, hi = hi, lo
    r = ModelRouter(MODELS, RoutingConfig(min_tier=lo, max_tier=hi))
    q = req(t, max_tokens=max_tokens)
    d = r.route(q, PromptAnalyzer().analyze(q), budget_pressure=pressure)
    names = [m.name for m in d.candidates]
    assert len(names) == len(set(names)) == len(MODELS)        # every model exactly once
    assert d.primary is d.candidates[0]
    assert TIERS.index(lo) <= TIERS.index(d.tier) <= TIERS.index(hi)
    assert d.primary.tier == d.tier                              # primary matches decided tier


@SETTINGS
@given(st.integers(0, 10**7), st.integers(0, 10**7), st.integers(0, 10**7))
def test_cost_is_monotonic_and_linear(i, o, extra):
    m = ModelSpec("x", "p", "small", 3.0, 15.0)
    assert m.cost(i + extra, o) >= m.cost(i, o)
    assert abs(m.cost(i, o) - (m.cost(i, 0) + m.cost(0, o))) < 1e-12


@SETTINGS
@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), max_size=60),
       st.floats(min_value=0.01, max_value=5))
def test_hard_budget_reservations_never_exceed_limit(costs, limit):
    c = CostController(budgets={"t": Budget(limit)})
    for x in costs:
        token = c.reserve("t", x)
        if token is not None:
            c.settle(token, x)
    assert c.spent("t") <= limit + 1e-9
