import time

import pytest

from smartgate import (Budget, CacheConfig, CostController, GatewayRequest, HashingEmbedder,
                       ModelRouter, PromptAnalyzer, RoutingRule, SemanticCache, SQLiteVectorStore,
                       detect_pii, redact_pii)
from smartgate.embeddings import cosine
from smartgate.errors import ProviderError
from smartgate.resilience import CircuitBreaker, RetryPolicy

from .conftest import MODELS


def req(text, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": text}], **kw)


# ------------------------------------------------------------------ analysis
class TestAnalysis:
    def test_simple_question_is_low_complexity(self):
        a = PromptAnalyzer().analyze(req("What is the capital of France?"))
        assert a.task_type == "qa"
        assert a.complexity < 0.35

    def test_keyword_needs_word_boundary(self):
        # "capital" contains "api" — must not be classified as code
        assert PromptAnalyzer().analyze(req("capital of Peru?")).task_type != "code"

    def test_complex_prompt_is_high_complexity(self):
        a = PromptAnalyzer().analyze(req(
            "Design an end-to-end architecture for a payments ledger. First analyze trade-offs of "
            "event sourcing, then implement it in Python, ensuring idempotency. Compare three approaches."))
        assert a.complexity >= 0.7

    def test_time_sensitive_not_cacheable(self):
        a = PromptAnalyzer().analyze(req("What is the latest stock price of ACME today?"))
        assert a.time_sensitive and not a.cacheable

    def test_domain_keywords_and_custom_scorer(self):
        def legal(request, analysis):
            return 0.5 if "indemnity" in request.last_user_message else 0
        a = PromptAnalyzer(extra_scorers=[legal], domain_keywords={"clause": 0.2}).analyze(
            req("Explain this indemnity clause"))
        assert a.signals["legal"] == 0.5 and a.signals["domain"] == 0.2

    def test_pii_detection_and_redaction(self):
        text = "Email me at jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111"
        assert {"email", "ssn", "credit_card"} <= set(detect_pii(text))
        red = redact_pii(text)
        assert "jane@example.com" not in red and "123-45-6789" not in red and "4111" not in red

    def test_non_luhn_number_is_not_a_card(self):
        assert "credit_card" not in detect_pii("order 1234 5678 9012 3456")


# ----------------------------------------------------------------- embeddings
def test_embedder_similarity_ordering():
    e = HashingEmbedder()
    base = e.embed("How do I reset my password?")
    assert cosine(base, e.embed("how can I reset my password")) > 0.95
    assert cosine(base, e.embed("how do I change my email address")) < 0.6


# ---------------------------------------------------------------------- cache
class TestCache:
    def test_exact_and_semantic_hit(self):
        c = SemanticCache(CacheConfig(similarity_threshold=0.9))
        c.store_response(req("How do I reset my password?"), "Click 'forgot password'.", "m")
        assert c.lookup(req("how do i reset my password")).exact
        hit = c.lookup(req("How can I reset my password?"))
        assert hit and not hit.exact and hit.similarity >= 0.9
        assert c.lookup(req("What is your refund policy?")) is None

    def test_tenant_isolation(self):
        c = SemanticCache()
        c.store_response(req("hello", tenant_id="a"), "hi", "m")
        assert c.lookup(req("hello", tenant_id="b")) is None
        assert c.lookup(req("hello", tenant_id="a"))

    def test_global_scope_shares(self):
        c = SemanticCache(CacheConfig(scope="global"))
        c.store_response(req("hello", tenant_id="a"), "hi", "m")
        assert c.lookup(req("hello", tenant_id="b"))

    def test_conversation_context_isolation(self):
        c = SemanticCache()
        r1 = GatewayRequest(messages=[{"role": "user", "content": "Talk about Python"},
                                      {"role": "assistant", "content": "ok"},
                                      {"role": "user", "content": "give an example"}])
        r2 = GatewayRequest(messages=[{"role": "user", "content": "Talk about Rust"},
                                      {"role": "assistant", "content": "ok"},
                                      {"role": "user", "content": "give an example"}])
        c.store_response(r1, "print('hi')", "m")
        assert c.lookup(r2) is None
        assert c.lookup(req("give an example", system="other")) is None

    def test_ttl_expiry(self):
        c = SemanticCache(CacheConfig(ttl_seconds=0.05))
        c.store_response(req("hello"), "hi", "m")
        time.sleep(0.08)
        assert c.lookup(req("hello")) is None

    def test_sqlite_store_persists(self, tmp_path):
        path = str(tmp_path / "c.db")
        SemanticCache(store=SQLiteVectorStore(path)).store_response(req("hello world"), "hi", "m")
        c2 = SemanticCache(store=SQLiteVectorStore(path))
        hit = c2.lookup(req("Hello, world!"))
        assert hit and hit.entry.response == "hi"
        assert c2.invalidate("default") == 1

    def test_lru_bound(self):
        c = SemanticCache(CacheConfig(max_entries=2))
        for t in ("alpha one", "beta two", "gamma three"):
            c.store_response(req(t), t, "m")
        assert c.store.size() == 2
        assert c.lookup(req("alpha one")) is None


# -------------------------------------------------------------------- routing
class TestRouting:
    def test_tiers_by_complexity(self):
        r = ModelRouter(MODELS)
        a = PromptAnalyzer()
        assert r.route(req("hi"), a.analyze(req("hi"))).primary.name == "small-a"
        hard = req("Design an end-to-end distributed architecture, first analyze trade-offs then "
                   "implement in Python and compare approaches, ensuring correctness.")
        assert r.route(hard, a.analyze(hard)).primary.tier == "large"

    def test_fallback_order_same_tier_then_up(self):
        d = ModelRouter(MODELS).route(req("hi"), PromptAnalyzer().analyze(req("hi")))
        assert [m.name for m in d.candidates] == ["small-a", "medium-a", "large-a"]

    def test_rule_min_tier_and_forced_model(self):
        rule = RoutingRule("vip", when=lambda r, a: "vip" in r.tags, min_tier="medium")
        r = ModelRouter(MODELS, rules=[rule])
        q = req("hi", tags=["vip"])
        assert r.route(q, PromptAnalyzer().analyze(q)).primary.name == "medium-a"
        q2 = req("hi", model="large-a")
        assert r.route(q2, PromptAnalyzer().analyze(q2)).primary.name == "large-a"

    def test_budget_pressure_downgrades(self):
        q = req("hi", tier="medium")
        r = ModelRouter(MODELS)
        # forced tier is respected even under pressure
        assert r.route(q, PromptAnalyzer().analyze(q), budget_pressure=True).primary.tier == "medium"
        hard = req("Analyze and compare the trade-offs, then design a step-by-step strategy.")
        an = PromptAnalyzer().analyze(hard)
        normal = r.route(hard, an).tier
        pressured = r.route(hard, an, budget_pressure=True).tier
        assert ["small", "medium", "large"].index(pressured) == ["small", "medium", "large"].index(normal) - 1

    def test_context_window_filter(self):
        from smartgate import ModelSpec
        models = [ModelSpec("tiny", "mock", "small", 0.1, 0.5, max_context_tokens=100), *MODELS[1:]]
        q = req("x " * 1000)
        assert ModelRouter(models).route(q, PromptAnalyzer().analyze(q)).primary.name != "tiny"


# ----------------------------------------------------------------------- cost
class TestCost:
    def test_hard_budget_and_per_request_cap(self):
        c = CostController(budgets={"t": Budget(1.0, max_cost_per_request=0.1)})
        assert c.allows("t", 0.05)
        assert not c.allows("t", 0.2)
        c.record("t", "m", 0.99, 1, 1)
        assert not c.allows("t", 0.05)
        assert c.should_downgrade("t")

    def test_soft_budget_never_blocks(self):
        c = CostController(budgets={"t": Budget(1.0, hard_limit=False)})
        c.record("t", "m", 5.0, 1, 1)
        assert c.allows("t", 1.0)

    def test_report(self):
        c = CostController(budgets={"t": Budget(10.0)})
        c.record("t", "m", 1.0, 10, 10)
        c.record("t", "m", 0.0, 0, 0, cache_hit=True, saved=1.0)
        rep = c.report("t")
        assert rep["cost_usd"] == 1.0 and rep["saved_usd"] == 1.0 and rep["budget"]["utilization"] == 0.1


# ----------------------------------------------------------------- resilience
class TestResilience:
    def test_retry_then_success(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ProviderError("x", retryable=True)
            return "ok"
        assert RetryPolicy(max_attempts=3).run(fn, sleep=lambda s: None) == "ok"

    def test_non_retryable_raises_immediately(self):
        calls = []

        def fn():
            calls.append(1)
            raise ProviderError("x", retryable=False)
        with pytest.raises(ProviderError):
            RetryPolicy(max_attempts=5).run(fn, sleep=lambda s: None)
        assert len(calls) == 1

    def test_circuit_breaker_cycle(self):
        now = [0.0]
        b = CircuitBreaker(failure_threshold=2, recovery_timeout=10, clock=lambda: now[0])
        b.failure(); b.failure()
        assert b.state == "open" and not b.allow()
        now[0] = 11
        assert b.state == "half_open" and b.allow()
        b.failure()
        assert b.state == "open"
        now[0] = 22
        b.success()
        assert b.state == "closed"
