"""Second-stage cache verification and the tightened time-sensitive rule."""
import pytest

from smartgate import (CacheConfig, GatewayRequest, LLMPairVerifier, MemoizedVerifier, MockProvider,
                       SemanticCache, SymmetricVerifier, gateway_from_config)
from smartgate.analysis import is_time_sensitive


def req(t, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": t}], **kw)


def cache(verifier=None, **cfg):
    c = SemanticCache(CacheConfig(similarity_threshold=0.5, **cfg), verifier=verifier)
    c.store_response(req("How can I lose weight in 2 weeks?"), "A", "m")
    return c


class TestVerifierHook:
    def test_rejection_turns_hit_into_miss_and_is_counted(self):
        c = cache(verifier=lambda new, old: 0.1)
        assert c.lookup(req("how can i lose weight in 2 weeks please")) is None
        assert c.stats["verifier_calls"] == 1 and c.stats["verifier_rejections"] == 1

    def test_acceptance_serves_and_receives_both_questions(self):
        seen = []

        def v(new, old):
            seen.append((new, old))
            return 0.9
        c = cache(verifier=v)
        assert c.lookup(req("How can I lose weight in 2 weeks fast?")).entry.response == "A"
        assert seen == [("How can I lose weight in 2 weeks fast?", "How can I lose weight in 2 weeks?")]

    def test_exact_hits_are_never_verified(self):
        calls = []
        c = cache(verifier=lambda a, b: calls.append(1) or 0.0)
        assert c.lookup(req("how can i lose weight in 2 weeks")) is not None
        assert calls == []

    def test_min_score_and_skip_above(self):
        c = cache(verifier=lambda a, b: 0.6, verify_min_score=0.7)
        assert c.lookup(req("How can I lose weight in 2 weeks fast?")) is None
        calls = []
        c2 = cache(verifier=lambda a, b: calls.append(1) or 0.0, verify_skip_above=0.3)
        assert c2.lookup(req("How can I lose weight in 2 weeks fast?")) is not None and calls == []

    def test_broken_verifier_fails_closed_unless_opted_out(self):
        def boom(a, b):
            raise RuntimeError("model down")
        q = req("How can I lose weight in 2 weeks fast?")
        c = cache(verifier=boom)
        assert c.lookup(q) is None and c.stats["verifier_errors"] == 1
        assert cache(verifier=boom, verify_fail_open=True).lookup(q) is not None

    def test_memoized_verifier_judges_each_pair_once(self):
        calls = []
        m = MemoizedVerifier(lambda a, b: calls.append(1) or 1.0)
        assert [m("a", "b"), m("a", "b"), m("a", "c")] == [1.0, 1.0, 1.0] and len(calls) == 2

    @pytest.mark.parametrize("answer,score", [("YES", 1.0), (" yes, same", 1.0), ("NO", 0.0),
                                              ("Not sure", 0.0), ("", 0.0)])
    def test_llm_wrapper_parses_answers(self, answer, score):
        assert LLMPairVerifier(lambda system, user: answer)("q1", "q2") == score

    def test_end_to_end_through_the_gateway(self):
        judged = []

        def judge(system, user):
            judged.append(user)
            return "NO" if "2 months" in user else "YES"
        gw = gateway_from_config({"provider_override": "mock", "cache": {"similarity_threshold": 0.5}},
                                 verifier=LLMPairVerifier(judge))
        gw.generate("How can I lose weight in 2 weeks?")
        assert gw.generate("How can I lose weight in 2 months?").cache_hit is False
        assert gw.generate("How can I lose weight in 2 weeks fast?").cache_hit is True
        assert gw.stats()["cache"]["verifier_rejections"] == 1 and "Q1:" in judged[0]


@pytest.mark.parametrize("text", [
    "Is the app down right now?", "What's the weather today?", "What's the latest iPhone model?",
    "Any update on my order status today?", "Is your server down?", "When will my flight arrive tonight?",
    "What is the current exchange rate for euros?", "Is the store open tomorrow?", "stock price of ACME",
    "Is there an outage in my area?", "How long is the delay at the moment?", "Where is my order right now?",
    "When will my package arrive today?", "Is the website working right now?",
    "Any news on the refund this week?", "what's the price of bitcoin now"])
def test_time_varying_questions_are_flagged(text):
    assert is_time_sensitive(text)


@pytest.mark.parametrize("text", [
    "I need help now", "I ordered it yesterday and it arrived broken", "How do I reset my password today?",
    "I need to cancel my subscription now please", "Can you help me change my email address right now",
    "My current plan is Basic, how do I upgrade?", "I live in Berlin and love your app",
    "Thanks, that fixed it today!", "Why was I charged twice?", "I recently moved, how do I update my address?",
    "please update my payment method now", "Cancel my order today please",
    "Internet worked for 2 days, now back to not working", "My recent update broke the app, how do I fix it?"])
def test_ordinary_requests_with_a_time_word_are_not_flagged(text):
    # found by the Twitter benchmark: "now" alone flagged 14% of real messages (39% of flags)
    assert not is_time_sensitive(text)


def test_symmetric_verifier_needs_both_orders_to_agree():
    order_sensitive = lambda a, b: 0.9 if a < b else 0.2  # noqa: E731
    assert order_sensitive("a", "b") == 0.9 and order_sensitive("b", "a") == 0.2
    sym = SymmetricVerifier(order_sensitive)
    assert sym("a", "b") == sym("b", "a") == 0.2
    assert SymmetricVerifier(lambda a, b: 0.8)("x", "y") == 0.8
