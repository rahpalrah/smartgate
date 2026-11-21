"""Regression tests for defects found by the advanced suite. Each one failed before its fix."""
import pytest

from smartgate import CacheConfig, GatewayRequest, SemanticCache, detect_pii, redact_pii
from smartgate.cache import store as store_mod


def req(t, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": t}], **kw)


def test_bug1_hash_collision_served_wrong_numbered_answer():
    # found by test_ledger_and_metrics_consistency_under_load: "feature 6" ~ "feature 57" = 0.92
    c = SemanticCache(CacheConfig(similarity_threshold=0.9))
    c.store_response(req("How do I configure feature 6?"), "answer-6", "m")
    assert c.lookup(req("How do I configure feature 57?")) is None


def test_bug2_negation_flip_served_opposite_answer():
    c = SemanticCache(CacheConfig(similarity_threshold=0.86))  # ecommerce preset threshold
    c.store_response(req("Is it safe to take ibuprofen?"), "yes", "m")
    assert c.lookup(req("Is it not safe to take ibuprofen?")) is None
    assert c.lookup(req("is it safe to take ibuprofen")) is not None  # still a true repeat


def test_bug3_card_next_to_stray_digit_was_not_detected():
    # found by hypothesis: '0 4111 1111 1111 1111 0'
    assert "credit_card" in detect_pii("0 4111 1111 1111 1111 0")
    assert "4111" not in redact_pii("ref 0 4111 1111 1111 1111 0")


def test_bug3b_order_numbers_are_not_cards():
    assert "credit_card" not in detect_pii("order 1234 5678 9012 3456")


@pytest.fixture
def no_numpy(monkeypatch):
    monkeypatch.setattr(store_mod, "_np", None)


def test_pure_python_fallback_index(no_numpy):
    c = SemanticCache(CacheConfig(similarity_threshold=0.9, max_entries=50))
    for i in range(80):
        c.store_response(req(f"question about topic {i}"), f"a{i}", "m")
    assert c.store.size() == 50
    assert c.lookup(req("Question about topic 79?")).entry.response == "a79"
    assert c.lookup(req("question about topic 0")) is None  # evicted (LRU)
    assert c.invalidate("default") == 50 and c.store.size() == 0
