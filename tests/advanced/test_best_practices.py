"""Tests for the industry-practice upgrades: action guard, calibration, seeding, cascade."""
from smartgate import (CacheConfig, CallableProvider, Gateway, GatewayRequest, ProviderResponse,
                       SemanticCache, Usage, calibrate_threshold)
from smartgate.embeddings import HashingEmbedder

from ..conftest import MODELS


def req(t, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": t}], **kw)


class TestActionGuard:
    def test_different_actions_never_share_an_answer(self):
        pairs = [("I want to edit my account", "I want to delete my account"),
                 ("can you check my invoice", "can you send my invoice"),
                 ("help me cancel my order", "help me track my order")]
        for a, b in pairs:
            c = SemanticCache(CacheConfig(similarity_threshold=0.5))
            c.store_response(req(a), "x", "m")
            assert c.lookup(req(b)) is None, (a, b)

    def test_synonyms_within_a_class_still_match(self):
        c = SemanticCache(CacheConfig(similarity_threshold=0.5))
        c.store_response(req("I want to change my shipping address"), "x", "m")
        assert c.lookup(req("I want to modify my shipping address"))

    def test_custom_classes_and_disable(self):
        c = SemanticCache(CacheConfig(similarity_threshold=0.5, extra_action_classes={"x-ray": ["xray"], "mri": ["mri"]}))
        c.store_response(req("book an xray appointment"), "x", "m")
        assert c.lookup(req("book an mri appointment")) is None
        off = SemanticCache(CacheConfig(similarity_threshold=0.5, guard_actions=False))
        off.store_response(req("I want to edit my account"), "x", "m")
        assert off.lookup(req("I want to delete my account"))


def test_seeded_answer_is_served():
    c = SemanticCache(CacheConfig(similarity_threshold=0.85))
    c.seed("What is your return policy?", "30 days, free returns.", tenant_id="shop", system="S")
    hit = c.lookup(req("what's your return policy", tenant_id="shop", system="S"))
    assert hit and hit.entry.response == "30 days, free returns." and hit.entry.model == "curated"


def test_calibration_respects_the_wrong_answer_budget():
    labeled = []
    for i in range(30):
        labeled += [(f"how do I reset my password {'please' * (i % 2)}", "reset"),
                    (f"where is my parcel number {i}", f"track-{i}"),
                    ("I want to edit my account", "edit"), ("I want to delete my account", "delete")]
    out = calibrate_threshold(labeled, HashingEmbedder(), max_wrong_per_1000=0.0)
    best = next(r for r in out["table"] if r["threshold"] == out["threshold"])
    assert out["meets_budget"] and best["wrong"] == 0 and best["hits"] > 0


def _gateway(answers):
    calls = []

    def fn(model, request):
        calls.append(model.name)
        return ProviderResponse(answers[model.tier], model.name, Usage(10, 10), "end_turn")
    return Gateway(models=MODELS, providers={"mock": CallableProvider("mock", fn)}), calls


def test_cascade_escalates_on_bad_answer_and_does_not_cache_it():
    gw, calls = _gateway({"small": "I'm unable to help with that.", "medium": "Go to Settings > Password.",
                          "large": "L"})
    r = gw.generate("How do I reset my password?")
    assert r.model == "medium-a" and calls == ["small-a", "medium-a"]
    assert r.cost_usd == gw.costs.total_cost  # both calls billed
    assert "rejected" in r.route_reason and r.quality_warning is None
    again = gw.generate("How do I reset my password?")
    assert again.cache_hit and again.text == "Go to Settings > Password."


def test_empathetic_apology_is_not_treated_as_refusal():
    gw, calls = _gateway({"small": "I'm sorry for the trouble! Click 'Forgot password' to reset it.",
                          "medium": "m", "large": "l"})
    assert gw.generate("How do I reset my password?").model == "small-a" and calls == ["small-a"]


def test_cascade_best_effort_when_nothing_passes():
    gw, _ = _gateway({"small": "", "medium": "", "large": ""})
    r = gw.generate("How do I reset my password?")
    assert r.quality_warning == "empty response"
    assert not gw.generate("How do I reset my password?").cache_hit  # never cached
