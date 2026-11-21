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


# ------------------------------------------------- round 2: limitations found by the benchmark
import pytest

from smartgate import (GroundingValidator, LLMJudgeValidator, QualityTracker, chain, confusion_report,
                       detect_pii, redact_pii, default_validator)
from smartgate.analysis import set_phone_regions


class TestInternationalPhones:
    @pytest.mark.parametrize("text", ["call +44 20 7946 0958", "+91 98765 43210", "+49 30 901820",
                                      "tel: +33 1 42 68 53 00", "+14155550123", "(212) 555-0100"])
    def test_detected_and_redacted(self, text):
        assert "phone" in detect_pii(text)
        assert not any(ch.isdigit() for ch in redact_pii(text).replace("[REDACTED_PHONE]", ""))

    def test_national_format_needs_region(self):
        pytest.importorskip("phonenumbers")
        try:
            assert "phone" not in detect_pii("ring 020 7946 0958")
            set_phone_regions(["US", "GB"])
            assert "phone" in detect_pii("ring 020 7946 0958")
        finally:
            set_phone_regions(["US"])

    def test_order_numbers_still_safe(self):
        for t in ("order 732201349959", "invoice 00004587345", "ref INV-48211"):
            assert "phone" not in detect_pii(t)


class TestActionMorphology:
    @pytest.mark.parametrize("a,b", [
        ("i need to egt the invoice", "i need to look for the invoice"),      # typo + look for
        ("I expect a refund of 15 dollars", "refund 15 dollars"),             # track vs request
        ("set my delivery address up", "change my delivery address"),         # split phrase
        ("change to the Platinum account", "editing the Platinum account"),   # switch vs edit
        ("help creating a free account", "help me use another free account"),
    ])
    def test_confusable_requests_are_separated(self, a, b):
        c = SemanticCache(CacheConfig(similarity_threshold=0.3))
        assert c.signature(a)[2] != c.signature(b)[2]

    def test_inflections_match_their_base(self):
        c = SemanticCache()
        assert c.signature("getting my bill")[2] == c.signature("get my bill")[2] == frozenset({"obtain"})
        assert c.signature("cancelled order")[2] == c.signature("cancel order")[2]

    def test_extra_classes_merge_into_existing(self):
        c = SemanticCache(CacheConfig(extra_action_classes={"track": ["chase"]}))
        assert "track" in c.signature("chase my parcel")[2] and "track" in c.signature("track my parcel")[2]


def test_confusion_report_names_the_confusing_words():
    labeled = [("refund 15 dollars", "get_refund"), ("refund 15 dollars please", "get_refund")] * 3
    labeled += [("refund 15 dollars now", "track_refund")]
    r = confusion_report(labeled, HashingEmbedder(), CacheConfig(similarity_threshold=0.5))
    assert r["wrong"] >= 1 and ("get_refund", "track_refund") in [p for p, _ in r["confused_labels"]]


class TestGrounding:
    def _resp(self, text):
        return ProviderResponse(text, "m", Usage(1, 1), "end_turn")

    def test_invented_eta_is_flagged(self):
        v = GroundingValidator()
        problem = v(req("ETA of order 326753?"), self._resp("The ETA for your order 326753 is 2 days."))
        assert problem and "2 days" in problem

    def test_facts_from_context_or_policy_pass(self):
        v = GroundingValidator(known_facts=["30 days"])
        assert v(req("Refund of $49.99 for order 326753"), self._resp("Your $49.99 refund for 326753 is on its way.")) is None
        assert v(req("return policy?"), self._resp("You can return items within 30 days.")) is None
        assert v(req("hi", system="Refunds take 5-7 business days."), self._resp("Refunds take 5-7 business days.")) is None

    def test_chain_and_judge(self):
        judge = LLMJudgeValidator(lambda r, a: 0.9 if "Settings" in a else 0.2, min_score=0.5)
        v = chain(default_validator, judge)
        assert v(req("reset password?"), self._resp("Go to Settings.")) is None and judge.last_score == 0.9
        assert v(req("reset password?"), self._resp("Not sure.")) == "judge score 0.20"


def test_grounding_escalates_through_gateway():
    from smartgate.quality import GroundingValidator, chain
    answers = {"small": "Your order arrives in 2 days.", "medium": "I can't see delivery dates; check the "
               "tracking link in your confirmation email.", "large": "L"}
    calls = []

    def fn(model, request):
        calls.append(model.tier)
        return ProviderResponse(answers[model.tier], model.name, Usage(5, 5), "end_turn")
    gw = Gateway(models=MODELS, providers={"mock": CallableProvider("mock", fn)},
                 validator=chain(default_validator, GroundingValidator()))
    r = gw.generate("When will my order arrive?")
    assert r.model == "medium-a" and calls == ["small", "medium"]


class TestAdaptiveRouting:
    def _gw(self, small_answer):
        def fn(model, request):
            text = small_answer if model.tier == "small" else "Use the reset link in Settings."
            return ProviderResponse(text, model.name, Usage(5, 5), "end_turn")
        return Gateway(models=MODELS, providers={"mock": CallableProvider("mock", fn)},
                       quality_tracker=QualityTracker(min_score=0.7, min_samples=5))

    def test_failing_small_tier_gets_bypassed_for_that_key_only(self):
        gw = self._gw("")  # small always returns empty -> validator fails
        for i in range(5):
            gw.generate(f"reset question {i}", use_cache=False, metadata={"route_key": "reset_password"})
        r = gw.generate("reset question 99", use_cache=False, metadata={"route_key": "reset_password"})
        assert r.model == "medium-a" and "quality floor" in r.route_reason and len(r.attempts) == 1
        other = gw.generate("other question", use_cache=False, metadata={"route_key": "track_order"})
        assert "quality floor" not in other.route_reason  # other keys unaffected

    def test_user_feedback_drives_the_floor(self):
        gw = self._gw("Use the reset link in Settings.")  # passes validation, but users dislike it
        first = gw.generate("q 0", use_cache=False, metadata={"route_key": "billing"})
        assert first.model == "small-a"
        assert gw.feedback(first.request_id, 0.0)
        served = [first.model]
        for i in range(1, 8):
            r = gw.generate(f"q {i}", use_cache=False, metadata={"route_key": "billing"})
            served.append(r.model)
            if r.model == "small-a":
                gw.feedback(r.request_id, 0.0)  # thumbs-down on every small-tier answer
        assert served[-1] == "medium-a" and "small-a" in served
        assert not gw.feedback("unknown-id", 1.0)


@pytest.mark.parametrize("answer,flagged", [
    ("I'm sorry, but I'm unable to assist with that. If you have any other questions, please let me know.", True),
    ("I'm sorry, but I don't have the ability to update orders. However, I can assist you with any other questions.", True),
    ("I don't have access to your order details, but you can track it from the link in your confirmation email.", False),
    ("I'm unable to cancel it myself. Please contact our support team at the Help page.", False),
])
def test_honest_limit_with_next_step_is_not_a_refusal(answer, flagged):
    # found by the round-3 benchmark: helpful redirects were being escalated as refusals
    r = ProviderResponse(answer, "m", Usage(1, 1), "end_turn")
    assert (default_validator(req("where is my order?"), r) is not None) is flagged
