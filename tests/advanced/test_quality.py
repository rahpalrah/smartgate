"""Quality benchmarks with hard pass/fail thresholds: routing, cache precision, cost savings."""
import json
import os
import random
from collections import Counter

import pytest

from smartgate import CacheConfig, GatewayRequest, MockProvider, SemanticCache, gateway_from_preset
from smartgate.config import PRESETS
from smartgate.types import TIERS

from .datasets import LEXICAL_PARAPHRASES, NEAR_MISS_PAIRS, ROUTING_SET

REPORT = {}


def req(t, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": t}], **kw)


@pytest.fixture(scope="module", autouse=True)
def write_report():
    yield
    path = os.path.join(os.path.dirname(__file__), "..", "..", "benchmark_report.json")
    with open(path, "w") as f:
        json.dump(REPORT, f, indent=2)


# --------------------------------------------------------------------- routing
def test_routing_accuracy_on_labeled_set():
    gw = gateway_from_preset("general", providers={"anthropic": MockProvider()})
    confusion = Counter()
    wrong = []
    for prompt, expected in ROUTING_SET:
        got = gw.analyze(prompt)["tier"]
        confusion[(expected, got)] += 1
        if got != expected:
            wrong.append((expected, got, round(gw.analyze(prompt)["analysis"]["complexity"], 2), prompt[:70]))
    n = len(ROUTING_SET)
    acc = sum(v for (e, g), v in confusion.items() if e == g) / n
    severe = sum(v for (e, g), v in confusion.items() if abs(TIERS.index(e) - TIERS.index(g)) == 2)
    # under-routing (hard prompt to a weaker model) hurts quality; over-routing only costs money
    under = sum(v for (e, g), v in confusion.items() if TIERS.index(g) < TIERS.index(e))
    REPORT["routing"] = {"n": n, "accuracy": round(acc, 3), "severe_misroutes": severe,
                         "under_routed": under,
                         "confusion": {f"{e}->{g}": v for (e, g), v in sorted(confusion.items())},
                         "errors": wrong}
    print(json.dumps(REPORT["routing"], indent=2))
    assert severe == 0, "a large-tier prompt went to small, or vice versa"
    assert acc >= 0.75


# ----------------------------------------------------------------------- cache
@pytest.mark.parametrize("threshold", [0.80, 0.86, 0.90, 0.97])
def test_cache_precision_no_false_hits_on_near_misses(threshold):
    """Not one wrong answer, even at thresholds looser than any preset uses."""
    false_hits = []
    for a, b in NEAR_MISS_PAIRS:
        c = SemanticCache(CacheConfig(similarity_threshold=threshold))
        c.store_response(req(a), f"answer for: {a}", "m")
        hit = c.lookup(req(b))
        if hit:
            false_hits.append((a, b, hit.similarity))
    REPORT.setdefault("cache_false_hits", {})[str(threshold)] = false_hits
    assert false_hits == []


def test_guard_is_what_prevents_false_hits():
    """Ablation: with the number/negation guard off, the same pairs DO false-hit."""
    hits = 0
    for a, b in NEAR_MISS_PAIRS:
        c = SemanticCache(CacheConfig(similarity_threshold=0.80, guard_numbers=False, guard_negation=False))
        c.store_response(req(a), "x", "m")
        hits += bool(c.lookup(req(b)))
    REPORT["cache_false_hits_without_guard@0.80"] = hits
    assert hits > 0


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_cache_recall_on_lexical_paraphrases(preset):
    cfg = CacheConfig(**PRESETS[preset].get("cache", {}))
    recalled = 0
    for a, b in LEXICAL_PARAPHRASES:
        c = SemanticCache(cfg)
        c.store_response(req(a), "x", "m")
        recalled += bool(c.lookup(req(b)))
    recall = recalled / len(LEXICAL_PARAPHRASES)
    REPORT.setdefault("cache_recall", {})[preset] = {"threshold": cfg.similarity_threshold,
                                                     "recall": recall}
    # strict presets (0.97) trade recall for safety; lenient ones must catch most rephrasings
    assert recall >= (0.5 if cfg.similarity_threshold >= 0.95 else 0.8)


# ------------------------------------------------------------------- economics
def test_cost_savings_vs_always_opus_baseline():
    """Simulated support traffic: Zipf-distributed repeat questions with rephrasings, plus a
    long tail of unique complex requests. Baseline = every request sent to Opus, no cache."""
    rnd = random.Random(42)
    faqs = [p for p, t in ROUTING_SET if t == "small"]
    variants = {p: [p, p.lower(), p.rstrip("?.") + "?", p.lower().rstrip("?.")] for p in faqs}
    medium = [p for p, t in ROUTING_SET if t == "medium"]
    large = [p for p, t in ROUTING_SET if t == "large"]
    weights = [1 / (i + 1) for i in range(len(faqs))]
    traffic = []
    for i in range(3000):
        r = rnd.random()
        if r < 0.70:
            traffic.append(rnd.choice(variants[rnd.choices(faqs, weights)[0]]))
        elif r < 0.90:
            traffic.append(rnd.choice(medium))
        else:
            traffic.append(f"{rnd.choice(large)} (case #{i})")  # unique => never cached

    gw = gateway_from_preset("customer_support", providers={"anthropic": MockProvider()})
    opus = gw.router.get("claude-opus-5-5")
    baseline = gateway_cost = 0.0
    by_model = Counter()
    quality_floor_violations = 0
    labels = dict(ROUTING_SET)
    for p in traffic:
        r = gw.generate(p, tenant_id="acme", max_tokens=800)
        base_prompt = p.split(" (case #")[0]
        out_tokens = r.usage.output_tokens or 30
        baseline += opus.cost(gw.analyze(p)["analysis"]["estimated_input_tokens"], out_tokens)
        gateway_cost += r.cost_usd
        by_model["cache" if r.cache_hit else r.model] += 1
        if labels.get(base_prompt) == "large" and not r.cache_hit and r.model != "claude-opus-5-5":
            quality_floor_violations += 1

    savings = 1 - gateway_cost / baseline
    REPORT["economics"] = {"requests": len(traffic), "baseline_usd": round(baseline, 4),
                           "gateway_usd": round(gateway_cost, 4), "savings_pct": round(100 * savings, 1),
                           "served_by": dict(by_model), "cache_hit_rate": round(gw.stats()["cache"]["hit_rate"], 3),
                           "hard_prompts_downrouted": quality_floor_violations}
    print(json.dumps(REPORT["economics"], indent=2))
    assert savings >= 0.5
    assert quality_floor_violations == 0
