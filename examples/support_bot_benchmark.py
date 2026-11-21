"""Real-data benchmark: e-commerce customer support on the Bitext dataset, with free local models.

Data:   bitext/Bitext-customer-support-llm-chatbot-training-dataset (Hugging Face, CDLA-Sharing-1.0)
        26,872 real-style customer messages labeled with 27 intents.
Models: free GGUF models run locally with llama-cpp-python (pip install llama-cpp-python):
        Qwen2.5 0.5B / 1.5B / 3B Instruct (Q4_K_M) as small / medium / large tiers,
        nomic-embed-text-v1.5 as an optional neural embedder for the semantic cache.
Prices: calls are free; each tier is "shadow priced" at Claude list prices so the report
        shows what the same traffic would cost on Haiku / Sonnet / Opus.

Part A (--part cache): replays ALL messages through the gateway. The stand-in provider
        answers with the message's intent label, so every cache hit is checked: a hit that
        returns another intent's answer is a WRONG ANSWER.
Part B (--part live):  replays a sample through the three real local models, measures real
        latency and token usage, and kills the large model mid-run to exercise failover.

    python examples/support_bot_benchmark.py --data bitext.csv --models-dir ./models --part all
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import threading
import time
from collections import Counter, defaultdict

from smartgate import (CacheConfig, CallableEmbedder, CallableProvider, Gateway, HashingEmbedder,
                       ModelSpec, PolicyConfig, ProviderError, ProviderResponse, SemanticCache, Usage,
                       gateway_from_config)
from smartgate.analysis import estimate_tokens

SYSTEM = ("You are a customer support assistant for an online store. "
          "Answer the customer helpfully in at most 3 sentences.")
TIERS = [  # local model file, tier, shadow price = Claude list price (USD per MTok in / out)
    ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "small", 0.10, 0.50, "claude-haiku-5-5"),
    ("qwen2.5-1.5b-instruct-q4_k_m.gguf", "medium", 2.00, 10.00, "claude-sonnet-5-5"),
    ("qwen2.5-3b-instruct-q4_k_m.gguf", "large", 4.00, 20.00, "claude-opus-5-5"),
]
OPUS_IN, OPUS_OUT = 4.00, 20.00

# ----------------------------------------------------------------------------- data
FILL = {
    "{{Person Name}}": ["Maria Lopez", "John Smith", "Aisha Khan", "Wei Chen", "Priya Patel", "Tom Becker"],
    "{{Account Type}}": ["Premium", "Basic", "Gold", "Platinum", "Business", "Free"],
    "{{Account Category}}": ["Premium", "Standard", "Pro", "Family", "Student"],
    "{{Currency Symbol}}": ["$", "€", "£"],
    "{{Delivery City}}": ["Berlin", "Austin", "Mumbai", "Toronto", "Lyon", "Osaka"],
    "{{Delivery Country}}": ["Germany", "USA", "India", "Canada", "France", "Japan"],
}


def load(path: str, seed: int = 7, pii_rate: float = 0.05):
    """Fill placeholders with realistic values, plant synthetic PII in a few messages, shuffle."""
    rnd = random.Random(seed)
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = []
    for r in rows:
        text = r["instruction"]
        for ph, choices in FILL.items():
            text = text.replace(ph, rnd.choice(choices))
        text = text.replace("{{Order Number}}", str(rnd.randint(100000, 999999)))
        text = text.replace("{{Invoice Number}}", f"INV-{rnd.randint(10000, 99999)}")
        text = text.replace("{{Refund Amount}}", str(rnd.choice([15, 29, 49.99, 120, 300])))
        text = re.sub(r"\{\{[^}]+\}\}", "", text)
        pii = None
        if rnd.random() < pii_rate:
            pii = rnd.choice([f"my email is cust{rnd.randint(1, 9999)}@example.com",
                              f"call me at 555-{rnd.randint(100, 999)}-{rnd.randint(1000, 9999)}"])
            text = f"{text} {pii}"
        out.append({"text": text, "intent": r["intent"], "category": r["category"],
                    "flags": r["flags"], "pii": pii})
    rnd.shuffle(out)
    return out


# ------------------------------------------------------------------------ embedders
class NomicEmbedder:
    """nomic-embed-text-v1.5 via llama.cpp; memoized so threshold sweeps embed each text once."""

    def __init__(self, path: str):
        from llama_cpp import Llama
        self.llm = Llama(model_path=path, embedding=True, n_ctx=512, n_threads=os.cpu_count(), verbose=False)
        self.memo, self.lock, self.dim = {}, threading.Lock(), 768

    def embed(self, text: str):
        with self.lock:
            if text not in self.memo:
                v = self.llm.embed("clustering: " + text)
                self.memo[text] = v[0] if v and isinstance(v[0], list) else v
            return self.memo[text]


class MemoHashing(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.memo = {}

    def embed(self, text):
        if text not in self.memo:
            self.memo[text] = super().embed(text)
        return self.memo[text]


# -------------------------------------------------------------------------- part A
def shadow_models(provider: str):
    return [ModelSpec(claude, provider, tier, pin, pout, max_context_tokens=32_000)
            for _, tier, pin, pout, claude in TIERS]


def run_cache_replay(data, embedder, threshold, label):
    intent_of = {}
    for d in data:
        intent_of.setdefault(d["text"], d["intent"])
    leaked = []

    def stand_in(model, request):  # answers with the intent: makes every cache hit checkable
        msg = request.last_user_message
        if "@example.com" in msg or re.search(r"555-\d{3}-\d{4}", msg):
            leaked.append(msg)
        return f"INTENT::{intent_of.get(msg, intent_of.get(redact_map.get(msg, ''), '?'))}"

    redact_map = {}
    gw = Gateway(models=shadow_models("standin"), providers={"standin": CallableProvider("standin", stand_in)},
                 cache=SemanticCache(CacheConfig(similarity_threshold=threshold, ttl_seconds=None,
                                                 max_entries=100_000),
                                     embedder=embedder),
                 policy=PolicyConfig(pii="redact"))
    from smartgate.analysis import redact_pii
    for d in data:  # the provider only sees redacted text; map it back to its label
        red = redact_pii(d["text"])
        if red != d["text"]:
            redact_map.setdefault(red, d["text"])

    hits = wrong = 0
    tiers = Counter()
    tier_by_cat = defaultdict(Counter)
    cost = baseline = 0.0
    wrong_examples = []
    t0 = time.perf_counter()
    for d in data:
        r = gw.generate(d["text"], system=SYSTEM, max_tokens=150)
        in_tok = estimate_tokens(SYSTEM + d["text"])
        out_tok = 60  # typical support reply length, same for baseline and gateway
        baseline += (in_tok * OPUS_IN + out_tok * OPUS_OUT) / 1e6
        if r.cache_hit:
            hits += 1
            if r.text != f"INTENT::{d['intent']}":
                wrong += 1
                if len(wrong_examples) < 6:
                    wrong_examples.append({"asked": d["text"], "intent": d["intent"], "got": r.text[8:],
                                           "similarity": r.cache_similarity})
        else:
            spec = gw.router.get(r.model)
            cost += spec.cost(in_tok, out_tok)
            tiers[spec.tier] += 1
            tier_by_cat[d["category"]][spec.tier] += 1
    n = len(data)
    return {
        "embedder": label, "threshold": threshold, "requests": n,
        "cache_hit_rate": round(hits / n, 4),
        "wrong_answers": wrong, "wrong_per_1000_requests": round(1000 * wrong / n, 2),
        "hit_precision": round(1 - wrong / hits, 4) if hits else None,
        "model_calls": n - hits, "calls_by_tier": dict(tiers),
        "shadow_cost_usd": round(cost, 4), "baseline_all_opus_no_cache_usd": round(baseline, 4),
        "savings_pct": round(100 * (1 - cost / baseline), 1),
        "pii_messages": sum(1 for d in data if d["pii"]), "pii_leaked_to_model": len(leaked),
        "guard_rejections": gw.cache.stats.get("guard_rejections", 0),
        "seconds": round(time.perf_counter() - t0, 1),
        "tier_by_category": {c: dict(v) for c, v in sorted(tier_by_cat.items())},
        "wrong_examples": wrong_examples,
    }


# -------------------------------------------------------------------------- part B
class LocalModels:
    def __init__(self, models_dir):
        from llama_cpp import Llama
        self.llms = {}
        for fname, tier, *_ , claude in TIERS:
            self.llms[claude] = Llama(model_path=os.path.join(models_dir, fname), n_ctx=2048,
                                      n_threads=os.cpu_count(), verbose=False)
        self.lock = threading.Lock()
        self.down = set()
        self.latency = defaultdict(list)

    def __call__(self, model, request):
        if model.name in self.down:
            raise ProviderError(f"{model.name} unavailable", retryable=True, kind="server")
        msgs = ([{"role": "system", "content": request.system}] if request.system else []) + \
               [{"role": m.role, "content": m.content} for m in request.messages]
        with self.lock:
            t0 = time.perf_counter()
            out = self.llms[model.name].create_chat_completion(messages=msgs, max_tokens=request.max_tokens,
                                                               temperature=0.2)
            self.latency[model.name].append(time.perf_counter() - t0)
        u = out["usage"]
        return ProviderResponse(out["choices"][0]["message"]["content"].strip(), model.name,
                                Usage(u["prompt_tokens"], u["completion_tokens"]), "end_turn")


def run_live(data, models_dir, embedder, threshold, n, kill_at):
    local = LocalModels(models_dir)
    gw = Gateway(models=shadow_models("local"), providers={"local": CallableProvider("local", local)},
                 cache=SemanticCache(CacheConfig(similarity_threshold=threshold, ttl_seconds=None), embedder=embedder),
                 policy=PolicyConfig(pii="redact"))
    from smartgate import ResilienceConfig, RetryPolicy
    gw.resilience = ResilienceConfig(retry=RetryPolicy(max_attempts=2, base_delay=0.1), breaker_failure_threshold=3)
    rnd = random.Random(11)
    sample = rnd.sample(data, n)
    # realistic repetition: 35% of traffic re-asks one of the 40 most common questions
    hot = sample[:40]
    traffic = [rnd.choice(hot) if rnd.random() < 0.35 else d for d in sample]
    rows, failures, baseline, cost = [], 0, 0.0, 0.0
    t_start = time.perf_counter()
    for i, d in enumerate(traffic):
        if i == kill_at:
            local.down.add("claude-opus-5-5")  # large tier goes down mid-run
        t0 = time.perf_counter()
        try:
            r = gw.generate(d["text"], system=SYSTEM, max_tokens=96)
        except Exception as e:  # noqa
            failures += 1
            continue
        wall = time.perf_counter() - t0
        u_in = r.usage.input_tokens or estimate_tokens(SYSTEM + d["text"])
        u_out = r.usage.output_tokens or 60
        baseline += (u_in * OPUS_IN + u_out * OPUS_OUT) / 1e6
        cost += r.cost_usd
        rows.append({"i": i, "model": "cache" if r.cache_hit else r.model, "latency_s": wall,
                     "fallback": any(not a.ok for a in r.attempts), "after_kill": i >= kill_at,
                     "intent": d["intent"], "q": d["text"], "a": r.text})
    hit_lat = [x["latency_s"] for x in rows if x["model"] == "cache"]
    miss_lat = [x["latency_s"] for x in rows if x["model"] != "cache"]
    after = [x for x in rows if x["after_kill"] and x["model"] != "cache"]
    return {
        "requests": len(traffic), "failed_requests": failures,
        "served_by": dict(Counter(x["model"] for x in rows)),
        "cache_hit_rate": round(len(hit_lat) / len(rows), 3),
        "latency_cache_hit_ms_p50": round(1000 * statistics.median(hit_lat), 2) if hit_lat else None,
        "latency_model_call_s_p50": round(statistics.median(miss_lat), 2) if miss_lat else None,
        "per_model_generation_s_p50": {m: round(statistics.median(v), 2) for m, v in local.latency.items()},
        "large_tier_killed_at_request": kill_at,
        "after_kill_large_tier_requests_rerouted": sum(1 for x in after if x["fallback"]),
        "after_kill_served_by": dict(Counter(x["model"] for x in after)),
        "circuit_breakers": gw.stats()["circuit_breakers"],
        "shadow_cost_usd": round(cost, 5), "baseline_all_opus_no_cache_usd": round(baseline, 5),
        "savings_pct": round(100 * (1 - cost / baseline), 1) if baseline else None,
        "wall_clock_min": round((time.perf_counter() - t_start) / 60, 1),
        "sample_answers": [{k: x[k] for k in ("model", "intent", "q", "a")}
                           for x in rows if x["model"] != "cache"][:6],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models-dir")
    ap.add_argument("--part", choices=["cache", "live", "all"], default="cache")
    ap.add_argument("--live-n", type=int, default=300)
    ap.add_argument("--out", default="support_benchmark_results.json")
    args = ap.parse_args()

    data = load(args.data)
    results = {"dataset_messages": len(data)}
    nomic = None
    if args.models_dir and os.path.exists(os.path.join(args.models_dir, "nomic-embed-text-v1.5.Q8_0.gguf")):
        nomic = NomicEmbedder(os.path.join(args.models_dir, "nomic-embed-text-v1.5.Q8_0.gguf"))

    if args.part in ("cache", "all"):
        runs = []
        hashing = MemoHashing()
        for th in (0.80, 0.86, 0.90, 0.95):
            runs.append(run_cache_replay(data, hashing, th, "hashing (built-in)"))
            print(json.dumps({k: runs[-1][k] for k in ("embedder", "threshold", "cache_hit_rate",
                                                         "wrong_per_1000_requests", "savings_pct")}), flush=True)
        if nomic:
            for th in (0.85, 0.90, 0.93, 0.95, 0.97):
                runs.append(run_cache_replay(data, CallableEmbedder(nomic.embed, 768), th, "nomic-embed-text-v1.5"))
                print(json.dumps({k: runs[-1][k] for k in ("embedder", "threshold", "cache_hit_rate",
                                                             "wrong_per_1000_requests", "savings_pct")}), flush=True)
        results["cache_replay"] = runs

    if args.part in ("live", "all"):
        emb = CallableEmbedder(nomic.embed, 768) if nomic else HashingEmbedder()
        results["live"] = run_live(data, args.models_dir, emb, 0.95 if nomic else 0.90,
                                   args.live_n, kill_at=args.live_n // 2)
        print(json.dumps({k: v for k, v in results["live"].items() if k != "sample_answers"}, indent=1))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
