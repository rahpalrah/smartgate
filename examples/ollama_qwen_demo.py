"""Run smartgate on local Qwen models via Ollama: free, offline, no API key, no packages.

Setup (once, in a terminal):
    ollama pull qwen2.5:1.5b qwen2.5:3b qwen2.5:7b      # or --profile light:  0.5b 1.5b 3b
    python -m smartgate ollama-check                    # confirms Ollama is up and has the models

Run:
    python examples/ollama_qwen_demo.py [--profile light|standard|strong] [--host host:port]

Shows: complexity routing across the three Qwen sizes, the semantic cache, cost (priced at Claude list
prices so the numbers mean something), and a cache-hit verifier that uses the medium Qwen model as judge.
"""
import argparse
import time

from smartgate import (CacheConfig, GatewayRequest, MemoizedVerifier, SemanticCache, gateway_from_ollama,
                       provider_verifier, qwen_models)

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="standard", choices=["light", "standard", "strong"])
ap.add_argument("--host", default=None, help="default: $OLLAMA_HOST or localhost:11434")
ap.add_argument("--models", default=None, help="three tags: small,medium,large")
args = ap.parse_args()
models = args.models.split(",") if args.models else None

gw = gateway_from_ollama(args.profile, host=args.host, preset="customer_support", models=models,
                         shadow_prices=True, check=True,           # check=True: clear error if a model is missing
                         verify_with="medium")                     # the medium Qwen double-checks every cache hit
SYSTEM = "You are a customer support assistant for an online store. Answer in at most 3 sentences."
print("tiers:", {m.tier: m.name for m in gw.router.models}, "\n")

questions = [
    ("easy FAQ", "How do I reset my password?"),
    ("same question, reworded -> cache", "how can I reset my password"),
    ("easy", "What are your store opening hours?"),
    ("medium", "Summarize the pros and cons of expedited shipping versus standard shipping for a customer "
               "deciding between them for a birthday gift."),
    ("hard", "Analyze why a customer might be charged twice for one order, compare the likely causes, and "
             "propose a step-by-step plan to resolve it, ensuring we avoid refunding the wrong amount."),
]
for label, q in questions:
    t0 = time.perf_counter()
    r = gw.generate(q, system=SYSTEM, tenant_id="demo", max_tokens=160)
    print(f"[{label}]\n  Q: {q[:90]}\n  -> {r.model:14} {'CACHE HIT' if r.cache_hit else f'{(time.perf_counter()-t0):5.1f}s'} "
          f"cost=${r.cost_usd:.6f} ({r.route_reason})\n  A: {r.text.strip()[:160]!r}\n")

# The verifier on its own: does one answer fit both questions? (judge = the medium tier, never the 0.5b model)
provider = gw.providers["ollama"]
judge = MemoizedVerifier(provider_verifier(provider, qwen_models(args.profile, models)[1]))
cache = SemanticCache(CacheConfig(similarity_threshold=0.5), verifier=judge)
cache.store_response(GatewayRequest(messages=[{"role": "user", "content": "How can I lose weight in 2 weeks?"}]), "A", "m")
for q in ("How can I lose weight in 2 months?", "How can I lose weight in 2 weeks fast?"):
    hit = cache.lookup(GatewayRequest(messages=[{"role": "user", "content": q}]))
    print(f"verifier: {q!r:45} -> {'served from cache' if hit else 'rejected (not the same question)'}")

s = gw.stats()
print("\nstats:", {"requests": s["counters"].get("requests"), "cache_hit_rate": s["cache"]["hit_rate"],
                   "shadow_cost_usd": s["cost"]["total_usd"], "saved_usd": s["cost"]["saved_usd"],
                   "p50_latency_ms": s["latency_ms"]["p50"]})
