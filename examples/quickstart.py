"""Quickstart. Runs offline with the mock provider; pass --live to call Claude
(needs only the ANTHROPIC_API_KEY environment variable; no package to install)."""
import sys

from smartgate import Budget, MockProvider, gateway_from_preset

live = "--live" in sys.argv
providers = None if live else {"anthropic": MockProvider()}
gw = gateway_from_preset("customer_support", providers=providers)
gw.set_budget("acme", Budget(limit_usd=25.0, period="monthly", downgrade_at=0.8))
gw.on("fallback", lambda event, data: print("  !! fallback:", data))

prompts = [
    "How do I reset my password?",
    "how can I reset my password",                      # semantic cache hit
    "I was charged twice and want a refund right away",  # escalation rule -> medium tier
    "Analyze our churn drivers from these notes and propose a step-by-step retention strategy, "
    "comparing at least three approaches with trade-offs.",  # high complexity -> large tier
]
for p in prompts:
    r = gw.generate(p, tenant_id="acme")
    print(f"{p[:60]!r:64} -> {r.model:18} cache={r.cache_hit!s:5} "
          f"cost=${r.cost_usd:.6f} saved=${r.saved_usd:.6f}\n    reason: {r.route_reason}")

print("\nstats:", gw.stats()["cache"], gw.stats()["cost"])
print("tenant report:", gw.costs.report("acme"))
