"""The same SDK across industries: presets + custom rules + custom providers."""
from smartgate import (CallableProvider, MockProvider, ModelSpec, RoutingRule, gateway_from_config,
                       gateway_from_preset)

mock = {"anthropic": MockProvider()}

# --- Healthcare: PII redacted, high cache threshold, clinical prompts forced to the large tier
health = gateway_from_preset("healthcare", providers=mock)
r = health.generate("Patient MRN 4482913, email pat@clinic.org: max safe dosage of acetaminophen?")
print("healthcare:", r.model, "|", r.text)          # PII never reaches the model

# --- Finance: regulatory keywords route at least to medium; per-tenant hard budgets from config
fin = gateway_from_config({
    "preset": "finance",
    "budgets": {"desk-7": {"limit_usd": 500, "period": "daily", "max_cost_per_request": 0.50}},
}, providers=mock)
print("finance:", fin.analyze("Summarize KYC obligations for a new broker-dealer", tenant_id="desk-7")["model"])

# --- E-commerce: aggressive caching for product Q&A, global cache scope shared across storefronts
shop = gateway_from_config({"preset": "ecommerce", "cache": {"scope": "global"}}, providers=mock)
shop.generate("Does the X200 blender come with a warranty?", tenant_id="store-eu")
print("ecommerce cache hit across tenants:",
      shop.generate("does the x200 blender come with warranty", tenant_id="store-us").cache_hit)

# --- Bring-your-own models: an on-prem model as the small tier, Claude for the rest
def on_prem_llm(model, request):
    return f"(on-prem) {request.last_user_message[:40]}"

hybrid = gateway_from_config({
    "models": [
        {"name": "local-llama", "provider": "onprem", "tier": "small",
         "input_cost_per_mtok": 0.0, "output_cost_per_mtok": 0.0, "max_context_tokens": 8000},
        {"name": "claude-sonnet-5-5", "provider": "anthropic", "tier": "medium",
         "input_cost_per_mtok": 2.0, "output_cost_per_mtok": 10.0, "max_context_tokens": 1_000_000},
        {"name": "claude-opus-5-5", "provider": "anthropic", "tier": "large",
         "input_cost_per_mtok": 4.0, "output_cost_per_mtok": 20.0, "max_context_tokens": 1_000_000},
    ]}, providers={"onprem": CallableProvider("onprem", on_prem_llm), **mock})
hybrid.router.add_rule(RoutingRule("internal-only", when=lambda req, a: "confidential" in req.tags,
                                   model="local-llama"))
print("hybrid:", hybrid.generate("Classify this ticket", tags=["confidential"]).model)
