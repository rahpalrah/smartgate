# smartgate — Semantic Cache & Model Routing Gateway SDK

A drop-in layer between your applications and LLMs that **answers repeat questions from a semantic cache**, **sends each prompt to the cheapest model that can handle it**, **enforces per-tenant budgets**, and **fails over automatically** when a model is down or refuses.

- **Zero-dependency core** (Python stdlib). `anthropic` and `pyyaml` are optional extras.
- **Two ways to use it:** embed as a Python library, or run it as an HTTP service any language can call over plain HTTP + JSON (Python client included).
- **Industry presets:** healthcare, finance, e-commerce, customer support, legal, education and SaaS/devtools, all overridable.
- **Pluggable everywhere:** providers, embedders, vector stores, spend ledgers, analyzers, routing rules and event hooks.

```
request ─► Prompt analysis ─► Policy (PII redact/block) ─► Semantic cache ──hit──► response ($0)
                                                               │ miss
                                                               ▼
               Budget gate ◄── Model router (complexity + rules + budget pressure)
                   │
                   ▼
     Provider call ─► retry w/ backoff ─► circuit breaker ─► fallback chain ─► record cost ─► cache store
```

## Install

From GitHub:
```bash
pip install "smartgate[anthropic] @ git+https://github.com/rahpalrah/smartgate.git"
```

From a clone, for development:
```bash
git clone https://github.com/rahpalrah/smartgate.git && cd smartgate
pip install -e ".[anthropic,fast,dev]"
```

| Extra | Adds |
|---|---|
| `anthropic` | Claude provider (official `anthropic` SDK) |
| `fast` | NumPy-accelerated cache search (~0.4 ms at 10k entries vs ~150 ms in pure Python) |
| `yaml` | YAML config files |
| `dev` | pytest, hypothesis |

Credentials for Claude resolve the standard way: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or an `ant auth login` profile.

## 60-second tour

```python
from smartgate import gateway_from_preset, Budget

gw = gateway_from_preset("customer_support")        # Claude Haiku/Sonnet/Opus 5.5 tiers
gw.set_budget("acme", Budget(limit_usd=50, period="monthly", downgrade_at=0.8))

r = gw.generate("How do I reset my password?", tenant_id="acme")
r.text, r.model, r.cost_usd, r.route_reason        # 'claude-haiku-5-5', 'complexity 0.20 -> small'

r = gw.generate("how can I reset my password", tenant_id="acme")
r.cache_hit, r.cache_similarity, r.saved_usd       # True, <similarity>, <cost of the first call>

gw.analyze("Design a multi-region failover architecture ...")   # dry run: tier, model, est. cost
gw.stats()                                          # hit rate, spend, savings, p50/p95, breaker states
```

Run without an API key: `python examples/quickstart.py`, which uses the offline `MockProvider`. Add `--live` to call Claude.

## Features

### 1. Semantic caching
- **Exact-match fast path** on normalized text, then **cosine similarity** search over embeddings.
- **Context-safe keys:** an entry is only reused when the tenant, system prompt and prior conversation turns match, so an answer never leaks into a different conversation.
- **Tenant isolation** by default (`scope="tenant"`); use `scope="global"` for shared public content such as product FAQs.
- **TTL**, LRU size bounds, manual invalidation per tenant.
- **Verification after retrieval:** a semantic match is served only if both questions agree on numbers ("order 5512" ≠ "order 5513"), negation ("safe" ≠ "not safe") and the requested action ("check my invoice" ≠ "send me my invoice", "edit" ≠ "delete"). Actions are grouped into synonym classes, so "change" still matches "modify". Add domain classes with `CacheConfig(extra_action_classes={...})`.
- **Calibrate the threshold on your own traffic** instead of guessing:
  ```python
  from smartgate import calibrate_threshold
  result = calibrate_threshold(labeled_queries, embedder, max_wrong_per_1000=0.5)  # [(text, label), ...]
  result["threshold"], result["table"]   # best threshold within the wrong-answer budget, full sweep
  ```
- **Seed approved answers:** `gw.cache.seed("What is your return policy?", approved_text, tenant_id="shop", system=SYSTEM)`.
- **Safety:** prompts containing PII or time-sensitive words ("today", "latest", "stock price") are not cached unless you opt in.
- **Backends:** `InMemoryVectorStore`, `SQLiteVectorStore` (durable), or implement `VectorStore` for Redis, pgvector, Qdrant or Pinecone.
- **Embeddings:** the built-in `HashingEmbedder` is deterministic and offline, and catches rephrasings and near-duplicates. For deeper paraphrase matching, plug in any neural model:
  ```python
  from smartgate import CallableEmbedder, SemanticCache
  from sentence_transformers import SentenceTransformer
  m = SentenceTransformer("all-MiniLM-L6-v2")
  cache = SemanticCache(embedder=CallableEmbedder(lambda t: m.encode(t), dim=384))
  ```

### 2. Prompt analysis
`PromptAnalyzer` runs in under 1 ms and returns:
- a `complexity` score from 0 to 1, built from the task-type prior, length, multi-step markers, code blocks, multiple intents, constraints, conversation depth and domain keywords;
- a `task_type`: qa, classification, extraction, summarization, translation, creative, code, reasoning, math or general;
- `pii` found (email, phone, SSN, Luhn-checked card numbers, IP, IBAN, MRN), `time_sensitive`, and `cacheable`.

Extend it without forking:
```python
PromptAnalyzer(domain_keywords={"indemnif": 0.25},
               extra_scorers=[lambda req, a: 0.3 if "board deck" in req.last_user_message else 0])
```

### 3. Model selection by complexity
- `small_max` / `medium_max` thresholds map complexity to a tier. `min_tier` and `max_tier` set a global floor and ceiling.
- **Rules** (first match wins) can force a `tier`, a `min_tier` or a specific `model`. They can match on `keywords`, `task_types`, `tenants`, `tags` or `min_complexity`, either in a config file or as Python callables.
- **Capability and context-window filtering:** a model is only chosen if the prompt fits its context window and it has the capabilities the task requires.
- Per-request overrides: `model=`, `tier=`, `use_cache=False`.

### 4. Cost controls
- `Budget(limit_usd, period="daily|monthly|total", hard_limit=True, downgrade_at=0.8, max_cost_per_request=...)`.
- **Soft landing:** once a tenant passes `downgrade_at`, routing steps down one tier. When the budget can't cover the chosen model, the gateway switches to a cheaper one before it ever returns `BudgetExceededError`.
- Every call is costed from real token usage and model prices. Each cache hit records the dollars it saved.
- `gw.costs.report(tenant)` gives spend by model, savings and budget utilization. Implement `SpendLedger` to keep spend in Redis or SQL and share it across replicas.

### 5. Fallback mechanisms
- **Retry** with exponential backoff, jitter, and `retry-after` support. Only transient errors are retried: rate limits, 5xx, timeouts and connection errors.
- **Circuit breaker** per model (closed → open → half-open), so a failing model is skipped instantly.
- **Fallback chain:** `same_tier_then_up` (the default: escalate for quality, then degrade for availability), `same_tier_then_down`, or `any`.
- **Cascade on bad answers:** every answer passes a validator (default: empty, refusal-style or unfilled-template answers fail). A failed answer is billed but never cached, and the request escalates to the next tier up. If no tier passes, the last answer is returned with `quality_warning` set. Plug in your own check (an LLM judge, a schema check) with `Gateway(validator=fn)`, or pass `validator=None` to turn it off.
- **Refusals** (`stop_reason == "refusal"`) fail over to the next model. Bad requests are raised immediately, because another model wouldn't fix them.

### 6. Policy and observability
- `PolicyConfig(pii="allow|redact|block", blocked_terms=[...], max_input_tokens=...)`. Redaction runs before the model call and before the cache key is built.
- Hooks: `gw.on("cache.hit" | "fallback" | "budget.exceeded" | "*", fn)`. Use them to bridge into OpenTelemetry, Prometheus, Datadog or an audit log. A hook that throws never breaks a request.

## Industry presets

| Preset | Cache threshold / TTL | PII | Routing highlights |
|---|---|---|---|
| `healthcare` | 0.97 / 6h | redact | floor `medium`; dosage, contraindication and drug-interaction prompts go to `large` |
| `finance` | 0.95 / 15m | redact | KYC, AML and compliance prompts at least `medium`; domain complexity boosts |
| `ecommerce` | 0.86 / 24h | redact | wider small tier; catalog copy goes to `medium` |
| `customer_support` | 0.88 / 12h | redact | refund, complaint and cancellation prompts escalate to `medium` |
| `legal` | 0.97 / 7d | redact | floor `medium`; clause and liability boosts |
| `education` | 0.88 / 7d | redact | math at least `medium` |
| `saas_devtools` | 0.93 / 1h | allow | code requires the `code` capability, at least `medium` |

Presets are merged under your own config, so anything can be overridden:
```python
gateway_from_config({"preset": "finance", "cache": {"ttl_seconds": 60},
                     "budgets": {"desk-7": {"limit_usd": 500, "period": "daily"}}})
```
See [`examples/config.example.json`](examples/config.example.json) for every option and [`examples/industries.py`](examples/industries.py) for worked examples.

## Bring your own models / providers
```python
from smartgate import Gateway, ModelSpec, CallableProvider
from smartgate.providers import AnthropicProvider

gw = Gateway(
    models=[ModelSpec("local-llama", "onprem", "small", 0, 0, max_context_tokens=8000),
            ModelSpec("claude-sonnet-5-5", "anthropic", "medium", 2, 10, max_context_tokens=1_000_000),
            ModelSpec("claude-opus-5-5", "anthropic", "large", 4, 20, max_context_tokens=1_000_000)],
    providers={"onprem": CallableProvider("onprem", my_inference_fn),
               "anthropic": AnthropicProvider()},   # or AnthropicProvider(client=AnthropicBedrockMantle(...))
)
```
Subclass `Provider` and implement `complete(model, request) -> ProviderResponse` to connect any vendor or internal model. Raise `ProviderError(retryable=..., fallback=...)` so the gateway knows whether to retry, fail over or stop. `ModelSpec.params` passes per-model API params, for example `{"output_config": {"effort": "low"}}` on the small tier.

## Run as a service (any language)
```bash
smartgate --config examples/config.example.json serve --port 8080 \
  --api-keys '{"key-acme":"acme","key-ops":"ops"}' --admin-keys key-ops
# offline demo: smartgate --preset customer_support --mock serve
```
```bash
curl -s localhost:8080/v1/generate -H "Authorization: Bearer key-acme" \
  -d '{"prompt":"Where is my order?","max_tokens":512}'
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/generate` | `{prompt \| messages, system, max_tokens, tier, model, use_cache, tags, metadata}` |
| `POST /v1/analyze` | Dry-run routing decision and cost estimate |
| `GET /v1/stats` · `GET /v1/costs` | Metrics, hit rate, spend and savings |
| `POST /v1/cache/invalidate` | Clears the caller's tenant (admin keys can target any tenant or all) |
| `GET /health` | Liveness |

When API keys are configured, the tenant always comes from the key and never from the request body, so one tenant can't spend another's budget. Error codes: `401` unauthorized, `402` budget exceeded, `422` policy violation, `503` all models failed.

Python client:
```python
from smartgate.client import GatewayClient
client = GatewayClient("http://localhost:8080", api_key="key-acme")
print(client.generate("Where is my order?")["text"])
```
Services in other languages call the same endpoints with any HTTP library.

## CLI
```bash
smartgate --mock analyze "Refactor this function to be thread-safe"
smartgate --preset healthcare ask "What are common symptoms of anemia?"
smartgate presets
```

## Benchmark on real data

On 26,872 real-style support messages (Bitext dataset), with thresholds calibrated on one half and results reported on the other: **26% of requests served from cache with 0.07 wrong answers per 1,000**, PII never reaching a model, 0 failed requests when a model was killed mid-run, and ~90% lower cost than sending everything to Opus. Full method, numbers and limits: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Testing

```bash
pytest -q                                              # 113 tests, ~15 s, fully offline
SMARTGATE_FUZZ=3000 pytest tests/advanced/test_properties.py   # deeper fuzzing
```

| Suite | Proves |
|---|---|
| `tests/test_*.py` | Unit and end-to-end behaviour of every component and the HTTP server |
| `advanced/test_properties.py` | Invariants for random inputs (hypothesis): complexity bounds, PII redaction completeness, tenant isolation, router chain validity, budget limits |
| `advanced/test_concurrency.py` | 32-64 threads: the hard budget is never overspent, the ledger equals billed responses, no cross-tenant cache leaks, exactly one model call per distinct prompt (request coalescing) |
| `advanced/test_chaos.py` | Random provider failures, refusals, outages; circuit breaker opens and recovers; ≥97% availability under mixed chaos |
| `advanced/test_quality.py` | Routing accuracy on a labeled set (87.5%, 0 severe misroutes), zero cache false hits on near-miss pairs, 82% cost savings vs. always-Opus. Writes `benchmark_report.json` |
| `advanced/test_system.py` | HTTP under 40 concurrent clients, budget enforced over HTTP, SQLite cache shared across processes, lookup latency at 10k entries |
| `advanced/test_regressions.py` | One test per bug found by the suites above |

## Production notes
- Hard budgets reserve each call's worst-case cost before it runs (input estimate × 1.25 + `max_tokens`) and settle the real cost afterwards. Enforcement is atomic within one process; across replicas, use a shared `SpendLedger` with an atomic reserve step.
- Semantic hits also require numbers and negations to match exactly ("feature 6" ≠ "feature 57", "safe" ≠ "not safe"). This is on by default (`guard_numbers`, `guard_negation`).
- In-memory cache and spend ledger are per process. For multiple replicas, use a shared `VectorStore` (Redis or pgvector) and a shared `SpendLedger`, or put a single gateway service in front of your apps.
- Tune `similarity_threshold` on your own traffic. Log `cache.hit` events with their similarity and review false hits before lowering it.
- The heuristic analyzer is a strong default. If you have labeled traffic, replace it with a trained classifier via `extra_scorers` or a `PromptAnalyzer` subclass.
- Streaming responses aren't routed through the cache yet. Call the provider directly for streaming endpoints.

## Layout
```
smartgate/
  gateway.py        orchestrator (analysis → policy → cache → route → budget → call → record)
  analysis.py       complexity, task type, PII, cacheability
  cache/            SemanticCache, verification guards, calibration, InMemory/SQLite stores
  quality.py        answer validator for cascade escalation
  embeddings.py     HashingEmbedder, CallableEmbedder
  routing.py        ModelRouter, RoutingRule, fallback ordering
  cost.py           Budget, CostController, SpendLedger
  resilience.py     RetryPolicy, CircuitBreaker
  providers/        Provider base, AnthropicProvider, CallableProvider, MockProvider
  config.py         dict/JSON/YAML config + industry presets
  server.py client.py cli.py observability.py
tests/              113 tests: unit, e2e, HTTP, property-based, concurrency, chaos, quality benchmarks
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
