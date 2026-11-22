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

*Integrating into an existing system? Jump to [Integrate into your code](#integrate-into-your-code).*

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
| `pii` | International phone-number detection (Google's libphonenumber) |
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

## Integrate into your code

This section is for engineers adding smartgate to an existing application. Pick the pattern that fits your system, then follow the "lines to change" for it.

### Step 0 — decide where it runs

| Your situation | Use | Why |
|---|---|---|
| One Python service calls an LLM | **Library**, in-process (patterns A–C below) | Fewest moving parts, no extra network hop |
| Several services, or services in Java / Go / Node / .NET / PHP | **Shared HTTP service** (pattern D) | One cache, one budget ledger, any language |
| Many replicas of a Python service | Library **plus** shared storage (pattern E) | Replicas must share the cache and the spend ledger |

```
In-process library                    Shared service (any language)

 your app ──► smartgate ──► LLM        app 1 ─┐
              (cache, routing,         app 2 ─┼─HTTP─► smartgate serve ──► LLMs
               budgets, fallback)      app 3 ─┘        (one cache, one ledger)
```

**What it handles today:** text in, text out (single prompt or a multi-turn message list, plus a system prompt).
**What it does not handle yet:** streaming responses, tool/function calls, and image or PDF inputs. Keep those calls on your existing client and route only plain-text requests through the gateway.

---

### Pattern A — replace a direct Anthropic SDK call (Python)

Before:
```python
import anthropic                                                        # line 1
client = anthropic.Anthropic()                                          # line 2

resp = client.messages.create(                                          # line 3
    model="claude-opus-5-5",                                            # line 4
    max_tokens=1024,
    system="You are a support bot.",
    messages=history,                                                   # [{"role": "user"|"assistant", "content": "..."}]
)
answer = resp.content[0].text                                           # line 5
```

After:
```python
from smartgate import gateway_from_preset                               # line 1  (replaces the import)
gw = gateway_from_preset("customer_support")                            # line 2  (create ONCE at startup, reuse everywhere)

r = gw.generate(                                                        # line 3
    history,                                                            #         same message list you already build
    system="You are a support bot.",                                    #         same system prompt
    tenant_id=current_customer_id,                                      # NEW: who is asking (see "Choosing tenant_id")
    max_tokens=1024,
)                                                                       # line 4: delete `model=`; the router picks it
answer = r.text                                                         # line 5
```

| Line | Change |
|---|---|
| 1–2 | Swap the SDK import and client for one gateway object. Build it **once** (module level or app startup); it is thread-safe. |
| 3 | `client.messages.create(...)` becomes `gw.generate(messages, ...)`. The messages list is passed positionally. |
| 4 | **Delete `model=`.** The router chooses the model per request. To pin one anyway: `model="claude-sonnet-5-5"` (exact model) or `tier="large"`. |
| new | Add `tenant_id=` so each customer gets an isolated cache and its own budget. |
| 5 | `resp.content[0].text` becomes `r.text`. |

Also available on the result: `r.model`, `r.cost_usd`, `r.cache_hit`, `r.saved_usd`, `r.usage`, `r.route_reason`, `r.request_id`, `r.quality_warning`.
Credentials are unchanged: the Claude provider reads `ANTHROPIC_API_KEY` (or an `ant auth login` profile) exactly like the SDK.

### Pattern B — web backend (FastAPI shown; Flask and Django are the same idea)

```python
from fastapi import FastAPI, HTTPException
from smartgate import (AllModelsFailedError, Budget, BudgetExceededError,
                       PolicyViolationError, gateway_from_preset)

gw = gateway_from_preset("ecommerce")          # module level: created once per process
app = FastAPI()

@app.post("/support/ask")
async def ask(body: dict):
    try:
        r = await gw.agenerate(                # agenerate = async wrapper; use generate() in sync code
            body["question"],
            system="You are a support bot.",
            tenant_id=body["customer_id"],
            max_tokens=400,
        )
    except BudgetExceededError:
        raise HTTPException(402, "AI budget exhausted for this customer")
    except PolicyViolationError as e:
        raise HTTPException(422, f"Blocked by policy: {e.findings}")
    except AllModelsFailedError:
        raise HTTPException(503, "AI temporarily unavailable")
    return {"answer": r.text, "cached": r.cache_hit, "model": r.model}
```
Lines to change in your handler: the line that calls the LLM, and the `except` block (map the three errors above to your own error responses). Everything else stays.

### Pattern C — bring your own models (any vendor, any in-house model)

1. **List your models** with their tier and price (USD per million tokens). This is the only place model names and prices live:
```python
from smartgate import Gateway, ModelSpec, CallableProvider
from smartgate.providers import AnthropicProvider

models = [
    ModelSpec("my-small-model", "inhouse",   "small",  0.0,  0.0,  max_context_tokens=8_000),
    ModelSpec("claude-sonnet-5-5", "anthropic", "medium", 2.0, 10.0, max_context_tokens=1_000_000),
    ModelSpec("claude-opus-5-5",   "anthropic", "large",  4.0, 20.0, max_context_tokens=1_000_000),
]
```
2. **Write one function per non-Claude provider.** It receives the chosen `ModelSpec` and the request and returns the reply text:
```python
def call_inhouse(model, request):
    # request.messages = [Message(role, content), ...]; request.system = str | None; request.max_tokens = int
    return my_http_client.post("https://llm.internal/v1/chat", json={
        "model": model.name, "system": request.system,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "max_tokens": request.max_tokens}).json()["text"]
```
3. **Register them:**
```python
gw = Gateway(models=models, providers={
    "inhouse": CallableProvider("inhouse", call_inhouse),
    "anthropic": AnthropicProvider(),                       # or AnthropicProvider(client=<Bedrock/Vertex client>)
})
```
For token counts, cost and error handling, return a `ProviderResponse` and raise `ProviderError(retryable=..., fallback=...)` instead (see "Bring your own models / providers" below). The `provider` string in each `ModelSpec` must match a key in `providers=`.

If you only use the default Claude catalog, the model names and prices in `DEFAULT_MODELS` (`smartgate/config.py`) are what the presets use; override them with a `"models": [...]` list in your config.

### Pattern D — call it from any other language over HTTP

Start the service once (any host that can reach your LLM provider):
```bash
export ANTHROPIC_API_KEY=...
smartgate --preset customer_support serve --host 0.0.0.0 --port 8080 \
  --api-keys '{"key-web":"storefront","key-mobile":"mobile-app"}' --admin-keys key-ops
```
Each API key maps to one tenant, and the server takes the tenant **from the key, never from the request body**, so one client cannot spend another's budget.

Then replace your LLM call with an HTTP call:
```bash
curl -s http://smartgate.internal:8080/v1/generate \
  -H "Authorization: Bearer key-web" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Where is my order?"}],"system":"You are a support bot.","max_tokens":300}'
```
```js
// Node 18+; the same shape works from Java, Go, C#, PHP, Ruby...
const res = await fetch("http://smartgate.internal:8080/v1/generate", {
  method: "POST",
  headers: { "Authorization": "Bearer key-web", "Content-Type": "application/json" },
  body: JSON.stringify({ messages: history, system: SYSTEM_PROMPT, max_tokens: 300 }),
});
const out = await res.json();           // out.text, out.model, out.cost_usd, out.cache_hit
if (res.status === 402) { /* budget exhausted */ }
```

| Lines to change in the calling code | |
|---|---|
| The URL | your provider's endpoint becomes `http(s)://<smartgate-host>/v1/generate` |
| The auth header | your provider key becomes the **smartgate API key** (`Authorization: Bearer ...`); the provider key now lives only on the server |
| The request body | `{"messages": [...], "system": "...", "max_tokens": N}`; optional: `tier`, `model`, `use_cache`, `tags`, `metadata` |
| The response parse | read `text` (also `model`, `cost_usd`, `cache_hit`, `route_reason`) |
| Error handling | `402` budget exhausted · `422` blocked by policy · `503` all models failed · `401` bad key · `400` bad request |

Other endpoints: `POST /v1/analyze` (dry run: which model and what cost, without calling one), `GET /v1/stats`, `GET /v1/costs`, `POST /v1/cache/invalidate`, `GET /health` (use for load-balancer checks).
There is a Python client too: `from smartgate.client import GatewayClient`.

### Pattern E — more than one replica or process

By default the cache and the spend ledger live in one process's memory. With several replicas, each would keep its own cache and its own budget, so a "$100 limit" could be spent once per replica. Choose one:

| Option | Change |
|---|---|
| **Run it as one shared service** (pattern D) | Nothing else to do: all clients share one cache and ledger. Simplest. |
| **Shared cache file on one host** | In config: `"cache_store": {"type": "sqlite", "path": "/var/lib/smartgate/cache.db"}` (several processes on the same machine can share it) |
| **Redis / Postgres / pgvector across hosts** | Implement the two small interfaces and pass them in: `VectorStore` (`smartgate/cache/store.py`) for the cache, `SpendLedger` (`smartgate/cost.py`) for settled spend. **Caveat:** the in-flight reservation that makes a hard budget exact is held in each process's memory, so with several replicas a limit can be exceeded by up to the worst-case cost of the requests in flight at the same moment. For a strict global limit, run one shared service (pattern D). |

---

### Choosing `tenant_id`

`tenant_id` decides who shares a cache and who shares a budget.

| You want | Use |
|---|---|
| Each customer or organization isolated from the others (the usual case) | `tenant_id=<customer or org id>` |
| Per-user isolation (private data in answers) | `tenant_id=<user id>` (lower cache hit rate) |
| One cache for public content such as FAQs | a single fixed `tenant_id`, or `"cache": {"scope": "global"}` in config |

Never reuse one tenant for data that must not mix. With `scope="global"` any tenant can receive another's cached answer.

### What else to change for your own use case

| Goal | Where / what |
|---|---|
| Start from an industry setup | `gateway_from_preset("healthcare" \| "finance" \| "ecommerce" \| "customer_support" \| "legal" \| "education" \| "saas_devtools")` |
| Keep settings outside code | `gateway_from_file("smartgate.json")` or set `SMARTGATE_CONFIG` / `SMARTGATE_PRESET` and call `gateway_from_env()`. See `examples/config.example.json` for every option. |
| Keep personal data away from the model | `"policy": {"pii": "redact"}` (or `"block"`). Add `"phone_regions": ["US", "GB"]` for national phone formats. |
| Spend limits | `gw.set_budget("tenant-id", Budget(limit_usd=100, period="monthly"))`, or `"budgets"` / `"default_budget"` in config |
| Always use at least a certain tier for some requests | `"rules": [{"name": "vip", "tenants": ["gold-co"], "min_tier": "medium"}]` in config |
| Stricter or looser cache matching | `"cache": {"similarity_threshold": 0.9}`; tune it on your own logs with `calibrate_threshold(...)` |
| Make cache hits safer (recommended for high-stakes domains) | Plug in a verifier: `gateway_from_config(cfg, verifier=SymmetricVerifier(LLMPairVerifier(my_judge)))` |
| Stronger semantic matching | `SemanticCache(embedder=CallableEmbedder(my_embed_fn, dim))`; the built-in embedder is lexical |
| Domain words that change the action | `"cache": {"extra_action_classes": {"track": ["chase"]}}` |
| Facts the bot may state without being flagged as invented | `"quality": {"grounding": true, "known_facts": ["30 days", "5-7 business days"]}` |
| Let routing learn which request types need a stronger model | `Gateway(quality_tracker=QualityTracker(...))`, tag requests with `metadata={"route_key": "billing"}`, and report outcomes with `gw.feedback(r.request_id, score)` |
| Send metrics to Datadog / Prometheus / OpenTelemetry | `gw.on("request.complete", fn)`, `gw.on("cache.hit", fn)`, `gw.on("fallback", fn)`, or `gw.on("*", fn)`; `gw.stats()` returns hit rate, spend, savings and circuit-breaker state |

### Gotchas that cost the most hit rate (or money)

- **Keep the system prompt stable.** The cache is scoped to the system prompt and earlier turns. If you paste user-specific or per-request data into the system prompt, nothing will ever repeat. Put variable data in the user message.
- **Set a realistic `max_tokens`.** A hard budget reserves the worst case (input estimate + `max_tokens`) before each call. A huge `max_tokens` makes requests look expensive and can block them early.
- **Time-sensitive and personal-data prompts are deliberately not cached** ("is it down right now?", messages with emails or phone numbers).
- **Reuse one `Gateway` object.** It is thread-safe; creating one per request throws away the cache.
- **Check the model names.** The default catalog uses `claude-haiku-5-5`, `claude-sonnet-5-5` and `claude-opus-5-5`; if your account uses other models, change the names and prices in your `"models"` list.

### Roll-out checklist

1. **Shadow it:** start with `use_cache=False` and the default routing; compare `r.cost_usd` and answers with your current setup.
2. **Turn on PII redaction** and per-customer `tenant_id`.
3. **Turn on caching** with a high threshold (0.97), strict presets first (`healthcare`, `legal`); lower it only after reviewing `cache.hit` events.
4. **Calibrate** the threshold on your own labeled queries with `calibrate_threshold(...)`.
5. **Add a verifier** and per-tenant **budgets** before widening the roll-out.
6. **Watch** `gw.stats()` / the event hooks: cache hit rate, fallbacks, quality escalations, spend per tenant.

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
- **Find what confuses the cache:** `confusion_report(labeled, embedder, config)` lists the label pairs and words behind wrong hits (e.g. "expect" vs "refund" = tracking vs requesting a refund), so you can add synonym groups with `extra_action_classes` (same-named groups merge). Action matching handles inflections ("getting", "creating", "cancelled"), adjacent-letter typos ("egt") and split phrases ("set my address up").
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
- **Grounding check (support, finance, healthcare, e-commerce presets):** an answer that states a duration, amount, date or reference number that appears nowhere in the conversation, system prompt or your `known_facts` (e.g. "30 days", "5-7 business days") is treated as invented and escalated. `GroundingValidator(known_facts=[...])`.
- **LLM judge:** `LLMJudgeValidator(judge_fn, min_score=0.6)` scores answers with any judge model; combine checks with `chain(default_validator, GroundingValidator(), judge)`.
- **Adaptive routing:** with `Gateway(quality_tracker=QualityTracker(min_score=0.8, min_samples=30))` the gateway keeps a quality score per route key (`metadata={"route_key": intent}`, else the task type) and tier, from validator outcomes and `gw.feedback(request_id, score)` (thumbs up/down, human review). When the cheap tier falls below the bar for a key, that key is routed one tier up; other keys keep the cheap tier.
- **Cascade on bad answers:** every answer passes a validator (default: empty, refusal-style or unfilled-template answers fail). A failed answer is billed but never cached, and the request escalates to the next tier up. If no tier passes, the last answer is returned with `quality_warning` set. Plug in your own check (an LLM judge, a schema check) with `Gateway(validator=fn)`, or pass `validator=None` to turn it off.
- **Refusals** (`stop_reason == "refusal"`) fail over to the next model. Bad requests are raised immediately, because another model wouldn't fix them.

### 6. Policy and observability
- `PolicyConfig(pii="allow|redact|block", blocked_terms=[...], max_input_tokens=..., phone_regions=["US", "GB"])`. Install the `pii` extra (Google's libphonenumber) for international phone numbers; `phone_regions` adds national formats such as UK `020 7946 0958`. Redaction runs before the model call and before the cache key is built.
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

On 26,872 real-style support messages (Bitext dataset), with settings derived from one half and results reported on the other half:
- **27% of requests served from cache with zero wrong answers** on FAQ-style support data (about 3% on open-ended real tweets). A guessed threshold would have served wrong answers to 1 in 16 cache hits.
- **No PII reached a model.**
- **0 failed requests** when the large model was killed mid-run.
- **Invented facts caught and escalated.**
- **Adaptive routing raised answer quality during the run.**
- **About 90% lower cost** than sending everything to Opus.

Full method, numbers and limits: [docs/BENCHMARK.md](docs/BENCHMARK.md). A second benchmark on two Kaggle datasets (Quora Question Pairs and Customer Support on Twitter), including where the system is weak and how a second-stage verifier cuts wrong cache hits by 83-90%, is in [docs/KAGGLE_BENCHMARK.md](docs/KAGGLE_BENCHMARK.md).

## Testing

```bash
pytest -q                                              # 184 tests, ~15 s, fully offline
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
- Hard budgets reserve each call's worst-case cost before it runs (input estimate × 1.25 + `max_tokens`) and settle the real cost afterwards. Enforcement is exact within one process. Across replicas a shared `SpendLedger` shares settled spend, but reservations stay per-process, so a limit can overshoot by the cost of in-flight requests; for a strict global limit run one shared service.
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
tests/              184 tests: unit, e2e, HTTP, property-based, concurrency, chaos, quality benchmarks
```

## License

MIT License — see [LICENSE](LICENSE).
