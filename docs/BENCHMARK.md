# Benchmark: e-commerce customer support on real data (free)

Reproduce:
```bash
python examples/support_bot_benchmark.py --data <bitext.csv> --models-dir <models> --part all
```

## Setup

| | |
|---|---|
| Data | [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (CDLA-Sharing-1.0): 26,872 messages, 27 intents, 11 categories, each with a human-written reference reply. Only ~10% of messages are exact repeats. |
| Realism | Placeholders filled with synthetic order numbers, names, amounts and cities; synthetic emails/phones planted in 5% of messages. |
| Method | Shuffled, split 50/50. Cache thresholds and synonym groups were **derived from the train half only**; **every number below is from the unseen test half**. |
| Cache correctness | A cache hit that returns another intent's answer is a **wrong answer**. |
| Answer quality | Semantic similarity (nomic-embed-text-v1.5) between each model answer and Bitext's human-written reference reply, plus the grounding check for invented facts. |
| Models | Free, local, CPU-only: Qwen2.5 0.5B / 1.5B / 3B Instruct (Q4_K_M) as small / medium / large tiers via llama-cpp-python. |
| Cost | Calls were free; tiers are *shadow priced* at Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5 list prices. Baseline = every request to Opus, no cache. |

## 1. Semantic cache (test half, 13,436 messages)

| Embedder | Configuration | Threshold | Served from cache | Wrong answers / 1,000 | Hit precision |
|---|---|---|---|---|---|
| nomic-embed-text-v1.5 | v1: number + negation guards, guessed threshold | 0.90 | 74.0% | **46.2** | 93.75% |
| nomic-embed-text-v1.5 | v2: + action guard, calibrated | 0.98 | 26.3% | 0.07 | 99.97% |
| nomic-embed-text-v1.5 | **v3: + inflections, typos, split phrases, train-derived synonyms** | 0.98 | **27.0%** | **0.00** | **100%** |
| Built-in (hashing) | v1 | 0.90 | 9.0% | 0.00 | 100% |
| Built-in (hashing) | v3 | 0.80 | 23.2% | 0.22 | 99.90% |

- With a guessed threshold, a neural embedder serves 74% of traffic but **1 in 16 cached answers is for a different request**. Hit rate alone is misleading.
- Verification and calibration removed every wrong answer on the held-out half while still serving 27% of requests from cache.
- PII: 679 test messages carried synthetic emails or phone numbers; **0 reached a model**.

## 2. Is the small tier good enough? (the same 60 test questions on every tier)

| Tier | Reference similarity (mean) | Answers with invented facts | p50 latency |
|---|---|---|---|
| small (0.5B) | 0.832 | 3.3% | 1.1 s |
| medium (1.5B) | 0.848 | 0.0% | 2.1 s |
| large (3B) | 0.870 | **6.7%** | 3.2 s |

The small tier is measurably below the large one, but not uniformly. Adaptive routing (section 3) decides per intent instead of all or nothing. The largest model invented the most facts, so a bigger model does not remove the need for the grounding check.

## 3. Live run: grounding + adaptive routing + failover (400 requests)

A simulated reviewer gave each answer thumbs-up when its reference similarity reached the large tier's median (minus 0.02), via `gw.feedback()`. The large tier was killed at request 200.

| Metric | Result |
|---|---|
| Failed requests | **0 / 400** |
| Served by | cache 120 · small 207 · medium 63 · large 10 |
| Cache-hit latency (p50) | 0.53 ms |
| Invented facts caught and escalated | 8 (e.g. "2-3 business days", "24 hours" for questions the bot cannot know) |
| Refusals escalated | 25 (empty "I'm unable to assist" answers; helpful "I can't see your order, but you can…" answers now pass) |
| Answers no tier could fix (returned flagged, uncached) | 3 |
| Intents moved up a tier by adaptive routing | 8: cancel_order, change_order, create_account, delivery_period, get_invoice, place_order, switch_account, track_order |
| Answer quality, first vs second half of the run | 0.856 → **0.866** (routing learns which intents need a stronger tier) |
| After the large tier was killed | Circuit breaker opened; medium and small absorbed its traffic with no errors |
| Shadow cost vs always-Opus | $0.050 vs $0.504 (**90% lower**) |

### Effect of the refusal fix (same run, before → after)

| | Before | After |
|---|---|---|
| Refusal escalations | 64 | 25 |
| Answers no tier could fix | 20 | 3 |
| Quality gain over the run | +0.002 | +0.010 |
| Savings vs always-Opus | 86% | 90% |

## Limits
- **Reference similarity is an automatic proxy.** It rewards answers that resemble the human reply. It is no substitute for a human review sample or a strong LLM judge (`LLMJudgeValidator`) before production.
- **The local 0.5B–3B models are much weaker than Claude.** These results show the gateway's mechanisms working; absolute answer quality on Claude models was not measured.
- Shadow prices use list prices; cache-replay costs assume a 60-token reply.
- One dataset, one domain, English only.
