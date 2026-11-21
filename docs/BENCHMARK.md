# Benchmark: e-commerce customer support on real data (free)

Reproduce: `python examples/support_bot_benchmark.py --data <bitext.csv> --models-dir <models> --part all --live-n 200`

## Setup

| | |
|---|---|
| Data | [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (CDLA-Sharing-1.0): 26,872 messages, 27 intents, 11 categories. Only ~10% are exact repeats. |
| Realism | Placeholders filled with synthetic order numbers, names, amounts and cities; synthetic emails/phones planted in 5% of messages. |
| Method | Shuffled, split 50/50. Similarity thresholds were **chosen on the train half** with `calibrate_threshold(max_wrong_per_1000=0.5)`; **every number below is from the unseen test half** (13,436 messages). |
| Correctness | Each cache hit is checked against the intent labels: a hit that returns another intent's answer is a **wrong answer**. |
| Models (live run) | Free, local, CPU-only: Qwen2.5 0.5B / 1.5B / 3B Instruct (Q4_K_M) as small / medium / large tiers, via llama-cpp-python. |
| Cost | Calls were free; tiers are *shadow priced* at Claude Haiku 5.5 / Sonnet 5.5 / Opus 5.5 list prices. Baseline = every request to Opus, no cache. |

## Semantic cache: before vs. after (test half, 13,436 messages)

| Embedder | Configuration | Threshold | Served from cache | Wrong answers / 1,000 requests | Hit precision |
|---|---|---|---|---|---|
| Built-in (hashing) | before: number + negation guards | 0.90 | 9.0% | 0.00 | 100% |
| Built-in (hashing) | **after**: + action guard, calibrated | 0.80 | **23.2%** | 0.52 | 99.78% |
| nomic-embed-text-v1.5 | before: number + negation guards | 0.90 | 74.0% | **46.2** | 93.75% |
| nomic-embed-text-v1.5 | **after**: + action guard, calibrated | 0.98 | **26.3%** | **0.07** | **99.97%** |

Takeaways:
- A neural embedder with a guessed threshold looks great on hit rate (74%) but **1 in 16 cached answers was for a different request** (e.g. "check my invoice" answered with "send my invoice"). Hit rate alone is a misleading metric.
- Verification (action guard) + calibration cut wrong answers from 46.2 to 0.07 per 1,000 (**660× fewer**) while still serving 26% of traffic from cache, about **3× the built-in baseline**.
- Remaining errors are near-synonymous intents ("I expect a rebate" = *track* refund vs. "rebate 120 dollars" = *get* refund). Domain-specific action classes (`extra_action_classes`) are the lever for those.
- PII: 679 test messages carried synthetic emails/phones; **0 reached a model**.

## Live run with real local models (200 requests)

| Metric | Result |
|---|---|
| Failed requests | **0 / 200** |
| Served by | cache 53 · small 124 · medium 13 · large 10 |
| Cache-hit latency (p50) | **0.29 ms** |
| Model-call latency (p50) | 1.42 s (small 1.06 s · medium 1.89 s · large 3.93 s generation) |
| Large tier killed at request 100 | Circuit breaker opened; the 5 large-tier requests after it were rerouted; 0 errors |
| Quality cascade | 21 answers rejected (empty / refusal-style) and escalated to a higher tier; 5 failed every tier and were returned flagged, uncached |
| Shadow cost vs. always-Opus | $0.0248 vs $0.2554 (**90.3% lower**) |

## Limits of this benchmark
- **Answer quality was not scored.** The 0.5B–3B local models stand in for real tiers; their answers are often generic and sometimes invented (one claimed an order ETA it could not know). This run tests the gateway (caching, routing, budgets, failover), not the models. Judging quality needs a human review sample or an LLM judge.
- Savings are dominated by **routing**: ~97% of support messages route to the small tier, which is ~40× cheaper than Opus at list prices. That is only a saving if the small model's answers are good enough for your users, which is the thing to validate on a reviewed sample before relying on it.
- Shadow prices use list prices and, in the cache replay, an assumed 60-token reply; real bills depend on actual output lengths and any discounts.
- Single dataset, single domain, English only. International phone formats such as `+44 20 7946 0958` are not yet detected as PII.
