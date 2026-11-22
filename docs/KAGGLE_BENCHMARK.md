# Benchmark on two Kaggle datasets (free, no API key)

Reproduce (download the data from Kaggle yourself; it is not redistributed here):
```bash
kaggle datasets download -d quora/question-pairs-dataset
kaggle datasets download -d thoughtvector/customer-support-on-twitter
python examples/kaggle_benchmark.py --quora-csv questions.csv --twitter-csv twcs.csv \
    --models-dir ./models --part all
```

| Dataset | Used for | License |
|---|---|---|
| [Quora Question Pairs](https://www.kaggle.com/datasets/quora/question-pairs-dataset) (404,351 pairs labeled duplicate / not) | Exact cache precision and recall | "Other (specified in description)" |
| [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (2.8M tweets, 108 brands) | The whole system on real multi-tenant traffic; the brand's real reply is the reference answer | CC BY-NC-SA 4.0 |

Twitter numbers below are from the final code (symbol guard and tightened time-sensitive rule). Method: 12,000 random Quora pairs, thresholds chosen on the first 6,000, **all numbers from the other 6,000**. For Twitter, 30,000 customer messages with a real brand reply from the 12 busiest brands (a random sample over all 108 brands has almost no repeat questions per brand), replayed in time order, **each brand a separate tenant**. 5% of messages got synthetic emails/phone numbers appended. Embedders: nomic-embed-text-v1.5 (neural, local) and the built-in hashing embedder.

## 1. Semantic cache accuracy on Quora (6,000 held-out pairs)

Quora's non-duplicate pairs are deliberately confusable (same topic, different question), which makes this a **stress test**, harder than live traffic. A cache "serves" a pair if the embedding is similar enough and the guards agree; a *wrong serve* is a served pair that Quora labels not-duplicate. Neural embedder, best-F1 threshold (0.90):

| Guard layers (cumulative) | Right when served | Duplicates caught | Wrong serves / 1,000 pairs |
|---|---|---|---|
| none | 65.7% | 82.6% | 159.3 |
| + numbers and negation | 67.6% | 76.1% | 134.7 |
| + action words | 67.6% | 69.3% | 122.8 |
| + symbols (final) | 67.8% | 68.5% | **120.7** |

- Each layer removes wrong serves; together **24% fewer**, paid for with recall (82.6% → 68.5%).
- **Even so, about a third of served answers are for a different question by Quora's labels, and no threshold reaches 98% precision.** Similarity plus rules cannot guarantee a safe hit for confusable questions, which is what the second-stage verifier below addresses. Quora's labels are also noisy (some identical questions are labeled "not duplicate"), so absolute numbers are pessimistic.
- The built-in embedder is much weaker on paraphrases (best F1 0.49 vs 0.68).
- The pairwise measurement matches the real `SemanticCache.lookup()` path on 400 sampled pairs (400/400 agree).

### Second-stage verifier on Quora (6,000 held-out pairs, 2,223 semantic candidates)

A verifier asks each candidate hit a stricter question than similarity does: "is one answer correct for both questions?" (`SemanticCache(verifier=...)`). Here a small local model (Qwen2.5) reads the two questions and answers YES or NO, and the cache serves the hit only if P(YES) is above a cut-off. Candidates are pairs with similarity at least 0.90 that pass all guards; the cut-off is applied on top. Wrong serves are per 1,000 pairs.

| Setup | Right when served | Duplicates caught | Wrong serves / 1,000 pairs | Time per check |
|---|---|---|---|---|
| Guards only (no verifier) | 67.8% | 68.5% | 120.7 | n/a |
| + 1.5B judge, cut-off 0.5 | 87.5% | 39.8% | **21.0** (-83%) | 0.21 s |
| + 1.5B judge, both question orders must agree, cut-off 0.5 | **90.0%** | 29.5% | **12.2** (-90%) | 0.41 s (2 calls) |
| + 1.5B judge, both orders, cut-off 0.95 | 94.9% | 8.4% | 1.7 (-99%) | 0.41 s |
| + 3B judge, cut-off 0.5 | 86.9% | 46.0% | 25.7 (-79%) | 0.41 s |

- **The verifier removes most wrong cache hits.** The price is recall: the cache serves fewer hits (68.5% of duplicates caught drops to 30-46%). Choose the cut-off for your domain: lenient for FAQ-style traffic, strict for medical, legal or financial answers.
- **The judge ranks candidate pairs better than similarity does** (AUC 0.79-0.80 vs 0.69).
- **The larger judge is not better at precision.** The 3B model is more confident, so its scores pile up near 0 and 1: raising its cut-off from 0.5 to 0.95 barely changes precision (87% to 88%), and no cut-off reaches 90%. It catches more duplicates at about 87% precision (46% vs 40%) but is twice as slow.
- **Judges are order-sensitive.** The 1.5B judge's verdict changed with the order of the two questions for 561 of 2,223 candidates (25%). Requiring both orders to agree (`SymmetricVerifier`) gives the highest precision for the same model, at double the cost and fewer hits.
- **Calibration is imprecise with a small sample.** A cut-off chosen on 600 labeled train candidates to reach 90% precision gave 87.0% on the held-out half. Use more labeled data for the choice.
- **Precision stops around 90-95% against Quora's labels**, partly because those labels are noisy and the pairs are deliberately confusable. Live traffic has far fewer confusable pairs.
- The real `SemanticCache.lookup()` path with the verifier plugged in matches the benchmark's pairwise decisions on 120 of 120 pairs for each judge.
- **Cost (an estimate from list prices, not measured on Claude):** a check is about 130 input tokens and 1 output token, roughly $0.00001 per candidate on a Haiku-class model, far less than the answer call it avoids. Local models, as used here, cost nothing per call but need CPU.

**Bug found and fixed.** Text normalization stripped symbols, so "O- blood group" and "O+ blood group", "code in C" and "C++", "Nike+" and "Nike" produced the same exact-match key and one's cached answer could be served for the other. Exact keys and semantic hits now compare meaning-bearing symbols (`guard_symbols`); hyphenated words, bracket spacing and trailing "?" still match. Exact-only precision rose from 92.0% to 95.5% on the held-out pairs.

## 2. The whole system on real support traffic (30,000 messages, 12 brands as tenants)

| Check | Result |
|---|---|
| Tenant isolation | **0 cross-brand answers** with per-tenant caches. With a shared cache the same traffic would have served **100** cross-brand answers (neural, 0.94) and 33 (built-in): isolation is doing real work. |
| PII | **0 of 1,509** planted emails/phone numbers reached a model |
| Time-sensitive messages | 1,319 flagged (4.4%) by the tightened rule (the old keyword rule flagged 4,324, 14.4%); **0 served from cache** |
| Hard budgets | 8 brands x 150 concurrent requests, $0.0015 limit each: 354 served, 846 blocked, highest spend 97% of limit, no overspend, ledger equals billed, no leaked reservations |
| Routing (29,108 model calls at 0.90) | small 71%, medium 24.6%, large 4.4% |
| Shadow cost vs. always-Opus | about 81 to 82% lower, almost entirely from routing |

### Cache hit rate on real, open-ended support tweets

| Embedder | Threshold | Guards | Served from cache | Semantic hits whose real brand replies agree (above random same-brand baseline) | Weak agreement (< 0.6) |
|---|---|---|---|---|---|
| neural | 0.90 | all | **3.0%** | 76.0% | 0.6% |
| neural | 0.90 | none | 7.8% | 76.4% | 1.3% |
| neural | 0.94 | all | 0.85% | 76.0% | 1.7% |
| neural | 0.97 | all | 0.26% | 75.5% | 0% |
| built-in | 0.86 | all | 0.19% | 68.8% | 0% |

- **Caching helps far less here than on FAQ-style data** (27% on the Bitext set). Real tweets are individual and conversational, so safe repeats are rare. The value of a semantic cache depends on how repetitive your traffic is; measure that before counting on it.
- The guards reject 3,669 candidate hits (about 81% of them) at 0.90. That is the safety/hit-rate trade, and the real brand replies suggest it is not over-cautious: hits the guards allow agree with the brand's real replies at the same rate as unguarded hits, with fewer weak ones.
- "Agreement" compares the brands' real replies to the two matched messages (mean similarity 0.79 vs 0.72 for random same-brand pairs). Replies in this dataset are templated, so this is a **coarse proxy**, not proof that cached answers are right.
- **Time-sensitive detection was too broad, and has been tightened.** The old keyword rule flagged 14.4% of messages, 39% of them just for the word "now" ("I need help now"). A message is now flagged only for a strong cue (outage, "is down", weather, "latest") or a time word together with a request for something that changes (status, price, ETA, availability): 4.4% flagged. In a hand check of 20 still-flagged messages about 70% were genuinely time-varying; the rest are false alarms that only cost a cache hit. Messages that stopped being flagged were almost all complaints and narratives.

## 3. Live run with real local models (400 requests, 8 brands)

Ran before the symbol guard and the tightened time-sensitive rule; the cache path is not what this part measures. Qwen2.5 0.5B / 1.5B / 3B stand in for small / medium / large, priced at Claude list prices. **The cache share in this run is built into its design: 35% of traffic re-asks one of 40 hot questions, so it is not a natural hit rate** (see section 2 for that).

| Metric | Result |
|---|---|
| Failed requests | **0 / 400**; the large tier was killed at request 200, its breaker opened, the others absorbed its traffic |
| Answer quality, same 60 questions per tier (similarity to the brand's real reply) | small 0.695, medium 0.719, large 0.731 |
| Invented-fact rate per tier | 1.7%, 1.7%, 0.0% |
| Escalations | 9 refusal-style, 2 unsupported specifics (e.g. "24-48 hours"); 1 answer failed every tier and was returned flagged |
| Adaptive routing | moved only 1 brand up a tier (2 requests); quality first half 0.723 vs second half 0.717, **no improvement** |
| Shadow cost vs. always-Opus | 90.3% lower |

The tiers are close by this metric (0.695 to 0.731), so on this data there was little for adaptive routing to correct. That is consistent with the small gap, not evidence that routing works better than on Bitext.

## What this tells you

1. **Works as designed:** tenant isolation, PII redaction, time-sensitive exclusion, hard concurrent budgets, failover and grounding checks all behaved correctly on real traffic.
2. **Cache benefit is traffic-dependent:** about 28% of requests on FAQ-style support data, about 3% on open-ended social-media support.
3. **Similarity alone is not enough** for confusable questions, even with the guards. A second-stage verifier cut wrong cache hits by 83-90% on Quora at the cost of serving fewer hits; add one before caching in high-stakes domains.
4. **Real data found a real bug** (symbols collapsed in cache keys), now fixed with a regression test.

## Limits
- Reference similarity is an automatic proxy; the local 0.5B to 3B models (also used as verifier judges) are far weaker than Claude models, which were not run. A stronger judge would likely do better; that is untested.
- Twitter results use English-looking messages from the 12 busiest brands; Quora labels are noisy.
- Datasets are used under their Kaggle licenses and are not included in this repository.
