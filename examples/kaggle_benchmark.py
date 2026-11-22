"""Kaggle benchmark: the whole system on two real datasets (free, no API key).

  quora    Quora Question Pairs (kaggle: quora/question-pairs-dataset). 404K pairs labeled
           duplicate / not duplicate, so cache precision and recall are measured EXACTLY.
  twitter  Customer Support on Twitter (kaggle: thoughtvector/customer-support-on-twitter,
           CC BY-NC-SA 4.0). 2.8M real tweets from 108 brands. Each brand is a tenant; the
           brand's real reply is the reference answer. Tests tenant isolation, caching,
           routing, PII redaction, time-sensitive prompts and budgets on real traffic.
  live     Real local models (Qwen2.5 0.5B/1.5B/3B via llama-cpp-python) on Twitter traffic:
           answer quality per tier, grounding, adaptive routing, failover.

    python examples/kaggle_benchmark.py --quora-csv questions.csv --twitter-csv twcs.csv \\
        --models-dir ./models --part all

Data is never redistributed or committed; download it from Kaggle yourself.
Needs: numpy; for twitter/live also llama-cpp-python and the GGUF models used by
examples/support_bot_benchmark.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import support_bot_benchmark as sb  # noqa: E402  (reuses local models, embedders, quality probe)
from smartgate import (Budget, BudgetExceededError, CacheConfig, CallableEmbedder, CallableProvider,  # noqa: E402
                       Gateway, GatewayRequest, Message, PolicyConfig, SemanticCache)
from smartgate.analysis import estimate_tokens  # noqa: E402

SYSTEM = "You are the customer support team of a consumer brand, replying on Twitter. Reply helpfully in at most 3 sentences."
sb.SYSTEM = SYSTEM  # the shared live-run helpers read this module global


# ================================================================================ QUORA
GUARD_CONFIGS = {  # cumulative layers; "legacy" exact matching = text normalization only
    "no guards (legacy exact match)": dict(guard_numbers=False, guard_negation=False, guard_actions=False, guard_symbols=False),
    "+ numbers + negation": dict(guard_numbers=True, guard_negation=True, guard_actions=False, guard_symbols=False),
    "+ actions": dict(guard_numbers=True, guard_negation=True, guard_actions=True, guard_symbols=False),
    "+ symbols (final)": dict(guard_numbers=True, guard_negation=True, guard_actions=True, guard_symbols=True),
}
GRID = np.round(np.arange(0.60, 1.0001, 0.01), 2)


def _metrics(served, y):
    tp = int((served & (y == 1)).sum()); fp = int((served & (y == 0)).sum()); fn = int((~served & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "served": tp + fp, "wrong_serves": fp, "wrong_per_1000_pairs": round(1000 * fp / len(y), 2)}


def quora_part(path, n_pairs, embedders, target_precision):
    rnd = random.Random(5)
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["question1"] and r["question2"]]
    rnd.shuffle(rows)
    rows = rows[:n_pairs]
    half = len(rows) // 2
    y = np.array([int(r["is_duplicate"]) for r in rows])
    train, test = slice(0, half), slice(half, None)
    from smartgate.embeddings import normalize_text
    exact_legacy = np.array([normalize_text(r["question1"]) == normalize_text(r["question2"]) for r in rows])
    exact_new = np.array([SemanticCache.exact_key(r["question1"]) == SemanticCache.exact_key(r["question2"]) for r in rows])
    out_exact = {"pairs_identical_after_normalization": int(exact_legacy.sum()),
                 "of_which_labeled_duplicate": int((exact_legacy & (y == 1)).sum()),
                 "now_kept_apart_by_symbol_check": int((exact_legacy & ~exact_new).sum()),
                 "of_those_labeled_not_duplicate": int((exact_legacy & ~exact_new & (y == 0)).sum())}
    out = {"pairs": len(rows), "duplicate_share": round(float(y.mean()), 3), "target_precision_on_train": target_precision,
           "rows": [], "lookup_path_selfcheck": None, "exact_match_check": out_exact}
    chosen = {}
    for emb_name, emb in embedders.items():
        t0 = time.perf_counter()
        v1 = np.array([emb.embed(r["question1"]) for r in rows], dtype=np.float32)
        v2 = np.array([emb.embed(r["question2"]) for r in rows], dtype=np.float32)
        sims = (v1 * v2).sum(1)
        print(f"[quora] embedded {len(rows)*2} questions with {emb_name} in {time.perf_counter()-t0:.0f}s", flush=True)
        for g_name, g in GUARD_CONFIGS.items():
            cache = SemanticCache(CacheConfig(**g), embedder=emb)
            ok = np.array([cache.signature(r["question1"]) == cache.signature(r["question2"]) for r in rows])

            exact = exact_new if g["guard_symbols"] else exact_legacy

            def serve(th, exact=exact, ok=ok):
                return exact | ((sims >= th) & ok)
            cal = []
            for th in GRID:
                m = _metrics(serve(th)[train], y[train]); cal.append((float(th), m))
            feasible = [(th, m) for th, m in cal if m["precision"] >= target_precision and m["served"] >= 20]
            th_prec = min(feasible, key=lambda x: x[0])[0] if feasible else float(GRID[-1])
            th_f1 = max(cal, key=lambda x: x[1]["f1"])[0]
            row = {"embedder": emb_name, "guards": g_name,
                   "at_precision_target": {"threshold": th_prec, "met_on_train": bool(feasible),
                                           **_metrics(serve(th_prec)[test], y[test])},
                   "at_best_f1": {"threshold": th_f1, **_metrics(serve(th_f1)[test], y[test])},
                   "guard_blocked_true_duplicates_test": int((~ok[test] & (y[test] == 1)).sum()),
                   "guard_blocked_false_duplicates_test": int((~ok[test] & (y[test] == 0)).sum())}
            out["rows"].append(row)
            print(json.dumps({k: row[k] for k in ("embedder", "guards", "at_precision_target")}), flush=True)
            if g_name.startswith("+ symbols"):
                chosen[emb_name] = (th_f1, cache, serve(th_f1))
    # the pairwise numbers above must equal what the real SemanticCache.lookup() path does
    emb_name = list(embedders)[0]
    th, cache, decisions = chosen[emb_name]  # checked at the best-F1 threshold (a realistic one)
    cache = SemanticCache(CacheConfig(similarity_threshold=th, ttl_seconds=None), embedder=embedders[emb_name])
    idx = rnd.sample(range(half, len(rows)), min(400, len(rows) - half))
    agree = 0
    for i in idx:
        cache.store.clear()
        cache.store_response(GatewayRequest(messages=[Message("user", rows[i]["question1"])]), "A", "m")
        hit = cache.lookup(GatewayRequest(messages=[Message("user", rows[i]["question2"])]))
        agree += bool(hit) == bool(decisions[i])
    out["lookup_path_selfcheck"] = {"embedder": emb_name, "threshold": th, "pairs": len(idx), "agree": agree}
    print(json.dumps(out["lookup_path_selfcheck"]), flush=True)
    return out


# ================================================================================ VERIFY
class LocalJudge:
    """YES/NO verifier from a local llama.cpp model: P(YES) read from the next-token logits.

    The fixed instructions come first so llama.cpp reuses their cached state; only the two
    questions are evaluated per call (about 0.2 s with a 1.5B model on 4 CPU cores).
    """

    def __init__(self, path):
        import math
        from llama_cpp import Llama
        from smartgate.cache.verify import VERIFY_SYSTEM
        self.math, self.system = math, VERIFY_SYSTEM
        self.llm = Llama(model_path=path, n_ctx=1024, n_threads=os.cpu_count(), verbose=False)
        self.yes = self.llm.tokenize(b"YES", add_bos=False, special=False)[0]
        self.no = self.llm.tokenize(b"NO", add_bos=False, special=False)[0]
        self.n_vocab, self.prev, self.calls, self.seconds = self.llm.n_vocab(), [], 0, 0.0

    def __call__(self, new_query, cached_query):
        from smartgate.cache.verify import pair_prompt
        t0 = time.perf_counter()
        text = (f"<|im_start|>system\n{self.system}<|im_end|>\n<|im_start|>user\n"
                f"{pair_prompt(new_query, cached_query)}<|im_end|>\n<|im_start|>assistant\n")
        toks = self.llm.tokenize(text.encode(), add_bos=False, special=True)
        n = 0
        while n < min(len(toks), len(self.prev)) and toks[n] == self.prev[n]:
            n += 1
        n = min(n, len(toks) - 1)
        self.llm._ctx.kv_cache_seq_rm(-1, n, -1)
        self.llm.n_tokens = n
        self.llm.eval(toks[n:])
        self.prev = toks
        lg = np.ctypeslib.as_array(self.llm._ctx.get_logits(), shape=(self.n_vocab,))
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return 1 / (1 + self.math.exp(float(lg[self.no]) - float(lg[self.yes])))


def _auc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def verify_part(path, n_pairs, nomic, models_dir, judge_files, retrieval_th, cal_n, work_dir,
                symmetric_for=(), targets=(0.90, 0.95)):
    from smartgate.embeddings import normalize_text  # noqa: F401
    rnd = random.Random(5)  # same sample as quora_part
    rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["question1"] and r["question2"]]
    rnd.shuffle(rows)
    rows = rows[:n_pairs]
    half = len(rows) // 2
    y = np.array([int(r["is_duplicate"]) for r in rows])
    emb = CallableEmbedder(nomic.embed, 768)
    v1 = np.array([emb.embed(r["question1"]) for r in rows], dtype=np.float32)
    v2 = np.array([emb.embed(r["question2"]) for r in rows], dtype=np.float32)
    sims = (v1 * v2).sum(1)
    final = CacheConfig(similarity_threshold=retrieval_th, **GUARD_CONFIGS["+ symbols (final)"])
    cache = SemanticCache(final, embedder=emb)
    ok = np.array([cache.signature(r["question1"]) == cache.signature(r["question2"]) for r in rows])
    exact = np.array([SemanticCache.exact_key(r["question1"]) == SemanticCache.exact_key(r["question2"]) for r in rows])
    cand = (sims >= retrieval_th) & ok & ~exact          # semantic candidates the verifier would see
    idx_all = np.arange(len(rows))
    train_idx = idx_all[:half][cand[:half]]
    test_idx = idx_all[half:][cand[half:]]
    cal_idx = np.array(sorted(random.Random(3).sample(list(train_idx), min(cal_n, len(train_idx)))))
    test = slice(half, None)
    out = {"pairs": len(rows), "retrieval_threshold": retrieval_th, "test_candidates": int(len(test_idx)),
           "calibration_sample": int(len(cal_idx)), "models": {}}

    def metrics(served):
        return _metrics(served[test], y[test])

    base_served = exact | cand
    out["no_verifier"] = metrics(base_served)
    out["test_true_duplicates"] = int(y[test].sum())
    print("[verify] no verifier:", json.dumps(out["no_verifier"]), flush=True)
    def evaluate(s_arr):
        sweep = [{"verify_min_score": vm, **metrics(exact | (cand & (s_arr >= vm)))}
                 for vm in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]]
        cal_s, cal_y = s_arr[cal_idx], y[cal_idx]
        calibrated = {}
        for tgt in targets:  # lowest cut-off whose precision on the train sample meets the target
            pick = next((float(vm) for vm in np.round(np.arange(0.02, 1.0, 0.02), 2)
                         if (cal_s >= vm).sum() >= 30 and cal_y[cal_s >= vm].mean() >= tgt), None)
            calibrated[str(tgt)] = ({"met_on_train": False} if pick is None else
                                    {"met_on_train": True, "verify_min_score": pick,
                                     **metrics(exact | (cand & (s_arr >= pick)))})
        return {"sweep": sweep, "calibrated_on_train_sample": calibrated,
                "auc_on_candidates": round(_auc(s_arr[test_idx], y[test_idx]), 4)}

    def load(fn):
        return {int(k): v for k, v in json.load(open(fn)).items()} if os.path.exists(fn) else {}

    def fill(judge, scores, fn, order):
        todo = [int(i) for i in list(cal_idx) + list(test_idx) if int(i) not in scores]
        for n, i in enumerate(todo, 1):
            q1, q2 = rows[i]["question1"], rows[i]["question2"]
            scores[i] = judge(q2, q1) if order == "prod" else judge(q1, q2)   # prod = (new=q2, cached=q1)
            if n % 300 == 0:
                json.dump(scores, open(fn, "w")); print(f"[verify] {os.path.basename(fn)}: {n}/{len(todo)}", flush=True)
        json.dump(scores, open(fn, "w"))
        arr = np.full(len(rows), -1.0)
        for i, v in scores.items():
            arr[i] = v
        return arr

    for jf in judge_files:
        name = jf.split("-instruct")[0]
        judge = LocalJudge(os.path.join(models_dir, jf))
        prod_fn = os.path.join(work_dir, f"verify_scores_{name}.json")
        prod = fill(judge, load(prod_fn), prod_fn, "prod")
        res = {"production_order": evaluate(prod),
               "similarity_auc_on_candidates": round(_auc(sims[test_idx], y[test_idx]), 4)}
        if name in symmetric_for:
            rev_fn = os.path.join(work_dir, f"verify_scores_{name}_rev.json")
            rev = fill(judge, load(rev_fn), rev_fn, "rev")
            res["symmetric_min_of_both_orders"] = evaluate(np.minimum(prod, rev))
            res["verdict_flips_between_orders_at_0.5"] = int(((prod[test_idx] >= .5) != (rev[test_idx] >= .5)).sum())
        res["seconds_per_verification"] = round(judge.seconds / max(judge.calls, 1), 3)
        # the real SemanticCache.lookup() path with the verifier plugged in (production order)
        chk = SemanticCache(CacheConfig(similarity_threshold=retrieval_th, ttl_seconds=None, verify_min_score=0.5,
                                        **GUARD_CONFIGS["+ symbols (final)"]), embedder=emb, verifier=judge)
        sample = random.Random(8).sample([int(i) for i in test_idx], min(120, len(test_idx)))
        agree = 0
        for i in sample:
            chk.store.clear()
            chk.store_response(GatewayRequest(messages=[Message("user", rows[i]["question1"])]), "A", "m")
            hit = chk.lookup(GatewayRequest(messages=[Message("user", rows[i]["question2"])]))
            agree += bool(hit) == bool(prod[i] >= 0.5)
        res["lookup_path_selfcheck"] = {"pairs": len(sample), "agree": agree}
        out["models"][name] = res
        print(f"[verify] {name}: AUC verifier {res['production_order']['auc_on_candidates']} vs similarity "
              f"{res['similarity_auc_on_candidates']}; selfcheck {agree}/{len(sample)}", flush=True)
        del judge
    return out


# =============================================================================== TWITTER
_URL = re.compile(r"https?://\S+")
_AT = re.compile(r"@\w+")
PII_LEAK = re.compile(r"@example\.com|555-\d{3}-\d{4}|\+44 20 7946 0958")


def _clean(t):
    return re.sub(r"\s+", " ", _AT.sub("", _URL.sub("", t))).strip()


def _english(t):
    return len(t) >= 15 and sum(ord(c) < 128 for c in t) / len(t) >= 0.9


def load_twitter(path, n_pairs, cache_file, top_brands=12):
    if os.path.exists(cache_file):
        return json.load(open(cache_file))
    t0 = time.perf_counter()
    tw = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tw[r["tweet_id"]] = (r["author_id"], r["inbound"] == "True", r["text"], r["created_at"],
                                 r["response_tweet_id"], r["in_response_to_tweet_id"])
    cands = []
    for tid, (author, inbound, text, created, resp, in_resp) in tw.items():
        if not inbound or in_resp or not resp:
            continue
        for rid in resp.split(","):
            y = tw.get(rid)
            if y and not y[1]:
                cands.append((tid, y[0], text, y[2], created))
                break
    # Dense traffic: a random sample over all 108 brands has almost no repeat questions per
    # brand. Keep the busiest brands (still a multi-tenant test) so repeats actually occur.
    busiest = {b for b, _ in Counter(c[1] for c in cands).most_common(top_brands)}
    cands = [c for c in cands if c[1] in busiest]
    rnd = random.Random(5)
    rnd.shuffle(cands)
    pairs = []
    for tid, brand, text, reply, created in cands:
        text, reply = _clean(text), _clean(reply)
        if _english(text) and _english(reply):
            pairs.append({"tid": tid, "brand": brand, "text": text, "reference": reply,
                          "ts": datetime.strptime(created, "%a %b %d %H:%M:%S %z %Y").timestamp()})
            if len(pairs) == n_pairs:
                break
    pairs.sort(key=lambda p: p["ts"])  # chronological: repeat bursts appear when outages happen
    prng = random.Random(9)
    for p in pairs:  # plant synthetic PII in 5% of messages
        p["pii"] = None
        if prng.random() < 0.05:
            p["pii"] = prng.choice([f"my email is cust{prng.randint(1, 9999)}@example.com",
                                    f"call me on 555-{prng.randint(100, 999)}-{prng.randint(1000, 9999)}",
                                    "ring +44 20 7946 0958"])
            p["text"] = f"{p['text']} {p['pii']}"
    json.dump(pairs, open(cache_file, "w"))
    print(f"[twitter] loaded {len(tw):,} tweets, {len(cands):,} customer->brand pairs, sampled {len(pairs):,} "
          f"in {time.perf_counter()-t0:.0f}s (top {top_brands} brands kept)", flush=True)
    return pairs


class _ReplyVectors:
    def __init__(self, embedder, pairs):
        self.emb, self.by_tid, self.memo = embedder, {p["tid"]: p for p in pairs}, {}

    def __call__(self, tid):
        if tid not in self.memo:
            self.memo[tid] = np.array(self.emb.embed(self.by_tid[tid]["reference"]), dtype=np.float32)
        return self.memo[tid]


def twitter_replay(pairs, embedder, label, threshold, guards, replies, baseline_sim):
    leaked = []

    def provider(model, request):
        msg = request.last_user_message
        if PII_LEAK.search(msg):
            leaked.append(msg)
        return f"ORIGIN::{request.metadata['tid']}::{request.tenant_id}"

    gw = Gateway(models=sb.shadow_models("standin"), providers={"standin": CallableProvider("standin", provider)},
                 cache=SemanticCache(CacheConfig(similarity_threshold=threshold, ttl_seconds=None,
                                                 max_entries=200_000, **guards), embedder=embedder),
                 policy=PolicyConfig(pii="redact"), validator=None)
    n = len(pairs)
    hits = exact_hits = tenant_violations = ts_total = ts_from_cache = 0
    sem_sims, tiers, tier_examples = [], Counter(), defaultdict(list)
    per_brand = defaultdict(lambda: [0, 0])
    cost = baseline = 0.0
    t0 = time.perf_counter()
    for p in pairs:
        r = gw.generate(p["text"], system=SYSTEM, tenant_id=p["brand"], max_tokens=150, metadata={"tid": p["tid"]})
        a = r.analysis or {}
        ts_total += bool(a.get("time_sensitive"))
        per_brand[p["brand"]][0] += 1
        in_tok = estimate_tokens(SYSTEM + p["text"])
        baseline += (in_tok * sb.OPUS_IN + 60 * sb.OPUS_OUT) / 1e6
        if r.cache_hit:
            hits += 1
            per_brand[p["brand"]][1] += 1
            ts_from_cache += bool(a.get("time_sensitive"))
            _, origin, origin_brand = r.text.split("::")
            tenant_violations += origin_brand != p["brand"]
            if r.cache_similarity == 1.0:
                exact_hits += 1
            elif origin != p["tid"]:
                sem_sims.append(float(replies(p["tid"]) @ replies(origin)))
        else:
            spec = gw.router.get(r.model)
            cost += spec.cost(in_tok, 60)
            tiers[spec.tier] += 1
            if len(tier_examples[spec.tier]) < 3:
                tier_examples[spec.tier].append(p["text"][:140])
    sem = np.array(sem_sims) if sem_sims else np.array([0.0])
    top = sorted(per_brand.items(), key=lambda kv: -kv[1][0])[:8]
    return {
        "embedder": label, "threshold": threshold, "guards": "all" if guards.get("guard_actions", True) else "none",
        "scope": guards.get("scope", "tenant"),
        "requests": n, "brands": len(per_brand), "cache_hit_rate": round(hits / n, 4),
        "exact_hits": exact_hits, "semantic_hits": len(sem_sims),
        "tenant_isolation_violations": tenant_violations,
        "semantic_hit_reply_similarity_mean": round(float(sem.mean()), 4),
        "random_same_brand_baseline": round(baseline_sim, 4),
        "semantic_hits_with_reply_agreement_above_baseline": round(float((sem >= baseline_sim).mean()), 4),
        "semantic_hits_with_weak_agreement_below_0.6": round(float((sem < 0.6).mean()), 4),
        "calls_by_tier": dict(tiers), "tier_examples": {k: v for k, v in tier_examples.items()},
        "pii_messages": sum(1 for p in pairs if p["pii"]), "pii_leaked_to_model": len(leaked),
        "time_sensitive_messages": ts_total, "time_sensitive_served_from_cache": ts_from_cache,
        "guard_rejections": gw.cache.stats["guard_rejections"],
        "shadow_cost_usd": round(cost, 4), "baseline_all_opus_no_cache_usd": round(baseline, 4),
        "savings_pct": round(100 * (1 - cost / baseline), 1),
        "top_brand_hit_rates": {b: round(v[1] / v[0], 3) for b, v in top},
        "seconds": round(time.perf_counter() - t0, 1),
    }


def twitter_budget_test(pairs, brands=8, per_brand=150, threads=16):
    """Hard per-brand budgets on real traffic with concurrent clients: never overspent."""
    top = [b for b, _ in Counter(p["brand"] for p in pairs).most_common(brands)]
    work = [(p["brand"], p["text"]) for b in top for p in [q for q in pairs if q["brand"] == b][:per_brand]]
    random.Random(2).shuffle(work)

    def provider(model, request):
        return "x" * 240  # ~60-token reply

    gw = Gateway(models=sb.shadow_models("standin"), providers={"standin": CallableProvider("standin", provider)},
                 policy=PolicyConfig(pii="redact"), validator=None)
    limit = 0.0015
    for b in top:
        gw.set_budget(b, Budget(limit_usd=limit, period="total", downgrade_at=2.0))
    ok, blocked, billed = Counter(), Counter(), defaultdict(float)
    lock = threading.Lock()

    def call(item):
        brand, text = item
        try:
            r = gw.generate(text, tenant_id=brand, max_tokens=150, use_cache=False)
            with lock:
                ok[brand] += 1; billed[brand] += r.cost_usd
        except BudgetExceededError:
            with lock:
                blocked[brand] += 1

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(call, work))
    return {"brands": len(top), "requests": len(work), "limit_usd_per_brand": limit,
            "served": sum(ok.values()), "blocked_by_budget": sum(blocked.values()),
            "max_spend_vs_limit": round(max(gw.costs.spent(b) for b in top) / limit, 4),
            "any_brand_overspent": any(gw.costs.spent(b) > limit + 1e-12 for b in top),
            "ledger_equals_billed": all(abs(gw.costs.spent(b) - billed[b]) < 1e-9 for b in top),
            "leaked_reservations": any(gw.costs.reserved(b) > 1e-12 for b in top)}


def twitter_part(pairs, embedders, nomic, ref_embedder):
    replies = _ReplyVectors(ref_embedder, pairs)
    rnd = random.Random(4)
    by_brand = defaultdict(list)
    for p in pairs:
        by_brand[p["brand"]].append(p["tid"])
    sims = []
    brands = [b for b, v in by_brand.items() if len(v) >= 2]
    for _ in range(3000):
        a, b = rnd.sample(by_brand[rnd.choice(brands)], 2)
        sims.append(float(replies(a) @ replies(b)))
    baseline_sim = statistics.mean(sims)
    print(f"[twitter] random same-brand reply similarity baseline: {baseline_sim:.3f}", flush=True)
    plan = [("nomic-embed-text-v1.5", t, dict()) for t in (0.90, 0.94, 0.97, 0.98)] if "nomic-embed-text-v1.5" in embedders else []
    plan += [("nomic-embed-text-v1.5", 0.90, GUARD_CONFIGS["no guards (legacy exact match)"])] if plan else []
    plan += [("hashing (built-in)", t, dict()) for t in (0.80, 0.86, 0.92)]
    plan += [("hashing (built-in)", 0.86, GUARD_CONFIGS["no guards (legacy exact match)"])]
    # shared (global) cache: counts the cross-brand answers that tenant scoping prevents
    plan += [("hashing (built-in)", 0.86, {"scope": "global"})]
    plan += [("nomic-embed-text-v1.5", 0.94, {"scope": "global"})] if "nomic-embed-text-v1.5" in embedders else []
    runs = []
    for name, th, guards in plan:
        runs.append(twitter_replay(pairs, embedders[name], name, th, guards, replies, baseline_sim))
        print(json.dumps({k: runs[-1][k] for k in ("embedder", "threshold", "guards", "scope", "cache_hit_rate",
                                                     "tenant_isolation_violations", "semantic_hits_with_reply_agreement_above_baseline",
                                                     "pii_leaked_to_model", "savings_pct")}), flush=True)
    return {"requests": len(pairs), "reply_similarity_random_same_brand_baseline": round(baseline_sim, 4),
            "replay": runs, "budget_test": twitter_budget_test(pairs)}


def live_part(pairs, models_dir, nomic, threshold, n_live, n_probe):
    top = [b for b, _ in Counter(p["brand"] for p in pairs).most_common(8)]
    data = [{"text": p["text"], "reference": p["reference"], "intent": p["brand"]} for p in pairs if p["brand"] in top]
    random.Random(1).shuffle(data)
    emb = CallableEmbedder(nomic.embed, 768)
    return sb.run_quality_live(data, models_dir, nomic, emb, threshold, n_probe=n_probe, n_live=n_live,
                               kill_at=n_live // 2)


# ================================================================================= MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quora-csv")
    ap.add_argument("--twitter-csv")
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--part", choices=["quora", "twitter", "live", "verify", "all"], default="all")
    ap.add_argument("--verify-threshold", type=float, default=0.90)
    ap.add_argument("--verify-cal-n", type=int, default=600)
    ap.add_argument("--symmetric-for", default="qwen2.5-1.5b", help="judges also scored in both orders")
    ap.add_argument("--judges", default="qwen2.5-1.5b-instruct-q4_k_m.gguf,qwen2.5-3b-instruct-q4_k_m.gguf")
    ap.add_argument("--n-quora", type=int, default=12000)
    ap.add_argument("--n-twitter", type=int, default=30000)
    ap.add_argument("--live-n", type=int, default=400)
    ap.add_argument("--probe-n", type=int, default=60)
    ap.add_argument("--live-threshold", type=float, default=0.97)
    ap.add_argument("--target-precision", type=float, default=0.98)
    ap.add_argument("--work-dir", default=".")
    ap.add_argument("--out", default="kaggle_results.json")
    args = ap.parse_args()

    nomic_path = os.path.join(args.models_dir, "nomic-embed-text-v1.5.Q8_0.gguf")
    results = {}
    out_path = os.path.join(args.work_dir, args.out)
    if os.path.exists(out_path):
        results = json.load(open(out_path))

    def save():
        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    if args.part in ("quora", "all"):
        nomic = sb.NomicEmbedder(nomic_path, memo_path=os.path.join(args.work_dir, "nomic_quora.pkl"))
        embedders = {"nomic-embed-text-v1.5": CallableEmbedder(nomic.embed, 768), "hashing (built-in)": sb.MemoHashing()}
        results["quora"] = quora_part(args.quora_csv, args.n_quora, embedders, args.target_precision)
        nomic.save(); save()

    if args.part == "verify":
        nomic = sb.NomicEmbedder(nomic_path, memo_path=os.path.join(args.work_dir, "nomic_quora.pkl"))
        results["verify"] = verify_part(args.quora_csv, args.n_quora, nomic, args.models_dir,
                                        args.judges.split(","), args.verify_threshold, args.verify_cal_n, args.work_dir,
                                        symmetric_for=args.symmetric_for.split(","))
        save()

    if args.part in ("twitter", "live", "all"):
        pairs = load_twitter(args.twitter_csv, args.n_twitter, os.path.join(args.work_dir, "twitter_pairs_top12.json"))
        nomic = sb.NomicEmbedder(nomic_path, memo_path=os.path.join(args.work_dir, "nomic_twitter.pkl"))
        nomic_cb = CallableEmbedder(nomic.embed, 768)
        if args.part in ("twitter", "all"):
            embedders = {"nomic-embed-text-v1.5": nomic_cb, "hashing (built-in)": sb.MemoHashing()}
            results["twitter"] = twitter_part(pairs, embedders, nomic, nomic_cb)
            nomic.save(); save()
        if args.part in ("live", "all"):
            results["live"] = live_part(pairs, args.models_dir, nomic, args.live_threshold, args.live_n, args.probe_n)
            nomic.save(); save()
            print(json.dumps(results["live"], indent=1, ensure_ascii=False)[:6000])
    save()
    print("wrote", out_path)


if __name__ == "__main__":
    main()
