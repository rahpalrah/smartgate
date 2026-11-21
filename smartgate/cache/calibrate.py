"""Threshold calibration on labeled traffic.

Give it queries labeled with what they mean (an intent, a ticket category, an FAQ id). It
replays them in order through a fresh cache per threshold: the first query of each meaning is
stored, later ones may hit. A hit whose label differs from the query's label is a wrong answer.
It returns the threshold with the most hits whose wrong-answer rate stays under your budget.

Use a train split to choose the threshold and report on a held-out split.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..embeddings import Embedder
from ..types import GatewayRequest, Message
from .semantic import CacheConfig, SemanticCache

DEFAULT_THRESHOLDS = (0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)


def replay(labeled: Sequence[Tuple[str, str]], embedder: Embedder, config: CacheConfig) -> Dict:
    cache = SemanticCache(replace(config, ttl_seconds=None, max_entries=max(config.max_entries, len(labeled) + 1)),
                          embedder=embedder)
    hits = wrong = 0
    for text, label in labeled:
        req = GatewayRequest(messages=[Message("user", text)])
        hit = cache.lookup(req)
        if hit:
            hits += 1
            wrong += hit.entry.response != label
        else:
            cache.store_response(req, label, "label")
    n = len(labeled)
    return {"threshold": config.similarity_threshold, "requests": n, "hit_rate": hits / n if n else 0.0,
            "wrong_per_1000": 1000 * wrong / n if n else 0.0,
            "precision": (1 - wrong / hits) if hits else 1.0, "hits": hits, "wrong": wrong,
            "guard_rejections": cache.stats["guard_rejections"]}


def calibrate_threshold(labeled: Sequence[Tuple[str, str]], embedder: Embedder,
                        max_wrong_per_1000: float = 0.5, base: Optional[CacheConfig] = None,
                        thresholds: Iterable[float] = DEFAULT_THRESHOLDS) -> Dict:
    """Returns {"threshold": best, "table": [replay results per threshold]}."""
    base = base or CacheConfig()
    table: List[Dict] = [replay(labeled, embedder, replace(base, similarity_threshold=t)) for t in thresholds]
    ok = [r for r in table if r["wrong_per_1000"] <= max_wrong_per_1000]
    best = max(ok, key=lambda r: r["hit_rate"]) if ok else max(table, key=lambda r: r["threshold"])
    return {"threshold": best["threshold"], "meets_budget": bool(ok), "table": table}
