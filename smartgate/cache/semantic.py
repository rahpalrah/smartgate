"""Semantic cache: exact-match fast path, then embedding similarity search."""
from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

import re

from ..embeddings import Embedder, HashingEmbedder, normalize_text

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_NEG = re.compile(r"\b(not|no|never|none|nor|without|cannot|neither|nothing|nobody|nowhere|"
                  r"isn|aren|wasn|weren|doesn|didn|won|wouldn|shouldn|couldn|can t|don)\b")


def guard_signature(text: str, numbers: bool = True, negation: bool = True) -> tuple:
    norm = normalize_text(text)
    nums = tuple(sorted(n.replace(",", "") for n in _NUM.findall(text))) if numbers else ()
    negs = len(_NEG.findall(norm)) % 2 if negation else 0  # parity: "not not" == affirmative-ish
    return nums, negs
from ..types import GatewayRequest
from .store import CacheEntry, InMemoryVectorStore, VectorStore


@dataclass
class CacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.90  # cosine; raise for high-stakes domains
    ttl_seconds: Optional[float] = 3600.0
    max_entries: int = 10_000
    scope: str = "tenant"  # "tenant" (isolated per tenant) | "global" (shared)
    cache_pii: bool = False  # never cache prompts containing PII unless explicitly allowed
    cache_time_sensitive: bool = False
    min_response_chars: int = 1
    # Semantic (non-exact) hits must agree on numbers and negation. Embeddings — hashed
    # or neural — routinely score "feature 6" vs "feature 57" or "safe" vs "not safe"
    # above 0.9; serving those from cache would return a wrong answer.
    guard_numbers: bool = True
    guard_negation: bool = True
    search_top_k: int = 5


@dataclass
class CacheHit:
    entry: CacheEntry
    similarity: float
    exact: bool


class SemanticCache:
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        store: Optional[VectorStore] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.config = config or CacheConfig()
        self.store: VectorStore = store or InMemoryVectorStore(self.config.max_entries)
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "exact_hits": 0, "semantic_hits": 0, "misses": 0, "stores": 0,
                      "guard_rejections": 0}

    # ------------------------------------------------------------------ keys
    def namespace(self, request: GatewayRequest, partition: str = "") -> str:
        tenant = request.tenant_id if self.config.scope == "tenant" else "*"
        return f"{tenant}:{request.context_fingerprint()}:{partition}"

    @staticmethod
    def exact_key(text: str) -> str:
        return hashlib.sha256(normalize_text(text).encode()).hexdigest()

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    # ------------------------------------------------------------- operations
    def lookup(self, request: GatewayRequest, partition: str = "",
               threshold: Optional[float] = None) -> Optional[CacheHit]:
        if not self.config.enabled:
            return None
        ns = self.namespace(request, partition)
        prompt = request.last_user_message
        entry = self.store.get_exact(ns, self.exact_key(prompt))
        if entry:
            self.store.touch(entry.id)
            self._bump("hits"); self._bump("exact_hits")
            return CacheHit(entry, 1.0, True)

        cfg = self.config
        limit = cfg.similarity_threshold if threshold is None else threshold
        sig = guard_signature(prompt, cfg.guard_numbers, cfg.guard_negation)
        for entry, score in self.store.search(ns, self.embedder.embed(prompt), top_k=cfg.search_top_k):
            if score < limit:
                break
            if guard_signature(entry.prompt, cfg.guard_numbers, cfg.guard_negation) != sig:
                self._bump("guard_rejections")
                continue
            self.store.touch(entry.id)
            self._bump("hits"); self._bump("semantic_hits")
            return CacheHit(entry, round(score, 4), False)
        self._bump("misses")
        return None

    def store_response(self, request: GatewayRequest, response_text: str, model: str,
                       cost_usd: float = 0.0, input_tokens: int = 0, output_tokens: int = 0,
                       partition: str = "", ttl_seconds: Optional[float] = None) -> Optional[CacheEntry]:
        if not self.config.enabled or len(response_text) < self.config.min_response_chars:
            return None
        import time

        ttl = self.config.ttl_seconds if ttl_seconds is None else ttl_seconds
        prompt = request.last_user_message
        entry = CacheEntry(
            id=uuid.uuid4().hex,
            namespace=self.namespace(request, partition),
            prompt=prompt,
            response=response_text,
            model=model,
            embedding=self.embedder.embed(prompt),
            exact_key=self.exact_key(prompt),
            expires_at=(time.time() + ttl) if ttl else None,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.store.add(entry)
        self._bump("stores")
        return entry

    def invalidate(self, tenant_id: Optional[str] = None) -> int:
        return self.store.clear(f"{tenant_id}:" if tenant_id else None)

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total else 0.0
