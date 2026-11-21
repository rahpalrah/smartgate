"""Semantic cache: exact-match fast path, then embedding search, then verification guards.

Retrieval alone is not enough: embeddings (hashed or neural) place "edit my account" next to
"delete my account" and "feature 6" next to "feature 57". Every semantic candidate therefore
passes a verification step before it is served:

* numbers must match           ("order 5512" != "order 5513")
* negation parity must match   ("safe" != "not safe")
* requested actions must match ("check my invoice" != "send me my invoice"), using
  synonym classes so "change" and "modify" still match each other.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ..embeddings import Embedder, HashingEmbedder, normalize_text
from ..types import GatewayRequest, Message
from .store import CacheEntry, InMemoryVectorStore, VectorStore

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_NEG = re.compile(r"\b(not|no|never|none|nor|without|cannot|neither|nothing|nobody|nowhere|"
                  r"isn|aren|wasn|weren|doesn|didn|won|wouldn|shouldn|couldn|can t|don)\b")

# Action classes: words inside one class are synonyms; two classes are different requests.
DEFAULT_ACTION_CLASSES: Dict[str, Sequence[str]] = {
    "cancel": ("cancel", "cancellation", "abort"),
    "track": ("track", "tracking", "status", "where is", "wheres", "arrive", "eta"),
    "modify": ("change", "modify", "edit", "update", "amend", "correct", "switch", "alter"),
    "delete": ("delete", "remove", "close", "deactivate", "terminate", "erase"),
    "create": ("create", "open", "register", "sign up", "signup", "set up", "setup", "new account"),
    "obtain": ("get", "send", "download", "mail", "email me", "receive", "print"),
    "inspect": ("check", "view", "see", "find", "locate", "look up", "lookup", "consult", "show"),
    "recover": ("recover", "reset", "forgot", "forgotten", "retrieve", "restore", "unlock"),
    "complain": ("complain", "complaint", "claim", "dissatisfied", "unhappy"),
    "refund": ("refund", "reimburse", "reimbursement", "money back"),
    "return": ("return", "exchange", "send back"),
    "pay": ("pay", "payment", "charge", "billing"),
    "subscribe": ("subscribe", "subscription", "newsletter", "unsubscribe"),
    "contact": ("contact", "speak", "talk", "call", "human", "agent", "representative"),
}
_SUFFIXES = ("", "s", "es", "ed", "d", "ing", "ling", "led", "ation", "ions", "ion", "er")


def _compile_actions(classes: Dict[str, Sequence[str]]) -> List[Tuple[str, re.Pattern]]:
    compiled = []
    for name, terms in classes.items():
        alts = []
        for t in terms:
            t = normalize_text(t)
            if " " in t:
                alts.append(re.escape(t))
            else:
                alts.append(re.escape(t) + "(?:" + "|".join(_SUFFIXES[1:]) + ")?")
        compiled.append((name, re.compile(r"\b(?:" + "|".join(alts) + r")\b")))
    return compiled


def guard_signature(text: str, numbers: bool = True, negation: bool = True,
                    actions: Optional[List[Tuple[str, re.Pattern]]] = None) -> tuple:
    norm = normalize_text(text)
    nums = tuple(sorted(n.replace(",", "") for n in _NUM.findall(text))) if numbers else ()
    negs = len(_NEG.findall(norm)) % 2 if negation else 0
    acts: FrozenSet[str] = frozenset(n for n, rx in actions if rx.search(norm)) if actions else frozenset()
    return nums, negs, acts


@dataclass
class CacheConfig:
    enabled: bool = True
    similarity_threshold: float = 0.90  # cosine; calibrate on your traffic (see calibrate.py)
    ttl_seconds: Optional[float] = 3600.0
    max_entries: int = 10_000
    scope: str = "tenant"  # "tenant" (isolated per tenant) | "global" (shared)
    cache_pii: bool = False  # never cache prompts containing PII unless explicitly allowed
    cache_time_sensitive: bool = False
    min_response_chars: int = 1
    guard_numbers: bool = True
    guard_negation: bool = True
    guard_actions: bool = True
    action_classes: Optional[Dict[str, List[str]]] = None  # None = DEFAULT_ACTION_CLASSES
    extra_action_classes: Dict[str, List[str]] = field(default_factory=dict)  # merged on top
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
        classes = dict(self.config.action_classes or DEFAULT_ACTION_CLASSES)
        classes.update(self.config.extra_action_classes)
        self._actions = _compile_actions(classes) if self.config.guard_actions else None
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

    def signature(self, text: str) -> tuple:
        cfg = self.config
        return guard_signature(text, cfg.guard_numbers, cfg.guard_negation, self._actions)

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

        limit = self.config.similarity_threshold if threshold is None else threshold
        sig = self.signature(prompt)
        for entry, score in self.store.search(ns, self.embedder.embed(prompt), top_k=self.config.search_top_k):
            if score < limit:
                break
            if self.signature(entry.prompt) != sig:
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

    def seed(self, question: str, answer: str, *, tenant_id: str = "default",
             system: Optional[str] = None, ttl_seconds: Optional[float] = None,
             model: str = "curated") -> Optional[CacheEntry]:
        """Pre-load an approved answer (e.g. from a reviewed FAQ). `system` must match the
        system prompt your app sends, because entries are scoped to it."""
        req = GatewayRequest(messages=[Message("user", question)], system=system, tenant_id=tenant_id)
        return self.store_response(req, answer, model, ttl_seconds=ttl_seconds)

    def invalidate(self, tenant_id: Optional[str] = None) -> int:
        return self.store.clear(f"{tenant_id}:" if tenant_id else None)

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total else 0.0
