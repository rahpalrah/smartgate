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
# Symbols that change meaning: "O-" vs "O+", "C" vs "C++", "Nike+" vs "Nike", "/=" vs "//=".
# normalize_text() strips punctuation so these would otherwise look identical. Always counted:
# + # = % & * < > | \\ ^ ~ and brackets; - and / only when not word-internal ("dishwasher-safe"
# is fine, "O-" is not). Brackets that wrap words ("(preemptive)") are ignored; "()" and "[]" count.
_SYM_ALWAYS = frozenset("+#=%&*<>|\\^~()[]{}")
_WRAPPED = re.compile(r"[(\[{][^)\]}]*[A-Za-z0-9][^)\]}]*[)\]}]")


def symbol_signature(text: str) -> str:
    text = _WRAPPED.sub(" ", text)
    n, syms = len(text), []
    for i, ch in enumerate(text):
        if ch in _SYM_ALWAYS:
            syms.append(ch)
        elif ch in "-/":
            prev = text[i - 1] if i else " "
            nxt = text[i + 1] if i + 1 < n else " "
            if not (prev.isalnum() and nxt.isalnum()):
                syms.append(ch)
    return "".join(sorted(syms))
_NEG = re.compile(r"\b(not|no|never|none|nor|without|cannot|neither|nothing|nobody|nowhere|"
                  r"isn|aren|wasn|weren|doesn|didn|won|wouldn|shouldn|couldn|can t|don)\b")

# Action classes: words inside one class are synonyms; two classes are different requests.
# "a ... b" matches a split phrase such as "set my address up".
DEFAULT_ACTION_CLASSES: Dict[str, Sequence[str]] = {
    "cancel": ("cancel", "cancellation", "abort"),
    "track": ("track", "status", "where is", "wheres", "arrive", "eta", "expect", "waiting for",
              "wait for", "await", "still waiting", "when will"),
    "modify": ("change", "modify", "edit", "update", "amend", "correct", "alter"),
    "switch": ("switch", "change to", "upgrade", "downgrade", "move to", "use another", "swap"),
    "delete": ("delete", "remove", "close", "deactivate", "terminate", "erase"),
    "create": ("create", "open", "register", "sign up", "signup", "set up", "setup", "set ... up",
               "new account"),
    "obtain": ("get", "send", "download", "mail", "email me", "receive", "print", "obtain"),
    "inspect": ("check", "view", "see", "find", "locate", "look", "look up", "lookup", "look for",
                "consult", "show"),
    "recover": ("recover", "reset", "forgot", "forgotten", "retrieve", "restore", "unlock"),
    "complain": ("complain", "complaint", "claim", "dissatisfied", "unhappy"),
    "refund": ("refund", "reimburse", "reimbursement", "rebate", "money back", "compensation"),
    "return": ("return", "exchange", "send back"),
    "pay": ("pay", "payment", "charge", "billing"),
    "subscribe": ("subscribe", "subscription", "newsletter", "unsubscribe"),
    "contact": ("contact", "speak", "talk", "call", "human", "agent", "representative"),
}
_VOWELS = set("aeiou")


def _word_forms(term: str) -> List[str]:
    """Inflections without a stemmer: get/gets/getting, create/creating/creation, cancel/cancelled."""
    forms = {term, term + "s", term + "es", term + "ed", term + "d", term + "ing", term + "er",
             term + "ion", term + "ions", term + "ation", term + "ment"}
    if term.endswith("e"):
        stem = term[:-1]
        forms |= {stem + "ing", stem + "ion", stem + "ation", stem + "ed", stem + "er"}
    if len(term) >= 3 and term[-1] not in _VOWELS and term[-2] in _VOWELS and term[-3] not in _VOWELS \
            and term[-1] not in "wxy":
        forms |= {term + term[-1] + suf for suf in ("ing", "ed", "er", "ation")}
    return sorted(forms, key=len, reverse=True)


def _phrase_regex(term: str) -> str:
    if " ... " in term:  # split phrase: up to 4 words in between
        a, b = (normalize_text(x) for x in term.split(" ... ", 1))
        return re.escape(a) + r"(?:\s+\w+){0,4}?\s+" + re.escape(b)
    t = normalize_text(term)
    if " " in t:
        return re.escape(t)
    return "(?:" + "|".join(re.escape(f) for f in _word_forms(t)) + ")"


def _compile_actions(classes: Dict[str, Sequence[str]]):
    compiled = [(name, re.compile(r"\b(?:" + "|".join(_phrase_regex(t) for t in terms) + r")\b"))
                for name, terms in classes.items()]
    # Typo tolerance: adjacent-letter swaps of single-word action terms ("egt" -> "get"),
    # unless the swapped spelling is itself another action term.
    singles = {normalize_text(t) for terms in classes.values() for t in terms
               if " " not in t and len(t) >= 3}
    typos: Dict[str, str] = {}
    for t in singles:
        for i in range(len(t) - 1):
            swapped = t[:i] + t[i + 1] + t[i] + t[i + 2:]
            if swapped != t and swapped not in singles:
                typos.setdefault(swapped, t)
    return compiled, typos


def guard_signature(text: str, numbers: bool = True, negation: bool = True, actions=None,
                    symbols: bool = False) -> tuple:
    norm = normalize_text(text)
    if actions:
        compiled, typos = actions
        norm = " ".join(typos.get(w, w) for w in norm.split())
        actions = compiled
    nums = tuple(sorted(n.replace(",", "") for n in _NUM.findall(text))) if numbers else ()
    negs = len(_NEG.findall(norm)) % 2 if negation else 0
    acts: FrozenSet[str] = frozenset(n for n, rx in actions if rx.search(norm)) if actions else frozenset()
    return nums, negs, acts, symbol_signature(text) if symbols else ""


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
    guard_symbols: bool = True
    # Second-stage check on semantic hits (see verify.py). A candidate is served only if the
    # verifier scores it >= verify_min_score. A verifier that raises counts as "not the same"
    # (fail closed) unless verify_fail_open. verify_skip_above skips verification for very
    # similar candidates to save calls (None = verify every semantic hit).
    verify_min_score: float = 0.5
    verify_skip_above: Optional[float] = None
    verify_fail_open: bool = False
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
        verifier=None,
    ) -> None:
        self.config = config or CacheConfig()
        self.verifier = verifier
        self.store: VectorStore = store or InMemoryVectorStore(self.config.max_entries)
        self.embedder: Embedder = embedder or HashingEmbedder()
        classes = {k: list(v) for k, v in (self.config.action_classes or DEFAULT_ACTION_CLASSES).items()}
        for name, terms in self.config.extra_action_classes.items():  # same name => add synonyms
            classes[name] = list(dict.fromkeys(classes.get(name, []) + list(terms)))
        self._actions = _compile_actions(classes) if self.config.guard_actions else None
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "exact_hits": 0, "semantic_hits": 0, "misses": 0, "stores": 0,
                      "guard_rejections": 0, "verifier_calls": 0, "verifier_rejections": 0,
                      "verifier_errors": 0}

    # ------------------------------------------------------------------ keys
    def namespace(self, request: GatewayRequest, partition: str = "") -> str:
        tenant = request.tenant_id if self.config.scope == "tenant" else "*"
        return f"{tenant}:{request.context_fingerprint()}:{partition}"

    @staticmethod
    def exact_key(text: str) -> str:
        # Normalized text + meaning-bearing symbols, so "O-" and "O+" never collide.
        return hashlib.sha256((normalize_text(text) + "\0" + symbol_signature(text)).encode()).hexdigest()

    def signature(self, text: str) -> tuple:
        cfg = self.config
        return guard_signature(text, cfg.guard_numbers, cfg.guard_negation, self._actions, cfg.guard_symbols)

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
            if self.verifier is not None and (self.config.verify_skip_above is None
                                              or score < self.config.verify_skip_above):
                self._bump("verifier_calls")
                try:
                    verdict = float(self.verifier(prompt, entry.prompt))
                except Exception:  # a broken verifier must never cause a wrong cache hit
                    self._bump("verifier_errors")
                    verdict = 1.0 if self.config.verify_fail_open else 0.0
                if verdict < self.config.verify_min_score:
                    self._bump("verifier_rejections")
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
