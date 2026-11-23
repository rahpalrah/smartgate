"""Second-stage verification of semantic cache hits.

Embedding similarity answers "are these questions about the same thing?". A cache needs a
stricter question: "is ONE answer correct for BOTH?". "How do I lose weight in 2 weeks?" and
"...in 2 months?" are close in any embedding space. A verifier asks that stricter question of
each candidate hit and the cache serves it only if the verifier agrees.

A verifier is any callable `(new_query, cached_query) -> float` in [0, 1] (1 = same question,
safe to reuse). Plug in a cross-encoder, a fine-tuned classifier, or an LLM; helpers below
wrap a plain text LLM. It runs only on semantic candidates (never on exact matches) and costs
one small model call per candidate, far less than the answer call it avoids.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Optional, Tuple

PairVerifier = Callable[[str, str], float]

VERIFY_SYSTEM = (
    "You decide whether a cached answer can be reused. Two questions are the SAME only if one "
    "answer is fully correct for both. Different numbers, versions, symbols (O+ vs O-, C vs C++), "
    "people, places, negation, scope or intent make them DIFFERENT. Reply with exactly YES or NO."
)


def pair_prompt(new_query: str, cached_query: str) -> str:
    return f"Q1: {new_query}\nQ2: {cached_query}\nSame question?"


class LLMPairVerifier:
    """Verifier from any text-completion function `complete(system, user) -> str`.

    Returns 1.0 for an answer starting with YES and 0.0 otherwise (anything unparseable counts
    as NO: when in doubt, don't reuse). For graded scores use a logprob-based verifier
    (see examples/kaggle_benchmark.py).
    """

    def __init__(self, complete: Callable[[str, str], str], system: str = VERIFY_SYSTEM) -> None:
        self.complete, self.system = complete, system

    def __call__(self, new_query: str, cached_query: str) -> float:
        answer = (self.complete(self.system, pair_prompt(new_query, cached_query)) or "").strip().upper()
        return 1.0 if answer.startswith("YES") else 0.0


class SymmetricVerifier:
    """Require agreement in both orders: min(v(a, b), v(b, a)).

    LLM judges are order-sensitive; on 220 real question pairs a 1.5B judge flipped its verdict
    with the order of the two questions for 24% of them. Taking the minimum favors precision
    (fewer wrong cache hits) at the price of two calls per candidate and fewer hits.
    """

    def __init__(self, verifier: PairVerifier) -> None:
        self.verifier = verifier

    def __call__(self, new_query: str, cached_query: str) -> float:
        return min(self.verifier(new_query, cached_query), self.verifier(cached_query, new_query))


class MemoizedVerifier:
    """LRU memo so the same (question, cached question) pair is judged once."""

    def __init__(self, verifier: PairVerifier, max_entries: int = 50_000) -> None:
        self.verifier, self.max_entries = verifier, max_entries
        self._memo: "OrderedDict[Tuple[str, str], float]" = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, new_query: str, cached_query: str) -> float:
        key = (new_query, cached_query)
        with self._lock:
            if key in self._memo:
                self._memo.move_to_end(key)
                return self._memo[key]
        score = self.verifier(new_query, cached_query)
        with self._lock:
            self._memo[key] = score
            while len(self._memo) > self.max_entries:
                self._memo.popitem(last=False)
        return score


def provider_verifier(provider, model, max_tokens: int = 3, symmetric: bool = False) -> PairVerifier:
    """Build a verifier from any smartgate provider and one of its models.

    The judge model answers YES or NO for each candidate pair. Works with the built-in
    `AnthropicProvider` (Claude over plain HTTPS) and `OpenAICompatibleProvider` (local servers),
    so no SDK or extra package is involved. Use a small, cheap model as the judge.

        judge = provider_verifier(AnthropicProvider(), ModelSpec("claude-haiku-5-5", "anthropic", "small", 0.1, 0.5))
        cache = SemanticCache(verifier=MemoizedVerifier(judge))

    Set `symmetric=True` to require YES in both question orders (fewer wrong hits, twice the calls).
    """
    from ..types import GatewayRequest, Message

    def complete(system: str, user: str) -> str:
        request = GatewayRequest(messages=[Message("user", user)], system=system, max_tokens=max_tokens)
        return provider.complete(model, request).text

    verifier: PairVerifier = LLMPairVerifier(complete)
    return SymmetricVerifier(verifier) if symmetric else verifier
