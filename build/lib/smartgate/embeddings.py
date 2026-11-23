"""Embedding backends for semantic caching.

`HashingEmbedder` is dependency-free and deterministic: good for tests, edge
deployments and near-duplicate detection. For true paraphrase matching plug in a
neural embedder via `CallableEmbedder` (sentence-transformers, Voyage, Cohere,
an internal service...).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, List, Protocol, Sequence

STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or but with at by from as "
    "it this that these those do does did can could would should will please me i you "
    "your my we our tell give show what's whats".split()
)
_CONTRACTIONS = {"what's": "what is", "it's": "it is", "how's": "how is", "who's": "who is",
                 "can't": "cannot", "don't": "do not", "won't": "will not", "i'm": "i am"}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> List[float]: ...


def _l2(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity for (already L2-normalized) vectors."""
    return sum(x * y for x, y in zip(a, b))


class HashingEmbedder:
    """Feature-hashed bag of words + word bigrams + character trigrams."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        norm = normalize_text(text)
        words = [w for w in norm.split() if w not in STOPWORDS] or norm.split()
        feats: List[tuple] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
        joined = " ".join(words)
        feats += [(f"c:{joined[i:i + 3]}", 0.3) for i in range(max(0, len(joined) - 2))]
        for feat, weight in feats:
            idx, sign = self._bucket(feat)
            vec[idx] += sign * weight
        return _l2(vec)


class CallableEmbedder:
    """Wrap any `fn(text) -> list[float]` as an Embedder (vectors are normalized)."""

    def __init__(self, fn: Callable[[str], Sequence[float]], dim: int) -> None:
        self.fn = fn
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        vec = list(map(float, self.fn(text)))
        if len(vec) != self.dim:
            raise ValueError(f"Embedder returned dim {len(vec)}, expected {self.dim}")
        return _l2(vec)
