"""Vector stores backing the semantic cache.

Implement `VectorStore` to back the cache with Redis, pgvector, Qdrant,
Pinecone, etc. Two implementations ship with the SDK:

* `InMemoryVectorStore` — thread-safe, LRU-bounded, per-process.
* `SQLiteVectorStore`  — durable, shareable between processes on one host.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from array import array
from collections import OrderedDict
from heapq import nlargest
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from ..embeddings import cosine


@dataclass
class CacheEntry:
    id: str
    namespace: str
    prompt: str
    response: str
    model: str
    embedding: List[float]
    exact_key: str
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hits: int = 0
    cost_usd: float = 0.0  # what the original call cost (=> savings per hit)
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: Dict = field(default_factory=dict)

    def expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at is not None and (now or time.time()) >= self.expires_at


class VectorStore(Protocol):
    def add(self, entry: CacheEntry) -> None: ...
    def get_exact(self, namespace: str, exact_key: str) -> Optional[CacheEntry]: ...
    def search(self, namespace: str, embedding: List[float], top_k: int = 1) -> List[Tuple[CacheEntry, float]]: ...
    def touch(self, entry_id: str) -> None: ...
    def delete(self, entry_id: str) -> None: ...
    def clear(self, namespace_prefix: Optional[str] = None) -> int: ...
    def size(self) -> int: ...


try:  # math.sumprod (Python 3.12+) runs the dot product in C; older versions use a slower fallback
    from math import sumprod as _dot
except ImportError:  # pragma: no cover
    from operator import mul as _mul

    def _dot(a, b):
        return sum(map(_mul, a, b))


class _NamespaceIndex:
    """Exact nearest-neighbour index for one namespace, standard library only.

    Vectors are stored as compact float32 arrays (about 4 bytes per dimension) and scanned with
    a C-speed dot product. O(1) add/remove (swap-with-last). A scan costs roughly 7-15 ms per
    1,000 entries at 512-768 dimensions, so this suits caches up to a few tens of thousands of
    entries per namespace; beyond that, back the cache with your own `VectorStore`.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: List[str] = []
        self.pos: Dict[str, int] = {}
        self.rows: List[array] = []

    def add(self, entry_id: str, vec: List[float]) -> None:
        self.pos[entry_id] = len(self.ids)
        self.ids.append(entry_id)
        self.rows.append(array("f", vec))

    def remove(self, entry_id: str) -> None:
        row = self.pos.pop(entry_id, None)
        if row is None:
            return
        last = len(self.ids) - 1
        if row != last:
            moved = self.ids[last]
            self.ids[row] = moved
            self.rows[row] = self.rows[last]
            self.pos[moved] = row
        self.ids.pop()
        self.rows.pop()

    def search(self, vec: List[float], k: int) -> List[Tuple[str, float]]:
        if not self.ids:
            return []
        query = array("f", vec)
        best = nlargest(k, ((_dot(query, row), i) for i, row in enumerate(self.rows)))
        return [(self.ids[i], float(score)) for score, i in best]


class InMemoryVectorStore:
    """Thread-safe, LRU-bounded store with a per-namespace vector index.

    Expired entries are dropped lazily when they surface in a search and by a periodic sweep.
    """

    SWEEP_EVERY = 1000

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._exact: Dict[Tuple[str, str], str] = {}
        self._index: Dict[str, _NamespaceIndex] = {}
        self._lock = threading.RLock()
        self._ops = 0

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            old_id = self._exact.get((entry.namespace, entry.exact_key))
            if old_id:
                self._remove(old_id)
            self._entries[entry.id] = entry
            self._exact[(entry.namespace, entry.exact_key)] = entry.id
            idx = self._index.get(entry.namespace)
            if idx is None:
                idx = self._index[entry.namespace] = _NamespaceIndex(len(entry.embedding))
            idx.add(entry.id, entry.embedding)
            while len(self._entries) > self.max_entries:
                self._remove(next(iter(self._entries)))
            self._maybe_sweep()

    def _remove(self, entry_id: str) -> None:
        e = self._entries.pop(entry_id, None)
        if e:
            self._exact.pop((e.namespace, e.exact_key), None)
            idx = self._index.get(e.namespace)
            if idx:
                idx.remove(entry_id)
                if not idx.ids:
                    del self._index[e.namespace]

    def _maybe_sweep(self) -> None:
        self._ops += 1
        if self._ops % self.SWEEP_EVERY == 0:
            now = time.time()
            for eid in [e.id for e in self._entries.values() if e.expired(now)]:
                self._remove(eid)

    def get_exact(self, namespace: str, exact_key: str) -> Optional[CacheEntry]:
        with self._lock:
            eid = self._exact.get((namespace, exact_key))
            entry = self._entries.get(eid) if eid else None
            if entry and entry.expired():
                self._remove(entry.id)
                return None
            return entry

    def search(self, namespace: str, embedding: List[float], top_k: int = 1):
        now = time.time()
        with self._lock:
            self._maybe_sweep()
            idx = self._index.get(namespace)
            if not idx:
                return []
            out = []
            for eid, score in idx.search(embedding, top_k + 8):
                e = self._entries[eid]
                if e.expired(now):
                    self._remove(eid)
                    continue
                out.append((e, score))
                if len(out) == top_k:
                    break
            return out

    def touch(self, entry_id: str) -> None:
        with self._lock:
            e = self._entries.get(entry_id)
            if e:
                e.hits += 1
                self._entries.move_to_end(entry_id)  # LRU recency

    def delete(self, entry_id: str) -> None:
        with self._lock:
            self._remove(entry_id)

    def clear(self, namespace_prefix: Optional[str] = None) -> int:
        with self._lock:
            ids = [e.id for e in self._entries.values()
                   if namespace_prefix is None or e.namespace.startswith(namespace_prefix)]
            for eid in ids:
                self._remove(eid)
            return len(ids)

    def size(self) -> int:
        return len(self._entries)


class SQLiteVectorStore:
    """Durable store. Brute-force cosine per namespace (fine to ~100k entries)."""

    def __init__(self, path: str = "smartgate_cache.db", max_entries: int = 100_000) -> None:
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                id TEXT PRIMARY KEY, namespace TEXT, exact_key TEXT, data TEXT,
                expires_at REAL, last_used REAL)"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_ns ON cache(namespace)")
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_exact ON cache(namespace, exact_key)")
        self._conn.commit()

    @staticmethod
    def _load(data: str) -> CacheEntry:
        return CacheEntry(**json.loads(data))

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache WHERE namespace=? AND exact_key=?",
                               (entry.namespace, entry.exact_key))
            self._conn.execute(
                "INSERT INTO cache VALUES (?,?,?,?,?,?)",
                (entry.id, entry.namespace, entry.exact_key, json.dumps(asdict(entry)),
                 entry.expires_at, time.time()),
            )
            (count,) = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
            if count > self.max_entries:
                self._conn.execute(
                    "DELETE FROM cache WHERE id IN (SELECT id FROM cache ORDER BY last_used ASC LIMIT ?)",
                    (count - self.max_entries,),
                )
            self._conn.commit()

    def _purge_expired(self) -> None:
        self._conn.execute("DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at <= ?", (time.time(),))

    def get_exact(self, namespace: str, exact_key: str) -> Optional[CacheEntry]:
        with self._lock:
            self._purge_expired()
            row = self._conn.execute("SELECT data FROM cache WHERE namespace=? AND exact_key=?",
                                     (namespace, exact_key)).fetchone()
            return self._load(row[0]) if row else None

    def search(self, namespace: str, embedding: List[float], top_k: int = 1):
        with self._lock:
            self._purge_expired()
            rows = self._conn.execute("SELECT data FROM cache WHERE namespace=?", (namespace,)).fetchall()
        scored = [(e, cosine(embedding, e.embedding)) for e in map(lambda r: self._load(r[0]), rows)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def touch(self, entry_id: str) -> None:
        with self._lock:
            row = self._conn.execute("SELECT data FROM cache WHERE id=?", (entry_id,)).fetchone()
            if not row:
                return
            entry = self._load(row[0])
            entry.hits += 1
            self._conn.execute("UPDATE cache SET data=?, last_used=? WHERE id=?",
                               (json.dumps(asdict(entry)), time.time(), entry_id))
            self._conn.commit()

    def delete(self, entry_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cache WHERE id=?", (entry_id,))
            self._conn.commit()

    def clear(self, namespace_prefix: Optional[str] = None) -> int:
        with self._lock:
            if namespace_prefix is None:
                cur = self._conn.execute("DELETE FROM cache")
            else:
                cur = self._conn.execute("DELETE FROM cache WHERE namespace LIKE ?", (namespace_prefix + "%",))
            self._conn.commit()
            return cur.rowcount

    def size(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
