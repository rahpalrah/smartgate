"""System-level: HTTP under concurrent load, multi-process durability, overhead at scale."""
import json
import multiprocessing as mp
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from smartgate import (Budget, CacheConfig, GatewayRequest, MockProvider, SemanticCache,
                       SQLiteVectorStore, gateway_from_preset)
from smartgate.client import GatewayClient, GatewayHTTPError
from smartgate.server import serve


@pytest.fixture
def http():
    gw = gateway_from_preset("customer_support", providers={"anthropic": MockProvider(latency=0.002)})
    gw.set_budget("capped", Budget(limit_usd=0.002, period="total"))
    keys = {f"key-{i}": f"tenant-{i}" for i in range(10)}
    keys["key-capped"] = "capped"
    httpd = serve(gw, port=0, api_keys=keys)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield gw, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_http_concurrent_load_and_tenant_isolation(http):
    gw, url = http
    jobs = [(f"key-{i % 10}", f"Where is order {i % 25}?") for i in range(1000)]
    results, errors = [], []
    lock = threading.Lock()

    def call(job):
        key, prompt = job
        try:
            r = GatewayClient(url, api_key=key).generate(prompt)
            with lock:
                results.append((key, r))
        except Exception as e:  # noqa
            with lock:
                errors.append(repr(e))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(40) as ex:
        list(ex.map(call, jobs))
    elapsed = time.perf_counter() - t0
    assert errors == []
    assert len(results) == 1000
    # every tenant's report counts exactly its own requests
    for i in range(10):
        rep = GatewayClient(url, api_key=f"key-{i}").costs()
        assert rep["requests"] == 100
        mine = sum(r["cost_usd"] for k, r in results if k == f"key-{i}")
        assert rep["cost_usd"] == pytest.approx(mine, abs=1e-6)
    hits = sum(r["cache_hit"] for _, r in results)
    assert hits >= 1000 - 10 * 25 - 40           # 250 distinct (tenant, order) pairs
    print(f"\nHTTP: 1000 req / 40 clients in {elapsed:.2f}s = {1000 / elapsed:.0f} req/s, hits={hits}")


def test_http_budget_enforced_under_concurrency(http):
    gw, url = http
    statuses = []
    lock = threading.Lock()

    def call(i):
        try:
            GatewayClient(url, api_key="key-capped").generate(f"unique question {i} about plans", use_cache=False)
            s = 200
        except GatewayHTTPError as e:
            s = e.status
        with lock:
            statuses.append(s)

    with ThreadPoolExecutor(40) as ex:
        list(ex.map(call, range(400)))
    assert set(statuses) == {200, 402}
    assert gw.costs.spent("capped") <= 0.002 + 1e-12
    assert gw.costs.reserved("capped") == pytest.approx(0)


def _writer(path, start, n):
    c = SemanticCache(store=SQLiteVectorStore(path))
    for i in range(start, start + n):
        c.store_response(GatewayRequest(messages=[{"role": "user", "content": f"question number {i}"}]),
                         f"answer {i}", "m")


def test_sqlite_cache_shared_across_processes(tmp_path):
    path = str(tmp_path / "shared.db")
    SQLiteVectorStore(path)  # create schema once
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(path, k * 100, 100)) for k in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    c = SemanticCache(store=SQLiteVectorStore(path))
    assert c.store.size() == 400
    for i in (0, 150, 299, 399):
        hit = c.lookup(GatewayRequest(messages=[{"role": "user", "content": f"Question number {i}?"}]))
        assert hit and hit.entry.response == f"answer {i}"


@pytest.mark.parametrize("entries", [1_000, 10_000])
def test_gateway_overhead_with_large_cache(entries):
    """Overhead added by the gateway itself (provider latency excluded)."""
    from smartgate.cache import store
    if store._np is None and entries > 1_000:
        pytest.skip("large-cache latency target assumes the NumPy index (pip install smartgate[fast])")
    c = SemanticCache(CacheConfig(max_entries=entries + 10))
    for i in range(entries):
        c.store_response(GatewayRequest(messages=[{"role": "user", "content": f"faq item {i} about topic {i % 97}"}]),
                         "a", "m")
    probes = [GatewayRequest(messages=[{"role": "user", "content": f"never seen prompt {i} zebra"}])
              for i in range(30)]
    t = []
    for p in probes:
        t0 = time.perf_counter()
        c.lookup(p)
        t.append((time.perf_counter() - t0) * 1000)
    p50 = statistics.median(t)
    print(f"\ncache miss lookup @ {entries} entries: p50={p50:.2f} ms  max={max(t):.2f} ms")
    assert p50 < (25 if store._np is not None else 60), f"semantic lookup too slow: {p50:.1f} ms"
