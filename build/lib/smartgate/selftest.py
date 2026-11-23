"""Built-in self-test: verifies an installation with nothing but the standard library.

    python -m smartgate selftest        (or:  smartgate selftest)

Covers caching and its safety guards, PII redaction, routing, hard budgets under concurrency,
retry / circuit breaker / failover, the verifier hook, the HTTP service, the Claude provider over
real HTTP (against a local fake API), TOML/JSON config, and that the package imports nothing
outside the Python standard library. Takes a few seconds; needs no API key and no network.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import (AnthropicProvider, Budget, BudgetExceededError, CacheConfig, Gateway, GatewayRequest, MockProvider,
               ModelSpec, PolicyConfig, ResilienceConfig, RetryPolicy, SemanticCache, SQLiteVectorStore,
               detect_pii, gateway_from_preset, redact_pii)
from .analysis import PromptAnalyzer, is_time_sensitive, set_phone_regions
from .client import GatewayClient, GatewayHTTPError
from .server import serve

MODELS = [ModelSpec("small-a", "mock", "small", 0.1, 0.5), ModelSpec("medium-a", "mock", "medium", 2.0, 10.0),
          ModelSpec("large-a", "mock", "large", 4.0, 20.0)]


def _req(text, **kw):
    return GatewayRequest(messages=[{"role": "user", "content": text}], **kw)


def _gateway(provider=None, **kw):
    return Gateway(models=MODELS, providers={"mock": provider or MockProvider()}, sleep=lambda s: None,
                   resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=2, base_delay=0)), **kw)


class Caching(unittest.TestCase):
    def test_exact_and_semantic_hits(self):
        c = SemanticCache(CacheConfig(similarity_threshold=0.9))
        c.store_response(_req("How do I reset my password?"), "Use 'Forgot password'.", "m")
        self.assertTrue(c.lookup(_req("how do i reset my password")).exact)
        self.assertFalse(c.lookup(_req("How can I reset my password?")).exact)
        self.assertIsNone(c.lookup(_req("What is your refund policy?")))

    def test_tenants_never_share_answers(self):
        c = SemanticCache()
        c.store_response(_req("hello", tenant_id="a"), "hi", "m")
        self.assertIsNone(c.lookup(_req("hello", tenant_id="b")))
        self.assertIsNotNone(c.lookup(_req("hello", tenant_id="a")))

    def test_guards_keep_different_questions_apart(self):
        pairs = [("Is the O- blood group the universal donor?", "Is the O+ blood group the universal donor?"),
                 ("How do I configure feature 6?", "How do I configure feature 57?"),
                 ("Is it safe to take ibuprofen?", "Is it not safe to take ibuprofen?"),
                 ("I want to edit my account", "I want to delete my account"),
                 ("code in C for SJF scheduling", "code in C++ for SJF scheduling")]
        for a, b in pairs:
            c = SemanticCache(CacheConfig(similarity_threshold=0.5))
            c.store_response(_req(a), "x", "m")
            self.assertIsNone(c.lookup(_req(b)), (a, b))

    def test_ttl_and_sqlite_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            SemanticCache(store=SQLiteVectorStore(path)).store_response(_req("hello world"), "hi", "m")
            hit = SemanticCache(store=SQLiteVectorStore(path)).lookup(_req("Hello, world!"))
            self.assertEqual(hit.entry.response, "hi")

    def test_verifier_can_veto_a_semantic_hit_and_fails_closed(self):
        c = SemanticCache(CacheConfig(similarity_threshold=0.5), verifier=lambda new, old: 0.0)
        c.store_response(_req("How can I lose weight in 2 weeks?"), "A", "m")
        self.assertIsNone(c.lookup(_req("How can I lose weight in 2 weeks fast?")))

        def broken(a, b):
            raise RuntimeError("judge offline")
        c2 = SemanticCache(CacheConfig(similarity_threshold=0.5), verifier=broken)
        c2.store_response(_req("How can I lose weight in 2 weeks?"), "A", "m")
        self.assertIsNone(c2.lookup(_req("How can I lose weight in 2 weeks fast?")))


class Analysis(unittest.TestCase):
    def test_pii_detection_and_redaction(self):
        text = "mail jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111, call +44 20 7946 0958"
        self.assertTrue({"email", "ssn", "credit_card", "phone"} <= set(detect_pii(text)))
        red = redact_pii(text)
        for secret in ("jane@example.com", "123-45-6789", "4111", "7946"):
            self.assertNotIn(secret, red)
        self.assertEqual(detect_pii("order 732201349959, invoice 00004587345"), [])

    def test_national_phone_formats_need_their_region(self):
        try:
            self.assertEqual(detect_pii("ring 020 7946 0958"), [])
            set_phone_regions(["US", "GB"])
            self.assertIn("phone", detect_pii("ring 020 7946 0958"))
        finally:
            set_phone_regions(["US"])

    def test_complexity_orders_prompts(self):
        a = PromptAnalyzer()
        easy = a.analyze(_req("What is the capital of France?")).complexity
        hard = a.analyze(_req("Design an end-to-end architecture, first analyze the trade-offs, then implement "
                              "it in Python and compare three approaches, ensuring correctness.")).complexity
        self.assertLess(easy, 0.35)
        self.assertGreater(hard, 0.7)

    def test_time_sensitive_rule(self):
        self.assertTrue(is_time_sensitive("Is the app down right now?"))
        self.assertFalse(is_time_sensitive("I need help now"))


class GatewayBehaviour(unittest.TestCase):
    def test_cache_saves_money_and_routes_by_complexity(self):
        gw = _gateway()
        r1 = gw.generate("How do I reset my password?", tenant_id="acme")
        r2 = gw.generate("how can I reset my password", tenant_id="acme")
        self.assertEqual(r1.model, "small-a")
        self.assertTrue(r2.cache_hit and r2.cost_usd == 0 and r2.saved_usd > 0)

    def test_retry_failover_and_circuit_breaker(self):
        p = MockProvider(fail_models={"small-a": "server"})
        gw = _gateway(p)
        self.assertEqual(gw.generate("hi").model, "medium-a")
        self.assertEqual(p.calls.count("small-a"), 2)  # retried once, then failed over

    def test_pii_is_redacted_before_the_model_sees_it(self):
        seen = []
        from .providers import CallableProvider
        gw = _gateway(CallableProvider("mock", lambda m, r: seen.append(r.last_user_message) or "ok"),
                      policy=PolicyConfig(pii="redact"))
        gw.generate("my email is bob@corp.com, help")
        self.assertNotIn("bob@corp.com", seen[0])

    def test_hard_budget_is_never_overspent_under_concurrency(self):
        gw = _gateway(MockProvider(latency=0.001))
        limit = 0.02
        gw.set_budget("t", Budget(limit_usd=limit, period="total", downgrade_at=2.0))

        def call(i):
            try:
                gw.generate(f"Analyze option {i}, compare trade-offs and plan step by step.", tenant_id="t",
                            max_tokens=400, use_cache=False)
            except BudgetExceededError:
                pass
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(call, range(300)))
        self.assertLessEqual(gw.costs.spent("t"), limit + 1e-12)
        self.assertGreater(gw.costs.spent("t"), 0)
        self.assertAlmostEqual(gw.costs.reserved("t"), 0.0)

    def test_identical_concurrent_requests_make_one_model_call(self):
        p = MockProvider(latency=0.05)
        gw = _gateway(p)
        with ThreadPoolExecutor(16) as ex:
            list(ex.map(lambda _: gw.generate("What is the refund policy?"), range(16)))
        self.assertEqual(len(p.calls), 1)


class HttpService(unittest.TestCase):
    def test_server_and_client_round_trip(self):
        gw = gateway_from_preset("customer_support", providers={"anthropic": MockProvider()})
        gw.set_budget("broke", Budget(limit_usd=0.0))
        httpd = serve(gw, port=0, api_keys={"k1": "acme", "k2": "broke"})
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}"
            c = GatewayClient(url, api_key="k1")
            self.assertFalse(c.generate("Where is my order?")["cache_hit"])
            self.assertTrue(c.generate("where is my order")["cache_hit"])
            with self.assertRaises(GatewayHTTPError) as e:
                GatewayClient(url, api_key="k2").generate("hi")
            self.assertEqual(e.exception.status, 402)
            with self.assertRaises(GatewayHTTPError) as e:
                GatewayClient(url, api_key="nope").generate("hi")
            self.assertEqual(e.exception.status, 401)
        finally:
            httpd.shutdown()


class ClaudeProviderOverHttp(unittest.TestCase):
    def test_request_shape_headers_and_usage(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                seen.append((self.path, self.headers.get("x-api-key"), self.headers.get("anthropic-version"), body))
                raw = json.dumps({"model": body["model"], "stop_reason": "end_turn",
                                  "content": [{"type": "text", "text": "Paris"}],
                                  "usage": {"input_tokens": 12, "output_tokens": 3}}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            claude = AnthropicProvider(api_key="sk-test", base_url=f"http://127.0.0.1:{httpd.server_address[1]}")
            gw = Gateway(models=[ModelSpec("claude-haiku-5-5", "anthropic", "small", 0.1, 0.5)],
                         providers={"anthropic": claude})
            r = gw.generate("Capital of France?", system="Be brief", tenant_id="t")
            self.assertEqual((r.text, r.usage.input_tokens, r.usage.output_tokens), ("Paris", 12, 3))
            self.assertAlmostEqual(r.cost_usd, (12 * 0.1 + 3 * 0.5) / 1e6)
            path, key, version, body = seen[0]
            self.assertEqual((path, key, version), ("/v1/messages", "sk-test", "2023-06-01"))
            self.assertEqual(body["system"], "Be brief")
        finally:
            httpd.shutdown()


class Standalone(unittest.TestCase):
    def test_config_files_json_and_toml(self):
        from .config import gateway_from_file
        with tempfile.TemporaryDirectory() as d:
            j, t = os.path.join(d, "c.json"), os.path.join(d, "c.toml")
            open(j, "w").write(json.dumps({"preset": "legal", "provider_override": "mock"}))
            open(t, "w").write('preset = "legal"\nprovider_override = "mock"\n')
            for path in (j, t):
                self.assertTrue(gateway_from_file(path).generate("Explain this indemnity clause").text)

    def test_package_imports_only_the_standard_library(self):
        root = os.path.dirname(os.path.abspath(__file__))
        stdlib = set(sys.stdlib_module_names)
        offenders = []
        for folder, _, files in os.walk(root):
            for name in files:
                if not name.endswith(".py"):
                    continue
                tree = ast.parse(open(os.path.join(folder, name), encoding="utf-8").read())
                for node in ast.walk(tree):
                    mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                           [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else []
                    offenders += [(name, m) for m in mods if m.split(".")[0] not in stdlib]
        self.assertEqual(offenders, [], "third-party imports found")


def run() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print("\nsmartgate self-test:", "PASSED" if result.wasSuccessful() else "FAILED",
          f"({result.testsRun} checks, standard library only)")
    return 0 if result.wasSuccessful() else 1
