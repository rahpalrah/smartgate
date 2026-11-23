"""Ollama + Qwen support, tested against a local fake that speaks Ollama's HTTP endpoints."""
import io
import json
import threading
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from smartgate import (ConfigError, OllamaProvider, ProviderError, gateway_from_ollama, qwen_models)
from smartgate.cli import main
from smartgate.providers.ollama import QWEN_PROFILES, normalize_host


class FakeOllama:
    def __init__(self, installed=("qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"), modern=True):
        self.installed, self.modern, self.seen = list(installed), modern, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models" and outer.modern:
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in outer.installed]})
                elif self.path == "/api/tags":
                    self._send(200, {"models": [{"name": m} for m in outer.installed]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append((self.path, body))
                if body["model"] not in outer.installed:
                    return self._send(404, {"error": {"message": f"model \"{body['model']}\" not found, try pulling it first",
                                                      "type": "api_error"}})
                self._send(200, {"model": body["model"], "choices": [{"message": {"role": "assistant",
                                 "content": f"[{body['model']}] hello"}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def ollama():
    o = FakeOllama()
    yield o
    o.close()


class TestHostAndModels:
    @pytest.mark.parametrize("given,expected", [
        (None, "http://localhost:11434"), ("myhost", "http://myhost:11434"), ("10.0.0.5:9999", "http://10.0.0.5:9999"),
        ("http://example.com", "http://example.com:80"), ("https://ollama.corp", "https://ollama.corp:443"),
        ("0.0.0.0:11434", "http://127.0.0.1:11434"), ("0.0.0.0", "http://127.0.0.1:11434")])
    def test_host_forms_follow_ollamas_defaults(self, given, expected, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
        assert normalize_host(given) == expected

    def test_host_comes_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "gpu-box:5555")
        assert OllamaProvider().host == "http://gpu-box:5555"

    def test_profiles_and_shadow_prices(self):
        for name, tags in QWEN_PROFILES.items():
            specs = qwen_models(name)
            assert [s.name for s in specs] == list(tags) and [s.tier for s in specs] == ["small", "medium", "large"]
            assert all(s.input_cost_per_mtok == 0 and s.provider == "ollama" for s in specs)
        priced = qwen_models("standard", shadow_prices=True, context_tokens=8192)
        assert [s.input_cost_per_mtok for s in priced] == [0.10, 2.00, 4.00] and priced[0].max_context_tokens == 8192
        assert [s.name for s in qwen_models(models=["a:1", "b:2", "c:3"])] == ["a:1", "b:2", "c:3"]
        with pytest.raises(ConfigError):
            qwen_models("nope")
        with pytest.raises(ConfigError):
            qwen_models(models=["only-one"])

    def test_installed_and_missing_models(self, ollama):
        p = OllamaProvider(ollama.host)
        assert p.installed_models() == ["qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]
        assert p.missing_models(["qwen2.5:3b", "qwen2.5:14b"]) == ["qwen2.5:14b"]
        ollama.installed.append("llama3.2:latest")
        assert p.missing_models(["llama3.2"]) == []            # a bare name means :latest

    def test_older_ollama_without_v1_models_falls_back_to_api_tags(self):
        o = FakeOllama(modern=False)
        try:
            assert "qwen2.5:3b" in OllamaProvider(o.host).installed_models()
        finally:
            o.close()

    def test_server_not_running_is_a_clear_connection_error(self):
        with pytest.raises(ProviderError) as e:
            OllamaProvider("127.0.0.1:9", timeout=1).installed_models()
        assert e.value.kind == "connection"


class TestGatewayOnOllama:
    def test_routing_cache_and_request_shape(self, ollama):
        gw = gateway_from_ollama(host=ollama.host, preset="customer_support", check=True)
        r1 = gw.generate("How do I reset my password?", tenant_id="acme", system="Be brief", max_tokens=300)
        r2 = gw.generate("how can I reset my password", tenant_id="acme", system="Be brief")
        assert r1.model == "qwen2.5:1.5b" and r1.provider == "ollama" and r1.text == "[qwen2.5:1.5b] hello"
        assert (r1.usage.input_tokens, r1.usage.output_tokens) == (20, 5) and r1.cost_usd == 0
        assert r2.cache_hit and len(ollama.seen) == 1
        path, body = ollama.seen[0]
        assert path == "/v1/chat/completions" and body["messages"][0] == {"role": "system", "content": "Be brief"}
        assert body["max_tokens"] == 300

    def test_hard_questions_go_to_the_large_model(self, ollama):
        gw = gateway_from_ollama(host=ollama.host)
        r = gw.generate("Design an end-to-end architecture for a payments ledger. First analyze the trade-offs, "
                        "then implement it in Python, ensuring idempotency. Compare three approaches.")
        assert r.model == "qwen2.5:7b"

    def test_missing_model_fails_over_to_the_next_tier(self):
        o = FakeOllama(installed=("qwen2.5:3b", "qwen2.5:7b"))      # the small model was never pulled
        try:
            r = gateway_from_ollama(host=o.host).generate("How do I reset my password?")
            assert r.model == "qwen2.5:3b" and [a.ok for a in r.attempts] == [False, True]
            assert "not found" in r.attempts[0].error
        finally:
            o.close()

    def test_check_names_the_exact_pull_commands(self):
        o = FakeOllama(installed=("qwen2.5:3b",))
        try:
            with pytest.raises(ConfigError) as e:
                gateway_from_ollama(host=o.host, check=True)
            assert "ollama pull qwen2.5:1.5b" in str(e.value) and "ollama pull qwen2.5:7b" in str(e.value)
        finally:
            o.close()
        with pytest.raises(ConfigError) as e:
            gateway_from_ollama(host="127.0.0.1:9", check=True)
        assert "Is it running" in str(e.value)

    def test_shadow_prices_make_budgets_meaningful(self, ollama):
        from smartgate import Budget, BudgetExceededError
        gw = gateway_from_ollama(host=ollama.host, shadow_prices=True)
        gw.set_budget("t", Budget(limit_usd=0.0, period="total"))
        with pytest.raises(BudgetExceededError):
            gw.generate("hello there", tenant_id="t")

    def test_config_overrides_are_merged(self, ollama):
        gw = gateway_from_ollama(host=ollama.host, preset="healthcare",
                                 overrides={"cache": {"similarity_threshold": 0.99}})
        assert gw.cache.config.similarity_threshold == 0.99 and gw.policy.pii == "redact"


class TestCommandLine:
    def run(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_ollama_check_ok_and_missing(self, ollama):
        code, out, _ = self.run("--ollama-host", ollama.host, "ollama-check")
        assert code == 0 and "All set" in out and out.count("ok") >= 3
        ollama.installed.remove("qwen2.5:7b")
        code, out, _ = self.run("--ollama-host", ollama.host, "ollama-check")
        assert code == 1 and "MISSING" in out and "ollama pull qwen2.5:7b" in out

    def test_ask_through_ollama(self, ollama):
        code, out, err = self.run("--ollama", "--ollama-host", ollama.host, "--preset", "customer_support",
                                  "ask", "How do I reset my password?")
        assert code == 0 and "[qwen2.5:1.5b] hello" in out and "model=qwen2.5:1.5b" in err

    def test_profile_and_custom_models(self, ollama):
        ollama.installed += ["a:1", "b:2", "c:3"]
        code, out, err = self.run("--ollama", "--ollama-host", ollama.host, "--ollama-models", "a:1,b:2,c:3",
                                  "ask", "hello")
        assert code == 0 and "[a:1] hello" in out

    def test_ollama_not_running_is_a_message_not_a_traceback(self):
        code, out, err = self.run("--ollama", "--ollama-host", "127.0.0.1:9", "ask", "hi")
        assert code == 2 and "Is it running" in err and "Traceback" not in err


class TestVerifyWith:
    def test_medium_model_double_checks_semantic_hits(self, ollama):
        gw = gateway_from_ollama(host=ollama.host, preset="customer_support", verify_with="medium")
        gw.generate("How do I reset my password?", tenant_id="t")
        # the fake judge answers "[qwen2.5:3b] hello" (not YES), so a reworded question must be a miss
        r = gw.generate("how can I reset my password", tenant_id="t")
        assert r.cache_hit is False
        judged = [b for _, b in ollama.seen if b["model"] == "qwen2.5:3b"]
        assert judged and "Same question?" in judged[0]["messages"][-1]["content"] and judged[0]["max_tokens"] == 3
        assert gw.stats()["cache"]["verifier_rejections"] == 1

    def test_bad_tier_name_is_rejected(self, ollama):
        with pytest.raises(ConfigError):
            gateway_from_ollama(host=ollama.host, verify_with="huge")

    def test_cli_flag(self, ollama):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["--ollama", "--ollama-host", ollama.host, "--verify-with", "medium", "ask", "hello there"])
        assert code == 0
