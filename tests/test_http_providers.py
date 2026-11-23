"""The built-in providers talk real HTTP to a local fake server (no SDK, no network, no API key)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from smartgate import (AllModelsFailedError, AnthropicProvider, Gateway, GatewayRequest, ModelSpec,
                       OpenAICompatibleProvider, ProviderError, ResilienceConfig, RetryPolicy)

SPEC = ModelSpec("claude-haiku-5-5", "anthropic", "small", 0.1, 0.5)


class FakeAPI:
    """Serves scripted replies and records every request it receives."""

    def __init__(self):
        self.script = []   # list of (status, body dict, extra headers, delay)
        self.seen = []     # recorded (path, headers, body)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                outer.seen.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
                status, reply, headers, delay = outer.script.pop(0) if outer.script else (200, outer.default(body), {}, 0)
                if delay:
                    time.sleep(delay)
                raw = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def default(self, body):
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": body.get("model"),
                "content": [{"type": "text", "text": "Paris"}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 12, "output_tokens": 3}}

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def api():
    a = FakeAPI()
    yield a
    a.close()


def request(text="capital of France?", **kw):
    return GatewayRequest(messages=[{"role": "user", "content": text}], **kw)


class TestClaudeOverHttp:
    def test_success_request_shape_and_headers(self, api, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        p = AnthropicProvider(api_key="sk-test", base_url=api.url)
        r = p.complete(ModelSpec("claude-opus-5-5", "anthropic", "large", 4, 20, max_output_tokens=500,
                                 params={"output_config": {"effort": "low"}}),
                       request(system="Be brief", max_tokens=9999))
        assert (r.text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason) == ("Paris", 12, 3, "end_turn")
        path, headers, body = api.seen[0]
        assert path == "/v1/messages"
        assert headers["x-api-key"] == "sk-test" and headers["anthropic-version"] == "2023-06-01"
        assert body["model"] == "claude-opus-5-5" and body["system"] == "Be brief"
        assert body["max_tokens"] == 500                      # clipped to the model's output limit
        assert body["output_config"] == {"effort": "low"}     # ModelSpec.params are merged in
        assert body["messages"] == [{"role": "user", "content": "capital of France?"}]
        assert "temperature" not in body

    def test_credentials_from_environment(self, api, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", api.url)
        AnthropicProvider().complete(SPEC, request())
        assert api.seen[0][1]["x-api-key"] == "sk-env"

    def test_bearer_token_fallback(self, api, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
        AnthropicProvider(base_url=api.url).complete(SPEC, request())
        assert api.seen[0][1]["authorization"] == "Bearer tok" and "x-api-key" not in api.seen[0][1]

    def test_missing_credentials_fail_clearly_without_a_request(self, api, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(base_url=api.url).complete(SPEC, request())
        assert e.value.kind == "auth" and e.value.fallback is False and api.seen == []

    @pytest.mark.parametrize("status,kind,retryable,fallback", [
        (400, "bad_request", False, False), (401, "auth", False, True), (403, "auth", False, True),
        (404, "not_found", False, True), (429, "rate_limit", True, True), (500, "server", True, True),
        (503, "server", True, True), (529, "server", True, True), (408, "timeout", True, True)])
    def test_http_errors_are_classified(self, api, status, kind, retryable, fallback):
        api.script.append((status, {"type": "error", "error": {"type": "x", "message": "boom"}}, {}, 0))
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(api_key="k", base_url=api.url).complete(SPEC, request())
        assert (e.value.kind, e.value.retryable, e.value.fallback, e.value.status_code) == (kind, retryable, fallback, status)
        assert "boom" in str(e.value)

    def test_retry_after_header_is_honored(self, api):
        api.script.append((429, {"error": {"message": "slow down"}}, {"retry-after": "7"}, 0))
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(api_key="k", base_url=api.url).complete(SPEC, request())
        assert e.value.retry_after == 7.0

    def test_refusal_becomes_a_fallback_error(self, api):
        api.script.append((200, {"model": "m", "content": [], "stop_reason": "refusal",
                                 "stop_details": {"type": "refusal", "category": "cyber"},
                                 "usage": {"input_tokens": 1, "output_tokens": 0}}, {}, 0))
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(api_key="k", base_url=api.url).complete(SPEC, request())
        assert e.value.kind == "refusal" and e.value.fallback and "cyber" in str(e.value)

    def test_timeout_and_connection_errors_are_retryable(self, api):
        api.script.append((200, api.default({"model": "m"}), {}, 1.0))
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(api_key="k", base_url=api.url, timeout=0.2).complete(SPEC, request())
        assert e.value.kind == "timeout" and e.value.retryable
        with pytest.raises(ProviderError) as e2:
            AnthropicProvider(api_key="k", base_url="http://127.0.0.1:9", timeout=1).complete(SPEC, request())
        assert e2.value.kind == "connection" and e2.value.retryable

    def test_garbage_response_is_a_retryable_server_error(self, api):
        api.script.append((200, ["not", "an", "object"], {}, 0))
        with pytest.raises(ProviderError) as e:
            AnthropicProvider(api_key="k", base_url=api.url).complete(SPEC, request())
        assert e.value.kind == "server" and e.value.retryable

    def test_multiple_text_blocks_are_joined_and_other_blocks_ignored(self, api):
        api.script.append((200, {"model": "m", "stop_reason": "end_turn", "usage": {},
                                 "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "a"},
                                             {"type": "text", "text": "b"}]}, {}, 0))
        assert AnthropicProvider(api_key="k", base_url=api.url).complete(SPEC, request()).text == "ab"


class TestGatewayEndToEndOverHttp:
    def gateway(self, api, models, **kw):
        return Gateway(models=models, providers={"anthropic": AnthropicProvider(api_key="k", base_url=api.url)},
                       resilience=ResilienceConfig(retry=RetryPolicy(max_attempts=2, base_delay=0)),
                       sleep=lambda s: None, **kw)

    MODELS = [ModelSpec("claude-haiku-5-5", "anthropic", "small", 0.1, 0.5),
              ModelSpec("claude-sonnet-5-5", "anthropic", "medium", 2, 10),
              ModelSpec("claude-opus-5-5", "anthropic", "large", 4, 20)]

    def test_cache_cost_and_real_usage_over_http(self, api):
        gw = self.gateway(api, self.MODELS)
        r1 = gw.generate("How do I reset my password?", tenant_id="acme")
        r2 = gw.generate("how can I reset my password", tenant_id="acme")
        assert r1.text == "Paris" and r1.model == "claude-haiku-5-5"
        assert r1.cost_usd == pytest.approx((12 * 0.1 + 3 * 0.5) / 1e6)
        assert r2.cache_hit and len(api.seen) == 1

    def test_retry_then_failover_across_models_over_http(self, api):
        gw = self.gateway(api, self.MODELS)
        api.script += [(503, {"error": {"message": "down"}}, {}, 0), (503, {"error": {"message": "down"}}, {}, 0)]
        r = gw.generate("How do I reset my password?")
        assert r.model == "claude-sonnet-5-5" and [a.ok for a in r.attempts] == [False, True]
        assert [b["model"] for _, _, b in api.seen] == ["claude-haiku-5-5", "claude-haiku-5-5", "claude-sonnet-5-5"]

    def test_bad_request_is_raised_not_retried(self, api):
        gw = self.gateway(api, self.MODELS)
        api.script.append((400, {"error": {"message": "invalid"}}, {}, 0))
        with pytest.raises(ProviderError):
            gw.generate("hi")
        assert len(api.seen) == 1

    def test_every_model_failing_reports_all_attempts(self, api):
        gw = self.gateway(api, self.MODELS)
        api.script += [(500, {"error": {"message": "x"}}, {}, 0)] * 6
        with pytest.raises(AllModelsFailedError) as e:
            gw.generate("hi")
        assert len(e.value.attempts) == 3


class TestOpenAICompatibleServer:
    def test_chat_completions_round_trip(self, api):
        api.script.append((200, {"model": "llama3.2:3b", "choices": [{"message": {"content": "Hi!"}, "finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 9, "completion_tokens": 2}}, {}, 0))
        spec = ModelSpec("llama3.2:3b", "local", "small", 0, 0, params={"temperature": 0.2})
        r = OpenAICompatibleProvider(api.url + "/v1", api_key="secret").complete(spec, request(system="Be brief"))
        assert (r.text, r.usage.input_tokens, r.usage.output_tokens, r.stop_reason) == ("Hi!", 9, 2, "end_turn")
        path, headers, body = api.seen[0]
        assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer secret"
        assert body["messages"][0] == {"role": "system", "content": "Be brief"} and body["temperature"] == 0.2

    def test_errors_use_the_same_classification(self, api):
        api.script.append((429, {"error": "busy"}, {"retry-after": "2"}, 0))
        with pytest.raises(ProviderError) as e:
            OpenAICompatibleProvider(api.url + "/v1").complete(ModelSpec("m", "l", "small", 0, 0), request())
        assert e.value.kind == "rate_limit" and e.value.retry_after == 2.0


class TestInjectedSdkStyleClient:
    """A client you already have (e.g. a Bedrock/Vertex SDK client) works without smartgate importing the SDK."""

    def test_exceptions_are_mapped_by_shape_not_by_import(self):
        from types import SimpleNamespace

        class RateLimitError(Exception):
            status_code = 429
            response = SimpleNamespace(headers={"retry-after": "4"})
            message = "slow"

        class APITimeoutError(Exception):
            pass

        def boom(exc):
            def create(**kw):
                raise exc
            return AnthropicProvider(client=SimpleNamespace(messages=SimpleNamespace(create=create)))
        with pytest.raises(ProviderError) as e:
            boom(RateLimitError()).complete(SPEC, request())
        assert e.value.kind == "rate_limit" and e.value.retry_after == 4.0
        with pytest.raises(ProviderError) as e2:
            boom(APITimeoutError()).complete(SPEC, request())
        assert e2.value.kind == "timeout"
        with pytest.raises(KeyError):                         # unknown errors are not swallowed
            boom(KeyError("bug")).complete(SPEC, request())


class TestProviderBackedVerifier:
    def test_claude_over_http_acts_as_the_cache_judge(self, api):
        from smartgate import CacheConfig, SemanticCache, provider_verifier
        answers = []

        def judge_reply(body):
            q = body["messages"][0]["content"]
            word = "NO" if "2 months" in q else "YES"
            answers.append(word)
            return {"model": body["model"], "stop_reason": "end_turn", "usage": {},
                    "content": [{"type": "text", "text": word}]}
        api.default = judge_reply
        judge = provider_verifier(AnthropicProvider(api_key="k", base_url=api.url),
                                  ModelSpec("claude-haiku-5-5", "anthropic", "small", 0.1, 0.5))
        cache = SemanticCache(CacheConfig(similarity_threshold=0.5), verifier=judge)
        cache.store_response(request("How can I lose weight in 2 weeks?"), "A", "m")
        assert cache.lookup(request("How can I lose weight in 2 months?")) is None
        assert cache.lookup(request("How can I lose weight in 2 weeks fast?")) is not None
        assert answers == ["NO", "YES"]
        body = api.seen[0][2]
        assert body["max_tokens"] == 3 and "Same question?" in body["messages"][0]["content"]
        assert "O+ vs O-" in body["system"]
