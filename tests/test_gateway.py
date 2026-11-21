import threading
from types import SimpleNamespace

import pytest

from smartgate import (AllModelsFailedError, Budget, BudgetExceededError, CallableProvider,
                       MockProvider, PolicyConfig, PolicyViolationError, ProviderError,
                       gateway_from_preset)
from smartgate.client import GatewayClient, GatewayHTTPError
from smartgate.server import serve


class TestGateway:
    def test_end_to_end_cache_saves_cost(self, make_gateway, provider):
        gw = make_gateway()
        r1 = gw.generate("How do I reset my password?", tenant_id="acme")
        assert not r1.cache_hit and r1.cost_usd > 0 and r1.model == "small-a"
        r2 = gw.generate("how can I reset my password", tenant_id="acme")
        assert r2.cache_hit and r2.cost_usd == 0 and r2.saved_usd == pytest.approx(r1.cost_usd)
        assert len(provider.calls) == 1
        s = gw.stats()
        assert s["cache"]["hits"] == 1 and s["cost"]["saved_usd"] > 0

    def test_use_cache_false_bypasses(self, make_gateway, provider):
        gw = make_gateway()
        gw.generate("hello there")
        gw.generate("hello there", use_cache=False)
        assert len(provider.calls) == 2

    def test_time_sensitive_not_cached(self, make_gateway, provider):
        gw = make_gateway()
        gw.generate("What's the weather today?")
        gw.generate("What's the weather today?")
        assert len(provider.calls) == 2

    def test_fallback_on_provider_failure(self, make_gateway):
        p = MockProvider(fail_models={"small-a": "server"})
        gw = make_gateway(provider=p)
        events = []
        gw.on("fallback", lambda e, d: events.append(d))
        r = gw.generate("hi")
        assert r.model == "medium-a"
        assert [a.ok for a in r.attempts] == [False, True]
        assert p.calls.count("small-a") == 2  # retried, then failed over
        assert events and "fell back" in r.route_reason

    def test_retry_recovers_same_model(self, make_gateway):
        p = MockProvider(fail_times={"small-a": 1})
        r = make_gateway(provider=p).generate("hi")
        assert r.model == "small-a" and p.calls == ["small-a", "small-a"]

    def test_non_fallback_error_propagates(self, make_gateway):
        p = MockProvider(fail_models={"small-a": "bad_request"})
        with pytest.raises(ProviderError):
            make_gateway(provider=p).generate("hi")
        assert p.calls == ["small-a"]

    def test_all_models_failed(self, make_gateway):
        p = MockProvider(fail_models={m: "server" for m in ("small-a", "medium-a", "large-a")})
        with pytest.raises(AllModelsFailedError):
            make_gateway(provider=p).generate("hi")

    def test_circuit_breaker_skips_dead_model(self, make_gateway):
        from smartgate import ResilienceConfig, RetryPolicy
        p = MockProvider(fail_models={"small-a": "server"})
        gw = make_gateway(provider=p, resilience=ResilienceConfig(
            retry=RetryPolicy(max_attempts=1), breaker_failure_threshold=2))
        for i in range(3):
            gw.generate(f"question number {i} unique", use_cache=False)
        assert p.calls.count("small-a") == 2  # third request skipped the open circuit
        assert gw.stats()["circuit_breakers"]["mock:small-a"] == "open"

    def test_budget_blocks_and_downgrades(self, make_gateway):
        gw = make_gateway()
        gw.set_budget("t", Budget(limit_usd=0.0001))
        with pytest.raises(BudgetExceededError):
            gw.generate("x " * 2000, tenant_id="t", max_tokens=4000)

    def test_budget_picks_cheaper_model(self, make_gateway):
        gw = make_gateway()
        gw.set_budget("t", Budget(limit_usd=100, max_cost_per_request=0.001))
        r = gw.generate("Hard question", tier="large", tenant_id="t", max_tokens=1000)
        assert r.model == "small-a" and "budget" in r.route_reason

    def test_pii_redact(self, make_gateway):
        seen = []
        p = CallableProvider("mock", lambda m, r: seen.append(r.last_user_message) or "ok")
        gw = make_gateway(provider=p, policy=PolicyConfig(pii="redact"))
        gw.generate("My email is bob@corp.com, help me")
        assert "bob@corp.com" not in seen[0] and "[REDACTED_EMAIL]" in seen[0]

    def test_pii_block(self, make_gateway):
        gw = make_gateway(policy=PolicyConfig(pii="block"))
        with pytest.raises(PolicyViolationError):
            gw.generate("SSN 123-45-6789")

    def test_blocked_terms(self, make_gateway):
        with pytest.raises(PolicyViolationError):
            make_gateway(policy=PolicyConfig(blocked_terms=["secret project"])).generate("Tell me about Secret Project")

    def test_async(self, make_gateway):
        import asyncio
        assert asyncio.run(make_gateway().agenerate("hi")).text

    def test_hook_errors_do_not_break_requests(self, make_gateway):
        gw = make_gateway().on("*", lambda e, d: 1 / 0)
        assert gw.generate("hi").text


class TestPresets:
    @pytest.mark.parametrize("preset", ["general", "healthcare", "finance", "ecommerce",
                                        "customer_support", "legal", "education", "saas_devtools"])
    def test_all_presets_build_and_run(self, preset):
        gw = gateway_from_preset(preset, providers={"anthropic": MockProvider()})
        assert gw.generate("Summarize the key points of our return policy").text

    def test_healthcare_rules(self):
        gw = gateway_from_preset("healthcare", providers={"anthropic": MockProvider()})
        assert gw.analyze("What is the max dosage of ibuprofen?")["model"] == "claude-opus-5-5"
        assert gw.analyze("hello")["tier"] == "medium"  # min_tier floor
        r = gw.generate("Patient jane@x.com reports a headache")
        assert "jane@x.com" not in r.text

    def test_declarative_rule_from_config(self):
        from smartgate import gateway_from_config
        gw = gateway_from_config({"provider_override": "mock",
                                  "rules": [{"name": "gold", "tenants": ["gold"], "model": "claude-opus-5-5"}]})
        assert gw.analyze("hi", tenant_id="gold")["model"] == "claude-opus-5-5"
        assert gw.analyze("hi", tenant_id="free")["model"] == "claude-haiku-5-5"


class TestAnthropicProvider:
    @pytest.fixture(autouse=True)
    def _needs_sdk(self):
        pytest.importorskip("anthropic", reason="pip install smartgate[anthropic]")

    def _client(self, stop_reason="end_turn"):
        def create(**kw):
            self.kwargs = kw
            return SimpleNamespace(
                model=kw["model"], stop_reason=stop_reason,
                stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
                content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="Paris")],
                usage=SimpleNamespace(input_tokens=12, output_tokens=3))
        return SimpleNamespace(messages=SimpleNamespace(create=create))

    def test_success_maps_response(self):
        from smartgate.providers.anthropic_provider import AnthropicProvider
        gw = gateway_from_preset("general", providers={"anthropic": AnthropicProvider(client=self._client())})
        r = gw.generate("capital of France?", system="Be brief")
        assert r.text == "Paris" and r.usage.input_tokens == 12
        assert self.kwargs["system"] == "Be brief" and "temperature" not in self.kwargs
        assert r.cost_usd == pytest.approx((12 * 0.10 + 3 * 0.50) / 1e6)

    def test_refusal_triggers_fallback(self):
        from smartgate.providers.anthropic_provider import AnthropicProvider
        refusing = AnthropicProvider(client=self._client("refusal"))
        gw = gateway_from_preset("general", providers={"anthropic": refusing})
        with pytest.raises(AllModelsFailedError) as ei:
            gw.generate("hi")
        assert all("refusal" in a.error for a in ei.value.attempts) and len(ei.value.attempts) == 3


class TestServer:
    @pytest.fixture
    def server(self):
        gw = gateway_from_preset("customer_support", providers={"anthropic": MockProvider()})
        gw.set_budget("broke", Budget(limit_usd=0.0))
        httpd = serve(gw, port=0, api_keys={"k-acme": "acme", "k-broke": "broke", "k-admin": "ops"},
                      admin_keys={"k-admin"})
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()

    def test_http_flow(self, server):
        c = GatewayClient(server, api_key="k-acme")
        r1 = c.generate("Where is my order?")
        r2 = c.generate(messages=[{"role": "user", "content": "where is my order"}])
        assert not r1["cache_hit"] and r2["cache_hit"]
        assert c.analyze("I want a refund")["tier"] == "medium"
        assert c.costs()["requests"] == 2
        assert c.stats()["cache"]["hits"] == 1
        assert c.invalidate_cache()["removed"] == 1

    def test_http_errors(self, server):
        with pytest.raises(GatewayHTTPError) as e:
            GatewayClient(server, api_key="nope").generate("hi")
        assert e.value.status == 401
        with pytest.raises(GatewayHTTPError) as e:
            GatewayClient(server, api_key="k-broke").generate("hi")
        assert e.value.status == 402
        with pytest.raises(GatewayHTTPError) as e:
            GatewayClient(server, api_key="k-acme")._call("POST", "/v1/generate", {})
        assert e.value.status == 400
