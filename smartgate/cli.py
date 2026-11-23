"""Command line: `smartgate serve | analyze | ask | presets`."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .errors import GatewayError
from .config import PRESETS, gateway_from_config, load_config_file


def _ollama_models(args):
    if not args.ollama_models:
        return None
    if args.ollama_models.strip() == "auto":
        return "auto"
    return [m.strip() for m in args.ollama_models.split(",")]


def _gateway(args):
    cfg = load_config_file(args.config) if args.config else {}
    if args.ollama:
        from .providers.ollama import gateway_from_ollama
        preset = args.preset or cfg.pop("preset", "general")
        cfg.pop("models", None)  # the three Ollama tiers define the models
        return gateway_from_ollama(args.ollama_profile, host=args.ollama_host, preset=preset,
                                   models=_ollama_models(args), shadow_prices=args.shadow_prices,
                                   context_tokens=args.ollama_context, verify_with=args.verify_with,
                                   overrides=cfg, check=True)
    if args.preset:
        cfg["preset"] = args.preset
    if args.mock:
        cfg["provider_override"] = "mock"
    return gateway_from_config(cfg)


def _ollama_check(args) -> int:
    from .providers.ollama import OllamaProvider, auto_models, qwen_models
    provider = OllamaProvider(args.ollama_host)
    try:
        installed = provider.installed_models()
    except Exception as exc:
        print(f"Cannot reach Ollama at {provider.host}: {exc}\nStart it with:  ollama serve")
        return 1
    wanted = _ollama_models(args)
    if wanted == "auto":
        wanted = auto_models(provider)
    specs = qwen_models(args.ollama_profile, wanted, args.ollama_context)
    missing = provider.missing_models(specs)
    print(f"Ollama at {provider.host}: {len(installed)} model(s) installed: {', '.join(installed) or 'none'}")
    for spec in specs:
        print(f"  {spec.tier:6} {spec.name:16} {'MISSING' if spec.name in missing else 'ok'}")
    if missing:
        print("\nInstall the missing models:\n" + "\n".join(f"  ollama pull {m}" for m in missing))
        return 1
    print("\nAll set. Try:  python -m smartgate --ollama ask \"How do I reset my password?\"")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="smartgate", description="Semantic cache & model routing gateway")
    p.add_argument("--config", help="JSON or TOML config file")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--mock", action="store_true", help="use the offline mock provider (no API calls)")
    p.add_argument("--ollama", action="store_true",
                   help="use local models on Ollama (default: Qwen2.5 small/medium/large); no API key needed")
    p.add_argument("--ollama-host", help="Ollama address (default: $OLLAMA_HOST or localhost:11434)")
    p.add_argument("--ollama-profile", default="standard", choices=["light", "standard", "strong"],
                   help="Qwen2.5 sizes for small/medium/large: light 0.5b/1.5b/3b, standard 1.5b/3b/7b, strong 3b/7b/14b")
    p.add_argument("--ollama-models", help='1 to 3 comma-separated Ollama tags (small,medium,large), or "auto" to use the Qwen '
                        "models Ollama already has (overrides the profile)")
    p.add_argument("--ollama-context", type=int, default=4096, help="context window Ollama runs with (default 4096)")
    p.add_argument("--verify-with", choices=["small", "medium", "large"],
                   help="with --ollama: this tier's model double-checks cache hits (use medium or large, not the 0.5b model)")
    p.add_argument("--shadow-prices", action="store_true",
                   help="with --ollama: price local models at Claude list prices so budgets and cost reports mean something")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--api-keys", help='JSON map of api_key -> tenant_id, e.g. \'{"k1":"acme"}\'')
    s.add_argument("--admin-keys", default="", help="comma separated admin keys")

    a = sub.add_parser("analyze")
    a.add_argument("prompt")
    q = sub.add_parser("ask")
    q.add_argument("prompt")
    q.add_argument("--tenant", default="default")
    sub.add_parser("presets")
    sub.add_parser("ollama-check", help="check that Ollama is running and has the Qwen models")
    sub.add_parser("selftest", help="verify this installation (standard library only, no network)")

    args = p.parse_args(argv)
    if args.cmd == "selftest":
        from .selftest import run
        return run()
    if args.cmd == "ollama-check":
        return _ollama_check(args)
    if args.cmd == "presets":
        print(json.dumps(PRESETS, indent=2))
        return 0

    try:
        return _run(args)
    except GatewayError as exc:  # configuration, budget, policy, provider failures: a message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args) -> int:
    gw = _gateway(args)
    if args.cmd == "analyze":
        print(json.dumps(gw.analyze(args.prompt), indent=2))
    elif args.cmd == "ask":
        r = gw.generate(args.prompt, tenant_id=args.tenant)
        print(r.text)
        print(f"\n-- model={r.model} cost=${r.cost_usd:.6f} cache_hit={r.cache_hit} "
              f"reason={r.route_reason}", file=sys.stderr)
    elif args.cmd == "serve":
        from .server import serve
        logging.basicConfig(level=logging.INFO)
        keys = json.loads(args.api_keys) if args.api_keys else None
        admin = {k for k in args.admin_keys.split(",") if k}
        httpd = serve(gw, args.host, args.port, keys, admin)
        print(f"smartgate listening on http://{args.host}:{args.port}", file=sys.stderr)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
