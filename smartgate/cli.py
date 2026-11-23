"""Command line: `smartgate serve | analyze | ask | presets`."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import PRESETS, gateway_from_config, load_config_file


def _gateway(args):
    cfg = load_config_file(args.config) if args.config else {}
    if args.preset:
        cfg["preset"] = args.preset
    if args.mock:
        cfg["provider_override"] = "mock"
    return gateway_from_config(cfg)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="smartgate", description="Semantic cache & model routing gateway")
    p.add_argument("--config", help="JSON or TOML config file")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--mock", action="store_true", help="use the offline mock provider (no API calls)")
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
    sub.add_parser("selftest", help="verify this installation (standard library only, no network)")

    args = p.parse_args(argv)
    if args.cmd == "selftest":
        from .selftest import run
        return run()
    if args.cmd == "presets":
        print(json.dumps(PRESETS, indent=2))
        return 0

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
