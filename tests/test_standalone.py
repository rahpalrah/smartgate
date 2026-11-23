"""smartgate must run with the standard library alone: no SDKs, no NumPy, nothing from site-packages."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOT = f"import sys; sys.path.insert(0, {ROOT!r}); "


def run_isolated(code, timeout=300):
    # -S: do not import site, so nothing installed in site-packages (pytest, numpy, ...) is visible
    return subprocess.run([sys.executable, "-S", "-c", BOOT + code], capture_output=True, text=True,
                          timeout=timeout, cwd=ROOT)


def test_selftest_passes_with_site_packages_disabled():
    r = run_isolated("from smartgate.cli import main; sys.exit(main(['selftest']))")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "PASSED" in r.stdout


def test_no_third_party_module_is_ever_loaded():
    r = run_isolated(
        "import smartgate, smartgate.server, smartgate.client, smartgate.cli, smartgate.config, smartgate.selftest\n"
        "bad = sorted({m.split('.')[0] for m in sys.modules if m.split('.')[0] not in sys.stdlib_module_names "
        "and m.split('.')[0] != 'smartgate' and not m.startswith('_')})\n"
        "print('BAD:', bad)")
    assert r.returncode == 0, r.stderr
    assert "BAD: []" in r.stdout, r.stdout


def test_isolated_environment_really_lacks_third_party_packages():
    for module in ("numpy", "anthropic", "phonenumbers", "yaml"):
        r = run_isolated(f"import {module}")
        assert r.returncode != 0, f"{module} unexpectedly importable; the isolation check would prove nothing"


def test_package_metadata_declares_no_dependencies():
    import tomllib
    meta = tomllib.load(open(os.path.join(ROOT, "pyproject.toml"), "rb"))["project"]
    assert meta["dependencies"] == []
    assert set(meta.get("optional-dependencies", {})) <= {"dev"}   # tooling for the test suite only
