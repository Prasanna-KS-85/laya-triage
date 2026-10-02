"""constraints-linux.txt, the Linux lock used by action.yml and ci.yml (PROJECT_SPEC.md §7.6, §7.7). No network."""
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
LINES = [ln for ln in (ROOT / "constraints-linux.txt").read_text().splitlines() if ln.strip() and not ln.startswith("#")]
PIN = re.compile(r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[A-Za-z0-9.+!_-]+)")


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()  # PEP 503


def pins():
    return {m["name"]: m["version"] for m in map(PIN.fullmatch, LINES)}


def test_every_line_is_name_equals_version():
    assert LINES and all(PIN.fullmatch(ln) for ln in LINES), [ln for ln in LINES if not PIN.fullmatch(ln)]


def test_names_are_normalised_unique_and_sorted():
    names = [PIN.fullmatch(ln)["name"] for ln in LINES]
    assert all(n == normalise(n) for n in names)
    assert len(set(names)) == len(names)
    assert names == sorted(names)


def test_runtime_dependencies_match_pyproject():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    wanted = {normalise(d.split("==")[0]): d.split("==")[1] for d in deps}
    assert set(wanted) == {"laya", "pyyaml"}
    p = pins()
    for name, version in wanted.items():
        assert p[name] == version, (name, p.get(name), version)


def test_torch_is_pinned_without_a_local_label():
    assert pins()["torch"] == "2.14.1"
    assert all("+" not in v for v in pins().values())


def test_no_cuda_packages_and_not_the_package_itself():
    names = set(pins())
    assert not {n for n in names if n.startswith("nvidia") or "triton" in n}
    assert "laya-triage" not in names and not any(ln.startswith("-e") or "@" in ln for ln in LINES)


def test_direct_pins_are_present():
    p = pins()
    for name in ("pip", "setuptools", "torch", "laya", "transformers", "tokenizers", "huggingface-hub", "hf-xet",
                 "safetensors", "numpy", "pyyaml"):
        assert name in p, name
