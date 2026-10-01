"""Phase 0 task 2 probe: load convaiinnovations/laya on CPU at a pinned revision.

Run from anywhere: .venv/bin/python results/phase0/load_probe.py [--cache DIR]
Logs every file under the HF hub cache (or HF_HOME) that Python opens (audit hook), plus the
weights path handed to safetensors (opened from Rust, so invisible to the audit hook), and
reports load time and peak RSS. Networking is off; the cache is only read. Reads no issue data.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# abspath, not resolve(): snapshot entries are symlinks into blobs/ and must keep their names.
HF_HOME = Path(os.path.abspath(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")))
parser = argparse.ArgumentParser()
parser.add_argument("--cache", type=Path, default=Path(os.environ.get("HF_HUB_CACHE", HF_HOME / "hub")),
                    help="HF hub cache to load from (default: %(default)s)")
CACHE = Path(os.path.abspath(parser.parse_args().cache))

# Networking off, CPU autocast off (fp32), cache chosen, set before laya/huggingface_hub import.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = str(CACHE)
os.environ.pop("LAYA_CPU_AMP", None)

REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
opened = []


def rel(path):
    """Path relative to the hub cache (or $HF_HOME), or None if outside both."""
    p = Path(os.path.abspath(path))
    for root, label in ((CACHE, ""), (HF_HOME, "$HF_HOME/")):
        if p.is_relative_to(root):
            return label + str(p.relative_to(root))
    return None


def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = os.fsdecode(args[0])
        if rel(p) is not None:
            opened.append((p, args[1] if len(args) > 1 else None))


sys.addaudithook(audit)

t_import = time.perf_counter()
import laya
import safetensors.torch as st
import torch

t_import = time.perf_counter() - t_import

_orig_load_file = st.load_file


def traced_load_file(filename, *a, **kw):
    opened.append((os.path.abspath(os.fsdecode(filename)), "rb (safetensors, Rust)"))
    return _orig_load_file(filename, *a, **kw)


st.load_file = traced_load_file

t0 = time.perf_counter()
agent = laya.load("convaiinnovations/laya", revision=REVISION, device="cpu")
t_load = time.perf_counter() - t0

params = list(agent.model.parameters())
maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
maxrss_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024

seen = []
for path, mode in opened:
    entry = (rel(path) or path, str(mode))
    if entry not in seen:
        seen.append(entry)

print(json.dumps({
    "laya": laya.__version__,
    "torch": torch.__version__,
    "cache": str(CACHE),
    "agent.revision": agent.revision,
    "device": str(agent.device),
    "amp_enabled": agent.amp_enabled,
    "param_dtypes": sorted({str(p.dtype) for p in params}),
    "n_params": sum(p.numel() for p in params),
    "cfg.max_len": agent.cfg.get("max_len"),
    "cfg.head_max_len": agent.cfg.get("head_max_len"),
    "torch_threads": torch.get_num_threads(),
    "import_s": round(t_import, 2),
    "load_s": round(t_load, 2),
    "peak_rss_MiB": round(maxrss_bytes / 2**20),
    "files_opened": seen,
}, indent=1))
