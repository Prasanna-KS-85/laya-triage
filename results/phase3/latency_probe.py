"""Cold load, peak RSS and per-issue CPU latency of a checkpoint (Phase 0 method, PROJECT_SPEC.md §6.2).

Run from anywhere (offline; the revision must be in the local HF cache):
  /usr/bin/time -l .venv/bin/python results/phase3/latency_probe.py --repo R --revision SHA \
      --expect-max-len 1024 --expect-head-max-len 256 --threads 2 --n 100 --out FILE.json
One fresh process: times `import laya` and `laya.load` (default threads), reads ru_maxrss right
after the load (as results/phase0/load_probe.py), then sets --threads, makes 3 warm-up calls and
times one predict() per issue on the first --n val issues. Writes ids-free numbers only.
"""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

# Networking off and CPU autocast off (fp32), set before laya/huggingface_hub are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

t_import = time.perf_counter()
import laya
import numpy as np
import run_eval
import torch
from baselines import load_split

from laya_triage.questions import ISSUE_TYPE_QUESTION

t_import = time.perf_counter() - t_import


def peak_rss_mib():
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return round((maxrss if sys.platform == "darwin" else maxrss * 1024) / 2**20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--expect-max-len", type=int, required=True)
    ap.add_argument("--expect-head-max-len", type=int, required=True)
    ap.add_argument("--threads", type=int, default=None, help="threads for the timed calls (default: library default)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    default_threads = torch.get_num_threads()
    t0 = time.perf_counter()
    agent = run_eval.load_agent(args.repo, args.revision, args.expect_max_len, args.expect_head_max_len)
    load_s = time.perf_counter() - t0
    rss_after_load = peak_rss_mib()

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    issues = load_split("val")[: args.n]
    for i in issues[: run_eval.WARMUP]:
        agent.predict(i.state, ISSUE_TYPE_QUESTION)
    ms = []
    for i in issues:
        t = time.perf_counter()
        agent.predict(i.state, ISSUE_TYPE_QUESTION)
        ms.append((time.perf_counter() - t) * 1000)

    out = {
        "model": args.repo, "revision": agent.revision, "max_len": agent.cfg["max_len"],
        "head_max_len": agent.cfg["head_max_len"], "device": str(agent.device), "dtype": str(agent.dtype),
        "amp_enabled": agent.amp_enabled, "laya": laya.__version__, "torch": torch.__version__,
        "import_s": round(t_import, 2), "load_s": round(load_s, 2), "load_threads": default_threads,
        "peak_rss_after_load_MiB": rss_after_load, "peak_rss_end_MiB": peak_rss_mib(),
        "latency_ms": {"issues": f"first {len(issues)} val.jsonl rows, one predict() each, "
                                 f"{run_eval.WARMUP} warm-up calls, model preloaded",
                       "threads": torch.get_num_threads(), "p50": float(np.percentile(ms, 50)),
                       "p95": float(np.percentile(ms, 95)), "mean": float(np.mean(ms)), "max": float(np.max(ms)),
                       "n": len(ms)},
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
