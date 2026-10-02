"""Run a Laya checkpoint on val or test on CPU fp32 (PROJECT_SPEC.md §9.4, §10.4, §13 Phase 3).

Usage (from the repo root):
  .venv/bin/python eval/run_eval.py --model-repo Prasanna85/laya-issue-triage --revision <sha> \
      --split val --expect-max-len 1024 --expect-head-max-len 256 --tag M1_<run_id> [--out DIR] [--download]
      [--threads N]

- Runs offline (HF_HUB_OFFLINE=1): the revision must be in the local HF cache. --download first
  fetches it with laya.load in a separate, online subprocess (a private repo needs HF_TOKEN in the
  environment or a cached login); the evaluation itself still runs offline.
- CPU, fp32, autocast off, one predict() per issue with the config's own max_len / head_max_len,
  as eval/baselines.py B2. The loaded config must have the expected max_len / head_max_len, every
  row's question must equal ISSUE_TYPE_QUESTION, and the input file must have its pinned SHA-256.
- Uncalibrated (T=1) probabilities come from the raw option logits (laya.calibrate.records_from_labeled,
  a second forward pass per issue). For every issue the run checks that softmax(logits / T), with
  the temperature laya applies to a 3-option choice question, reproduces laya's returned
  probabilities to within their 4-decimal rounding and gives the same label.

Writes <out>/predictions_<split>_<tag>.csv (id, repo, gold, pred, p_<label>, answer_confidence,
p_<label>_T1; no issue text) and a .meta.json beside it. An existing predictions file is never
overwritten. --split test is refused unless config/triage.default.yml is committed and unmodified,
the tag has no test predictions yet, and test.jsonl has a pinned SHA-256.
"""

import argparse
import os
import sys

# Networking off and CPU autocast off (fp32), set before laya/huggingface_hub are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)

import csv
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from baselines import DATA, QID, load_split

from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "phase3"
THRESHOLD_CONFIG = ROOT / "config" / "triage.default.yml"
# SHA-256 of the processed inputs (results/phase3/items_sha256.json for val). test.jsonl is never
# hashed by tooling here; the owner pins it before the single test run.
INPUT_SHA256 = {
    "val": "330bab114a6ff83cf508ef948603b53a6425f012f4e9df14e69490cc02d60d6b",
    "test": "ede236c9cc1875a41b394119bf49f203877b4249e8dd413d373545640c8794b2",
}
WARMUP = 3
ROUNDING_TOL = 5e-5 + 1e-9  # laya rounds probabilities to 4 decimals
CSV_COLUMNS = ["id", "repo", "gold", "pred", *(f"p_{c}" for c in LABELS), "answer_confidence",
               *(f"p_{c}_T1" for c in LABELS)]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def check_test_allowed(out_csv):
    rel = str(THRESHOLD_CONFIG.relative_to(ROOT))
    if not THRESHOLD_CONFIG.exists():
        raise SystemExit(f"refusing to run test: {rel} does not exist (thresholds come first, §10.4)")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True, check=False)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, check=False)
    if tracked.returncode or clean.returncode:
        raise SystemExit(f"refusing to run test: {rel} must be committed and unmodified")
    if out_csv.exists():
        raise SystemExit(f"refusing to re-run test: {out_csv.name} exists")
    if INPUT_SHA256["test"] is None:
        raise SystemExit("refusing to run test: pin the SHA-256 of data/processed/test.jsonl in INPUT_SHA256 first")


def download(repo, revision):
    """Fetch the revision into the HF cache from a child process with networking on."""
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    code = "import sys, laya; laya.load(sys.argv[1], revision=sys.argv[2], device='cpu')"
    subprocess.run([sys.executable, "-c", code, repo, revision], env=env, check=True)


def load_agent(repo, revision, expect_max_len, expect_head_max_len):
    import laya
    import torch

    agent = laya.load(repo, revision=revision, device="cpu")
    assert agent.revision == revision, (agent.revision, revision)
    assert str(agent.device) == "cpu" and agent.amp_enabled is False and agent.dtype == torch.float32
    got = (agent.cfg.get("max_len"), agent.cfg.get("head_max_len"))
    if got != (expect_max_len, expect_head_max_len):
        raise SystemExit(f"config max_len/head_max_len {got} != expected {(expect_max_len, expect_head_max_len)}")
    return agent


def choice_temperature(agent):
    """The temperature laya divides a 3-option choice question's logits by (agent._decode_answers)."""
    from laya.common import QTYPES, temp_bucket

    qt = QTYPES["choice"]
    return float(agent.temperature_by_options.get(temp_bucket(qt, len(LABELS)), agent.temperature[qt]))


def raw_logits(agent, state):
    import torch
    from laya.calibrate import records_from_labeled

    dummy_target = {QID: [0.0] * len(LABELS)}  # targets are only copied into the record
    with torch.no_grad():  # laya's predict paths run under no_grad; records_from_labeled does not set it
        ((qtype, logits, _target, k),) = records_from_labeled(agent, [(state, ISSUE_TYPE_QUESTION, dummy_target)])
    assert qtype == 0 and k == len(LABELS), (qtype, k)
    return np.asarray(logits, dtype=np.float64)


def versions():
    import huggingface_hub
    import laya
    import tokenizers
    import torch
    import transformers
    return {"python": sys.version.split()[0], "laya": laya.__version__, "torch": torch.__version__,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__}


def run(args):
    out_csv = args.out / f"predictions_{args.split}_{args.tag}.csv"
    if args.split == "test":
        check_test_allowed(out_csv)
    if out_csv.exists() or out_csv.with_suffix(".meta.json").exists():
        raise SystemExit(f"refusing to overwrite {out_csv}")
    path = DATA / f"{args.split}.jsonl"
    digest = sha256_file(path)
    if digest != INPUT_SHA256[args.split]:
        raise SystemExit(f"{path.name}: sha256 {digest} != pinned {INPUT_SHA256[args.split]}")
    issues = load_split(args.split)  # asserts every row's question equals ISSUE_TYPE_QUESTION
    if args.download:
        download(args.model_repo, args.revision)

    import torch

    if args.threads is not None:
        torch.set_num_threads(args.threads)
    t_start = time.perf_counter()
    agent = load_agent(args.model_repo, args.revision, args.expect_max_len, args.expect_head_max_len)
    load_s = time.perf_counter() - t_start
    t_choice = choice_temperature(agent)
    for i in issues[:WARMUP]:
        agent.predict(i.state, ISSUE_TYPE_QUESTION)

    rows, ms, truncated, max_dev = [], [], 0, 0.0
    t_loop = time.perf_counter()
    for n, i in enumerate(issues, 1):
        t = time.perf_counter()
        out = agent.predict(i.state, ISSUE_TYPE_QUESTION)
        ms.append((time.perf_counter() - t) * 1000)
        a = out["answers"][QID]
        p = {c: float(a["probabilities"][c]) for c in LABELS}
        conf = float(a["answer_confidence"])
        assert conf == p[a["choice"]], (i.id, conf, p)
        truncated += bool(out["usage"]["truncated"])

        z = raw_logits(agent, i.state)
        p_t1 = softmax(z)
        p_cal = softmax(z / t_choice)
        dev = max(abs(p_cal[k] - p[c]) for k, c in enumerate(LABELS))
        if dev > ROUNDING_TOL or LABELS[int(np.argmax(z))] != a["choice"]:
            raise SystemExit(f"{i.id}: logits / T do not reproduce laya's output (max dev {dev:.2e})")
        max_dev = max(max_dev, dev)
        rows.append([i.id, i.repo, i.label, a["choice"], *(repr(p[c]) for c in LABELS), repr(conf),
                     *(repr(float(v)) for v in p_t1)])
        if n % 50 == 0 or n == len(issues):
            print(f"{args.tag} {args.split}: {n}/{len(issues)} issues, {time.perf_counter() - t_loop:.0f} s", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "x", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)
    meta = {
        "tag": args.tag, "split": args.split, "model": args.model_repo, "revision": agent.revision,
        "max_len": agent.cfg["max_len"], "head_max_len": agent.cfg["head_max_len"],
        "question": "ISSUE_TYPE_QUESTION", "input": {"file": path.name, "sha256": digest, "rows": len(issues)},
        "device": str(agent.device), "dtype": str(agent.dtype), "amp_enabled": agent.amp_enabled,
        "torch_threads": torch.get_num_threads(),
        "config_temperature": agent.cfg.get("temperature"),
        "config_temperature_by_options": agent.cfg.get("temperature_by_options"),
        "applied_choice_temperature_3_options": t_choice,
        "t1_method": "softmax of raw option logits from laya.calibrate.records_from_labeled (second forward pass)",
        "t1_check_max_abs_dev_vs_returned": max_dev,
        "share_truncated": truncated / len(issues),
        "latency_ms": {"timed": f"agent.predict per issue, model preloaded, {WARMUP} warm-up calls",
                       "p50": float(np.percentile(ms, 50)), "p95": float(np.percentile(ms, 95)),
                       "mean": float(np.mean(ms)), "max": float(np.max(ms)), "n": len(ms)},
        "load_s": round(load_s, 1), "wall_s": round(time.perf_counter() - t_start, 1),
        "probabilities": "p_<label>: as returned by laya (rounded to 4 decimals); p_<label>_T1: unrounded",
        "versions": versions(),
    }
    out_csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out_csv}; latency p50 {meta['latency_ms']['p50']:.0f} ms, p95 {meta['latency_ms']['p95']:.0f} ms; "
          f"T={t_choice:.4f}; max |softmax(z/T) - p| {max_dev:.1e}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--revision", required=True, help="40-hex commit SHA (never a branch)")
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--expect-max-len", type=int, required=True)
    ap.add_argument("--expect-head-max-len", type=int, required=True)
    ap.add_argument("--tag", required=True, help="names the output files, e.g. M1_run01")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--download", action="store_true", help="fetch the revision first (child process, online)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: library default)")
    args = ap.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be >= 1")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        ap.error("--revision must be a 40-character commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.tag):
        ap.error("--tag may contain only letters, digits, '_', '.', '-'")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
