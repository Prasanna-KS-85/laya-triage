"""Phase 0 tasks 3-4: zero-shot wording comparison and CPU latency.

Run from anywhere: .venv/bin/python results/phase0/phase0_eval.py [--out DIR]
Input per issue: raw title + raw body[:6000] (§8.4 step 5 only). One predict() per issue.
Writes per-run prediction CSVs (row index, gold, predicted label, probabilities; no issue text)
and summary.json to --out (default: this directory).
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Networking off and CPU autocast off (fp32), set before laya/huggingface_hub are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)

import laya
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from wordings import WORDINGS

REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
LABELS = ("bug", "feature", "question")
SETTINGS = {"512/192": (512, 192), "1024/256": (1024, 256)}
WARMUP_IDS = (1, 2, 3)  # train rows 1-3; not in the sample (checked below)

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, default=HERE, help="output directory (default: %(default)s)")
OUT = parser.parse_args().out
OUT.mkdir(parents=True, exist_ok=True)

csv.field_size_limit(10**9)
with open(REPO_ROOT / "data/raw/issues_train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ids = [int(x) for x in (REPO_ROOT / "results/phase0_sample_ids.txt").read_text().split()]
assert len(ids) == 100 and not set(WARMUP_IDS) & set(ids)


def question(w):
    return {"issue_type": {"type": "choice", **w}}


def state(i):
    return {"title": rows[i]["title"], "body": rows[i]["body"][:6000]}


def macro_f1(gold, pred):
    f1s = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return sum(f1s.values()) / len(LABELS), f1s


def run_file(wname, sname):
    return OUT / f"predictions_{wname.replace(chr(39), 'p')}_{sname.replace('/', '-')}.csv"


agent = laya.load("convaiinnovations/laya", revision=REV, device="cpu")
assert agent.revision == REV and str(agent.device) == "cpu"
assert agent.amp_enabled is False and agent.dtype == torch.float32
default_threads = torch.get_num_threads()
gold = [rows[i]["label"] for i in ids]

# ---- Task 3: 3 wordings x 2 settings ----
runs, choices_by_run = {}, {}
for sname, (ml, hml) in SETTINGS.items():
    for wname, w in WORDINGS.items():
        q = question(w)
        preds, truncated, confs = [], [], []
        for i in ids:
            out = agent.predict(state(i), q, max_len=ml, head_max_len=hml)
            a = out["answers"]["issue_type"]
            preds.append((i, rows[i]["label"], a["choice"], a["probabilities"]))
            truncated.append(out["usage"]["truncated"])
            confs.append(a["answer_confidence"])
        with open(run_file(wname, sname), "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["row_id", "gold", "pred", *(f"p_{c}" for c in LABELS)])
            for i, g, p, probs in preds:
                wr.writerow([i, g, p, *(probs[c] for c in LABELS)])
        pred = [p for _, _, p, _ in preds]
        mf1, f1s = macro_f1(gold, pred)
        key = f"{wname}@{sname}"
        runs[key] = {
            "macro_f1": round(mf1, 4),
            "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
            "f1_per_class": {c: round(v, 4) for c, v in f1s.items()},
            "predicted_counts": {c: pred.count(c) for c in LABELS},
            "share_truncated": round(sum(truncated) / len(truncated), 4),
            "mean_answer_confidence": round(statistics.mean(confs), 4),
        }
        choices_by_run[key] = pred
        print(key, json.dumps(runs[key]), flush=True)

# ---- Task 4: latency, W0, model preloaded ----
q0 = question(WORDINGS["W0"])
latency = {}
for sname in ("1024/256", "512/192"):
    ml, hml = SETTINGS[sname]
    for threads in (default_threads, 2):
        torch.set_num_threads(threads)
        for i in WARMUP_IDS:
            agent.predict(state(i), q0, max_len=ml, head_max_len=hml)
        ms, choices = [], []
        for i in ids:
            s = state(i)
            t = time.perf_counter()
            out = agent.predict(s, q0, max_len=ml, head_max_len=hml)
            ms.append((time.perf_counter() - t) * 1000)
            choices.append(out["answers"]["issue_type"]["choice"])
        ref = choices_by_run[f"W0@{sname}"]
        key = f"W0@{sname} threads={torch.get_num_threads()}"
        latency[key] = {
            "p50_ms": round(float(np.percentile(ms, 50)), 1),
            "p95_ms": round(float(np.percentile(ms, 95)), 1),
            "mean_ms": round(statistics.mean(ms), 1),
            "max_ms": round(max(ms), 1),
            "n": len(ms),
            "same_choices_as_accuracy_run": sum(a == b for a, b in zip(choices, ref)),
        }
        print(key, json.dumps(latency[key]), flush=True)
torch.set_num_threads(default_threads)

meta = {"laya": laya.__version__, "torch": torch.__version__, "revision": agent.revision,
        "device": str(agent.device), "dtype": str(agent.dtype), "amp_enabled": agent.amp_enabled,
        "default_threads": default_threads, "python": sys.version.split()[0],
        "n_issues": len(ids), "gold_counts": {c: gold.count(c) for c in LABELS}}
with open(OUT / "summary.json", "w") as f:
    json.dump({"meta": meta, "runs": runs, "latency": latency}, f, indent=1)
print("META", json.dumps(meta))
