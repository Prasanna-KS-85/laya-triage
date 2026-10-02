"""Refit the choice temperature of R1 on val (Phase 3c-B Stop A2; PROJECT_SPEC.md §9.3). Weights unchanged.

Run from anywhere: .venv/bin/python results/phase3/refit_temperature.py
Offline: loads Prasanna85/laya-issue-triage @ R1 from the local HF cache (CPU fp32), collects the raw
T=1 option logits of the 300 val issues with laya.calibrate.records_from_labeled (as eval/run_eval.py
does; targets = one-hot gold), and fits the type-level choice temperature with laya's own fitter
laya.calibrate.fit_temperature_map (NLL + LBFGS on log T, clamped to [TEMP_MIN, TEMP_MAX]; the port of
the notebook's fitter). Cross-checks it with a 0.01 grid of val NLL, compares NLL / ECE / Brier /
accuracy at the notebook T and at the refit T (in-sample: fitted on val), and writes
results/phase3/refit/rl_agent_config.json (the snapshot's config with only temperature[0] replaced)
and results/phase3/refit/refit_summary.json. Numbers only; the HF cache is only read.
"""
import csv
import difflib
import json
import os
import sys
from pathlib import Path

# Networking off and CPU autocast off (fp32), set before laya/huggingface_hub are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import metrics
import numpy as np
import run_eval
import torch
from baselines import QID, load_split
from huggingface_hub import snapshot_download
from laya.calibrate import fit_temperature_map, records_from_labeled
from laya.common import TEMP_MAX, TEMP_MIN, ece_score

from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

REPO, REV = "Prasanna85/laya-issue-triage", "a704b3eadd185f1fa028576cfa50605b6eada4a4"
R1_CSV = ROOT / "results/phase3/val_R1/predictions_val_M1_R1.csv"
OUT = ROOT / "results/phase3/refit"
GRID_STEP = 0.01
AGREE_TOL = 0.02


def softmax_rows(z):
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def scores(logits, y, t):
    p = softmax_rows(logits / t)
    pred = p.argmax(axis=1)
    onehot = np.eye(len(LABELS))[y]
    return {"T": t, "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
            "ece": ece_score(p.max(axis=1), (pred == y).astype(float), bins=metrics.ECE_BINS),
            "brier": float(((p - onehot) ** 2).sum(axis=1).mean()), "accuracy": float((pred == y).mean())}


def main():
    issues = load_split("val")
    y = np.array([LABELS.index(i.label) for i in issues])
    agent = run_eval.load_agent(REPO, REV, 1024, 256)
    with torch.no_grad():
        records = records_from_labeled(
            agent, [(i.state, ISSUE_TYPE_QUESTION, {QID: np.eye(len(LABELS))[LABELS.index(i.label)]}) for i in issues])
    assert len(records) == len(issues) and all(q == 0 and k == 3 for q, _, _, k in records)
    logits = np.array([np.asarray(z, dtype=np.float64) for _, z, _, _ in records])

    # Consistency with the saved R1 predictions: same T=1 probabilities, same labels.
    with open(R1_CSV, newline="") as f:
        r1 = list(csv.DictReader(f))
    assert [r["id"] for r in r1] == [i.id for i in issues]
    p_t1_saved = np.array([[float(r[f"p_{c}_T1"]) for c in LABELS] for r in r1])
    t1_dev = float(np.abs(softmax_rows(logits) - p_t1_saved).max())
    r1_pred = [r["pred"] for r in r1]

    fit = fit_temperature_map(records)
    t_val_raw = float(fit["temperature"][0])
    t_val = round(t_val_raw, 4)
    grid = np.round(np.arange(TEMP_MIN, TEMP_MAX + 1e-9, GRID_STEP), 2)
    nll = [scores(logits, y, t)["nll"] for t in grid]
    t_grid = float(grid[int(np.argmin(nll))])
    assert abs(t_val - t_grid) <= AGREE_TOL, (t_val, t_grid)

    t_nb = float(agent.cfg["temperature"][0])
    pred_val = [LABELS[i] for i in softmax_rows(logits / t_val).argmax(axis=1)]
    same_labels = sum(a == b for a, b in zip(pred_val, r1_pred))
    table = [scores(logits, y, t_nb), scores(logits, y, t_val)]

    # Config: the snapshot's file with only temperature[0] replaced, written the way train_ddp.py wrote it.
    # Offline: resolves the cached snapshot path; nothing is downloaded or modified.
    snap = Path(snapshot_download(REPO, revision=REV, allow_patterns=["rl_agent_config.json"]))
    original = (snap / "rl_agent_config.json").read_text()
    cfg = json.loads(original)
    assert json.dumps(cfg, indent=2) == original, "snapshot config is not json.dump(indent=2) output"
    cfg["temperature"][0] = t_val
    new = json.dumps(cfg, indent=2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rl_agent_config.json").write_text(new)
    diff = "".join(difflib.unified_diff(original.splitlines(True), new.splitlines(True),
                                        "snapshot/rl_agent_config.json", "refit/rl_agent_config.json"))

    summary = {
        "model": REPO, "revision": REV, "n_val": len(issues), "note": "in-sample: fitted on val",
        "fitter": "laya.calibrate.fit_temperature_map (type-level fit_one_temperature: NLL + LBFGS on log T, "
                  f"clamped to [{TEMP_MIN}, {TEMP_MAX}]); records from laya.calibrate.records_from_labeled",
        "fit_output": {"temperature": fit["temperature"], "temperature_by_options": fit["temperature_by_options"],
                       "n_by_bucket": fit["n_by_bucket"]},
        "T_notebook": t_nb, "T_val_raw": t_val_raw, "T_val": t_val,
        "grid_check": {"grid": f"{TEMP_MIN}..{TEMP_MAX} step {GRID_STEP}", "T_min_nll": t_grid,
                       "agree_within": AGREE_TOL, "abs_diff": round(abs(t_val - t_grid), 4)},
        "labels_equal_R1": f"{same_labels}/{len(issues)}",
        "t1_logits_vs_saved_R1_p_T1_max_abs_dev": t1_dev,
        "table": table, "config_diff": diff,
    }
    (OUT / "refit_summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"fitter: {summary['fitter']}")
    print(f"T_val = {t_val:.4f} (raw {t_val_raw:.6f}); grid argmin {t_grid:.2f}; |diff| {abs(t_val - t_grid):.4f} "
          f"(<= {AGREE_TOL}); fit output {fit['temperature']}, buckets {fit['temperature_by_options']}")
    print(f"T=1 logits vs saved R1 p_T1: max |dev| {t1_dev:.2e}; argmax at T_val equals R1 on {same_labels}/{len(issues)}")
    print("\nval, in-sample (fitted on val), from unrounded logits:")
    print(f"{'T':>8} {'NLL':>7} {'ECE':>7} {'Brier':>7} {'acc':>7}")
    for r in table:
        print(f"{r['T']:>8.4f} {r['nll']:>7.4f} {r['ece']:>7.4f} {r['brier']:>7.4f} {r['accuracy']:>7.4f}")
    print("\n" + diff)



if __name__ == "__main__":
    main()
