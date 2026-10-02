"""Phase 3 val report: M1 vs B1 vs B2 from saved val predictions (PROJECT_SPEC.md §10.2, §10.4).

Run from anywhere: .venv/bin/python results/phase3/val_report.py [--run R1] [--figures DIR]
Reads results/phase3/val_<run>/predictions_val_M1_<run>.csv (+ .meta.json) and
results/phase2/predictions_val_{B1,B2_512-192,B2_1024-256}.csv; ids, labels and numbers only.
Writes results/phase3/val_<run>/metrics_val.json and prints:
- §10.2 metrics per system (eval/metrics.py), M1 ECE/Brier also uncalibrated (T=1 columns);
- paired M1 vs B1: exact McNemar p and a paired bootstrap 95% CI of the cross-repo macro-F1
  difference (2,000 resamples of issues with replacement, numpy RandomState(42));
- information only: NLL / ECE / accuracy of softmax(log p_T1 / T) for a few T (the model's fitted T
  is not changed and this scan is not used for thresholds);
- §10.4 thresholds on M1 and B1 (eval/gating.py), in-sample on val, and the YAML that would go to
  config/triage.default.yml (not written here).
--figures DIR also writes the §10.3 figures as PNG to DIR. --compare-run R compares the M1 predictions with
run R's (labels, T=1 columns, calibrated probabilities). --write-config writes the proposed YAML to
config/triage.default.yml (refuses to replace a different existing file).
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import gating
import metrics
from baselines import print_report, read_predictions
from laya.common import ece_score

from laya_triage.questions import LABELS

PHASE2 = ROOT / "results" / "phase2"
BASELINES = ("B1", "B2_512-192", "B2_1024-256")
N_BOOT = 2000
SEED = 42
T_SCAN = (1.0, 1.5, 2.0, 3.0, "fitted", 5.0, 6.0, 8.0)
REPO = "Prasanna85/laya-issue-triage"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_m1(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t1 = np.array([[float(r[f"p_{c}_T1"]) for c in LABELS] for r in rows])
    conf = [float(r["answer_confidence"]) for r in rows]
    return t1, conf


def mcnemar_exact(gold, pred_a, pred_b):
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    b = sum(pa == g != pb for g, pa, pb in zip(gold, pred_a, pred_b))  # a right, b wrong
    c = sum(pb == g != pa for g, pa, pb in zip(gold, pred_a, pred_b))  # b right, a wrong
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n)
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": p}


def paired_bootstrap_xrepo(gold, pred_a, pred_b, repos, n_boot=N_BOOT, seed=SEED):
    """95% CI of cross-repo macro-F1(a) - (b), resampling issues with replacement (same indices for both)."""
    n = len(gold)
    idx = np.random.RandomState(seed).randint(0, n, size=(n_boot, n))
    diffs = []
    for row in idx:
        g, r = [gold[i] for i in row], [repos[i] for i in row]
        diffs.append(metrics.cross_repo_macro_f1(g, [pred_a[i] for i in row], r)
                     - metrics.cross_repo_macro_f1(g, [pred_b[i] for i in row], r))
    diffs = np.array(diffs)
    return {"point": metrics.cross_repo_macro_f1(gold, pred_a, repos) - metrics.cross_repo_macro_f1(gold, pred_b, repos),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_resamples_a_better": float((diffs > 0).mean()), "n_boot": n_boot, "seed": seed}


def temperature_scan(t1, gold, t_fitted):
    """NLL, ECE (15 bins) and accuracy of softmax(log(p_T1) / T); argmax, so accuracy, is unchanged."""
    y = np.array([LABELS.index(g) for g in gold])
    logp = np.log(np.clip(t1, 1e-300, None))
    out = []
    for t in T_SCAN:
        tv = t_fitted if t == "fitted" else t
        p = _softmax_rows(logp / tv)
        pred = p.argmax(axis=1)
        out.append({"T": tv, "fitted": t == "fitted", "nll": float(-np.log(p[np.arange(len(y)), y]).mean()),
                    "ece": ece_score(p.max(axis=1), (pred == y).astype(float), bins=metrics.ECE_BINS),
                    "accuracy": float((pred == y).mean())})
    out.sort(key=lambda r: r["T"])
    grid = np.round(np.arange(0.5, 10.0001, 0.01), 2)
    nll = [float(-np.log(_softmax_rows(logp / tv)[np.arange(len(y)), y]).mean()) for tv in grid]
    return out, {"grid": "0.50..10.00 step 0.01", "T_min_nll": float(grid[int(np.argmin(nll))]), "nll": min(nll)}


def _softmax_rows(z):
    p = np.exp(z - z.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def gating_report(gold, pred, conf):
    rows = [{"gold": g, "pred": p, "answer_confidence": c} for g, p, c in zip(gold, pred, conf)]
    fit = gating.fit_thresholds(rows)
    out = {"per_label": {}}
    for label in LABELS:
        f = fit[label]
        by_tau = {t["tau"]: t for t in f["table"]}
        out["per_label"][label] = {
            "n_predicted": f["n_predicted"], "n_candidates": f["n_candidates"], "seed": f["seed"],
            "tau_lb": f["tau_lb"], "at_tau_lb": by_tau.get(f["tau_lb"]),
            "tau_point": f["tau_point"], "at_tau_point": by_tau.get(f["tau_point"]),
        }
    for rule in ("tau_lb", "tau_point"):
        taus = {lb: (fit[lb][rule] if fit[lb][rule] is not None else gating.NEVER) for lb in LABELS}
        out[rule] = {"taus": taus, "applied": gating.apply_thresholds(rows, taus)}
    th, cov, prec = gating.coverage_precision_curve([r["answer_confidence"] for r in rows],
                                                    [r["gold"] == r["pred"] for r in rows])
    out["curve"] = {"threshold": th.tolist(), "coverage": cov.tolist(), "precision": prec.tolist()}
    out["note"] = "in-sample: fitted on val"
    return out


def threshold_yaml(revision, max_len, taus):
    fmt = {lb: (f"{t:.2f}" if t == gating.NEVER else f"{t:.4f}") for lb, t in taus.items()}
    return f"""model:
  repo: "{REPO}"
  revision: "{revision}"        # pinned; never "main" in production
  max_len: {max_len}
  backend: torch                  # torch | onnx (v1.2)

labels:                           # model class -> GitHub label name
  bug: "type: bug"
  feature: "type: feature"
  question: "type: question"
  escalate: "triage: needs-human"

gating:
  thresholds:                     # §10.4 bootstrap lower bound on val (1.01 = never auto-apply)
    bug: {fmt["bug"]}
    feature: {fmt["feature"]}
    question: {fmt["question"]}

behaviour:
  mode: dry-run                   # dry-run | apply   (default dry-run for safety)
  comment_on_escalate: true
  skip_if_labeled: true           # skip if any labels.{{bug,feature,question}} present
  skip_authors: ["dependabot[bot]", "renovate[bot]"]
  create_missing_labels: false
  escalate_on_error: false

preprocess:
  max_code_lines: 20
  max_body_chars: 6000
"""


def figures(out_dir, gold, repos, m1_pred, m1_conf, t1, b1_pred, b1_conf, gate):
    import plots

    out_dir.mkdir(parents=True, exist_ok=True)
    correct = np.array([g == p for g, p in zip(gold, m1_pred)], dtype=float)
    plots.save(plots.reliability_diagram({"M1 R1, T=1": (t1.max(axis=1), correct),
                                          "M1 R1, fitted T": (np.array(m1_conf), correct)}),
               out_dir / "reliability_M1_R1.png")
    ap = gate["tau_lb"]["applied"]
    pt = gate["tau_point"]["applied"]
    ops = {"tau_lb (lower bound)": (ap["coverage"], ap["precision"]),
           "tau_point": (pt["coverage"], pt["precision"])}
    b1_correct = np.array([g == p for g, p in zip(gold, b1_pred)], dtype=float)
    plots.save(plots.coverage_precision_plot({"M1 R1": (np.array(m1_conf), correct), "B1": (np.array(b1_conf), b1_correct)},
                                             ops),
               out_dir / "coverage_precision_M1_R1.png")
    plots.save(plots.confusion_matrices({"M1 R1": (gold, m1_pred), "B1": (gold, b1_pred)}),
               out_dir / "confusion_M1_R1_vs_B1.png")
    plots.save(plots.per_repo_f1_bars({"M1 R1": (gold, m1_pred, repos), "B1": (gold, b1_pred, repos)}),
               out_dir / "per_repo_f1_M1_R1_vs_B1.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="R1")
    ap.add_argument("--figures", type=Path, default=None, help="write PNG figures to this directory")
    ap.add_argument("--compare-run", default=None, help="compare M1 predictions with this run's (e.g. R1)")
    ap.add_argument("--write-config", action="store_true", help="write config/triage.default.yml")
    args = ap.parse_args(argv)
    run_dir = ROOT / "results" / "phase3" / f"val_{args.run}"
    m1_csv = run_dir / f"predictions_val_M1_{args.run}.csv"
    m1_meta = json.loads(m1_csv.with_suffix(".meta.json").read_text())

    ids, gold, m1_pred, m1_probs, repos = read_predictions(m1_csv)
    t1, m1_conf = load_m1(m1_csv)
    assert all(c == p[y] for c, p, y in zip(m1_conf, m1_probs, m1_pred))
    report, inputs = {}, {str(m1_csv.relative_to(ROOT)): sha256_file(m1_csv)}
    report["M1"] = {"metrics": metrics.summarize(gold, m1_pred, m1_probs, repos), "meta": m1_meta}
    t1_probs = [{c: float(v) for c, v in zip(LABELS, row)} for row in t1]
    assert [LABELS[int(i)] for i in t1.argmax(axis=1)] == m1_pred
    report["M1"]["uncalibrated_T1"] = {"ece": metrics.ece(gold, m1_pred, t1_probs),
                                      "brier": metrics.brier(gold, t1_probs)}
    preds = {"M1": m1_pred}
    b1_conf = None
    for s in BASELINES:
        path = PHASE2 / f"predictions_val_{s}.csv"
        b_ids, b_gold, b_pred, b_probs, b_repos = read_predictions(path)
        assert (b_ids, b_gold, b_repos) == (ids, gold, repos), f"{s}: ids/gold/repos differ from M1"
        report[s] = {"metrics": metrics.summarize(gold, b_pred, b_probs, repos)}
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)
        preds[s] = b_pred
        if s == "B1":
            b1_conf = [p[y] for p, y in zip(b_probs, b_pred)]

    t_fit = m1_meta["applied_choice_temperature_3_options"]
    y = np.array([LABELS.index(g) for g in gold])
    logp = np.log(t1)
    report["M1"]["nll"] = {
        "calibrated": float(-np.log(_softmax_rows(logp / t_fit)[np.arange(len(y)), y]).mean()),
        "T1": float(-logp[np.arange(len(y)), y].mean()),
        "note": "from the unrounded T=1 columns; calibrated = softmax(log p_T1 / T_config)",
    }
    print_report("val", report)
    print("\nM1 uncalibrated (T=1): ECE {ece:.4f}  Brier {brier:.4f}".format(**report["M1"]["uncalibrated_T1"]))
    print(f"M1 NLL: calibrated (T={t_fit}) {report['M1']['nll']['calibrated']:.4f}, T=1 {report['M1']['nll']['T1']:.4f}")
    comparison = None
    if args.compare_run:
        other_csv = ROOT / "results" / "phase3" / f"val_{args.compare_run}" / f"predictions_val_M1_{args.compare_run}.csv"
        o_ids, _, o_pred, o_probs, _ = read_predictions(other_csv)
        o_t1, _ = load_m1(other_csv)
        assert o_ids == ids
        comparison = {
            "run": args.compare_run,
            "labels_identical": sum(a == b for a, b in zip(m1_pred, o_pred)),
            "t1_max_abs_diff": float(np.abs(t1 - o_t1).max()),
            "calibrated_max_abs_diff": max(abs(p[c] - q[c]) for p, q in zip(m1_probs, o_probs) for c in LABELS),
            "calibrated_rows_differing": sum(any(p[c] != q[c] for c in LABELS) for p, q in zip(m1_probs, o_probs)),
        }
        print(f"vs {args.compare_run}: labels identical {comparison['labels_identical']}/{len(ids)}; "
              f"T=1 columns max |diff| {comparison['t1_max_abs_diff']:.2e}; calibrated probabilities differ on "
              f"{comparison['calibrated_rows_differing']} rows (max |diff| {comparison['calibrated_max_abs_diff']:.4f})")

    paired = {"mcnemar_exact": mcnemar_exact(gold, m1_pred, preds["B1"]),
              "bootstrap_xrepo_macro_f1_diff": paired_bootstrap_xrepo(gold, m1_pred, preds["B1"], repos)}
    mc, bs = paired["mcnemar_exact"], paired["bootstrap_xrepo_macro_f1_diff"]
    print(f"\nPaired M1 vs B1 (val, n={len(gold)}): McNemar exact p = {mc['p_value']:.4g} "
          f"(M1 right & B1 wrong {mc['a_right_b_wrong']}, M1 wrong & B1 right {mc['a_wrong_b_right']}); "
          f"cross-repo macro-F1 diff {bs['point']:+.4f}, 95% CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}], "
          f"M1 better in {bs['share_resamples_a_better']:.1%} of resamples")

    scan, fine = temperature_scan(t1, gold, t_fit)
    print("\nTemperature scan (information only; the model's T is unchanged):")
    print(f"{'T':>8} {'NLL':>7} {'ECE':>7} {'acc':>7}")
    for r in scan:
        print(f"{r['T']:>8.4f} {r['nll']:>7.4f} {r['ece']:>7.4f} {r['accuracy']:>7.4f}" + ("  <- fitted" if r["fitted"] else ""))
    print(f"NLL-minimising T on {fine['grid']}: {fine['T_min_nll']:.2f} (NLL {fine['nll']:.4f})")

    gate = {"M1": gating_report(gold, m1_pred, m1_conf), "B1": gating_report(gold, preds["B1"], b1_conf)}
    for name, g in gate.items():
        print(f"\nThresholds {name} (§10.4; in-sample: fitted on val)")
        print(f"{'label':<9} {'pred':>4} {'tau_lb':>7} {'|S|':>4} {'prec':>6} {'LB':>6} | {'tau_pt':>7} {'|S|':>4} {'prec':>6} {'LB':>6}")
        for lb, v in g["per_label"].items():
            a, b = v["at_tau_lb"] or {}, v["at_tau_point"] or {}
            tp = "-" if v["tau_point"] is None else f"{v['tau_point']:.4f}"
            print(f"{lb:<9} {v['n_predicted']:>4} {v['tau_lb']:>7.4f} {a.get('support', '-'):>4} "
                  f"{a.get('precision', float('nan')):>6.3f} {a.get('lower_bound', float('nan')):>6.3f} | "
                  f"{tp:>7} {b.get('support', '-'):>4} {b.get('precision', float('nan')):>6.3f} "
                  f"{b.get('lower_bound', float('nan')):>6.3f}")
        for rule in ("tau_lb", "tau_point"):
            r = g[rule]["applied"]
            prec = "-" if r["precision"] is None else f"{r['precision']:.4f}"
            per = ", ".join(f"{lb} {v['coverage']:.3f}/{'-' if v['precision'] is None else format(v['precision'], '.3f')}"
                            for lb, v in r["per_label"].items())
            print(f"  {rule:<9}: coverage {r['coverage']:.4f}, precision {prec}, applied {r['applied']}, "
                  f"escalated {r['escalated']}  (per label coverage/precision: {per})")

    taus = gate["M1"]["tau_lb"]["taus"]
    yaml = threshold_yaml(m1_meta["revision"], m1_meta["max_len"], taus)
    print("\nProposed config/triage.default.yml:\n" + yaml)
    if args.write_config:
        cfg_path = ROOT / "config" / "triage.default.yml"
        if cfg_path.exists() and cfg_path.read_text() != yaml:
            raise SystemExit(f"refusing to replace a different {cfg_path.relative_to(ROOT)}")
        cfg_path.write_text(yaml)
        print(f"wrote {cfg_path.relative_to(ROOT)}")

    out = {"split": "val", "run": args.run, "n": len(gold), "inputs_sha256": inputs,
           "systems": report, "paired_M1_vs_B1": paired, "comparison": comparison,
           "temperature_scan": {"note": "information only; model temperature unchanged", "rows": scan, "fine": fine},
           "gating": gate, "proposed_config_yaml": yaml}
    (run_dir / "metrics_val.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {(run_dir / 'metrics_val.json').relative_to(ROOT)}")
    if args.figures:
        figures(args.figures, gold, repos, m1_pred, m1_conf, t1, preds["B1"], b1_conf, gate["M1"])
        print(f"wrote figures to {args.figures}")


if __name__ == "__main__":
    main()
