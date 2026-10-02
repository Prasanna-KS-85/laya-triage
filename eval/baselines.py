"""Phase 2 baselines B0, B1, B2 (PROJECT_SPEC.md §10.1, §13 Phase 2).

Usage (from the repo root):
  .venv/bin/python eval/baselines.py run --split val [--systems B0,B1,B2_512-192,B2_1024-256]
  .venv/bin/python eval/baselines.py report --split val
  .venv/bin/python eval/baselines.py latency-b1

- B0: majority class of train.jsonl (ties -> LABELS order); probabilities = train class priors.
- B1: TF-IDF (word 1-2 grams) + LogisticRegression(class_weight=None), fit on train.jsonl only,
  text = title + "\\n" + body of the processed state. On val, the grid below is fit and the config
  is selected by val cross-repo macro-F1 and written to results/phase2/b1_config.json. On test,
  that frozen config is refit on train only; test runs refuse to start unless b1_config.json is
  committed and unmodified.
- B2: convaiinnovations/laya base checkpoint at REV, zero-shot, ISSUE_TYPE_QUESTION, CPU fp32,
  offline, one predict() per issue (as results/phase0/phase0_eval.py).

`run` writes results/phase2/predictions_<split>_<system>.csv (id, repo, gold, pred, p_<label>; no
issue text) plus a .meta.json beside it. `report` computes metrics from those CSVs only and writes
results/phase2/metrics_<split>.json. Test predictions are never overwritten.
"""

import os

# Networking off and CPU autocast off (fp32), set before laya/huggingface_hub are imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.pop("LAYA_CPU_AMP", None)

import argparse
import csv
import json
import subprocess
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import metrics

from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "phase2"
B1_CONFIG = OUT / "b1_config.json"
QID = "issue_type"
REPO_ID = "convaiinnovations/laya"
REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SEED = 42
B2_SETTINGS = {"B2_512-192": (512, 192), "B2_1024-256": (1024, 256)}
SYSTEMS = ("B0", "B1", *B2_SETTINGS)
GRID_C = (0.3, 1, 3, 10, 30)
GRID_MIN_DF = (1, 2)
GRID_SUBLINEAR = (False, True)
MAX_ITER = 10_000
CSV_COLUMNS = ["id", "repo", "gold", "pred", *(f"p_{c}" for c in LABELS)]


@dataclass(frozen=True)
class Issue:
    id: str
    repo: str
    label: str
    state: dict  # issue text: in memory only, never printed or written


def load_split(split):
    issues = []
    with open(DATA / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["questions"]
            if q != ISSUE_TYPE_QUESTION or tuple(q[QID]["criteria"]) != LABELS:
                raise SystemExit(f"{split} row {r['id']}: question differs from ISSUE_TYPE_QUESTION")
            if split == "train":
                probs = r["gold"][QID]["probabilities"]
                (label,) = [c for c in LABELS if probs[c] == 1.0]
                repo = r["repo"]
            else:
                label = r["expected"][QID]
                (tag,) = [t for t in r["tags"] if t.startswith("repo:")]
                repo = tag.removeprefix("repo:")
            issues.append(Issue(r["id"], repo, label, r["state"]))
    return issues


def text(issue):
    return issue.state["title"] + "\n" + issue.state["body"]


def write_predictions(split, system, issues, preds, probs, meta):
    path = OUT / f"predictions_{split}_{system}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for i, y, p in zip(issues, preds, probs):
            w.writerow([i.id, i.repo, i.label, y, *(repr(float(p[c])) for c in LABELS)])
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    s = metrics.summarize([i.label for i in issues], preds, probs, [i.repo for i in issues])
    print(f"{system}: wrote {path.relative_to(ROOT)}; cross-repo macro-F1 {s['cross_repo_macro_f1']:.4f}, "
          f"accuracy {s['accuracy']:.4f}, predicted {s['predicted_counts']}", flush=True)


def versions():
    import joblib
    import laya
    import numpy
    import scipy
    import sklearn
    import threadpoolctl
    import torch
    return {"python": sys.version.split()[0], "laya": laya.__version__, "torch": torch.__version__,
            "numpy": numpy.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__}


# ---- B0 ----

def run_b0(split, train, issues):
    counts = Counter(i.label for i in train)
    majority = max(LABELS, key=lambda c: (counts[c], -LABELS.index(c)))
    priors = {c: counts[c] / len(train) for c in LABELS}
    meta = {"system": "B0", "split": split, "train_counts": {c: counts[c] for c in LABELS},
            "majority": majority, "probabilities": "train class priors"}
    write_predictions(split, "B0", issues, [majority] * len(issues), [priors] * len(issues), meta)


# ---- B1 ----

def fit_b1(train, C, min_df, sublinear_tf):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=sublinear_tf)
    X = vec.fit_transform([text(i) for i in train])
    clf = LogisticRegression(C=C, class_weight=None, random_state=SEED, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X, [i.label for i in train])
    assert tuple(clf.classes_) == LABELS, clf.classes_
    info = {"n_iter": int(clf.n_iter_.max()), "vocabulary": len(vec.vocabulary_),
            "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
            "other_warnings": sorted({f"{w.category.__name__}: {w.message}" for w in caught
                                      if not issubclass(w.category, ConvergenceWarning)})}
    return vec, clf, info


def predict_b1(vec, clf, issues):
    X = vec.transform([text(i) for i in issues])
    P = clf.predict_proba(X)
    preds = [LABELS[int(row.argmax())] for row in P]
    assert preds == list(clf.predict(X))
    return preds, [{c: float(row[k]) for k, c in enumerate(LABELS)} for row in P]


def b1_select_on_val(train, val):
    grid = []
    for C in GRID_C:
        for min_df in GRID_MIN_DF:
            for sub in GRID_SUBLINEAR:
                vec, clf, info = fit_b1(train, C, min_df, sub)
                preds, probs = predict_b1(vec, clf, val)
                s = metrics.summarize([i.label for i in val], preds, probs, [i.repo for i in val])
                grid.append({"C": C, "min_df": min_df, "sublinear_tf": sub,
                             "val_cross_repo_macro_f1": s["cross_repo_macro_f1"],
                             "val_pooled_macro_f1": s["pooled_macro_f1"], "val_accuracy": s["accuracy"],
                             "val_ece": s["ece"], **info})
    # Max cross-repo macro-F1; ties -> smaller C, then smaller min_df, then sublinear_tf False.
    best = min(grid, key=lambda g: (-round(g["val_cross_repo_macro_f1"], 12), g["C"], g["min_df"],
                                    g["sublinear_tf"]))
    print(f"\nB1 grid on val ({len(grid)} configs, fit on train only, max_iter {MAX_ITER}):")
    print(f"{'C':>5} {'min_df':>6} {'sublin':>6} | {'xrepo-F1':>8} {'pooled-F1':>9} {'acc':>6} {'ECE':>6} "
          f"| {'n_iter':>6} {'vocab':>7} {'conv.warn':>9}")
    for g in grid:
        mark = "  <- selected" if g is best else ""
        print(f"{g['C']:>5} {g['min_df']:>6} {g['sublinear_tf']!s:>6} | "
              f"{g['val_cross_repo_macro_f1']:>8.4f} {g['val_pooled_macro_f1']:>9.4f} "
              f"{g['val_accuracy']:>6.4f} {g['val_ece']:>6.4f} | {g['n_iter']:>6} {g['vocabulary']:>7} "
              f"{g['convergence_warnings']:>9}{mark}")
    other = sorted({w for g in grid for w in g["other_warnings"]})
    print(f"other warnings during fit: {other or 'none'}")
    with open(OUT / "b1_grid_val.csv", "w", newline="") as f:
        cols = [k for k in grid[0] if k != "other_warnings"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(grid)
    return best


def b1_config_dict(best):
    import sklearn
    return {
        "system": "B1",
        "text": "state.title + '\\n' + state.body (data/processed, PROJECT_SPEC.md §8.4)",
        "fit_on": "data/processed/train.jsonl only",
        "vectorizer": {"class": "sklearn.feature_extraction.text.TfidfVectorizer", "ngram_range": [1, 2],
                       "min_df": best["min_df"], "sublinear_tf": best["sublinear_tf"]},
        "classifier": {"class": "sklearn.linear_model.LogisticRegression", "C": best["C"],
                       "class_weight": None, "random_state": SEED, "max_iter": MAX_ITER},
        "other_parameters": f"scikit-learn {sklearn.__version__} defaults",
        "selection": {"split": "val", "metric": "cross-repo macro-F1",
                      "ties": "smaller C, then smaller min_df, then sublinear_tf False",
                      "grid": {"C": list(GRID_C), "min_df": list(GRID_MIN_DF),
                               "sublinear_tf": list(GRID_SUBLINEAR)},
                      "val_cross_repo_macro_f1": best["val_cross_repo_macro_f1"]},
        "versions": versions(),
    }


def run_b1(split, train, issues):
    if split == "val":
        best = b1_select_on_val(train, issues)
        cfg = b1_config_dict(best)
        B1_CONFIG.write_text(json.dumps(cfg, indent=1) + "\n")
        print(f"wrote {B1_CONFIG.relative_to(ROOT)}: C={best['C']} min_df={best['min_df']} "
              f"sublinear_tf={best['sublinear_tf']}")
    else:
        cfg = json.loads(B1_CONFIG.read_text())
    C = cfg["classifier"]["C"]
    min_df = cfg["vectorizer"]["min_df"]
    sub = cfg["vectorizer"]["sublinear_tf"]
    vec, clf, info = fit_b1(train, C, min_df, sub)
    preds, probs = predict_b1(vec, clf, issues)
    meta = {"system": "B1", "split": split, "config": "results/phase2/b1_config.json",
            "C": C, "min_df": min_df, "sublinear_tf": sub, **info, "versions": versions()}
    write_predictions(split, "B1", issues, preds, probs, meta)


# ---- B2 ----

def run_b2(split, system, issues):
    import laya
    import torch

    max_len, head_max_len = B2_SETTINGS[system]
    agent = laya.load(REPO_ID, revision=REV, device="cpu")
    assert agent.revision == REV and str(agent.device) == "cpu"
    assert agent.amp_enabled is False and agent.dtype == torch.float32
    preds, probs, truncated = [], [], 0
    t0 = time.perf_counter()
    for n, i in enumerate(issues, 1):
        out = agent.predict(i.state, ISSUE_TYPE_QUESTION, max_len=max_len, head_max_len=head_max_len)
        a = out["answers"][QID]
        p = {c: float(a["probabilities"][c]) for c in LABELS}
        assert a["answer_confidence"] == p[a["choice"]], (i.id, a["answer_confidence"], p)
        preds.append(a["choice"])
        probs.append(p)
        truncated += bool(out["usage"]["truncated"])
        if n % 50 == 0 or n == len(issues):
            print(f"{system} {split}: {n}/{len(issues)} issues, {time.perf_counter() - t0:.0f} s", flush=True)
    meta = {"system": system, "split": split, "model": REPO_ID, "revision": agent.revision,
            "max_len": max_len, "head_max_len": head_max_len, "question": "ISSUE_TYPE_QUESTION",
            "device": str(agent.device), "dtype": str(agent.dtype), "amp_enabled": agent.amp_enabled,
            "torch_threads": torch.get_num_threads(), "share_truncated": truncated / len(issues),
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "probabilities": "as returned by laya (rounded to 4 decimals)", "versions": versions()}
    write_predictions(split, system, issues, preds, probs, meta)


def cmd_latency_b1(n=100, warmup=3):
    """B1 CPU latency per issue incl. vectorising: frozen config fit on train, first n val issues."""
    import numpy as np

    cfg = json.loads(B1_CONFIG.read_text())
    vec, clf, _ = fit_b1(load_split("train"), cfg["classifier"]["C"], cfg["vectorizer"]["min_df"],
                         cfg["vectorizer"]["sublinear_tf"])
    issues = load_split("val")
    for i in issues[:warmup]:
        clf.predict_proba(vec.transform([text(i)]))
    ms = []
    for i in issues[:n]:
        s = text(i)
        t = time.perf_counter()
        clf.predict_proba(vec.transform([s]))
        ms.append((time.perf_counter() - t) * 1000)
    out = {"system": "B1", "issues": f"first {n} val.jsonl rows, one call per issue",
           "warmup": warmup, "timed": "vectorizer.transform + predict_proba (text built beforehand)",
           "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
           "mean_ms": float(np.mean(ms)), "max_ms": float(max(ms)), "n": len(ms), "versions": versions()}
    (OUT / "latency_B1.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=1))


# ---- test guard ----

def check_test_allowed(systems):
    rel = str(B1_CONFIG.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True,
                             check=False)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, check=False)
    if tracked.returncode or clean.returncode:
        raise SystemExit(f"refusing to touch test: {rel} must be committed and unmodified")
    for s in systems:
        if (OUT / f"predictions_test_{s}.csv").exists():
            raise SystemExit(f"refusing to re-run {s} on test: predictions_test_{s}.csv exists")


def cmd_run(split, systems):
    OUT.mkdir(parents=True, exist_ok=True)
    if split == "test":
        check_test_allowed(systems)
    issues = load_split(split)
    print(f"{split}: {len(issues)} issues; label counts {dict(Counter(i.label for i in issues))}", flush=True)
    train = load_split("train") if {"B0", "B1"} & set(systems) else None
    for s in systems:
        if s == "B0":
            run_b0(split, train, issues)
        elif s == "B1":
            run_b1(split, train, issues)
        else:
            run_b2(split, s, issues)


# ---- report ----

def read_predictions(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([r["id"] for r in rows], [r["gold"] for r in rows], [r["pred"] for r in rows],
            [{c: float(r[f"p_{c}"]) for c in LABELS} for r in rows], [r["repo"] for r in rows])


def cmd_report(split):
    report, ids_ref = {}, None
    for s in SYSTEMS:
        path = OUT / f"predictions_{split}_{s}.csv"
        if not path.exists():
            print(f"{s}: no predictions for {split}, skipped")
            continue
        ids, gold, pred, probs, repos = read_predictions(path)
        if ids_ref is None:
            ids_ref = ids
        assert ids == ids_ref, f"{s}: ids differ from the other systems"
        report[s] = {"metrics": metrics.summarize(gold, pred, probs, repos),
                     "meta": json.loads(path.with_suffix(".meta.json").read_text())}
    out = OUT / f"metrics_{split}.json"
    out.write_text(json.dumps({"split": split, "systems": report}, indent=1) + "\n")
    print_report(split, report)
    print(f"\nwrote {out.relative_to(ROOT)}")


def print_report(split, report):
    print(f"\n{split}: n = {next(iter(report.values()))['metrics']['n']}")
    hdr = f"{'system':<12} {'acc':>6} {'xrepo-F1':>8} {'pooled-F1':>9} {'F1 bug':>6} {'F1 feat':>7} {'F1 q':>6} " \
          f"{'ECE':>6} {'Brier':>6}  pred bug/feat/q"
    print(hdr)
    for s, r in report.items():
        m = r["metrics"]
        pc = m["per_class"]
        cnt = "/".join(str(m["predicted_counts"][c]) for c in LABELS)
        print(f"{s:<12} {m['accuracy']:>6.4f} {m['cross_repo_macro_f1']:>8.4f} {m['pooled_macro_f1']:>9.4f} "
              f"{pc['bug']['f1']:>6.4f} {pc['feature']['f1']:>7.4f} {pc['question']['f1']:>6.4f} "
              f"{m['ece']:>6.4f} {m['brier']:>6.4f}  {cnt}")
    print("\nper-class precision / recall")
    for s, r in report.items():
        pc = r["metrics"]["per_class"]
        print(f"{s:<12} " + "  ".join(f"{c} {pc[c]['precision']:.4f}/{pc[c]['recall']:.4f}" for c in LABELS))
    repos = list(next(iter(report.values()))["metrics"]["per_repo_macro_f1"])
    print("\nper-repo macro-F1")
    print(f"{'system':<12} " + " ".join(f"{r.split('/')[1]:>10}" for r in repos))
    for s, r in report.items():
        pr = r["metrics"]["per_repo_macro_f1"]
        print(f"{s:<12} " + " ".join(f"{pr[x]:>10.4f}" for x in repos))
    print("\nconfusion (rows gold, cols pred; bug/feature/question)")
    for s, r in report.items():
        print(f"{s:<12} {r['metrics']['confusion']['rows_gold_cols_pred']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "report", "latency-b1"))
    ap.add_argument("--split", choices=("val", "test"), help="required for run and report")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="comma-separated subset of %(default)s")
    args = ap.parse_args(argv)
    systems = args.systems.split(",")
    if not set(systems) <= set(SYSTEMS):
        ap.error(f"unknown systems: {sorted(set(systems) - set(SYSTEMS))}")
    if args.command == "latency-b1":
        cmd_latency_b1()
    elif args.split is None:
        ap.error("--split is required for run and report")
    elif args.command == "run":
        cmd_run(args.split, systems)
    else:
        cmd_report(args.split)


if __name__ == "__main__":
    main()
