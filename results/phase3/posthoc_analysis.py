"""Post-hoc exploratory analyses of Phase 3 (NOT part of the pre-declared test protocol; changes no criterion,
config or committed result). No model runs: predictions come from the committed CSVs.

Run from anywhere: .venv/bin/python results/phase3/posthoc_analysis.py
Inputs: results/phase3/{val,test}_R1b/predictions_{val,test}_M1_R1b.csv, results/phase2/predictions_{val,test}_B1.csv,
results/phase3/val_R1b/metrics_val.json and test_R1b/metrics_test.json (for cross-checks), results/phase2/b1_config.json,
and data/processed/{train,val,test}.jsonl (analysis A only; text stays in memory, never printed or written).
Writes aggregates only (no ids, no text) to results/phase3/posthoc/posthoc.json and report.md.

C. Sanity: confusion matrices of M1 R1b and B1 on test (eval/metrics.py and an independent count), the M1 test
   gating counts at the config thresholds and at val τ_point (eval/gating.py), checked against metrics_test.json.
A. Near-duplicates: per val / test issue, the max TF-IDF cosine similarity to any train issue (title + "\\n" + body,
   vectorizer settings of results/phase2/b1_config.json, fit on train only). Aggregates, and accuracy of M1 and
   B1 per tertile; tertile cut points from the val similarity distribution.
B. One global cutoff per system, pre-stated rule: gating.select_threshold on all val issues pooled (candidates =
   distinct val answer_confidence values, |S| >= 20, 5th percentile of precision over 2,000 resamples, seed 42,
   smallest τ with bound >= 0.90). Evaluated once on test with that τ. Separately, descriptive reads of the test
   curve (optimistic).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import gating
import metrics
from baselines import load_split, read_predictions, text

from laya_triage.questions import LABELS

OUT = ROOT / "results" / "phase3" / "posthoc"
PRED = {
    ("M1", "val"): ROOT / "results" / "phase3" / "val_R1b" / "predictions_val_M1_R1b.csv",
    ("M1", "test"): ROOT / "results" / "phase3" / "test_R1b" / "predictions_test_M1_R1b.csv",
    ("B1", "val"): ROOT / "results" / "phase2" / "predictions_val_B1.csv",
    ("B1", "test"): ROOT / "results" / "phase2" / "predictions_test_B1.csv",
}
METRICS_VAL = ROOT / "results" / "phase3" / "val_R1b" / "metrics_val.json"
METRICS_TEST = ROOT / "results" / "phase3" / "test_R1b" / "metrics_test.json"
B1_CONFIG = ROOT / "results" / "phase2" / "b1_config.json"
SIM_THRESHOLDS = (0.8, 0.9)
GLOBAL_SEED = 42
TERTILES = ("low", "mid", "high")


def load(system, split):
    """Predictions as parallel lists plus answer_confidence = probability of the predicted label."""
    ids, gold, pred, probs, repos = read_predictions(PRED[(system, split)])
    conf = [p[y] for p, y in zip(probs, pred)]
    return {"ids": ids, "gold": gold, "pred": pred, "conf": conf, "repos": repos}


def independent_confusion(gold, pred):
    """Rows gold, cols pred in LABELS order, by plain counting (cross-check of metrics.confusion_counts)."""
    c = Counter(zip(gold, pred))
    return [[c[(g, p)] for p in LABELS] for g in LABELS]


def gating_rows(d):
    return [{"gold": g, "pred": p, "answer_confidence": c} for g, p, c in zip(d["gold"], d["pred"], d["conf"])]


def n_correct(applied):
    """Correct auto-applied issues from apply_thresholds output (precision × applied, exact integers)."""
    return {"all": round((applied["precision"] or 0) * applied["applied"]),
            **{lb: round((v["precision"] or 0) * v["applied"]) for lb, v in applied["per_label"].items()}}


# ---- A: near-duplicates ----

def max_similarity_to_train(train_texts, query_texts, ngram_range, min_df, sublinear_tf):
    """Max cosine similarity of each query text to any train text; TF-IDF fit on train only (l2-normalised rows,
    so cosine = dot product)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(ngram_range=tuple(ngram_range), min_df=min_df, sublinear_tf=sublinear_tf)
    xt = vec.fit_transform(train_texts)
    xq = vec.transform(query_texts)
    return np.asarray((xq @ xt.T).max(axis=1).todense()).ravel()


def similarity_summary(sims):
    sims = np.asarray(sims, dtype=float)
    out = {"n": int(sims.size), "mean": float(sims.mean()), "median": float(np.median(sims)),
           "p90": float(np.percentile(sims, 90))}
    for t in SIM_THRESHOLDS:
        out[f"share_above_{t}"] = float((sims > t).mean())
    return out


def tertile_cuts(reference_sims):
    """Cut points at the 1/3 and 2/3 percentiles (numpy linear interpolation) of the reference distribution."""
    return [float(np.percentile(reference_sims, 100 / 3)), float(np.percentile(reference_sims, 200 / 3))]


def tertile_of(sims, cuts):
    """'low' (<= cut1), 'mid' (cut1, cut2], 'high' (> cut2) for each similarity."""
    sims = np.asarray(sims, dtype=float)
    return np.where(sims <= cuts[0], "low", np.where(sims <= cuts[1], "mid", "high"))


def accuracy_by_tertile(gold, pred, tertiles):
    out = {}
    for t in TERTILES:
        ix = [i for i, x in enumerate(tertiles) if x == t]
        out[t] = {"n": len(ix), "accuracy": metrics.accuracy([gold[i] for i in ix], [pred[i] for i in ix]) if ix else None}
    return out


# ---- B: global cutoff ----

def global_cutoff(conf, correct, seed=GLOBAL_SEED):
    """The pre-stated rule on all issues pooled: gating.select_threshold. None when no τ qualifies."""
    fit = gating.select_threshold(conf, correct, seed)
    if fit["tau_lb"] >= gating.NEVER:
        return {"tau": None, "n_candidates_with_support": len(fit["table"])}
    row = next(t for t in fit["table"] if t["tau"] == fit["tau_lb"])
    return {"tau": row["tau"], "support": row["support"], "coverage": row["support"] / len(conf),
            "precision": row["precision"], "lower_bound": row["lower_bound"],
            "n_candidates_with_support": len(fit["table"])}


def max_coverage_at_precision(conf, correct, target):
    """Descriptive read of a curve: the largest coverage over global thresholds with precision >= target
    (None if never reached), with that point's threshold and precision."""
    th, cov, prec = gating.coverage_precision_curve(conf, correct)
    ok = np.flatnonzero(prec >= target - gating.FLOAT_SLACK)
    if ok.size == 0:
        return None
    i = ok[np.argmax(cov[ok])]
    return {"threshold": float(th[i]), "coverage": float(cov[i]), "precision": float(prec[i])}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = {k: load(*k) for k in PRED}
    for split in ("val", "test"):
        assert d[("M1", split)]["ids"] == d[("B1", split)]["ids"] and d[("M1", split)]["gold"] == d[("B1", split)]["gold"]
    mt = json.loads(METRICS_TEST.read_text())
    mv = json.loads(METRICS_VAL.read_text())
    out = {"note": "post-hoc exploratory; not part of the pre-declared protocol; changes no criterion or config"}

    # C
    sanity = {}
    for s in ("M1", "B1"):
        t = d[(s, "test")]
        lib, ind = metrics.confusion_counts(t["gold"], t["pred"]), independent_confusion(t["gold"], t["pred"])
        committed = mt["systems"][s]["metrics"]["confusion"]["rows_gold_cols_pred"]
        assert lib == ind == committed, (s, lib, ind, committed)
        sanity[f"confusion_{s}"] = lib
    m1t, b1t = d[("M1", "test")], d[("B1", "test")]
    gf = [i for i, g in enumerate(m1t["gold"]) if g == "feature"]
    sanity["gold_feature"] = {"n": len(gf), "same_prediction": sum(m1t["pred"][i] == b1t["pred"][i] for i in gf),
                              "both_feature": sum(m1t["pred"][i] == b1t["pred"][i] == "feature" for i in gf)}
    for rule in ("tau_lb", "tau_point"):
        a = gating.apply_thresholds(gating_rows(m1t), mv["gating"]["M1"][rule]["taus"])
        key = {"tau_lb": "config", "tau_point": "M1_tau_point"}[rule]
        assert a == mt["gating"][key]["applied"], rule
        sanity[f"M1_test_{rule}"] = {"taus": mv["gating"]["M1"][rule]["taus"], "applied": a["applied"], "n": a["n"],
                                     "correct": n_correct(a)}
    out["C_sanity"] = sanity

    # A
    b1c = json.loads(B1_CONFIG.read_text())["vectorizer"]
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    for s in ("val", "test"):
        assert [i.id for i in splits[s]] == d[("M1", s)]["ids"], f"{s}.jsonl order != predictions"
    train_texts = [text(i) for i in splits["train"]]
    sims = {s: max_similarity_to_train(train_texts, [text(i) for i in splits[s]], b1c["ngram_range"], b1c["min_df"],
                                       b1c["sublinear_tf"]) for s in ("val", "test")}
    del splits, train_texts
    cuts = tertile_cuts(sims["val"])
    near = {"vectorizer": b1c, "summary": {s: similarity_summary(sims[s]) for s in sims}, "tertile_cuts_from_val": cuts,
            "accuracy_by_tertile": {}}
    for s in ("val", "test"):
        tert = tertile_of(sims[s], cuts)
        near["accuracy_by_tertile"][s] = {sys_: accuracy_by_tertile(d[(sys_, s)]["gold"], d[(sys_, s)]["pred"], tert)
                                          for sys_ in ("M1", "B1")}
        near["accuracy_by_tertile"][s]["overall"] = {sys_: metrics.accuracy(d[(sys_, s)]["gold"], d[(sys_, s)]["pred"])
                                                     for sys_ in ("M1", "B1")}
    out["A_near_duplicates"] = near

    # B
    glob = {}
    b1_declared_prec = mt["gating"]["B1_same_rule"]["applied"]["precision"]
    for s in ("M1", "B1"):
        v, t = d[(s, "val")], d[(s, "test")]
        v_ok = [g == p for g, p in zip(v["gold"], v["pred"])]
        t_ok = [g == p for g, p in zip(t["gold"], t["pred"])]
        fit = global_cutoff(v["conf"], v_ok)
        entry = {"val": fit, "test": None}
        if fit["tau"] is not None:
            a = gating.apply_thresholds(gating_rows(t), {lb: fit["tau"] for lb in LABELS})
            entry["test"] = {"applied": a["applied"], "escalated": a["escalated"], "coverage": a["coverage"],
                             "precision": a["precision"],
                             "per_label": {lb: {"applied": x["applied"], "precision": x["precision"]}
                                           for lb, x in a["per_label"].items()}}
        entry["descriptive_reads_test_curve_optimistic"] = {
            "max_coverage_at_precision_0.90": max_coverage_at_precision(t["conf"], t_ok, gating.TARGET_PRECISION)}
        if s == "M1":
            entry["descriptive_reads_test_curve_optimistic"]["max_coverage_at_B1_predeclared_test_precision"] = {
                "target": b1_declared_prec, "point": max_coverage_at_precision(t["conf"], t_ok, b1_declared_prec)}
        glob[s] = entry
    out["B_global_cutoff"] = glob

    (OUT / "posthoc.json").write_text(json.dumps(out, indent=1) + "\n")
    md = report(out)
    (OUT / "report.md").write_text(md + "\n")
    print(md)
    print(f"\nwrote {(OUT / 'posthoc.json').relative_to(ROOT)}, {(OUT / 'report.md').relative_to(ROOT)}")


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def report(out):
    c, a, b = out["C_sanity"], out["A_near_duplicates"], out["B_global_cutoff"]
    lines = ["Post-hoc exploratory analysis (not pre-declared). Aggregates only.", "",
             "C. Sanity (test)", "",
             (f"- M1 confusion {c['confusion_M1']}; B1 confusion {c['confusion_B1']} (metrics.confusion_counts == "
              "independent count == metrics_test.json)."),
             (f"- Gold feature ({c['gold_feature']['n']}): M1 and B1 predict the same label on "
              f"{c['gold_feature']['same_prediction']}, both `feature` on {c['gold_feature']['both_feature']}."),
             *(f"- M1 at val {r}: applied {c[f'M1_test_{r}']['applied']} / {c[f'M1_test_{r}']['n']}, correct "
               f"{c[f'M1_test_{r}']['correct']}" for r in ("tau_lb", "tau_point")),
             "", "A. Max TF-IDF cosine similarity to any train issue", "",
             "| Split | n | mean | median | p90 | share > 0.8 | share > 0.9 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s, v in a["summary"].items():
        lines.append(f"| {s} | {v['n']} | {f4(v['mean'])} | {f4(v['median'])} | {f4(v['p90'])} | "
                     f"{f4(v['share_above_0.8'])} | {f4(v['share_above_0.9'])} |")
    cuts = a["tertile_cuts_from_val"]
    lines += ["", f"Tertile cut points (val distribution): {cuts[0]:.4f}, {cuts[1]:.4f}", "",
              "| Split | System | low: n / acc | mid: n / acc | high: n / acc | overall acc |", "|---|---|---|---|---|---:|"]
    for s, by in a["accuracy_by_tertile"].items():
        for sys_ in ("M1", "B1"):
            cells = " | ".join(f"{by[sys_][t]['n']} / {f4(by[sys_][t]['accuracy'])}" for t in TERTILES)
            lines.append(f"| {s} | {sys_} | {cells} | {f4(by['overall'][sys_])} |")
    lines += ["", "B. One global cutoff (rule pre-stated; fitted on val, evaluated once on test)", "",
              ("| System | val τ | val \\|S\\| | val coverage | val precision | val bound | test applied | test coverage | "
               "test precision | per label applied / precision (bug, feature, question) |"),
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s, e in b.items():
        v, t = e["val"], e["test"]
        if v["tau"] is None:
            lines.append(f"| {s} | none qualifies | | | | | | | | |")
            continue
        per = "; ".join(f"{x['applied']} / {f4(x['precision'])}" for x in t["per_label"].values())
        lines.append(f"| {s} | {v['tau']:.4f} | {v['support']} | {f4(v['coverage'])} | {f4(v['precision'])} | "
                     f"{f4(v['lower_bound'])} | {t['applied']} | {f4(t['coverage'])} | {f4(t['precision'])} | {per} |")
    lines += ["", "Descriptive only (reads the test curve; optimistic):", ""]
    for s, e in b.items():
        r = e["descriptive_reads_test_curve_optimistic"]
        p = r["max_coverage_at_precision_0.90"]
        lines.append(f"- {s}: largest test coverage with precision ≥ 0.90: "
                     + ("not reached" if p is None else f"{f4(p['coverage'])} (precision {f4(p['precision'])}, "
                                                        f"τ {p['threshold']:.4f})"))
        if "max_coverage_at_B1_predeclared_test_precision" in r:
            q = r["max_coverage_at_B1_predeclared_test_precision"]
            lines.append(f"- M1: largest test coverage with precision ≥ B1's pre-declared test precision "
                         f"{q['target']:.4f}: " + ("not reached" if q["point"] is None else
                                                   f"{f4(q['point']['coverage'])} (precision "
                                                   f"{f4(q['point']['precision'])}, τ {q['point']['threshold']:.4f})"))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
