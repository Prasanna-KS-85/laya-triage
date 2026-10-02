"""Phase 3 test report: M1 (R1b) vs B0/B1/B2/B3 on test (PROJECT_SPEC.md §6.3, §10; results/phase3.md
"Test protocol (pre-declared)"). Computes exactly the pre-declared primary items 1-4 and secondary items 5-9.

Run from anywhere: .venv/bin/python results/phase3/test_report.py
Reads (ids, labels and numbers only; never test.jsonl):
- results/phase3/test_R1b/predictions_test_M1_R1b.csv (+ .meta.json), from the single test run;
- results/phase2/predictions_test_{B0,B1,B2_512-192,B2_1024-256}.csv (committed Phase 2 runs, not re-run);
- config/triage.default.yml (config thresholds), results/phase3/val_R1b/metrics_val.json (val τ_point of M1,
  val τ_lb of B1); the published B3 rows from results/phase2.md (copied there in Phase 2);
- results/phase3/val_R1/latency_M1_R1_threads{default,2}.json (Stop A latency, cold load) and
  results/phase2/latency_B1.json.
Thresholds read from files must equal the values pre-declared in results/phase3.md (DECLARED, 4 decimals).
Writes results/phase3/test_R1b/metrics_test.json and report_test.md (the printed tables), and the four §10.3
figures to results/figures/. Nothing is fitted on test.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gating
import metrics
from baselines import read_predictions
from val_report import load_m1, mcnemar_exact, paired_bootstrap_xrepo

from laya_triage.questions import LABELS

RUN = "R1b"
TEST_DIR = ROOT / "results" / "phase3" / f"test_{RUN}"
M1_CSV = TEST_DIR / f"predictions_test_M1_{RUN}.csv"
VAL_METRICS = ROOT / "results" / "phase3" / f"val_{RUN}" / "metrics_val.json"
CONFIG = ROOT / "config" / "triage.default.yml"
PHASE2 = ROOT / "results" / "phase2"
PHASE2_MD = ROOT / "results" / "phase2.md"
STOP_A_LATENCY = {t: ROOT / "results" / "phase3" / "val_R1" / f"latency_M1_R1_threads{t}.json"
                  for t in ("default", "2")}
FIGURES = ROOT / "results" / "figures"
BASELINES = ("B0", "B1", "B2_512-192", "B2_1024-256")
PAIRED = ("B1", "B2_512-192", "B2_1024-256")  # protocol item 3
N_TEST = 1500
ECE_TARGET = 0.10  # §6.3 criterion 2
NFR1_P95_MS = 1000  # NFR-1: ≤ 1 s p95 on a GitHub-hosted Linux runner
# Pre-declared in results/phase3.md before any M1 test prediction existed.
DECLARED = {
    "config": {"bug": 0.6033, "feature": 1.01, "question": 1.01},
    "M1_tau_point": {"bug": 0.5027, "feature": 0.7320, "question": 0.6088},
    "B1_tau_lb": {"bug": 0.7698, "feature": 0.6886, "question": 0.8692},
}
NAMES = {"B0": "B0 majority", "B1": "B1 TF-IDF+LR", "B2_512-192": "B2 zero-shot 512/192 (native)",
         "B2_1024-256": "B2 zero-shot 1024/256", "M1": "M1 R1b (ours)"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def config_thresholds(text):
    """gating.thresholds of a triage config YAML as {label: float}; keys must be exactly LABELS."""
    th = ((yaml.safe_load(text) or {}).get("gating") or {}).get("thresholds") or {}
    if set(th) != set(LABELS):
        raise ValueError(f"config thresholds must have exactly the keys {LABELS}, got {sorted(th)}")
    return {lb: float(th[lb]) for lb in LABELS}


def check_declared(taus, declared, name):
    """Raise unless every threshold rounds (4 decimals) to the pre-declared value."""
    bad = {lb: (taus[lb], declared[lb]) for lb in LABELS if round(taus[lb], 4) != round(declared[lb], 4)}
    if bad or set(taus) != set(LABELS):
        raise SystemExit(f"{name}: thresholds differ from the pre-declared values: {bad or sorted(taus)}")
    return taus


def _number(cell):
    """'**0.8270**' or '0.8263 [c]' -> 0.827 / 0.8263."""
    m = re.fullmatch(r"\**\s*([0-9]+\.[0-9]+)\s*\**(?:\s*\[[a-z]\])?", cell.strip())
    if not m:
        raise ValueError(f"not a number cell: {cell!r}")
    return float(m.group(1))


B3_COLUMNS = ("cross_repo_macro_f1", "pooled_macro_f1", "accuracy", "f1_bug", "f1_feature", "f1_question")


def parse_b3_rows(md_text):
    """The published B3 rows of results/phase2.md "Results table v0" (rows starting '| B3 ') and the
    provenance-unknown SetFit footnote row ('| SetFit, `output/setfit/results.json` |'): the system name
    and the first six numeric columns (B3_COLUMNS). Pooled, accuracy and per-class values are derived [c]."""
    section = md_text.split("## Results table v0", 1)
    if len(section) != 2:
        raise ValueError("no 'Results table v0' section")
    body = section[1].split("\n#", 1)[0]  # up to the next heading of any level
    out = {"published": [], "footnote": []}
    for line in body.splitlines():
        if line.startswith("| B3 "):
            kind = "published"
        elif line.startswith("| SetFit, `output/setfit/results.json` |"):
            kind = "footnote"
        else:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out[kind].append({"system": cells[0], **dict(zip(B3_COLUMNS, map(_number, cells[1:7])))})
    if not out["published"]:
        raise ValueError("no B3 rows found")
    return out


def gating_view(gold, pred, conf, taus):
    """gating.apply_thresholds on test predictions with fixed (val-fitted) thresholds."""
    rows = [{"gold": g, "pred": p, "answer_confidence": c} for g, p, c in zip(gold, pred, conf)]
    return {"taus": dict(taus), "applied": gating.apply_thresholds(rows, taus)}


def judge_criteria(xrepo, paired, ece_cal, config_gate, p95_ms):
    """§6.3 criteria 1-4 from test numbers. xrepo: {system: cross-repo macro-F1} incl. M1 and PAIRED;
    paired: {system: {"bootstrap_xrepo_macro_f1_diff": {...}, ...}}; config_gate: gating_view at the
    config thresholds; p95_ms: local CPU p95 of the test run."""
    c1 = {s: {"m1": xrepo["M1"], "other": xrepo[s], "diff": xrepo["M1"] - xrepo[s],
              "ci95": paired[s]["bootstrap_xrepo_macro_f1_diff"]["ci95"], "m1_higher": xrepo["M1"] > xrepo[s]}
          for s in PAIRED}
    per_label = {}
    for lb, v in config_gate["applied"]["per_label"].items():
        exists = v["tau"] < gating.NEVER
        met = exists and v["precision"] is not None and v["precision"] >= gating.TARGET_PRECISION
        per_label[lb] = {"val_threshold_exists": exists, "tau": v["tau"], "applied": v["applied"],
                         "coverage": v["coverage"], "precision": v["precision"], "met": met}
    k = sum(v["met"] for v in per_label.values())
    return {
        "1": {"criterion": "M1 cross-repo macro-F1 > B1 and > B2 (both settings), paired CIs reported",
              "comparisons": c1, "status": "MET" if all(v["m1_higher"] for v in c1.values()) else "NOT MET"},
        "2": {"criterion": f"calibrated ECE on test <= {ECE_TARGET}", "ece": ece_cal,
              "status": "MET" if ece_cal <= ECE_TARGET else "NOT MET"},
        "3": {"criterion": "val thresholds with >= 90% precision; test coverage and precision per label",
              "per_label": per_label, "k": k, "status": f"met for {k} of {len(LABELS)}"},
        "4": {"criterion": f"NFR-1: p95 <= {NFR1_P95_MS} ms per issue on a GitHub-hosted Linux runner, "
                           "or documented with the v1.2 ONNX plan",
              "local_p95_ms": p95_ms, "local_p95_within_bound": p95_ms <= NFR1_P95_MS,
              "status": "PENDING (runner not measured; Phase 4)"},
    }


def md_table(header, rows):
    """GitHub markdown table; cells are str()-ed."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def figures(gold, repos, m1_pred, m1_conf, t1_conf, b1_pred, config_gate):
    import plots

    FIGURES.mkdir(parents=True, exist_ok=True)
    correct = np.array([g == p for g, p in zip(gold, m1_pred)], dtype=float)
    t = config_gate["taus"]
    paths = {
        "reliability": FIGURES / f"reliability_M1_{RUN}_test.png",
        "coverage_precision": FIGURES / f"coverage_precision_M1_{RUN}_test.png",
        "confusion": FIGURES / f"confusion_M1_{RUN}_vs_B1_test.png",
        "per_repo_f1": FIGURES / f"per_repo_f1_M1_{RUN}_vs_B1_test.png",
    }
    plots.save(plots.reliability_diagram({f"M1 {RUN}, T=1": (np.array(t1_conf), correct),
                                          f"M1 {RUN}, calibrated": (np.array(m1_conf), correct)},
                                         title="Reliability (test)"), paths["reliability"])
    ap = config_gate["applied"]
    op = f"config τ (bug {t['bug']:.4f}, feature {t['feature']:.2f}, question {t['question']:.2f})"
    plots.save(plots.coverage_precision_plot({f"M1 {RUN}, one global τ": (np.array(m1_conf), correct)},
                                             {op: (ap["coverage"], ap["precision"])},
                                             title="Coverage vs precision (test)"), paths["coverage_precision"])
    plots.save(plots.confusion_matrices({f"M1 {RUN}": (gold, m1_pred), "B1": (gold, b1_pred)},
                                        title="Confusion, normalised by gold (test)"), paths["confusion"])
    plots.save(plots.per_repo_f1_bars({f"M1 {RUN}": (gold, m1_pred, repos), "B1": (gold, b1_pred, repos)},
                                      title="Per-repo macro-F1 (test)"), paths["per_repo_f1"])
    return {k: rel(v) for k, v in paths.items()}


def main():
    m1_meta = json.loads(M1_CSV.with_suffix(".meta.json").read_text())
    cfg_text = CONFIG.read_text()
    assert m1_meta["split"] == "test" and m1_meta["input"]["rows"] == N_TEST, m1_meta["input"]
    assert m1_meta["revision"] == yaml.safe_load(cfg_text)["model"]["revision"]

    ids, gold, m1_pred, m1_probs, repos = read_predictions(M1_CSV)
    t1, m1_conf = load_m1(M1_CSV)
    assert len(ids) == N_TEST and all(c == p[y] for c, p, y in zip(m1_conf, m1_probs, m1_pred))
    assert [LABELS[int(i)] for i in t1.argmax(axis=1)] == m1_pred
    t1_probs = [{c: float(v) for c, v in zip(LABELS, row)} for row in t1]
    inputs = {rel(M1_CSV): sha256_file(M1_CSV), rel(CONFIG): sha256_file(CONFIG),
              rel(VAL_METRICS): sha256_file(VAL_METRICS), rel(PHASE2_MD): sha256_file(PHASE2_MD)}

    # Item 1: M1 metrics, calibrated and at T=1.
    systems = {"M1": {"metrics": metrics.summarize(gold, m1_pred, m1_probs, repos), "meta": m1_meta,
                      "uncalibrated_T1": {"ece": metrics.ece(gold, m1_pred, t1_probs),
                                          "brier": metrics.brier(gold, t1_probs)}}}
    preds, confs = {"M1": m1_pred}, {"M1": m1_conf}
    for s in BASELINES:
        path = PHASE2 / f"predictions_test_{s}.csv"
        b_ids, b_gold, b_pred, b_probs, b_repos = read_predictions(path)
        assert (b_ids, b_gold, b_repos) == (ids, gold, repos), f"{s}: ids/gold/repos differ from M1"
        systems[s] = {"metrics": metrics.summarize(gold, b_pred, b_probs, repos)}
        inputs[rel(path)] = sha256_file(path)
        preds[s], confs[s] = b_pred, [p[y] for p, y in zip(b_probs, b_pred)]

    # Items 2, 5, 6, 7: gating with val-fitted thresholds only.
    val = json.loads(VAL_METRICS.read_text())["gating"]
    taus_cfg = check_declared(config_thresholds(cfg_text), DECLARED["config"], "config")
    taus_point = check_declared(val["M1"]["tau_point"]["taus"], DECLARED["M1_tau_point"], "M1 val tau_point")
    taus_b1 = check_declared(val["B1"]["tau_lb"]["taus"], DECLARED["B1_tau_lb"], "B1 val tau_lb")
    assert taus_cfg == val["M1"]["tau_lb"]["taus"], "config thresholds != val τ_lb in metrics_val.json"
    gate = {"config": gating_view(gold, m1_pred, m1_conf, taus_cfg),
            "M1_tau_point": gating_view(gold, m1_pred, m1_conf, taus_point),
            "B1_same_rule": gating_view(gold, preds["B1"], confs["B1"], taus_b1)}
    th, cov, prec = gating.coverage_precision_curve(m1_conf, [g == p for g, p in zip(gold, m1_pred)])
    gate["M1_curve"] = {"threshold": th.tolist(), "coverage": cov.tolist(), "precision": prec.tolist(),
                        "config_operating_point": {k: gate["config"]["applied"][k] for k in ("coverage", "precision")}}

    # Item 3: paired comparisons.
    paired = {s: {"mcnemar_exact": mcnemar_exact(gold, m1_pred, preds[s]),
                  "bootstrap_xrepo_macro_f1_diff": paired_bootstrap_xrepo(gold, m1_pred, preds[s], repos)}
              for s in PAIRED}

    # Item 8: latency.
    stop_a = {t: json.loads(p.read_text()) for t, p in STOP_A_LATENCY.items()}
    b1_lat = json.loads((PHASE2 / "latency_B1.json").read_text())
    latency = {
        "test_run": {**m1_meta["latency_ms"], "torch_threads": m1_meta["torch_threads"], "load_s": m1_meta["load_s"],
                     "wall_s": m1_meta["wall_s"]},
        "stop_a_100_val_issues_R1": {t: {**d["latency_ms"], "import_s": d["import_s"], "load_s": d["load_s"],
                                         "peak_rss_after_load_MiB": d["peak_rss_after_load_MiB"]}
                                     for t, d in stop_a.items()},
        "B1": {k: b1_lat[k] for k in ("p50_ms", "p95_ms", "n")},
        "hardware": "local Apple M4 Pro CPU, fp32; not a GitHub runner",
    }
    for p in (*STOP_A_LATENCY.values(), PHASE2 / "latency_B1.json"):
        inputs[rel(p)] = sha256_file(p)

    b3 = parse_b3_rows(PHASE2_MD.read_text())
    m1 = systems["M1"]["metrics"]
    xrepo = {s: systems[s]["metrics"]["cross_repo_macro_f1"] for s in systems}
    criteria = judge_criteria(xrepo, paired, m1["ece"], gate["config"], latency["test_run"]["p95"])
    figs = figures(gold, repos, m1_pred, m1_conf, t1.max(axis=1), preds["B1"], gate["config"])

    md = report_markdown(systems, gate, paired, latency, b3, criteria, figs, len(ids))
    print(md)
    out = {"split": "test", "run": RUN, "n": len(ids), "protocol": "results/phase3.md, Test protocol (pre-declared)",
           "inputs_sha256": inputs, "systems": systems, "paired_M1_vs": paired, "gating": gate,
           "latency": latency, "B3_copied_from_phase2_md": b3, "criteria_6_3": criteria, "figures": figs}
    (TEST_DIR / "metrics_test.json").write_text(json.dumps(out, indent=1) + "\n")
    (TEST_DIR / "report_test.md").write_text(md + "\n")
    print(f"\nwrote {rel(TEST_DIR / 'metrics_test.json')}, {rel(TEST_DIR / 'report_test.md')}, "
          f"figures {', '.join(figs.values())}")


def _gate_rows(view):
    a = view["applied"]
    rows = [[lb, f"{v['tau']:.4f}" if v["tau"] < gating.NEVER else "1.01 (never)", v["predicted"], v["applied"],
             f4(v["coverage"]), f4(v["precision"])] for lb, v in a["per_label"].items()]
    rows.append(["**overall**", "", a["n"], f"{a['applied']} applied / {a['escalated']} escalated",
                 f"**{f4(a['coverage'])}**", f"**{f4(a['precision'])}**"])
    return rows


def report_markdown(systems, gate, paired, latency, b3, criteria, figs, n):
    m1 = systems["M1"]["metrics"]
    cfg = gate["config"]["applied"]
    b1g = gate["B1_same_rule"]["applied"]
    lat = latency["test_run"]
    out = [(f"Test, n = {n} (M1 = R1b `{systems['M1']['meta']['revision']}`; baselines from committed Phase 2 "
            "predictions; B3 copied from results/phase2.md)")]

    # §10.5 results table (item 4).
    def cov_cell(a):
        return f"{f4(a['coverage'])} (prec {f4(a['precision'])})"

    rows = []
    for s in ("B0", "B1", "B2_512-192", "B2_1024-256", "M1"):
        m = systems[s]["metrics"]
        pc = m["per_class"]
        cov = {"M1": cov_cell(cfg), "B1": cov_cell(b1g) + " [s7]"}.get(s, "—")
        p50 = {"M1": f"{lat['p50']:.0f}", "B1": f"{latency['B1']['p50_ms']:.2f} [a]"}.get(s, "— [b]" if s != "B0" else "—")
        ece = "—" if s == "B0" else f4(m["ece"])
        brier = "—" if s == "B0" else f4(m["brier"])
        name = f"**{NAMES[s]}**" if s == "M1" else NAMES[s]
        rows.append([name, f4(m["cross_repo_macro_f1"]), f4(m["pooled_macro_f1"]), f4(m["accuracy"]),
                     f4(pc["bug"]["f1"]), f4(pc["feature"]["f1"]), f4(pc["question"]["f1"]), ece, brier, cov, p50])
        if s == "B2_1024-256":
            for r in b3["published"]:
                rows.append([r["system"], f4(r["cross_repo_macro_f1"]),
                             *(f"{f4(r[k])} [c]" for k in B3_COLUMNS[1:]), "—", "—", "—", "—"])
    out += ["", "§10.5 results table (test)", "",
            md_table(["System", "Cross-repo macro-F1", "Pooled macro-F1", "Acc", "F1 bug", "F1 feature", "F1 question",
                      "ECE", "Brier", "Coverage @ config τ (precision of auto-applied)", "CPU p50 ms"], rows)]
    for r in b3["footnote"]:
        out.append(f"\nFootnote row (provenance unknown; not averaged or chosen): {r['system']}: cross-repo "
                   f"{f4(r['cross_repo_macro_f1'])}, pooled {f4(r['pooled_macro_f1'])} [c], acc {f4(r['accuracy'])} [c], "
                   f"F1 bug/feature/question {f4(r['f1_bug'])} / {f4(r['f1_feature'])} / {f4(r['f1_question'])} [c].")
    out += ["",
            "- [a] B1 latency from results/phase2/latency_B1.json (first 100 val issues, local CPU).",
            "- [b] B2 latency not measured in Phase 2 or 3 (results/phase2.md cites Phase 0 under different conditions).",
            ("- [c] Derived, not published (results/phase2.md [c]): pooled per-class values recovered from the "
             "published per-repo P/R/F1; ECE, Brier and coverage unavailable for B3."),
            "- [s7] Secondary view 7: B1 under the same rule with its own val τ_lb.",
            "- B0's ECE/Brier are computed from train priors and carry no information (see metrics_test.json)."]

    pc = m1["per_class"]
    out += ["", "M1 per-class (test)", "",
            md_table(["Label", "Precision", "Recall", "F1", "Support", "Predicted"],
                     [[c, f4(pc[c]["precision"]), f4(pc[c]["recall"]), f4(pc[c]["f1"]), pc[c]["support"],
                       m1["predicted_counts"][c]] for c in LABELS])]

    pr_rows = []
    for s, name in (("M1", NAMES["M1"]), ("B1", NAMES["B1"]), ("B2_512-192", NAMES["B2_512-192"]),
                    ("B2_1024-256", NAMES["B2_1024-256"]), ("B0", NAMES["B0"])):
        prm = systems[s]["metrics"]["per_repo_macro_f1"]
        pr_rows.append([name, *(f4(v) for v in prm.values()), f4(systems[s]["metrics"]["cross_repo_macro_f1"])])
    out += ["", "Per-repo macro-F1 (test)", "",
            md_table(["System", *(r.split("/")[-1] for r in m1["per_repo_macro_f1"]), "Cross-repo"], pr_rows)]

    out += ["", "Confusion counts (rows gold, cols predicted; bug/feature/question)", "",
            md_table(["System", "Confusion"],
                     [[NAMES[s], f"`{systems[s]['metrics']['confusion']['rows_gold_cols_pred']}`"] for s in ("M1", "B1")])]

    t = systems["M1"]["meta"]["applied_choice_temperature_3_options"]
    u = systems["M1"]["uncalibrated_T1"]
    out += ["", "M1 calibration (test, 15 bins)", "",
            md_table(["Probabilities", "ECE", "Brier"],
                     [[f"calibrated (T = {t})", f4(m1["ece"]), f4(m1["brier"])],
                      ["uncalibrated (T = 1)", f4(u["ece"]), f4(u["brier"])]])]

    p_rows = []
    for s in PAIRED:
        mc, bs = paired[s]["mcnemar_exact"], paired[s]["bootstrap_xrepo_macro_f1_diff"]
        p_rows.append([f"M1 vs {NAMES[s]}", f"{bs['point']:+.4f}", f"[{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}]",
                       f"{bs['share_resamples_a_better']:.1%}", mc["a_right_b_wrong"], mc["a_wrong_b_right"],
                       f"{mc['p_value']:.3g}"])
    out += ["", "Paired comparisons (test; bootstrap 2,000 resamples of issues, seed 42; exact McNemar)", "",
            md_table(["Comparison", "Δ cross-repo macro-F1", "95% CI", "M1 better in", "M1 right / other wrong",
                      "M1 wrong / other right", "McNemar p"], p_rows)]

    out += ["", "Gating at the config thresholds (primary item 2)", "",
            md_table(["Label", "τ", "Predicted", "Applied", "Coverage", "Precision"], _gate_rows(gate["config"]))]
    out += ["", "Secondary 5: M1 at val τ_point", "",
            md_table(["Label", "τ", "Predicted", "Applied", "Coverage", "Precision"], _gate_rows(gate["M1_tau_point"]))]
    out += ["", "Secondary 7: B1 at its own val τ_lb", "",
            md_table(["Label", "τ", "Predicted", "Applied", "Coverage", "Precision"], _gate_rows(gate["B1_same_rule"]))]

    curve = gate["M1_curve"]
    cv, pv, tv = (np.array(curve[k]) for k in ("coverage", "precision", "threshold"))
    c_rows = []
    for target in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
        i = int(np.searchsorted(cv, target - 1e-12))
        c_rows.append([f"{target:.2f}", f"{tv[i]:.4f}", f4(cv[i]), f4(pv[i])])
    out += ["", (f"Secondary 6: M1 coverage–precision curve, one global τ ({len(tv)} points in metrics_test.json; "
                 "rows = first point reaching each coverage)"), "",
            md_table(["Coverage ≥", "τ", "Coverage", "Precision"], c_rows),
            f"\nConfig operating point: coverage {f4(cfg['coverage'])}, precision {f4(cfg['precision'])}."]

    sa = latency["stop_a_100_val_issues_R1"]
    out += ["", "Secondary 8: latency (local Apple M4 Pro CPU, fp32, model preloaded; not a GitHub runner)", "",
            md_table(["Measurement", "Threads", "n", "p50 ms", "p95 ms", "mean ms", "max ms"],
                     [["Test run, M1 R1b (meta)", lat["torch_threads"], lat["n"], f"{lat['p50']:.0f}", f"{lat['p95']:.0f}",
                       f"{lat['mean']:.0f}", f"{lat['max']:.0f}"],
                      *[[f"Stop A, R1, first 100 val issues ({k})", d["threads"], d["n"], f"{d['p50']:.0f}",
                         f"{d['p95']:.0f}", f"{d['mean']:.0f}", f"{d['max']:.0f}"] for k, d in sa.items()]]),
            (f"\nTest-run load {lat['load_s']} s, wall {lat['wall_s']} s. Cold load (Stop A, fresh process, default "
             f"threads): import laya {sa['default']['import_s']} s, laya.load {sa['default']['load_s']} s, peak RSS "
             f"after load {sa['default']['peak_rss_after_load_MiB']} MiB.")]

    out += ["", "Figures: " + ", ".join(figs.values())]

    c = criteria
    c1 = "; ".join(f"vs {NAMES[s]} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                   for s, v in c["1"]["comparisons"].items())
    c3 = "; ".join(f"{lb}: τ {format(v['tau'], '.4f') if v['val_threshold_exists'] else '1.01 (no val threshold)'}, "
                   f"applied {v['applied']}, coverage {f4(v['coverage'])}, precision {f4(v['precision'])} "
                   f"-> {'met' if v['met'] else 'not met'}" for lb, v in c["3"]["per_label"].items())
    out += ["", "§6.3 criteria 1-4 (test)", "",
            md_table(["#", "Criterion", "Numbers", "Status"],
                     [["1", c["1"]["criterion"], f"M1 {f4(m1['cross_repo_macro_f1'])}; {c1}", c["1"]["status"]],
                      ["2", c["2"]["criterion"], f"ECE {f4(c['2']['ece'])}", c["2"]["status"]],
                      ["3", c["3"]["criterion"], c3, c["3"]["status"]],
                      ["4", c["4"]["criterion"],
                       (f"local p95 {c['4']['local_p95_ms']:.0f} ms "
                        f"({'≤' if c['4']['local_p95_within_bound'] else '>'} {NFR1_P95_MS} ms locally)"),
                       c["4"]["status"]]])]
    return "\n".join(out)


if __name__ == "__main__":
    main()
