"""Build docs/MODEL_CARD.md, the Hugging Face README of Prasanna85/laya-issue-triage (PROJECT_SPEC.md §13 Phase 3
task 5). The owner uploads it; this script never touches the network.

Run from anywhere: .venv/bin/python results/phase3/make_model_card.py [--check]
Every number comes from a file, never from this script:
- results/phase3/test_R1b/metrics_test.json: test metrics, paired comparisons, gating on test, latency, B3 rows;
- results/phase3/val_R1b/metrics_val.json: val metrics and the §10.4 threshold fit;
- config/triage.default.yml: model repo / revision / max_len, shipped thresholds, preprocessing limits;
- results/phase3/refit/refit_summary.json: notebook vs val-refit temperature;
- results/phase3/items_sha256.json: training rows and label counts, laya / transformers pins of the items build;
- results/phase2.md: how the B3 baselines were trained (rows per classifier, input text), via `b3_training`;
- results/experiments.md, row R1: epochs, label smoothing, accumulation, seed, base revision, trained revision,
  optimizer steps, Kaggle GPUs, train time;
- PROJECT_SPEC.md §8.3 (official train size, dropped rows) and §9.3 (effective batch, Kaggle versions), the only
  places those settings are recorded in the repo;
- docs/snippets/classify_one.py: the laya-triage package example, inserted verbatim (the same text as README.md and
  docs/USING_THE_MODEL.md, built by docs/make_readme.py).
A missing key, row or pattern raises (KeyError / ValueError) naming the file. Consistency checks: config revision
== test-run revision, config thresholds == thresholds applied on test, train + val + dropped == official train,
spec accumulation steps == ledger grad_accum, spec GPUs == ledger GPUs.
--check exits 1 if docs/MODEL_CARD.md differs from a fresh build instead of writing it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))

import gating
from metrics import ECE_BINS

from laya_triage.preprocess import CODE_HEAD_LINES, CODE_TAIL_LINES
from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

CARD = ROOT / "docs" / "MODEL_CARD.md"
SOURCES = {
    "test": ROOT / "results" / "phase3" / "test_R1b" / "metrics_test.json",
    "val": ROOT / "results" / "phase3" / "val_R1b" / "metrics_val.json",
    "config": ROOT / "config" / "triage.default.yml",
    "refit": ROOT / "results" / "phase3" / "refit" / "refit_summary.json",
    "items": ROOT / "results" / "phase3" / "items_sha256.json",
    "experiments": ROOT / "results" / "experiments.md",
    "phase2": ROOT / "results" / "phase2.md",
    "spec": ROOT / "PROJECT_SPEC.md",
    "classify_snippet": ROOT / "docs" / "snippets" / "classify_one.py",
}
BASELINES = (("B1", "B1 TF-IDF + logistic regression"), ("B2_512-192", "B2 Laya base, zero-shot, 512/192 (native)"),
             ("B2_1024-256", "B2 Laya base, zero-shot, 1024/256"))
CFG = "config/triage.default.yml"
# Citation texts supplied verbatim by the owner (not generated; edit only to replace them verbatim).
LAYA_CITATION = ("Convai Innovations / NandhaKishorM. Laya: non-autoregressive decision models, v0.3.23 (Apache-2.0). "
                 "https://github.com/NandhaKishorM/laya . Checkpoints: https://huggingface.co/convaiinnovations/laya")
NLBSE24_BIBTEX = """@inproceedings{nlbse2024,
  author={Kallis, Rafael and Colavito, Giuseppe and Al-Kaswan, Ali and Pascarella, Luca and Chaparro, Oscar and Rani, Pooja},
  title={The NLBSE'24 Tool Competition},
  booktitle={Proceedings of The 3rd International Workshop on Natural Language-based Software Engineering (NLBSE'24)},
  year={2024}
}"""
LEDGER = "results/experiments.md R1"
GITHUB_URL = "https://github.com/Prasanna-KS-85/laya-triage"  # also used by docs/make_readme.py


def need(d, path, src):
    """d[k1][k2]... for a '.'-separated path, or a tuple path when a key contains '.'; KeyError names file and path."""
    cur = d
    for k in (path if isinstance(path, tuple) else path.split(".")):
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            raise KeyError(f"{src}: missing key {path!r} (at {k!r})") from None
    return cur


def search(pattern, text, src):
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"{src}: pattern not found: {pattern}")
    return m


def ledger_row(md, run_id, src="results/experiments.md"):
    """{column header: cell} of the ledger row whose run_id is `run_id`."""
    header = None
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == "run_id":
            header = cells
        elif header and cells and cells[0] == run_id:
            if len(cells) != len(header):
                raise ValueError(f"{src}: row {run_id} has {len(cells)} cells, header has {len(header)}")
            return dict(zip(header, cells))
    raise ValueError(f"{src}: no row {run_id!r}")


def b3_training(phase2_md):
    """How the published B3 baselines were trained, as recorded in results/phase2.md: rows per classifier (SetFit
    template, one classifier per repository) and whether the SetFit input is raw text. Raises if phase2.md changes."""
    src = "results/phase2.md"
    rows = search(r"classifier is trained only on \*\*that repo's (\d+) official-train rows\*\*", phase2_md, src)
    search(r"one\s+classifier per repo", phase2_md, src)
    search(r"raw\s+`title \+ \" \" \+ body` \(no cleaning\)", phase2_md, src)
    search(r"setups of the RoBERTa and fastText baselines were not read", phase2_md, src)
    search(r"including rows 559, 900, 901 and 1114,\s+whose content is also in official test", phase2_md, src)
    return {"rows_per_classifier": int(rows.group(1))}


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def facts(test, val, config, refit, items, experiments_md, spec_md):
    """The non-metric values the card uses, read and cross-checked from the parsed sources."""
    t = "metrics_test.json"
    meta = need(test, "systems.M1.meta", t)
    taus = {lb: float(need(config, f"gating.thresholds.{lb}", CFG)) for lb in LABELS}
    rev = need(config, "model.revision", CFG)
    if rev != need(meta, "revision", t):
        raise ValueError(f"config model.revision {rev} != test-run revision {meta['revision']}")
    if taus != need(test, "gating.config.taus", t):
        raise ValueError("config thresholds != thresholds applied on test")
    r1 = ledger_row(experiments_md, "R1")
    notes = need(r1, "notes", LEDGER)
    eb = search(r"\| Effective batch \| (\d+) sequences \((\d+) per micro-batch × (\d+) GPUs × (\d+) accumulation steps\)",
                spec_md, "PROJECT_SPEC.md §9.3")
    kaggle = search(r"\(Kaggle: Python ([\d.]+), torch ([^,]+), (\d+)× T4\)", spec_md, "PROJECT_SPEC.md §9.3")
    val_seed = int(search(r"stratified by `repo × label`, seed (\d+)", spec_md, "PROJECT_SPEC.md §8.3").group(1))
    official = search(r"Official train has ([\d,]+) rows\. The (\d+) rows", spec_md, "PROJECT_SPEC.md §8.3")
    n_official, n_dropped = int(official.group(1).replace(",", "")), int(official.group(2))
    eps = need(r1, "eps", LEDGER)
    out = {
        "repo": need(config, "model.repo", CFG), "revision": rev, "max_len": need(config, "model.max_len", CFG),
        "head_max_len": need(meta, "head_max_len", t),
        "max_code_lines": need(config, "preprocess.max_code_lines", CFG),
        "max_body_chars": need(config, "preprocess.max_body_chars", CFG),
        "taus": taus,
        "trained_revision": need(r1, "HF revision", LEDGER).strip("`"),
        "base_revision": need(r1, "base_rev", LEDGER).strip("`"),
        "epochs": int(need(r1, "epochs", LEDGER)), "eps": eps, "grad_accum": int(need(r1, "grad_accum", LEDGER)),
        "seed": int(need(r1, "seed", LEDGER)),
        "steps": int(search(r"(\d+) optimizer steps", notes, LEDGER).group(1)),
        "train_seconds": float(search(r"train_seconds ([\d.]+)", notes, LEDGER).group(1)),
        "n_gpus": int(search(r"Kaggle (\d+)× T4", notes, LEDGER).group(1)),
        "n_calibration": int(search(r"on (\d+) calibration items", notes, LEDGER).group(1)),
        "val_seed": val_seed, "hardware": need(test, "latency.hardware", t),
        "eff_batch": int(eb.group(1)), "micro_batch": int(eb.group(2)),
        "kaggle_python": kaggle.group(1), "kaggle_torch": kaggle.group(2),
        "items_laya": need(items, "laya", "items_sha256.json"),
        "items_transformers": need(items, "transformers", "items_sha256.json"),
        "train_label_counts": need(items, ("items", eps, "label_counts"), "items_sha256.json"),
        "n_official": n_official, "n_dropped": n_dropped,
        "n_train": need(items, ("files", "train.jsonl", "rows"), "items_sha256.json"),
        "n_val": need(val, "n", "metrics_val.json"), "n_test": need(test, "n", t),
        "T_notebook": need(refit, "T_notebook", "refit_summary.json"), "T_val": need(refit, "T_val", "refit_summary.json"),
        "T_applied": need(meta, "applied_choice_temperature_3_options", t),
        "share_truncated": need(meta, "share_truncated", t), "eval_versions": need(meta, "versions", t),
    }
    if out["n_train"] + out["n_val"] + n_dropped != n_official:
        raise ValueError(f"train {out['n_train']} + val {out['n_val']} + dropped {n_dropped} != official {n_official}")
    if int(eb.group(4)) != out["grad_accum"] or int(eb.group(3)) != out["n_gpus"]:
        raise ValueError("PROJECT_SPEC.md §9.3 GPUs / accumulation steps != experiments.md R1")
    if out["T_applied"] != out["T_val"]:
        raise ValueError(f"test-run temperature {out['T_applied']} != val refit {out['T_val']}")
    return out


def build_card(test, val, config, refit, items, experiments_md, spec_md, phase2_md, classify_snippet):
    """The model card (markdown with HF front matter) from the parsed sources; raises on any missing value."""
    t, v = "metrics_test.json", "metrics_val.json"
    F = facts(test, val, config, refit, items, experiments_md, spec_md)
    b3t = b3_training(phase2_md)
    tag = search(r"uses: Prasanna-KS-85/laya-triage@<[^>]*> # (v[\d.]+)", spec_md, "PROJECT_SPEC.md §12.5").group(1)
    sysm = {s: need(test, f"systems.{s}.metrics", t) for s in ("M1", *(b for b, _ in BASELINES))}
    m1 = sysm["M1"]
    t1 = need(test, "systems.M1.uncalibrated_T1", t)
    paired = {b: need(test, f"paired_M1_vs.{b}", t) for b, _ in BASELINES}
    gate = need(test, "gating.config.applied", t)
    gate_point = need(test, "gating.M1_tau_point.applied", t)
    lat = need(test, "latency", t)
    b3 = need(test, "B3_copied_from_phase2_md.published", t)
    crit = need(test, "criteria_6_3", t)
    mode = need(config, "behaviour.mode", CFG)
    setfit = [r for r in b3 if need(r, "system", t).split()[1] == "SetFit"]
    if len(setfit) != 1:
        raise ValueError(f"{t}: expected one published SetFit row, found {len(setfit)}")
    setfit_gap = need(m1, "cross_repo_macro_f1", t) - need(setfit[0], "cross_repo_macro_f1", t)
    cis = [need(paired[b], "bootstrap_xrepo_macro_f1_diff.ci95", t) for b, _ in BASELINES]
    ci_text = ("all three CIs exclude 0" if all(lo > 0 or hi < 0 for lo, hi in cis)
               else "not every CI excludes 0")
    beats = ("beats B1 and both B2 settings" if need(crit, "1.status", t) == "MET"
             else "does not beat B1 and both B2 settings")
    val_fit = {lb: need(val, f"gating.M1.per_label.{lb}", v) for lb in LABELS}
    val_xrepo = need(val, "systems.M1.metrics.cross_repo_macro_f1", v)
    bug_lb = need(val_fit["bug"], "at_tau_lb", v)
    repos = list(need(m1, "per_repo_macro_f1", t))
    pc = need(m1, "per_class", t)
    supports = {need(pc, f"{lb}.support", t) for lb in LABELS}
    if len(supports) != 1 or next(iter(supports)) % len(repos):
        raise ValueError(f"test is not balanced per class and repo: supports {supports}, {len(repos)} repos")
    per_cell = next(iter(supports)) // len(repos)
    question_json = json.dumps(ISSUE_TYPE_QUESTION, indent=4, ensure_ascii=False)
    tr = F["train_label_counts"]
    ev = F["eval_versions"]
    qid = next(iter(ISSUE_TYPE_QUESTION))

    def row(name, m, ece=True):
        c = need(m, "per_class", t)
        return (f"| {name} | {f4(need(m, 'cross_repo_macro_f1', t))} | {f4(need(m, 'pooled_macro_f1', t))} | "
                f"{f4(need(m, 'accuracy', t))} | " + " | ".join(f4(need(c, f'{lb}.f1', t)) for lb in LABELS) +
                f" | {f4(need(m, 'ece', t)) if ece else '—'} |")

    results = [row("**M1 (this model)**", m1), *(row(name, sysm[b]) for b, name in BASELINES)]
    for r in b3:
        results.append(f"| B3 {need(r, 'system', t).split()[1]} (NLBSE'24, published) | {f4(need(r, 'cross_repo_macro_f1', t))} | "
                       + " | ".join(f"{f4(need(r, k, t))} [c]" for k in ("pooled_macro_f1", "accuracy", "f1_bug",
                                                                        "f1_feature", "f1_question")) + " | — |")
    per_repo = [f"| {name} | " + " | ".join(f4(need(sysm[s], ('per_repo_macro_f1', r), t)) for r in repos) +
                f" | {f4(need(sysm[s], 'cross_repo_macro_f1', t))} |"
                for s, name in (("M1", "**M1**"), *BASELINES)]
    paired_rows = []
    for b, name in BASELINES:
        bs, mc = need(paired[b], "bootstrap_xrepo_macro_f1_diff", t), need(paired[b], "mcnemar_exact", t)
        ci = need(bs, "ci95", t)
        paired_rows.append(f"| M1 vs {name} | {need(bs, 'point', t):+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                           f"{need(mc, 'p_value', t):.2g} |")
    n_boot, seed = (need(paired["B1"], f"bootstrap_xrepo_macro_f1_diff.{k}", t) for k in ("n_boot", "seed"))
    gate_rows = []
    for lb in LABELS:
        g = need(gate, f"per_label.{lb}", t)
        tau = F["taus"][lb]
        gate_rows.append(f"| {lb} | {f'never ({gating.NEVER})' if tau >= gating.NEVER else f'{tau:.4f}'} | "
                         f"{f4(bug_lb['lower_bound']) if lb == 'bug' else f'none ≥ {gating.TARGET_PRECISION:.2f}'} | {need(g, 'applied', t)} | "
                         f"{f4(need(g, 'coverage', t))} | {f4(need(g, 'precision', t))} |")
    sa = need(lat, "stop_a_100_val_issues_R1", t)
    tr_lat = need(lat, "test_run", t)
    bug_test_prec = need(gate, "per_label.bug.precision", t)
    q_labels = ", ".join(f"`{lb}`" for lb in LABELS)

    return f"""---
license: apache-2.0
base_model: convaiinnovations/laya
language:
- en
tags:
- laya
- text-classification
- github-issues
- issue-triage
- nlbse
---

# laya-issue-triage

<!-- Generated by results/phase3/make_model_card.py from files under results/ and config/; do not edit by hand. -->

## Summary

A fine-tuned [Laya](https://huggingface.co/convaiinnovations/laya) decision model that classifies a newly
opened GitHub issue as one of {q_labels} from its title and body. It is the model behind the
[laya-triage]({GITHUB_URL}) GitHub Action, which auto-applies a label only when `answer_confidence` reaches a
per-label threshold and otherwise escalates the issue to a human. Source code, install instructions and the
full evaluation write-up: {GITHUB_URL}.

- Revision `{F['revision']}`: weights of training run R1 (`{F['trained_revision']}`) with the choice
  temperature refit on validation data ({F['T_notebook']:.4f} → {F['T_val']}); nothing else differs.
- On the NLBSE'24 test split ({F['n_test']:,} issues, 5 repositories) it reaches cross-repo macro-F1
  **{f4(m1['cross_repo_macro_f1'])}**, accuracy {f4(m1['accuracy'])}, and calibrated ECE {f4(m1['ece'])}.
- With the shipped thresholds, it auto-labels {f4(gate['coverage'])} of test issues (only `bug`) at precision
  **{f4(gate['precision'])}**. This **misses the {gating.TARGET_PRECISION:.2f} precision target** (see Gating).

## Intended use

- Suggesting or applying a type label (`bug`, `feature`, `question`) to new issues in English-language
  software repositories, through the laya-triage Action with the confidence gate and a human escalation path.
  The Action's default config sets `mode: {mode}`.
- Research on calibrated, gated issue classification.

## Not intended use

- Fully automatic labelling without a confidence gate or human review.
- Any decision about people (for example rating contributors or prioritising by author).
- Issue types other than these three (documentation, security reports, and so on), or non-English issues.
- Repositories very different from the {len(repos)} in the training data without first checking precision on a
  sample of that repository's own issues.

## How to load

CPU, fp32, pinned revision. Download once, then run offline. The question below, including its wording and
the option order {q_labels}, is part of the model's contract: it was frozen before training and must be
passed exactly as written. The state is the issue cleaned by laya-triage's `preprocess()` (see Training data).

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"  # after one online download of this revision
os.environ.pop("LAYA_CPU_AMP", None)  # keep CPU inference in fp32

import laya

REPO = "{F['repo']}"
REVISION = "{F['revision']}"
ISSUE_TYPE_QUESTION = {question_json}

agent = laya.load(REPO, revision=REVISION, device="cpu")  # max_len {F['max_len']}, head_max_len {F['head_max_len']}
state = {{"title": "...", "body": "..."}}  # from laya_triage.preprocess.preprocess(title, body)
answer = agent.predict(state, ISSUE_TYPE_QUESTION)["answers"]["{qid}"]
answer["choice"], answer["answer_confidence"], answer["probabilities"]
```

Gate on `answer_confidence` (the calibrated probability of the chosen label), not on `confidence`.

### With the laya-triage package

The [laya-triage]({GITHUB_URL}) package wraps the same steps: it brings the frozen question and `preprocess()`, so
nothing is retyped. Install it with `pip install "laya_triage @ git+{GITHUB_URL}@{tag}"` (on Linux, install
the CPU-only torch wheel first; see the repository's docs/USING_THE_MODEL.md), then:

```python
{classify_snippet.rstrip()}
```

## Training data

- [NLBSE'24 issue report classification](https://github.com/nlbse2024/issue-report-classification): issues
  from bitcoin, react, vscode, opencv and tensorflow, labelled `bug`, `feature` or `question`.
- Of the {F['n_official']:,} official-train rows, {F['n_dropped']} are dropped (their content also appears in
  the official test set). {F['n_val']} rows are held out as validation (stratified by repo × label, seed {F['val_seed']}),
  which leaves **{F['n_train']:,} training rows** (bug {tr['bug']}, feature {tr['feature']}, question {tr['question']}).
  The official test set ({F['n_test']:,} rows) is used only for the single final evaluation.
- Preprocessing (shared by training, evaluation and the Action): strip HTML comments; collapse fenced code
  blocks longer than {F['max_code_lines']} lines to their first {CODE_HEAD_LINES} and last {CODE_TAIL_LINES} lines; replace images and bare
  URLs with placeholders; normalise whitespace; cap the body at {F['max_body_chars']:,} characters. The state
  is `{{"title": ..., "body": ...}}`, and Laya truncates it to {F['max_len']} tokens ({100 * F['share_truncated']:.1f}% of
  test states are truncated).
- **License and redistribution.** The dataset's upstream `LICENSE` file is empty, so no license is granted.
  The issue text belongs to the source projects and their authors. No data is redistributed with this model
  or its repository; only synthetic IDs and content hashes are published.

## Training procedure

| Setting | Value |
|---|---|
| Base | `convaiinnovations/laya` @ `{F['base_revision']}` |
| Recipe | Laya Kaggle notebook (RLCD policy-gradient term + soft cross-entropy on one-hot gold), label smoothing ε {F['eps']} |
| Epochs / optimizer steps | {F['epochs']} / {F['steps']} |
| Effective batch | {F['eff_batch']} ({F['micro_batch']} per micro-batch × {F['n_gpus']} GPUs × {F['grad_accum']} accumulation steps) |
| Sequence budget | max_len {F['max_len']}, head_max_len {F['head_max_len']} |
| Seed | {F['seed']} (GPU kernels are not deterministic, so runs are seeded but not bit-for-bit reproducible) |
| Hardware / time | Kaggle, {F['n_gpus']}× T4; train time {F['train_seconds']:.0f} s |
| Versions (training) | laya {F['items_laya']}, transformers {F['items_transformers']}, Python {F['kaggle_python']}, torch {F['kaggle_torch']} |
| Versions (evaluation) | laya {ev['laya']}, torch {ev['torch']}, transformers {ev['transformers']}, Python {ev['python']}; CPU fp32 |
| Calibration | The notebook's choice temperature {F['T_notebook']:.4f} (fitted on {F['n_calibration']} items held out from the training rows) was replaced by **{F['T_val']}**, refit by NLL on the {F['n_val']} validation issues with `laya.calibrate`. Weights unchanged. |

## Evaluation

Single run on the NLBSE'24 official test split ({F['n_test']:,} issues: {per_cell} per class in each of {len(repos)} repositories), CPU fp32,
after the thresholds were frozen. The headline metric is **cross-repo macro-F1**: the mean of the {len(repos)}
per-repository macro-F1 scores, as in the NLBSE'24 competition.

| System | Cross-repo macro-F1 | Pooled macro-F1 | Accuracy | F1 bug | F1 feature | F1 question | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(results)}

- [c] Derived, not published. NLBSE'24 publishes per-repository, per-class P/R/F1 only; pooled values were
  recovered exactly from them, because every test repository has {per_cell} issues per class.
- **Protocol differences.** B3 trains one classifier per repository, each on only that repository's
  {b3t['rows_per_classifier']} official-train rows (together the {len(repos)} classifiers use all {F['n_official']:,} rows, including the
  {F['n_dropped']} whose content also appears in test). The SetFit input is raw text; the RoBERTa and fastText
  setups were not inspected. B3 is copied rather than re-run. M1 and B1 are one model across all {len(repos)} repositories, trained on {F['n_train']:,} rows with model selection,
  temperature and thresholds fitted on the {F['n_val']}-issue validation split. B2 is the base checkpoint
  without fine-tuning. B3 publishes no probabilities, so it has no ECE.
- Per class (M1), precision / recall: {'; '.join(f"{lb} {f4(pc[lb]['precision'])} / {f4(pc[lb]['recall'])}" for lb in LABELS)}.

Per-repository macro-F1 (test):

| System | {' | '.join(r.split('/')[-1] for r in repos)} | Cross-repo |
|---|{'---:|' * (len(repos) + 1)}
{chr(10).join(per_repo)}

Calibration (test, {ECE_BINS} bins):

| Probabilities | ECE | Brier |
|---|---:|---:|
| Calibrated (T = {F['T_applied']}, shipped) | {f4(m1['ece'])} | {f4(m1['brier'])} |
| Uncalibrated (T = 1) | {f4(t1['ece'])} | {f4(t1['brier'])} |

Paired comparisons (test): difference in cross-repo macro-F1, with a 95% CI from a paired bootstrap
({n_boot:,} resamples of issues, seed {seed}), and the exact McNemar p-value.

| Comparison | Δ cross-repo macro-F1 | 95% CI | McNemar p |
|---|---:|---|---:|
{chr(10).join(paired_rows)}

M1 {beats} on the headline metric, and {ci_text}. Relative to the published SetFit
baseline (B3) it is {100 * setfit_gap:+.1f} points (not a paired comparison; see the protocol differences).

## Gating

Per-label thresholds on `answer_confidence` were fitted on validation before the test run: the smallest
threshold whose bootstrap lower bound of precision ({gating.LOWER_PERCENTILE}th percentile of {gating.N_BOOTSTRAP:,} resamples, at
least {gating.MIN_SUPPORT} issues) is ≥ {gating.TARGET_PRECISION:.2f}; {gating.NEVER} means the label is never auto-applied.

| Label | Threshold | Val lower bound | Applied on test | Test coverage | Test precision |
|---|---|---:|---:|---:|---:|
{chr(10).join(gate_rows)}
| **all** | | | {gate['applied']} of {gate['n']:,} | **{f4(gate['coverage'])}** | **{f4(gate['precision'])}** |

**The {gating.TARGET_PRECISION:.2f} precision target was missed on test.** The `bug` threshold had validation precision
{f4(bug_lb['precision'])} (lower bound {f4(bug_lb['lower_bound'])}, {bug_lb['support']} issues, in-sample) but gave
{f4(bug_test_prec)} on test. `feature` and `question` never reached the lower bound on validation, so they are
always escalated (project success criterion 3: {crit['3']['status']} labels). For comparison, the point-estimate
thresholds from validation would auto-label {f4(gate_point['coverage'])} of test issues at precision
{f4(gate_point['precision'])}. Anyone using this model should verify precision on their own repository before
switching the Action from dry-run to apply.

## Limitations

- **Balanced data.** Test has exactly {per_cell} issues per class per repository; train is near-balanced
  (bug {tr['bug']}, feature {tr['feature']}, question {tr['question']}). Real repositories have very different class mixes, so precision at a given threshold will differ.
- **Random, not temporal, split.** The numbers do not measure drift to future issues.
- **{len(repos)} repositories.** Large, well-known projects only; performance elsewhere is unknown.
- **Label noise**, especially for `question`. Labels come from upstream repository labels. Some issues are
  plausibly mislabelled, and the author's own template choice often disagrees with a `question` label.
  `question` is the weakest class (test F1 {f4(pc['question']['f1'])}).
- **Validation overstated test performance.** Cross-repo macro-F1 was {f4(val_xrepo)} on the
  {F['n_val']}-issue validation split and {f4(m1['cross_repo_macro_f1'])} on test.
- **Thresholds fitted on validation do not transfer exactly.** The `bug` threshold lost
  {100 * (bug_lb['precision'] - bug_test_prec):.1f} points of precision from validation to test.
- **English only.** Trained and evaluated on English issues.
- **Latency was measured on a local CPU, not on a GitHub runner** ({F['hardware'].split(';')[0]}, model preloaded):
  p50 {tr_lat['p50']:.0f} ms and p95 {tr_lat['p95']:.0f} ms per issue at {tr_lat['torch_threads']} threads
  (test run); p50 {sa['2']['p50']:.0f} ms and p95 {sa['2']['p95']:.0f} ms at 2 threads (100 validation issues).
  Cold load: `laya.load` {sa['default']['load_s']} s, peak RSS {sa['default']['peak_rss_after_load_MiB']:,} MiB.
- **{100 * F['share_truncated']:.1f}% of test issues are truncated** at {F['max_len']} tokens; text beyond that
  is not seen.

## Citation

Laya:

> {LAYA_CITATION}

NLBSE'24:

```bibtex
{NLBSE24_BIBTEX}
```
"""


def load_sources():
    return {
        "test": json.loads(SOURCES["test"].read_text()),
        "val": json.loads(SOURCES["val"].read_text()),
        "config": yaml.safe_load(SOURCES["config"].read_text()),
        "refit": json.loads(SOURCES["refit"].read_text()),
        "items": json.loads(SOURCES["items"].read_text()),
        "experiments_md": SOURCES["experiments"].read_text(),
        "phase2_md": SOURCES["phase2"].read_text(),
        "spec_md": SOURCES["spec"].read_text(),
        "classify_snippet": SOURCES["classify_snippet"].read_text(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/MODEL_CARD.md is out of date")
    args = ap.parse_args(argv)
    card = build_card(**load_sources())
    rel = CARD.relative_to(ROOT)
    if args.check:
        if not CARD.exists() or CARD.read_text() != card:
            sys.exit(f"{rel} is out of date; run results/phase3/make_model_card.py")
        print(f"{rel} is up to date")
        return
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(card)
    print(f"wrote {rel} ({len(card.splitlines())} lines)")


if __name__ == "__main__":
    main()
