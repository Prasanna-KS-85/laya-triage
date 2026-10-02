# Phase 2 results: baselines

Spec: PROJECT_SPEC.md §10.1–§10.2, §13 Phase 2 (v1.0.5). Status: **stop B** (test run once per
system on 2026-10-02, after `b1_config.json` was committed in `e33a1fc`). All numbers below come from
`results/phase2/` files produced by `eval/baselines.py`, except B3 (copied from the NLBSE'24 repo).

## Setup

### Packages (macOS arm64, `.venv`)

`eval` extra added to `pyproject.toml`: `scikit-learn==1.9.1`. Installed with
`.venv/bin/python -m pip install "scikit-learn==1.9.1"`; pip did not stall this time (no curl
fallback needed). `pip check`: `No broken requirements found.`

| Package | Version | How it got there |
|---|---|---|
| scikit-learn | **1.9.1** | pinned in `eval` extra |
| scipy | 1.17.1 | resolved (sklearn dependency) |
| joblib | 1.6.0 | resolved (sklearn dependency) |
| threadpoolctl | 3.7.0 | resolved (sklearn dependency) |
| narwhals | 2.26.0 | resolved (sklearn dependency) |
| cloudpickle | 3.1.2 | resolved (joblib dependency) |

Unchanged from Phase 0: Python 3.11.17, laya 0.3.23, torch 2.14.1, numpy 2.4.6.

### `eval/metrics.py`

Pure functions over parallel sequences (gold, predicted, probability dicts keyed by `LABELS`, repo):
accuracy, per-class P/R/F1, macro-F1 (always averaged over all 3 labels; empty denominators give 0),
per-repo macro-F1, **cross-repo macro-F1 = arithmetic mean of the per-repo macro-F1 scores**, pooled
macro-F1, confusion counts (rows gold, columns predicted), multi-class Brier
(mean of Σ_k (p_k − y_k)², range [0, 2]), predicted-class counts, and ECE with 15 bins via
`laya.common.ece_score` (exists in laya 0.3.23: 15 equal-width bins on [0, 1], first bin closed on
the left), computed on answer_confidence = probability of the predicted label.
Probabilities must sum to 1 within 5e-4 because laya rounds each probability to 4 decimals.

`tests/test_eval_metrics.py`: 16 tests on hand-computed fixtures, no data files; includes a case
where the per-repo mean (7/12) differs from pooled macro-F1 (8/15). One-off cross-check against
sklearn (`f1_score`, `precision_recall_fscore_support`, `confusion_matrix`, `accuracy_score`) on
200 random cases: max abs difference 2.2e-16.

## Systems (val)

Inputs: `data/processed/{train,val}.jsonl` (already preprocessed). Every row's embedded question was
asserted equal to `ISSUE_TYPE_QUESTION` (including option order) on load.

- **B0** majority class of train (bug 398, feature 398, question 400 → `question`). Probabilities =
  train class priors (0.3328 / 0.3328 / 0.3344), so ECE and Brier are defined but meaningless.
- **B1** TF-IDF word 1–2 grams + `LogisticRegression(class_weight=None, random_state=42,
  max_iter=10000)`, other parameters at scikit-learn 1.9.1 defaults (lbfgs, L2, lowercase, l2-norm).
  Text = `title + "\n" + body` of the processed state. Fit on train.jsonl only (1,196 rows).
- **B2** `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, zero-shot,
  `ISSUE_TYPE_QUESTION`, `laya.load(..., device="cpu")`, `HF_HUB_OFFLINE=1`, `LAYA_CPU_AMP` unset;
  asserted `amp_enabled=False`, `dtype=float32`; one `predict()` per issue, 8 torch threads.
  Load emits the same `choice:11+` temperature warning as in Phase 0 (does not apply to 3 options).
  Predicted label = laya's `choice`; probabilities as returned (4 decimals).
  Truncated states: 39.3% at 512/192, 11.3% at 1024/256 (matches DATA_CARD). Wall time 50 s / 64 s.

### B1 grid (val, 300 issues; fit on train only)

Selection: max val cross-repo macro-F1; ties → smaller C, smaller min_df, sublinear_tf False (no tie
occurred). No `ConvergenceWarning` and no other warnings in any of the 20 fits (max `n_iter` 74).
Also saved as `results/phase2/b1_grid_val.csv`.

| C | min_df | sublinear_tf | val cross-repo F1 | val pooled F1 | val acc | val ECE | n_iter | vocabulary |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 1 | False | 0.6184 | 0.6269 | 0.6333 | 0.1700 | 24 | 103,055 |
| 0.3 | 1 | True | 0.6411 | 0.6514 | 0.6567 | 0.1986 | 18 | 103,055 |
| 0.3 | 2 | False | 0.6556 | 0.6600 | 0.6633 | 0.1853 | 15 | 20,331 |
| 0.3 | 2 | True | 0.6933 | 0.6995 | 0.7033 | 0.2273 | 20 | 20,331 |
| 1 | 1 | False | 0.6703 | 0.6751 | 0.6800 | 0.1367 | 31 | 103,055 |
| 1 | 1 | True | 0.6923 | 0.6969 | 0.7000 | 0.1813 | 29 | 103,055 |
| 1 | 2 | False | 0.6895 | 0.6928 | 0.6967 | 0.1519 | 21 | 20,331 |
| 1 | 2 | True | 0.7174 | 0.7210 | 0.7233 | 0.1818 | 23 | 20,331 |
| 3 | 1 | False | 0.6908 | 0.6960 | 0.7000 | 0.0835 | 36 | 103,055 |
| 3 | 1 | True | 0.7070 | 0.7104 | 0.7133 | 0.0997 | 48 | 103,055 |
| 3 | 2 | False | 0.6860 | 0.6902 | 0.6933 | 0.0856 | 45 | 20,331 |
| 3 | 2 | True | 0.7356 | 0.7385 | 0.7400 | 0.1147 | 30 | 20,331 |
| 10 | 1 | False | 0.6948 | 0.6991 | 0.7033 | 0.0532 | 64 | 103,055 |
| 10 | 1 | True | 0.7247 | 0.7277 | 0.7300 | 0.0698 | 53 | 103,055 |
| 10 | 2 | False | 0.6970 | 0.7010 | 0.7033 | 0.0572 | 42 | 20,331 |
| **10** | **2** | **True** | **0.7567** | **0.7591** | **0.7600** | **0.0607** | 46 | 20,331 |
| 30 | 1 | False | 0.6782 | 0.6827 | 0.6867 | 0.0561 | 54 | 103,055 |
| 30 | 1 | True | 0.7206 | 0.7242 | 0.7267 | 0.0499 | 64 | 103,055 |
| 30 | 2 | False | 0.7189 | 0.7221 | 0.7233 | 0.0508 | 74 | 20,331 |
| 30 | 2 | True | 0.7364 | 0.7392 | 0.7400 | 0.0545 | 37 | 20,331 |

**Selected: C = 10, min_df = 2, sublinear_tf = True** → `results/phase2/b1_config.json`.
Caveat: 300 val issues (60 per repo); neighbouring configs differ by 1–4 points, within noise.

### Val results (300 issues, 100 per class, 60 per repo)

| System | Acc | Cross-repo macro-F1 | Pooled macro-F1 | F1 bug | F1 feature | F1 question | ECE | Brier | Predicted bug / feature / question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B0 majority | 0.3333 | 0.1667 | 0.1667 | 0.0000 | 0.0000 | 0.5000 | (0.0011) | (0.6667) | 0 / 0 / 300 |
| B1 TF-IDF+LR | 0.7600 | **0.7567** | 0.7591 | 0.7340 | 0.7810 | 0.7624 | 0.0607 | 0.3350 | 88 / 110 / 102 |
| B2 zero-shot 512/192 | 0.6200 | 0.5772 | 0.5858 | 0.6484 | 0.7453 | 0.3636 | 0.0842 | 0.5145 | 156 / 112 / 32 |
| B2 zero-shot 1024/256 | 0.6300 | 0.5919 | 0.5989 | 0.6562 | 0.7524 | 0.3881 | 0.0849 | 0.5099 | 156 / 110 / 34 |

Per-class precision / recall:

| System | bug P / R | feature P / R | question P / R |
|---|---|---|---|
| B0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.3333 / 1.0000 |
| B1 | 0.7841 / 0.6900 | 0.7455 / 0.8200 | 0.7549 / 0.7700 |
| B2 512/192 | 0.5321 / 0.8300 | 0.7054 / 0.7900 | 0.7500 / 0.2400 |
| B2 1024/256 | 0.5385 / 0.8400 | 0.7182 / 0.7900 | 0.7647 / 0.2600 |

Per-repo macro-F1:

| System | bitcoin | react | vscode | opencv | tensorflow |
|---|---:|---:|---:|---:|---:|
| B0 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 |
| B1 | 0.6926 | 0.8036 | 0.7254 | 0.7426 | 0.8193 |
| B2 512/192 | 0.5864 | 0.6934 | 0.4619 | 0.4616 | 0.6827 |
| B2 1024/256 | 0.5864 | 0.6934 | 0.4952 | 0.5017 | 0.6827 |

Confusion (rows gold bug/feature/question, columns predicted): B1 `[[69,17,14],[7,82,11],[12,11,77]]`;
B2 512/192 `[[83,16,1],[14,79,7],[59,17,24]]`; B2 1024/256 `[[84,15,1],[14,79,7],[58,16,26]]`.

Observations (val only):

- B2 shows the Phase 0 pattern on cleaned input: it over-predicts `bug` (156 of 300) and misses most
  `question` issues (recall 0.24–0.26; 58–59 of 100 gold questions predicted `bug`).
- 1024/256 vs 512/192 for B2: +1.5 cross-repo points; the gain is only in vscode and opencv.
- B1 is balanced across classes (predicted 88/110/102) and beats B2 by 16–18 cross-repo points.

## B3: published NLBSE'24 baselines (verification)

Read only `reference/nlbse24/README.md` and `reference/nlbse24/output/**/results.json`
(`output/results.json`, `output/{setfit,roberta,fasttext}/results.json`).

**(a) F1 definition.** README: "Repository classification performance is measured using the F1
score over all the three classes (averaged). Cross-repository performance is measured as the
arithmetic mean of the five repository F1 scores." Checked numerically in all four files, per repo:

- the `average` F1 equals the mean of the 3 per-class F1 values (**macro-F1**) in 20/20 repo entries
  (and `macro avg` in sklearn's report in `output/results.json`);
- it does **not** equal micro-F1 (= accuracy = average recall here, since every test repo has 100
  issues per class), e.g. SetFit react 0.8718 macro vs 0.8733 accuracy, nor the harmonic mean of
  averaged P and R;
- `overall` = arithmetic mean over the 5 repos for every entry (verified), so cross-repo = mean of
  the 5 per-repo macro-F1 scores. This is our §10.2 headline definition.

The README also says macro and micro "will be the same since the dataset is balanced"; that holds
for recall only, not for F1. The published numbers are macro.

**(b) SetFit 0.8270.** Mean of the 5 per-repo macro-F1 in `output/results.json` = 0.827046 → **0.8270**,
and all 60 per-class P/R/F1 values in the README SetFit table match that file to 4 decimals.
**Discrepancy:** `output/setfit/results.json` holds a *different* SetFit result, cross-repo
**0.8240** (per repo below). Which run produced which file is not stated in the files I was allowed
to read. The spec's B3 number (0.8270) is the README / `output/results.json` one.
**Owner ruling (stop A):** B3 headline = 0.8270; 0.8240 is listed only as a footnote row labelled
"provenance unknown", neither averaged nor chosen. (The SetFit template's `OUTPUT_PATH` is
`output/setfit`, so that file is consistent with a run of the template, but nothing ties either
file to a specific run.)

Published per-repo macro-F1 (official test, one classifier per repo):

| Source | react | tensorflow | vscode | bitcoin | opencv | Cross-repo |
|---|---:|---:|---:|---:|---:|---:|
| SetFit, README table = `output/results.json` | 0.8718 | 0.8644 | 0.8262 | 0.7555 | 0.8173 | **0.8270** |
| SetFit, `output/setfit/results.json` (footnote: provenance unknown) | 0.8751 | 0.8679 | 0.8031 | 0.7507 | 0.8231 | 0.8240 |
| RoBERTa, `output/roberta/results.json` | 0.8441 | 0.8585 | 0.7643 | 0.7526 | 0.7422 | 0.7923 |
| fastText, `output/fasttext/results.json` | 0.7876 | 0.7063 | 0.7275 | 0.6618 | 0.7088 | 0.7184 |

Overall per-class F1 (mean over repos): SetFit (README) bug 0.8426 / feature 0.8558 / question
0.7827; RoBERTa 0.8059 / 0.8058 / 0.7653; fastText 0.7321 / 0.7394 / 0.6837.

**Protocol differences** (to carry into the test table): the published baselines train **one
classifier per repo** (README: "we expect 5 classifiers per model submission"), and in the SetFit
template (`reference/nlbse24/2-Template-SetFit.ipynb`, read with owner permission) each
classifier is trained only on **that repo's 300 official-train rows** (`ds.filter(repo_eq(repo))`),
not on all 1,500; ours (B1, M1) is **one model over all 5 repos**, trained on 1,196 rows (the 4
test-overlapping rows dropped and 300 held out as val). The template's text is the raw
`title + " " + body` (no cleaning); its per-repo `average` is sklearn's `weighted avg`, which equals
the macro average here because every test repo has 100 issues per class. B3 numbers are copied, not re-run, and
none of the published files gives probabilities, so ECE/Brier/coverage are not available for B3.

## Test: run log (stop B)

One invocation, `.venv/bin/python eval/baselines.py run --split test` (all four systems), then
`... report --split test`. **No crash, no rerun, no tuning.** The test guard confirmed
`b1_config.json` was committed and unmodified and that no `predictions_test_*.csv` existed.

- B0: majority = `question` (from train counts, as on val).
- B1: frozen config (C = 10, min_df = 2, sublinear_tf = True) refit on train.jsonl only; `n_iter`
  46, no warnings.
- B2: same load and assertions as on val; truncated states 39.1% (512/192) and 11.1% (1024/256),
  matching DATA_CARD; wall time 246 s and 319 s for 1,500 issues at 8 threads. Same `choice:11+`
  load warning.
- `eval/baselines.py` gained a `latency-b1` subcommand while the test run was in progress; the run
  had already imported the module, and the change adds a new command only (no change to `run`).
- `metrics_val.json` regenerated by `report --split val` is byte-identical to the committed file.

## Results table v0 (test, 1,500 issues, Protocol A)

PROJECT_SPEC.md §10.5 template, plus Brier. Coverage @ ≥ 90% precision is "—" for all systems here;
Phase 3 computes it for B1/B2 from the saved prediction CSVs with `eval/gating.py`.

| System | Cross-repo macro-F1 (headline) | Pooled macro-F1 | Acc | F1 bug | F1 feature | F1 question | ECE | Brier | Coverage @ ≥90% prec | CPU p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| B0 majority (`question`) | 0.1667 | 0.1667 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | — | — | — | — |
| B1 TF-IDF+LR | **0.7655** | 0.7666 | 0.7673 | 0.7843 | 0.7925 | 0.7230 | 0.0493 | 0.3273 | — | 0.29 [a] |
| B2 Laya zero-shot 512/192 (native) | 0.6108 | 0.6160 | 0.6453 | 0.6693 | 0.7757 | 0.4029 | 0.0684 | 0.4967 | — | 206 / 276 [b] |
| B2 Laya zero-shot 1024/256 | 0.6257 | 0.6310 | 0.6567 | 0.6773 | 0.7827 | 0.4330 | 0.0647 | 0.4877 | — | 201 / 276 [b] |
| B3 SetFit baseline (NLBSE'24 README = `output/results.json`) | **0.8270** | 0.8263 [c] | 0.8267 [c] | 0.8425 [c] | 0.8555 [c] | 0.7809 [c] | — | — | — | — |
| B3 RoBERTa (`output/roberta/results.json`) | 0.7923 | 0.7926 [c] | 0.7927 [c] | 0.8052 [c] | 0.8064 [c] | 0.7663 [c] | — | — | — | — |
| B3 fastText (`output/fasttext/results.json`) | 0.7184 | 0.7193 [c] | 0.7193 [c] | 0.7362 [c] | 0.7390 [c] | 0.6827 [c] | — | — | — | — |
| **M1 ours** | | | | | | | | | | |

Footnote row (**provenance unknown**; not averaged with or chosen over the 0.8270 row):

| System | Cross-repo macro-F1 | Pooled macro-F1 [c] | Acc [c] | F1 bug [c] | F1 feature [c] | F1 question [c] |
|---|---:|---:|---:|---:|---:|---:|
| SetFit, `output/setfit/results.json` | 0.8240 | 0.8238 | 0.8240 | 0.8328 | 0.8583 | 0.7802 |

- [a] B1 latency: `eval/baselines.py latency-b1` → `results/phase2/latency_B1.json`. Frozen config fit
  on train; first 100 val issues, 3 warm-up calls, one call per issue, `vectorizer.transform` +
  `predict_proba` timed (text string built beforehand). Local Apple M4 Pro CPU, not a GitHub runner.
  p50 0.29 ms, p95 0.38 ms, max 0.48 ms.
- [b] **Not measured in Phase 2.** Cited from Phase 0 (`results/phase0.md`, task 4): **W0 wording**
  (not the frozen W1'), 100 **uncleaned** train issues, Apple-silicon CPU fp32, model preloaded;
  p50 at **8 threads / 2 threads**. p95: 245 / 316 ms (512/192), 500 / 740 ms (1024/256).
- [c] **Derived, not published.** NLBSE'24 publishes per-repo, per-class P/R/F1 only. Because every
  test repo has exactly 100 issues per class, per-repo true positives (R × 100) and predicted counts
  (TP / P) are recoverable exactly (all came out integer to 1.4e-14, and each repo's predicted
  counts sum to 300); pooled per-class F1, pooled macro-F1 and accuracy are computed from those.
  The published "overall" per-class F1 is a mean over repos (SetFit 0.8426 / 0.8558 / 0.7827), a
  different quantity from our pooled per-class F1.
- B0's probabilities are train priors, so its ECE (0.0011) and Brier (0.6667) are in
  `metrics_test.json` but shown as "—".

Test per-class precision / recall:

| System | bug P / R | feature P / R | question P / R | Predicted bug / feature / question |
|---|---|---|---|---|
| B0 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.3333 / 1.0000 | 0 / 0 / 1500 |
| B1 | 0.8102 / 0.7600 | 0.7500 / 0.8400 | 0.7452 / 0.7020 | 469 / 560 / 471 |
| B2 512/192 | 0.5554 / 0.8420 | 0.7391 / 0.8160 | 0.7316 / 0.2780 | 758 / 552 / 190 |
| B2 1024/256 | 0.5620 / 0.8520 | 0.7537 / 0.8140 | 0.7525 / 0.3040 | 758 / 540 / 202 |

Test confusion (rows gold bug/feature/question, columns predicted; also in `metrics_test.json`):
B1 `[[380,55,65],[25,420,55],[64,85,351]]`; B2 512/192 `[[421,64,15],[56,408,36],[281,80,139]]`;
B2 1024/256 `[[426,59,15],[58,407,35],[274,74,152]]`.

### Per-repo macro-F1 (test), ours vs published

| System | react | tensorflow | vscode | bitcoin | opencv | Cross-repo |
|---|---:|---:|---:|---:|---:|---:|
| B1 TF-IDF+LR | 0.7903 | 0.8012 | 0.7490 | 0.7489 | 0.7383 | 0.7655 |
| B2 zero-shot 512/192 | 0.7081 | 0.6833 | 0.5053 | 0.6213 | 0.5358 | 0.6108 |
| B2 zero-shot 1024/256 | 0.7358 | 0.6901 | 0.5106 | 0.6298 | 0.5622 | 0.6257 |
| B3 SetFit (README) | 0.8718 | 0.8644 | 0.8262 | 0.7555 | 0.8173 | 0.8270 |
| B3 RoBERTa | 0.8441 | 0.8585 | 0.7643 | 0.7526 | 0.7422 | 0.7923 |
| B3 fastText | 0.7876 | 0.7063 | 0.7275 | 0.6618 | 0.7088 | 0.7184 |
| SetFit `output/setfit/` (provenance unknown) | 0.8751 | 0.8679 | 0.8031 | 0.7507 | 0.8231 | 0.8240 |

B1 minus published, per repo: vs SetFit (README) −8.2 / −6.3 / −7.7 / −0.7 / −7.9 points; vs
RoBERTa −5.4 / −5.7 / −1.5 / −0.4 / −0.4; vs fastText +0.3 / +9.5 / +2.2 / +8.7 / +3.0.

### Protocol differences (every row of the table above)

1. **Per-repo vs pooled training.** B3 trains one classifier per repo, each on only that repo's 300
   official-train rows (SetFit template, verified); B1/B2/M1 are one model across all 5 repos.
   The training setups of the RoBERTa and fastText baselines were not read (only their
   `results.json`).
2. **Training rows.** B3: all 1,500 official-train rows, including rows 559, 900, 901 and 1114,
   whose content is also in official test (3 test issues). B1: 1,196 rows (those 4 dropped, 300 held
   out as val). B2: no training.
3. **Input text.** B3 SetFit: raw `title + " " + body`. Ours: processed state (§8.4: HTML comments
   stripped, long code blocks collapsed, URLs/images replaced, whitespace normalised, 6,000-char cap),
   B1 as `title + "\n" + body`, B2 as laya's JSON state.
4. **Model selection.** B1's hyperparameters were chosen on our val split (20 configs) and frozen
   before test. SetFit uses fixed template hyperparameters (1 epoch, 20 iterations,
   all-mpnet-base-v2).
5. **Copied vs run.** B3 numbers are copied (not re-run). Their "average" is sklearn `weighted avg`
   (= macro here). No probabilities are published, so B3 has no ECE, Brier or coverage; pooled and
   per-class columns are derived [c].
6. **Truncation (B2).** 512/192 drops part of the state for 39.1% of test issues, 1024/256 for 11.1%.
7. **Latency.** B1 measured here; B2 cited from Phase 0 under different conditions [b]; all local
   Apple-silicon CPU, not a GitHub runner (Q4 open).

## Phase 2 gate checklist (PROJECT_SPEC.md §13)

| Gate item | Status | Evidence |
|---|---|---|
| Table filled for B0–B3 | **Done** | Results table v0 above; M1 row empty (Phase 3); coverage "—" (Phase 3, `gating.py`) |
| B1 number vs published B3 (owner judges plausibility) | **Reported** | B1 test cross-repo **0.7655**: −6.2 points vs SetFit 0.8270, −2.7 vs RoBERTa 0.7923, +4.7 vs fastText 0.7184. Val 0.7567 → test 0.7655 (no drop after selection on val). Per-repo comparison above |
| What B2 shows about fine-tuning | **Stated** | See below |
| Test run once per system, frozen config | **Done** | Run log above; one invocation, no rerun |
| Caveats | **Listed** | Below |

**What B2 shows about the value of fine-tuning.** The base checkpoint zero-shot reaches 0.611
(native 512/192) / 0.626 (1024/256) cross-repo macro-F1: far above the 0.167 floor, so the encoder
and question carry real signal, but **14.0–15.5 points below B1** and **20.1–21.6 points below SetFit**.
The gap is almost entirely the `question` class: recall 0.28–0.30, with 274–281 of 500 gold
questions predicted `bug` (bug is predicted 758 times for 500 gold). `feature` is already close to
B1 (F1 0.78 vs 0.79). Zero-shot Laya is therefore not a usable triage model on this data. M1 must
gain ≥ 14 points over B2 just to match B1 (§6.3 criterion 1), and the bug/question boundary is
where fine-tuning has to deliver. A longer budget alone (1024/256) adds only 1.5 points. B2's ECE
(0.065–0.068) is low despite its accuracy, so ECE alone says nothing about usefulness; gating
(Phase 3) is what matters.

**Caveats.**

- Single test run per system, no confidence intervals. With 300 issues per repo, per-repo
  differences of a few points are within noise.
- The test set is balanced (100 per repo × class); real repos are not (R11).
- B1's selection used a 300-issue val split; adjacent grid configs were within 1–4 points.
- B3 is not strictly comparable (protocol differences 1–5). Its rows reflect a different training
  regime, not just a different model.
- B3's 0.8270 vs 0.8240: two different SetFit results exist in the upstream repo; only 0.8270 is
  used as the headline (owner ruling).
- B2 latency is from Phase 0 (W0 wording, uncleaned input), not re-measured with the frozen wording.
- B0 probabilities are train priors; its ECE/Brier carry no information.

## Files (`results/phase2/`)

| File | Content |
|---|---|
| `predictions_{val,test}_{B0,B1,B2_512-192,B2_1024-256}.csv` | `id, repo, gold, pred, p_bug, p_feature, p_question`; no issue text |
| `predictions_{val,test}_*.meta.json` | how each run was produced (settings, versions, truncation, wall time) |
| `b1_config.json` | frozen B1 settings + selection rule + versions (committed before test) |
| `b1_grid_val.csv` | the 20-config grid above |
| `metrics_val.json`, `metrics_test.json` | all metrics per system, incl. confusion counts, computed from the CSVs |
| `latency_B1.json` | B1 latency measurement [a] |

Rerun: `.venv/bin/python eval/baselines.py run --split val` then `... report --split val`;
`... latency-b1`. B0/B1 take ~12 s; B2 ~2 min on val (both settings), ~9.5 min on test.
Test runs refuse to start unless `b1_config.json` is committed and unmodified, and never overwrite
existing `predictions_test_*.csv`.
