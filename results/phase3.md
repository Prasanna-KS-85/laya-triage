# Phase 3

## Environment (local evaluation, 2026-10-02)

Apple M4 Pro (12 cores), macOS 27.0 arm64, CPU fp32 (`LAYA_CPU_AMP` unset), 8 torch threads (default).

| Package | Version |
|---|---|
| Python | 3.11.17 |
| laya | 0.3.23 |
| torch | 2.14.1 |
| transformers | 5.18.0 |
| tokenizers | 0.23.2 |
| huggingface_hub | 1.33.0 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 (new in the `eval` extra) |
| pillow | 12.3.0 (matplotlib dependency) |
| pytest / ruff | 9.1.1 / 0.16.10 |

## Evaluation tooling check on the base checkpoint (val, not a result)

`eval/run_eval.py --model-repo convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
--split val --expect-max-len 512 --expect-head-max-len 192 --tag dryrun_base`, written to a temp directory
(not committed), compared with `results/phase2/predictions_val_B2_512-192.csv`:

- ids, repos and gold identical; predicted labels identical on 300/300; max |Δp| = 0.0.
- Uncalibrated (T=1) probabilities: softmax of the raw option logits from `laya.calibrate.records_from_labeled`.
  On every issue, softmax(logits / 1.7602) reproduced laya's returned probabilities within their 4-decimal
  rounding (max deviation 5.0e-5) with the same label. 1.7602 is the base config's `choice:3-5` bucket.
- 39.3% of states truncated at 512/192 (as in Phase 2). Latency per `predict()`, model preloaded, 3 warm-up
  calls: p50 182 ms, p95 234 ms (plus a second forward per issue for the T=1 logits, not timed). Wall time 104 s.

## R1: fine-tuned model on val (Stop A, test not run)

Model `Prasanna85/laya-issue-triage` @ `a704b3eadd185f1fa028576cfa50605b6eada4a4` (run R1: notebook commit
`a2451b3`, `SMOKE = False`, eps 0.0, 4 epochs, `GRAD_ACCUM` 4, seed 42). Training facts are in
`results/experiments.md`. Test hash pinned in `eval/run_eval.py` (owner-computed; test.jsonl not opened here).

**Caveats.** n = 300 val issues (60 per repo). Thresholds below are **in-sample: fitted on val** and
measured on the same issues. Latency and memory are from the local Apple M4 Pro CPU, not a GitHub runner.

### Snapshot verification

`.venv/bin/python results/phase3/verify_snapshot.py` (offline; cache, cached remote file listing, items
hashes, local pins): **25 PASS, 0 FAIL**. It checks the exact 6 files plus `.gitattributes` and no
checkpoint directory; sizes, git blob ids and the LFS SHA-256 of `model.safetensors` against the cached
remote listing (total 846,199,192 B); 206 tensors with names and shapes equal to the base, all F16 (the
base has 205 F16 + 1 F32); `run_meta.json` (smoke false, items hash = eps 0.0 entry, plan
1196/119/68/64, constants, versions = local pins, data hashes, train_seconds 712.0, choice T 4.0188);
and `rl_agent_config.json` (1024/256, fine_tuned, model_name, no `temperature_by_options`).
- `tokenizer/tokenizer.json` has the same git blob id as the base's, so the tokenizer is unchanged.
- `rl_agent_config.json` still carries the base's `training` block (updates 7313, hours 1.96, ...).
  laya does not read it; it describes the base, not R1.

### Val metrics (CPU fp32; `results/phase3/val_R1/metrics_val.json`, `val_report.py`)

| System | Acc | Cross-repo macro-F1 | Pooled macro-F1 | F1 bug | F1 feature | F1 question | ECE | Brier | Pred bug/feat/q |
|---|---|---|---|---|---|---|---|---|---|
| **M1 R1** (fitted T 4.0188) | **0.8700** | **0.8691** | **0.8699** | 0.8832 | 0.8738 | 0.8528 | 0.1288 | 0.2474 | 97/106/97 |
| M1 R1, uncalibrated T=1 | (same) | (same) | (same) | | | | 0.1043 | 0.2321 | |
| B1 TF-IDF+LR | 0.7600 | 0.7567 | 0.7591 | 0.7340 | 0.7810 | 0.7624 | 0.0607 | 0.3350 | 88/110/102 |
| B2 zero-shot 512/192 | 0.6200 | 0.5772 | 0.5858 | 0.6484 | 0.7453 | 0.3636 | 0.0842 | 0.5145 | 156/112/32 |
| B2 zero-shot 1024/256 | 0.6300 | 0.5919 | 0.5989 | 0.6562 | 0.7524 | 0.3881 | 0.0849 | 0.5099 | 156/110/34 |

Per-repo macro-F1 (bitcoin / react / vscode / opencv / tensorflow): M1 0.8513 / 0.8976 / 0.8818 / 0.7819 /
0.9332; B1 0.6926 / 0.8036 / 0.7254 / 0.7426 / 0.8193. M1 confusion (rows gold, cols pred):
`[[87, 6, 7], [4, 90, 6], [6, 10, 84]]`.

**Paired M1 vs B1 (val):** exact McNemar p = 6.7e-5 (M1 right and B1 wrong on 50 issues; the reverse on 17).
Cross-repo macro-F1 difference +0.1124, paired bootstrap 95% CI [+0.0625, +0.1650] (2,000 resamples of
issues, seed 42; M1 better in 100% of resamples).

### Temperature scan (information only; the model's temperature is unchanged and not used for thresholds)

softmax(log p_T1 / T) on val. Accuracy is 0.8700 at every T.

| T | 1 | 1.5 | 2 | 3 | **4.0188 (fitted)** | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| NLL | 0.7207 | 0.5370 | 0.4686 | 0.4482 | 0.4813 | 0.5259 | 0.5718 | 0.6522 |
| ECE | 0.1043 | 0.0820 | **0.0473** | 0.0821 | 0.1288 | 0.1782 | 0.2227 | 0.2914 |

The NLL-minimising T on a 0.01 grid is 2.70 (NLL 0.4455). The fitted 4.02, from 119 calibration items,
over-softens on val: confidence sits below accuracy (under-confident). Its ECE is worse than T=1.

### Thresholds (§10.4, `eval/gating.py`; in-sample: fitted on val)

| Label | Predicted | τ_lb | \|S\| | Precision | Lower bound | τ_point | \|S\| | Precision | Lower bound |
|---|---|---|---|---|---|---|---|---|---|
| M1 bug | 97 | 0.5218 | 77 | 0.948 | 0.909 | 0.4712 | 87 | 0.908 | 0.851 |
| M1 feature | 106 | 1.01 (never) | — | — | — | 0.6315 | 92 | 0.902 | 0.859 |
| M1 question | 97 | 1.01 (never) | — | — | — | 0.5172 | 90 | 0.900 | 0.844 |
| B1 bug | 88 | 0.7698 | 34 | 0.971 | 0.912 | 0.6293 | 53 | 0.906 | 0.849 |
| B1 feature | 110 | 0.6886 | 61 | 0.951 | 0.902 | 0.6624 | 66 | 0.909 | 0.848 |
| B1 question | 102 | 0.8692 | 21 | 1.000 | 1.000 | 0.6688 | 50 | 0.900 | 0.820 |

| System, rule | Coverage | Precision | Applied / escalated |
|---|---|---|---|
| M1, τ_lb (§10.4) | 0.2567 | 0.9481 | 77 / 223 (bug only) |
| M1, τ_point | 0.8967 | 0.9033 | 269 / 31 |
| B1, τ_lb | 0.3867 | 0.9655 | 116 / 184 |
| B1, τ_point | 0.5633 | 0.9053 | 169 / 131 |

For M1, precision stays between about 0.88 and 0.95 from 5% coverage upward. No `feature` or `question`
subset with |S| ≥ 20 reaches a bootstrap lower bound of 0.90, so §10.4 never auto-applies those labels.
The proposed `config/triage.default.yml` (not written yet) is stored in `metrics_val.json`
(`proposed_config_yaml`): thresholds bug 0.5218, feature 1.01, question 1.01, mode dry-run.

### Latency and memory (local CPU fp32, model preloaded, 3 warm-up calls)

| Measurement | Threads | p50 ms | p95 ms | mean ms |
|---|---|---|---|---|
| Full val run, 300 issues (`predictions_val_M1_R1.meta.json`) | 8 (default) | 205 | 495 | 247 |
| First 100 val issues (`latency_M1_R1_threadsdefault.json`) | 8 | 223 | 465 | 248 |
| First 100 val issues (`latency_M1_R1_threads2.json`) | 2 | 288 | 690 | 327 |

Cold load, Phase 0 method (fresh process, offline, `results/phase3/latency_probe.py` under
`/usr/bin/time -l`, default threads): `import laya` 0.47 s, `laya.load` 3.14 s, peak RSS after load
2,863 MiB. `/usr/bin/time -l` whole process: max RSS 3,002,155,008 B, peak footprint 2,632,944,496 B. A
second run (2 threads) loaded in 2.75 s with peak RSS 2,864 MiB. The OS page cache was warm in both.
11.3% of val states are truncated at 1024/256.

## Owner decisions after Stop A (2026-10-02)

1. **R1 accepted.** Val cross-repo macro-F1 0.8691 ≥ 0.787; no §9.5 experiments were run.
2. **§10.4 stays as written.** The shipped config uses τ_lb. τ_point, the full coverage–precision curve
   and B1 under the same rule are reported on test as pre-declared secondary views.
3. **Temperature refit on val.** The notebook-fitted choice temperature 4.0188 (119 calibration items) is
   replaced by one fitted on val with standard temperature scaling; weights are unchanged.

## R1b: choice temperature refit on val (Stop A2)

`.venv/bin/python results/phase3/refit_temperature.py` (offline; HF cache only read):
- **Logits:** raw T=1 logits of the 300 val issues from `laya.calibrate.records_from_labeled`, targets
  one-hot gold. They reproduce R1's saved `p_*_T1` columns exactly (max |dev| 0.0).
- **Fit:** laya's own fitter, `laya.calibrate.fit_temperature_map`. Its type-level `fit_one_temperature`
  is NLL + LBFGS on log T, clamped to [0.5, 5.0], and is laya's port of the notebook fitter.
  300 records is below the 2,000-record bucket floor, so no `temperature_by_options` is produced.
- **T_val = 2.6968** (raw 2.696783). The 0.01-grid NLL minimum is 2.70, |difference| 0.0032 ≤ 0.02.
- **Labels:** argmax at T_val equals R1 on 300/300.

Val, **in-sample (fitted on val)**, computed from the unrounded logits:

| T | NLL | ECE (15 bins) | Brier | Accuracy |
|---|---|---|---|---|
| 4.0188 (notebook fit) | 0.4813 | 0.1288 | 0.2474 | 0.8700 |
| **2.6968 (val refit)** | **0.4455** | **0.0699** | **0.2232** | 0.8700 |

`results/phase3/refit/rl_agent_config.json` is the snapshot's config with only `temperature[0]` replaced
(4.018789291381836 → 2.6968). A unified diff shows that one line, and the key order is unchanged. Summary:
`results/phase3/refit/refit_summary.json`.

**Caveats.**
- Val ECE after the refit is in-sample: T was fitted on the same 300 issues. Only test ECE counts for
  §6.3 criterion 2.
- The refit changes every `answer_confidence`, so the τ_lb thresholds from Stop A (fitted at T 4.0188)
  no longer apply. They must be recomputed from val predictions of the new revision (R1b) before
  `config/triage.default.yml` is written.

## R1b on val (Stop A3; test not run)

Revision `76ece1fb0eb8b32bd5d8c509293c1692a2534805` of `Prasanna85/laya-issue-triage` (uploaded by the
owner): R1 with only `rl_agent_config.json` `temperature[0]` = 2.6968.

### Snapshot verification

`.venv/bin/python results/phase3/verify_refit_snapshot.py`: **14 PASS, 0 FAIL**.
- The file list equals R1's and the cached remote listing; sizes and git blob ids match the listing.
- `model.safetensors` SHA-256 is `5cd2cc8f…fd4e9` (R1's), in the remote LFS entry too, and it is the same
  cache blob as R1's.
- `rl_agent_config.json` is byte-identical to `results/phase3/refit/rl_agent_config.json`, and differs
  from R1's only in `temperature[0]` (4.018789291381836 → 2.6968).
- `run_meta.json`, `encoder/config.json`, both tokenizer files and `.gitattributes` are byte-identical
  to R1's.
- `run_meta.json` still records the notebook fit, `temperature` [4.018789291381836, 1.2, 1.2]. It is
  left as is; the shipped temperature is the config's.

### R1 vs R1b on val (in-sample: T fitted on these 300 issues)

`eval/run_eval.py --tag M1_R1b` → `results/phase3/val_R1b/`. Its built-in check passed on every issue:
softmax(logits / 2.6968) reproduces the returned probabilities within the 4-decimal rounding (max
deviation 5.0e-5). Against `M1_R1`: labels identical on 300/300, unrounded T=1 columns identical (max
|diff| 0.0), and calibrated probabilities differ on all 300 rows (max |diff| 0.1321).

| Revision | T | Acc | Cross-repo F1 | Pooled F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|---|---|
| R1 `a704b3e` | 4.0188 | 0.8700 | 0.8691 | 0.8699 | 0.1288 | 0.4813 | 0.2474 |
| **R1b `76ece1f`** | **2.6968** | 0.8700 | 0.8691 | 0.8699 | **0.0699** | **0.4455** | **0.2232** |
| either, uncalibrated | 1 | 0.8700 | 0.8691 | 0.8699 | 0.1043 | 0.7207 | 0.2321 |

ECE and Brier are computed from the CSV probabilities (4 decimals), NLL from the unrounded T=1 columns.
Per-class, per-repo and confusion numbers equal R1's (labels identical); see
`results/phase3/val_R1b/metrics_val.json`. Latency of this run (default 8 threads, model preloaded,
3 warm-up calls, 300 issues): p50 182 ms, p95 435 ms, mean 215 ms; load 2.5 s; local Apple M4 Pro CPU.

### Thresholds on R1b (§10.4; in-sample: fitted on val)

| Label | Predicted | τ_lb | \|S\| | Precision | Lower bound | τ_point | \|S\| | Precision | Lower bound |
|---|---|---|---|---|---|---|---|---|---|
| M1 bug | 97 | **0.6033** | 77 | 0.948 | 0.909 | 0.5027 | 90 | 0.900 | 0.844 |
| M1 feature | 106 | **1.01** (never) | — | — | — | 0.7320 | 92 | 0.902 | 0.859 |
| M1 question | 97 | **1.01** (never) | — | — | — | 0.6088 | 90 | 0.900 | 0.844 |
| B1 bug | 88 | 0.7698 | 34 | 0.971 | 0.912 | 0.6293 | 53 | 0.906 | 0.849 |
| B1 feature | 110 | 0.6886 | 61 | 0.951 | 0.902 | 0.6624 | 66 | 0.909 | 0.848 |
| B1 question | 102 | 0.8692 | 21 | 1.000 | 1.000 | 0.6688 | 50 | 0.900 | 0.820 |

| System, rule | Coverage | Precision | Applied / escalated |
|---|---|---|---|
| M1 R1b, τ_lb (config) | 0.2567 | 0.9481 | 77 / 223 (bug only) |
| M1 R1b, τ_point | 0.9067 | 0.9007 | 272 / 28 |
| B1, τ_lb | 0.3867 | 0.9655 | 116 / 184 |
| B1, τ_point | 0.5633 | 0.9053 | 169 / 131 |

- `feature` and `question` get 1.01: no subset with |S| ≥ 20 reaches a lower bound of 0.90.
- Under τ_lb, R1b applies 77 `bug` issues, as R1 did with its own thresholds. 76 of them are the same
  issues; one differs each way, because temperature scaling can reorder confidences across issues.
  73 of the 77 are correct in both cases.
- The full coverage–precision curves (M1 and B1) are in `metrics_val.json` under `gating.*.curve`.

### Shipped config: `config/triage.default.yml`

Written by `val_report.py --run R1b --write-config`, §12.3 structure. Not committed yet; §10.4 step 7
requires the commit before test, and `eval/run_eval.py` now refuses `--split test` unless this file is
committed and unmodified and its `model.revision` / `model.max_len` equal `--revision` /
`--expect-max-len`.

```yaml
model:
  repo: "Prasanna85/laya-issue-triage"
  revision: "76ece1fb0eb8b32bd5d8c509293c1692a2534805"        # pinned; never "main" in production
  max_len: 1024
  backend: torch                  # torch | onnx (v1.2)

labels:                           # model class -> GitHub label name
  bug: "type: bug"
  feature: "type: feature"
  question: "type: question"
  escalate: "triage: needs-human"

gating:
  thresholds:                     # §10.4 bootstrap lower bound on val (1.01 = never auto-apply)
    bug: 0.6033
    feature: 1.01
    question: 1.01

behaviour:
  mode: dry-run                   # dry-run | apply   (default dry-run for safety)
  comment_on_escalate: true
  skip_if_labeled: true           # skip if any labels.{bug,feature,question} present
  skip_authors: ["dependabot[bot]", "renovate[bot]"]
  create_missing_labels: false
  escalate_on_error: false

preprocess:
  max_code_lines: 20
  max_body_chars: 6000
```

## Test protocol (pre-declared)

Declared on 2026-10-02, before any test prediction of M1 exists. The single test run reports exactly
the items below, and nothing is added, dropped or re-chosen after test numbers are seen.

**Run.** Once, after `config/triage.default.yml` is committed, from the repo root, with exactly this command:

```
.venv/bin/python eval/run_eval.py --model-repo Prasanna85/laya-issue-triage --revision 76ece1fb0eb8b32bd5d8c509293c1692a2534805 --split test --expect-max-len 1024 --expect-head-max-len 256 --tag M1_R1b --out results/phase3/test_R1b
```

If the run crashes, the identical command is re-run once after fixing the crash; crash and fix are
logged in results/experiments.md; no other re-run is allowed. (`run_eval.py` writes the predictions CSV
and its `.meta.json` atomically, CSV last, so a crash leaves no CSV that would block that re-run.)

Settings: CPU fp32, default threads, test.jsonl SHA-256
`ede236c9…794b2` (pinned), 1,500 issues. Baselines are not re-run: B0, B1 and both B2 settings use the
committed `results/phase2/predictions_test_*.csv`, and B3 numbers are copied from Phase 2.

**Primary (M1 = R1b on test):**
1. Accuracy, cross-repo macro-F1 (headline), pooled macro-F1, per-class precision / recall / F1,
   per-repo macro-F1, ECE (15 bins) calibrated and at T=1, Brier calibrated and at T=1,
   predicted-class counts, confusion counts.
2. Coverage and precision (overall and per label) at the config thresholds (bug 0.6033, feature 1.01,
   question 1.01), with the applied and escalated counts.
3. Paired comparisons on test, each with an exact McNemar p and a paired bootstrap 95% CI of the
   cross-repo macro-F1 difference (2,000 resamples of issues with replacement, seed 42):
   M1 vs B1; M1 vs B2 512/192 (native, §10.1); M1 vs B2 1024/256.
4. The §10.5 results table with B0, B1, B2 (both settings), the published B3 rows and M1, next to the
   Phase 2 protocol-difference notes (per-repo vs pooled training, training rows, input text, model
   selection, copied vs run, truncation, latency).

**Secondary (pre-declared views):**
5. τ_point coverage and precision on test (val τ_point: bug 0.5027, feature 0.7320, question 0.6088).
6. The full coverage–precision curve of M1 on test (one global threshold), with the config operating
   point marked.
7. B1 under the same rule with its own val thresholds (bug 0.7698, feature 0.6886, question 0.8692):
   coverage and precision on test.
8. CPU latency (p50 / p95 per issue, model preloaded) from the test run's meta, plus the Stop A
   100-issue latency and cold-load numbers. Local Apple M4 Pro, not a GitHub runner.
9. The four §10.3 figures, written to `results/figures/`: reliability before and after calibration,
   coverage–precision with the operating point, normalised confusion M1 vs B1, per-repo macro-F1 M1
   vs B1.

**Judging §6.3 criteria 1–4** (from the numbers above only):
1. M1 cross-repo macro-F1 on test > B1's and > B2's (both settings), with the paired CIs reported.
2. Calibrated M1 ECE on test ≤ 0.10.
3. Thresholds fitted on val with ≥ 90% precision exist: report the test coverage and precision at the
   config thresholds. Precision below 0.90 on test is reported as a miss, not re-tuned.
4. Latency as in item 8, judged against NFR-1, or documented with the v1.2 ONNX plan.
