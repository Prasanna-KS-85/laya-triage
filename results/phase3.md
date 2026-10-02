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
