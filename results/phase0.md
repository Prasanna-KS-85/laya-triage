# Phase 0 results

Spec: PROJECT_SPEC.md §13 Phase 0. Resolves Q2 (versions, macOS) and Q3 (checkpoint size vs
Actions cache); Q4 and Q7 still open.

## Task 1: repo skeleton + environment (done 2026-10-02)

### Environment (macOS arm64)

| Item | Value |
|---|---|
| Machine | Apple silicon, arm64, 24 GiB RAM |
| OS | macOS 27.0 (`macOS-27.0-arm64-arm-64bit`) |
| Python | 3.11.17 (Homebrew `python@3.11`), venv at `.venv` |
| torch default threads | 8 (MPS available but not used; spec requires CPU fp32) |

### Package versions (macOS arm64)

These are the versions pip resolved on macOS arm64. They are **not** a lock file. The lock file
will be generated on Linux/CI in Phase 4, where torch wheels differ.

| Package | Version | How it got there |
|---|---|---|
| laya | **0.3.23** | pinned in `pyproject.toml` (`laya==0.3.23`) |
| torch | **2.14.1** | resolved from `torch>=2.0.0` (laya requirement) |
| transformers | **5.18.0** | resolved from `transformers>=4.48.0` (laya requirement) |
| huggingface_hub | 1.33.0 | resolved |
| tokenizers | 0.23.2 | resolved |
| safetensors | 0.8.0 | resolved |
| numpy | 2.4.6 | resolved |
| pytest | 9.1.1 | pinned in `dev` extra |
| ruff | 0.16.10 | pinned in `dev` extra |

Verification:

```
$ .venv/bin/python -I -c "import laya; print(laya.__version__)"
0.3.23
$ .venv/bin/python -m pip check
No broken requirements found.
```

### Package source check: PyPI `laya` 0.3.23 vs github.com/NandhaKishorM/laya

Compared PyPI JSON metadata for 0.3.23 against `pyproject.toml` at git tag `v0.3.23`
(`ae3222b3fcdf424254a2c726d72161f671a86d95`). `pyproject.toml` on `main`
(`4aa6761be8173de4ce6d92c31b3e40b6eaf59a7c` at time of check) is byte-identical to the tag.

> **Correction (2026-10-02, Phase 3):** `ae3222b3fcdf424254a2c726d72161f671a86d95` is the annotated
> tag object of `v0.3.23`, not a commit. The tag points to commit
> `d8a2e59781ca135169a36095056132e273cd9938` (`git ls-remote`: `refs/tags/v0.3.23^{}`). A raw-file URL
> built with `ae3222b…` returns 404; use the commit SHA. The paragraph above is kept as originally written.

| Field | PyPI 0.3.23 | GitHub `v0.3.23` pyproject | Match |
|---|---|---|---|
| name / version | laya / 0.3.23 | laya / 0.3.23 | yes |
| author | Convai Innovations | Convai Innovations | yes |
| license | Apache-2.0 | Apache-2.0 | yes |
| requires-python | >=3.10 | >=3.10 | yes |
| core dependencies | torch>=2.0.0, transformers>=4.48.0, safetensors>=0.4.0, huggingface_hub>=0.20.0, numpy>=1.20.0 | same | yes |
| project URLs | Homepage: huggingface.co/convaiinnovations/laya; Demo: huggingface.co/spaces/convaiinnovations/laya-demo | same | yes |

**Mismatches: none.** Note: neither PyPI nor the upstream `pyproject.toml` links back to the GitHub
repo; both point to Hugging Face. That is consistent, not a mismatch.

PyPI artifact hashes (sha256), uploaded 2026-10-01:

- `laya-0.3.23-py3-none-any.whl` `30247fd93dec16b131d8483b1621db198600e90c777ad7d9992e48fc118db677`
- `laya-0.3.23.tar.gz` `5812dfd7bc27032a0b969b7ee2de655460e4d925ad2d55f97a869cf15ad0deba`

### Install notes

- Downloads from `files.pythonhosted.org` ran at about 41 KiB/s on this network, and pip stalled
  twice mid-download on the torch wheel (127 MB). The torch wheel was fetched with resumable
  `curl`, verified against PyPI's sha256
  (`b6074b130fd26bc50f5d171f07dfed70db22dd0c354ab34767729249f9e0f042`), and installed from that
  local file. All other packages came from pip's index or cache as usual.
- Disk free (home volume): 25 GiB before install, 24 GiB after.

### Skeleton deviations from §11 (agreed with owner)

Not created yet, because placeholders could be mistaken for real content: `action.yml`,
`config/triage.default.yml`, `.github/workflows/ci.yml`, `.github/workflows/triage.yml`,
`training/finetune_kaggle.ipynb`, `results/metrics_val.json`, `results/metrics_test.json`. Empty
folders hold a `.gitkeep`.

## Task 2: model load on CPU (done 2026-10-02)

### Checkpoint revision

| Item | Value |
|---|---|
| Repo | `convaiinnovations/laya` |
| Revision (commit SHA) | **`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`** |
| Same as laya 0.3.23's reviewed pin? | Yes: `laya/revisions.py` `PINNED_REVISIONS["convaiinnovations/laya"]` is this SHA |
| `agent.revision` reported after load | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` |

The snapshot was already in the local HF cache (downloaded outside this session), so download
time was not measured. It was loaded with `laya.load("convaiinnovations/laya",
revision="55cf4c4e…", device="cpu")` and `HF_HUB_OFFLINE=1`, so no network was used.

### Load measurements (macOS arm64, Apple-silicon CPU, fp32)

| Metric | Run A (real cache) | Run B (isolated minimal cache, page cache warm) |
|---|---|---|
| `import laya` (incl. torch) | 0.77 s | 0.41 s |
| `laya.load(...)` | 3.58 s | 1.46 s |
| Peak RSS (`ru_maxrss`) | 2,792 MiB | 2,794 MiB |
| Peak RSS / peak footprint (`/usr/bin/time -l`) | 2,928,082,944 B / 2,000,980,968 B | not measured |
| Device | cpu | cpu |
| Parameter dtype in memory | float32 | float32 |
| Parameters | 421,293,827 | 421,293,827 |
| torch threads | 8 | 8 |

- Weights on disk are F16 (205 tensors) plus 1 F32 tensor; laya up-casts to fp32 on CPU.
- `device="cpu"` must be passed explicitly: with `device=None` laya picks MPS on this Mac
  (and CUDA on a GPU box), which would break the §9.4 / §10.4 "CPU fp32" rule.
- Load emits: `RuntimeWarning: laya: this checkpoint ships invalid temperatures or values outside
  [0.5, 5]; using choice:11+=0.10058280825614929 -> 0.5`. This entry is for choice questions with
  11+ options and does not apply to our 3-option question.

### Snapshot contents: 38 files, 2,372,594,769 bytes

Sizes are the resolved blob sizes. **R** = read by `laya.load` for the root (English) checkpoint.

| Bytes | Path | R |
|---:|---|:-:|
| 1,913 | `.gitattributes` | |
| 278,159 | `assets/laya_benchmark_common.png` | |
| 260,892 | `assets/laya_benchmark.png` | |
| 487,005 | `assets/laya_vs_jev_full.png` | |
| 216,412 | `assets/laya_vs_jev.png` | |
| 27,666 | `assets/logo-lockup-dark.png` | |
| 1,590 | `assets/logo-lockup-dark.svg` | |
| 26,761 | `assets/logo-lockup.png` | |
| 1,590 | `assets/logo-lockup.svg` | |
| 17,378 | `assets/logo-mark-ink.png` | |
| 1,320 | `assets/logo-mark-ink.svg` | |
| 1,380 | `assets/logo-mark-mono.svg` | |
| 17,962 | `assets/logo-mark.png` | |
| 1,320 | `assets/logo-mark.svg` | |
| 3,937 | `email_utils.py` | |
| 2,083 | `encoder/config.json` | **R** |
| 467,466 | `eval/benchmark_comparison.png` | |
| 65,436 | `eval/reliability_eval_in.png` | |
| 72,014 | `eval/reliability_eval_zs.png` | |
| 2,976 | `eval/results.json` | |
| 1,608 | `eval/results.md` | |
| 842,609,210 | `model.safetensors` | **R** |
| 1,938 | `multilingual/encoder/config.json` | |
| 643,835,514 | `multilingual/model.safetensors` | |
| 472 | `multilingual/rl_agent_config.json` | |
| 524 | `multilingual/tokenizer/tokenizer_config.json` | |
| 34,363,188 | `multilingual/tokenizer/tokenizer.json` | |
| 23,187 | `README.md` | |
| 4,732 | `rl_agent_api.py` | |
| 745 | `rl_agent_config.json` | **R** |
| 19,139 | `rl_common.py` | |
| 308 | `tokenizer/tokenizer_config.json` | **R** |
| 3,583,228 | `tokenizer/tokenizer.json` | **R** |
| 2,084 | `typed-decisions/encoder/config.json` | |
| 842,609,220 | `typed-decisions/model.safetensors` | |
| 847 | `typed-decisions/rl_agent_config.json` | |
| 337 | `typed-decisions/tokenizer/tokenizer_config.json` | |
| 3,583,228 | `typed-decisions/tokenizer/tokenizer.json` | |

The repo bundles three checkpoints: root (English), `typed-decisions/`, and `multilingual/`.

### Which files `laya.load` reads, and how this was determined

1. **Code:** `laya/agent.py` downloads with `allow_patterns = ["rl_agent_config.json",
   "model.safetensors", "tokenizer/*", "encoder/*"]` (prefixed by `subfolder` if given), then reads
   `rl_agent_config.json`, `tokenizer/`, `encoder/` (if present) and `model.safetensors`.
2. **Run A, real cache:** a Python audit hook logged every `open()`, and `safetensors.torch.load_file`
   was wrapped because it opens the weights from Rust. Files opened under the HF cache:
   `trees/55cf4c4e….json`, `tokenizer/tokenizer_config.json`, `rl_agent_config.json`,
   `tokenizer/tokenizer.json`, `encoder/config.json`, `model.safetensors`. It also attempted
   `~/.cache/huggingface/token` (file does not exist, so no token was used) and read
   `~/.cache/huggingface/.agent_harnesses.json`; neither is part of the model cache.
3. **Run B, isolated cache:** a fresh `HF_HUB_CACHE` holding only the files below loaded
   successfully offline at the pinned SHA. Without `trees/<sha>.json` it failed with
   `OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models/convaiinnovations/laya/tree/55cf4c4e…`:
   with `allow_patterns`, huggingface_hub 1.33.0 needs the repo file listing, from that cached
   file or from the network.

Limits of the method: kernel-level tracing (`fs_usage`, `dtruss`) needs root on macOS and was not
used. Opens made from native code other than the wrapped safetensors call would not appear in
the audit log; Run B shows the subset is **sufficient** regardless.

### Minimal subset (resolves Q3)

| Bytes | Cache path (under `models--convaiinnovations--laya/`) |
|---:|---|
| 842,609,210 | `snapshots/<sha>/model.safetensors` |
| 3,583,228 | `snapshots/<sha>/tokenizer/tokenizer.json` |
| 2,083 | `snapshots/<sha>/encoder/config.json` |
| 745 | `snapshots/<sha>/rl_agent_config.json` |
| 308 | `snapshots/<sha>/tokenizer/tokenizer_config.json` |
| 6,088 | `trees/<sha>.json` (needed for offline loads with huggingface_hub 1.33.0) |
| 40 | `refs/main` (present in Run B; not separately tested as required) |
| **846,201,702** | **total: 846.2 MB (807.0 MiB), 35.7% of the 2.37 GB snapshot** |

**Q3:** a fresh `laya.load` downloads only this subset (about 846 MB), not the whole 2.37 GB repo,
so it is the expected download size and Actions cache entry. That is well under GitHub's
default 10 GB per-repository Actions cache quota. Download time on a runner is still unmeasured
(Phase 4).

### Findings to resolve (not acted on)

- **Checkpoint vs spec `max_len`.** The root checkpoint's `rl_agent_config.json` has `max_len: 512`,
  `head_max_len: 192`. §9.3/§9.4 assume `max_len 1024`, `head_max_len 256`, which match the
  `typed-decisions/` sub-checkpoint (`model_name: laya-typed-decisions`, `fine_tuned: true`) rather
  than the root one. §9.1 names the root checkpoint as the base. Which checkpoint and which
  `max_len` apply is an open question for the owner.
- **Offline cache must include `trees/<sha>.json`.** An Actions cache that keeps only `snapshots/`
  and `blobs/` would fail offline with huggingface_hub 1.33.0. Caching the whole
  `~/.cache/huggingface/hub` directory (as §7.7 already says) avoids this.

## Owner decision: `max_len` (2026-10-02)

Resolves the `max_len` finding above. Fine-tuning starts from the **root** checkpoint
(`convaiinnovations/laya` @ `55cf4c4e…`) and sets `max_len 1024` / `head_max_len 256` at training
time, as the Laya notebook does (`train_ddp.py` overrides `cfg["max_len"]=1024`,
`cfg["head_max_len"]=256`). The root's native `512` / `192` applies to zero-shot runs only.
PROJECT_SPEC.md §9.1, §9.3 and §9.4 stay as written.

## Dataset: NLBSE'24 raw files

Location: `data/raw/` (gitignored, not committed).

| Fact | Value | Source |
|---|---|---|
| Origin | ZIP of github.com/nlbse2024/issue-report-classification, branch `main`; no commit SHA available, so the file hashes below are the provenance record | owner-reported |
| Upstream `LICENSE` file | empty (no license text) | owner-reported |
| `issues_train.csv` sha256 | `18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5` | verified (`shasum -a 256`) |
| `issues_test.csv` sha256 | `4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85` | verified |
| Rows | train 1,500 / test 1,500 | verified (Python `csv`) |
| Columns | `repo, created_at, label, title, body` (both files) | verified |
| Issue IDs / URLs | none; no ID, number or URL column | verified |
| Labels | `bug` 500, `feature` 500, `question` 500 (each file) | verified |
| Balance | 5 repos × 3 labels = 15 cells, exactly 100 rows each (each file) | verified |
| Repos | bitcoin/bitcoin, facebook/react, microsoft/vscode, opencv/opencv, tensorflow/tensorflow | verified |
| Empty bodies | train 0, test 2 | verified |

Row IDs used in this project are the **0-based row index** in each CSV (data rows, header excluded).

### Duplicate `(title, body)` pairs (exact string match), verified

| Pair | Train rows (label) | Test rows (label) | Repo | Title |
|---|---|---|---|---|
| 1 | 559 (bug) | 558 (bug) | tensorflow/tensorflow | `Could not load dynamic library 'cudart64_110.dll'; dlerror: …` |
| 2 | 900, 901 (feature, feature) | 900 (feature) | bitcoin/bitcoin | `#F` |
| 3 | 1114 (bug) | 963 (**feature**) | bitcoin/bitcoin | `.` |

- 3 pairs appear in both train and test; pair 3 has **conflicting labels**.
- 1 pair is duplicated within train (pair 2, rows 900 and 901); it is also a train/test pair.
- 0 pairs are duplicated within test.
- Train rows excluded from the Phase 0 wording sample: **559, 900, 901, 1114**.

## Task 3: zero-shot wording comparison

### Smoke test (one prediction, §9.2 wording, native 512/192)

Setup: `laya.load("convaiinnovations/laya", revision="55cf4c4e…", device="cpu")`,
`HF_HUB_OFFLINE=1`, `LAYA_CPU_AMP` unset. Checked: `amp_enabled=False`, `dtype=float32` (laya
enables CPU autocast only when `LAYA_CPU_AMP=bf16`, so the checkpoint's `amp_dtype: bf16` does not
apply on CPU). Input: train row 0 (facebook/react, gold `bug`), raw title + body, body capped at
6,000 chars.

```json
{"model": "laya-rl-agent",
 "answers": {"issue_type": {"type": "choice", "choice": "bug",
   "probabilities": {"bug": 0.8531, "feature": 0.0789, "question": 0.0679},
   "confidence": 0.5279, "answer_confidence": 0.8531,
   "action": {"act_probability": 1.0}}},
 "usage": {"input_tokens": 499, "output_tokens": 0, "state_tokens": 431,
   "state_tokens_dropped": 0, "truncated": false, "truncated_questions": []}}
```

The predicted label is returned under `"choice"`. `usage.truncated` / `state_tokens_dropped` report
token truncation directly. First-call latency (cold) was 279 ms; this is not a latency measurement.

### Candidate wordings (owner-approved 2026-10-02, fixed before any run on the sample)

All three use the keys `bug`, `feature`, `question` in that order, as `type: choice`.

| | Instructions | `bug` | `feature` | `question` | Q+option tokens |
|---|---|---|---|---|---:|
| **W0** (§9.2) | What kind of GitHub issue is described in \`title\` and \`body\`? | something is broken, crashes, errors, or behaves differently than documented | a request for new functionality or an enhancement to existing behaviour | the author asks how to do something or seeks help or clarification | 68 |
| **W1'** | You are triaging a newly opened GitHub issue. Using its \`title\` and \`body\`, decide which type of issue it is. | reports a defect: a crash, error message, failing build or test, regression, or behaviour that contradicts the documentation | proposes something new: a new capability, option or API, or an improvement to how existing behaviour works | asks for help: how to use or configure something, why it behaves a certain way, or troubleshooting the author's own setup | 113 |
| **W2** | Classify this GitHub issue by its \`title\` and \`body\`. | a bug report: something does not work as intended | a feature request: asks for new or improved functionality | a question: asks for help, guidance, or an explanation | 62 |

W1' is the originally proposed W1 with its negations removed at the owner's request (W1 had 129
tokens), because Laya's docs list forced-choice negation as a known weakness. No question was
truncated (`truncated_questions = []`) at either setting.

### Sample

- File: `results/phase0_sample_ids.txt`, 100 lines, 0-based row indices into
  `data/raw/issues_train.csv`, sorted ascending. sha256
  `eac3b6b82774953a6af0a7680eeca77e6905c7a90e05d4043c21e989724f1f6a`. Phase 1 must force these rows
  into `train`.
- Source: official train only. Excluded rows 559, 900, 901, 1114 (any `(title, body)` also in test
  or twice in train; the rule was recomputed from the data and matched this list).
- Stratified by label with seed 42. Eligible rows: bug 498, feature 498, question 500.
  Proportional allocation with largest-remainder rounding gives **bug 33, feature 33, question 34**.
  Draw: `random.Random(42)`, then `rng.sample(sorted eligible indices, quota)` for bug, feature,
  question in that order.
- Repo mix (not stratified): bitcoin 22, react 24, vscode 12, opencv 14, tensorflow 28. Some
  repo × label cells are thin (opencv `bug`: 1, vscode `feature`/`question`: 3 each).

### Input and limitation

State = `{"title": <raw title>, "body": <raw body>[:6000]}`. That is §8.4 step 5 only: **no other
cleaning** (no HTML-comment stripping, code/log collapsing, URL/image replacement, whitespace
normalisation or title stripping), because `preprocess.py` is a Phase 1 task and §16 rule 5 forbids
a second cleaning path. Issue-template boilerplate and long logs therefore use up the token budget
here, so these zero-shot numbers are not what the model will see after Phase 1 preprocessing.
3 of the 100 bodies exceed 6,000 characters.

### Setup

`laya.load("convaiinnovations/laya", revision="55cf4c4e…", device="cpu")`, `HF_HUB_OFFLINE=1`,
`LAYA_CPU_AMP` unset; checked `amp_enabled=False`, `dtype=float32`. One `agent.predict(state, q,
max_len=…, head_max_len=…)` per issue. 512/192 is the root checkpoint's native setting; 1024/256
is the training/production setting (owner decision above). Gold counts: bug 33, feature 33,
question 34.

### Results: 3 wordings × 2 settings (zero-shot, 100 train issues)

| Run | Macro-F1 | Accuracy | F1 bug | F1 feature | F1 question | Predicted bug / feature / question | Truncated | Mean `answer_confidence` |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| W0 @ 512/192 | 0.6332 | 0.67 | 0.7021 | 0.8254 | 0.3721 | 61 / 30 / 9 | 46% | 0.7499 |
| **W1' @ 512/192** | **0.7464** | **0.76** | 0.7407 | 0.8986 | 0.6000 | 48 / 36 / 16 | 52% | 0.7022 |
| W2 @ 512/192 | 0.6002 | 0.64 | 0.6804 | 0.7869 | 0.3333 | 64 / 28 / 8 | 46% | 0.7364 |
| W0 @ 1024/256 | 0.6221 | 0.66 | 0.6804 | 0.8525 | 0.3333 | 64 / 28 / 8 | 16% | 0.7416 |
| **W1' @ 1024/256** | **0.7291** | **0.75** | 0.7470 | 0.8986 | 0.5417 | 50 / 36 / 14 | 16% | 0.6910 |
| W2 @ 1024/256 | 0.5955 | 0.63 | 0.6600 | 0.7931 | 0.3333 | 67 / 25 / 8 | 16% | 0.7276 |

"Truncated" = share of the 100 issues with `usage.truncated == true` (state tokens dropped to fit
`max_len`).

Confusion matrices, W0 vs W1' (rows = gold bug / feature / question, columns = predicted):

| Setting | W0 | W1' |
|---|---|---|
| 512/192 | `[[33,0,0],[6,26,1],[22,4,8]]` | `[[30,2,1],[2,31,0],[16,3,15]]` |
| 1024/256 | `[[33,0,0],[6,26,1],[25,2,7]]` | `[[31,1,1],[2,31,0],[17,4,13]]` |

Paired comparison, W1' vs W0 on the same 100 issues:

| Setting | Macro-F1 diff | W1' right, W0 wrong | W0 right, W1' wrong | Exact McNemar p (accuracy) | Paired bootstrap 95% CI of diff (2,000 resamples, seed 42) |
|---|---:|---:|---:|---:|---|
| 512/192 | +11.3 pts | 13 | 4 | 0.049 | [+2.7, +20.8] pts |
| 1024/256 | +10.7 pts | 12 | 3 | 0.035 | [+2.4, +19.7] pts |

Observations:

- No run collapses to one class, but every wording **over-predicts `bug`** (48–67 predicted vs 33
  gold). The main error is gold `question` predicted as `bug` (16–25 of 34).
- W1' gains mostly on `question` (F1 0.37 → 0.60 at 512/192) and `feature`, at a small cost on
  gold `bug` recall (33 → 30 or 31 of 33).
- W1' wins at 512/192 even though its longer question truncates more issues there (52% vs 46%).
- Going from 512/192 to 1024/256 cuts truncation from 46–52% to 16% but does not raise zero-shot
  macro-F1 (slightly lower for all three wordings). The root checkpoint was trained at 512, which
  may explain this; it says nothing about the fine-tuned model.
- None of the runs is near chance (all macro-F1 ≥ 0.59).

### Decision rule outcome (task 5 input; awaiting owner approval)

Rule: keep W0 unless another wording beats it by ≥ 5 macro-F1 points at **both** settings; keep
W0 if all runs are near chance (macro-F1 ≤ ~0.45).

- W1' beats W0 by **+11.3** (512/192) and **+10.7** (1024/256) points: the rule is met at both settings.
- W2 is below W0 at both settings (−3.3, −2.7 points).
- Not near chance.

**Per the rule, the winner is W1'.** Caveats: n = 100 and the bootstrap intervals are wide
(lower bounds about +2.5 points); the input is uncleaned (see above). `questions.py` has **not**
been edited; that is task 5, after owner approval.

## Task 4: CPU latency

**Local Apple-silicon CPU fp32, not a GitHub runner.** Q4 (runner CPU/RAM) stays open until
Phase 4.

Method: W0 wording, model preloaded, 3 warm-up predictions on train rows 1–3 (not in the sample),
then one timed `agent.predict` per issue over the same 100 issues (`time.perf_counter` around the
call only; state built beforehand). Percentiles: `numpy.percentile`, linear interpolation.
Thread count set with `torch.set_num_threads`; default was 8.

| Setting | Threads | p50 ms | p95 ms | Mean ms | Max ms | Predictions identical to accuracy run |
|---|---:|---:|---:|---:|---:|---:|
| **1024/256 (production)** | 8 (default) | 201.4 | 500.3 | 242.1 | 536.9 | 100 / 100 |
| **1024/256 (production)** | 2 | 276.2 | 740.4 | 350.8 | 770.9 | 100 / 100 |
| 512/192 (reference) | 8 (default) | 206.2 | 244.6 | 179.0 | 275.8 | 100 / 100 |
| 512/192 (reference) | 2 | 276.3 | 315.6 | 229.3 | 335.7 | 100 / 100 |

- NFR-1 (≤ 1 s p95): met locally at both thread counts at 1024/256. The 2-thread run is only a
  rough runner proxy; real runner numbers are Phase 4.
- p95 at 1024/256 is about 2× its 512/192 value because the 16–52% of issues that hit the 512
  limit run to full length at 1024.
- Changing the thread count did not change any prediction (100/100 identical), which is
  consistent with NFR-8 on this machine.

## Task 5: frozen wording (done 2026-10-02)

Owner approved W1' on 2026-10-02.

- `src/laya_triage/questions.py` now defines `ISSUE_TYPE_QUESTION` (§9.2 structure, `type: choice`)
  and `LABELS = ("bug", "feature", "question")`. The four strings were generated from the `W1'`
  entry in `results/phase0/wordings.py`, not retyped.
- One-off check (not a repo test): importing `laya_triage.questions` and comparing with `W1'`
  printed `OK: questions.py is byte-identical to the W1' entry in results/phase0/wordings.py`
  (UTF-8 bytes of instructions and each criterion, key order = `LABELS`, and the full dict equals
  the question the evaluation sent to laya).
- PROJECT_SPEC.md v1.0.2: §9.2 code block replaced with `questions.py`; §8.5 example strings
  updated; §9.1, §10.1 (B2), §17 (Q2, Q3) and the §16 changelog updated.
- The frozen contract (wording, keys, order) now falls under §16 rule 4.

## Experiment files and how to rerun

`results/phase0/`:

| File | Content |
|---|---|
| `phase0_eval.py` | Tasks 3–4 script (6 accuracy runs, then 4 latency passes) |
| `wordings.py` | W0, W1', W2 exactly as run |
| `predictions_<W0\|W1p\|W2>_<512-192\|1024-256>.csv` | One per run: `row_id, gold, pred, p_bug, p_feature, p_question`. No issue text. `W1p` = W1'. |
| `summary.json` | Metrics, latency and run metadata; the source of the task 3–4 tables above |

The prediction CSVs and `summary.json` were converted from the original run's output (field
subset only; `W1` renamed `W1'`).

Path fixes made when moving the script out of the session scratchpad: the original hard-coded the
absolute repo path (`REPO_ROOT = "/Users/…/Laya Triage"`), wrote outputs next to itself, and relied
on the caller's `PYTHONPATH` (to import `wordings`), `HF_HUB_OFFLINE=1` and unset `LAYA_CPU_AMP`.
The repo version resolves paths from its own location, imports `wordings` from its own directory,
sets `HF_HUB_OFFLINE=1` and removes `LAYA_CPU_AMP` itself, and takes `--out`.

To rerun (from the repo root):

```
.venv/bin/python results/phase0/phase0_eval.py --out /tmp/phase0_rerun
```

Task 2 load probe (offline, reads the cache only, no issue data): `.venv/bin/python results/phase0/load_probe.py [--cache DIR]` prints load time, peak RSS and the cache files `laya.load` opens.

Requirements: the `.venv` from task 1; `data/raw/issues_train.csv` with the sha256 above;
`results/phase0_sample_ids.txt`; and `convaiinnovations/laya` @ `55cf4c4e…` in the local HF cache
(networking is off). Omitting `--out` overwrites the committed files in `results/phase0/`.
Predictions are deterministic on this machine; latency numbers will differ run to run.

Rerun check (2026-10-02): the final repo script (after an import-order fix for ruff), run with
`--out` to a scratch directory, reproduced all 6 prediction CSVs byte-for-byte (`cmp`), and `runs`
and `meta` in its `summary.json` were identical to the committed one. This was checked twice, before
and after the import fix. Latency varied as expected: for example, 1024/256 at 2 threads gave
p95 720 ms and 724 ms against 740 ms in the original run. The tables above keep the original run's
numbers.

## Phase 0 gate (PROJECT_SPEC.md §13)

| Gate item | Status | Evidence |
|---|---|---|
| Inference works on CPU | **Pass** | Task 2: loads on CPU at the pinned SHA, fp32, autocast off. Task 3: 600 predictions over 3 wordings × 2 settings, plus 400 timed predictions, with no errors. |
| Latency is recorded | **Pass** (local only) | Task 4 table: W0 @ 1024/256 p50 201 ms / p95 500 ms at 8 threads, p50 276 ms / p95 740 ms at 2 threads; 512/192 reference included. |
| Wording is frozen | **Pass** | Task 5: `questions.py` = W1' (byte-identical check), PROJECT_SPEC.md §9.2 v1.0.2. |
| Model size fits the Actions cache | **Pass** | Task 2: `laya.load` needs 846,201,702 bytes (846 MB; full snapshot 2.37 GB), against GitHub's default 10 GB per-repository cache quota. |

Caveats:

- **Small sample.** n = 100 train issues. The W1' win is consistent (12–13 vs 3–4 discordant
  issues), but the paired bootstrap 95% CIs are wide, with lower bounds of about **+2.5 points**,
  so the true margin over W0 may be smaller than 5 points.
- **Uncleaned input.** Zero-shot used raw text (6,000-char cap only). Phase 1 preprocessing may
  change the absolute numbers and possibly the ranking.
- **Latency is local Apple-silicon CPU fp32, not a GitHub runner.** Q4 stays open until Phase 4.
- **Thin latency margin.** p95 at 2 threads, 1024/256 = **740 ms** against the 1 s NFR-1 target. A
  slower runner CPU could exceed it. The mitigation is the v1.2 ONNX INT8 plan (§4.1, R3).
