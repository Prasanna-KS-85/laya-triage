# Laya Triage — Project Specification & System Design

> **Status:** v1.0.24 draft (source of truth) · **Last updated:** 2026-10-04 · **Owner:** Prasanna · Phase 4 gate passed; v1.0.0-rc1; Phase 5 in progress (short README plus docs/DEEP_DIVE.md and two usage guides; HF model public, model card uploaded)
> **Working name:** `laya-triage` (rename freely; update this line and §11 when you do)

This document is the **single source of truth** for the project. Every human and every coding agent
(Claude, Claude Code, others) working on this repository must follow it. If code and this document
disagree, the document wins until it is deliberately updated. Rules for changing it are in §16.

---

## Table of contents

1. [Summary](#1-summary)
2. [Problem statement](#2-problem-statement)
3. [Solution overview](#3-solution-overview)
4. [Scope](#4-scope)
5. [Background: Laya concepts this project depends on](#5-background-laya-concepts-this-project-depends-on)
6. [Requirements](#6-requirements)
7. [System architecture](#7-system-architecture)
8. [Data specification](#8-data-specification)
9. [Model specification](#9-model-specification)
10. [Evaluation specification](#10-evaluation-specification)
11. [Repository structure](#11-repository-structure)
12. [Interfaces and configuration](#12-interfaces-and-configuration)
13. [Phased delivery plan](#13-phased-delivery-plan)
14. [Architecture decision records](#14-architecture-decision-records)
15. [Risks and mitigations](#15-risks-and-mitigations)
16. [Rules for coding agents](#16-rules-for-coding-agents)
17. [Open questions and items to verify](#17-open-questions-and-items-to-verify)
18. [Glossary](#18-glossary)
19. [References](#19-references)

---

## 1. Summary

**Laya Triage** is a GitHub Action that automatically classifies newly opened GitHub issues as
`bug`, `feature`, or `question` using a **fine-tuned Laya decision model** (a non-autoregressive,
encoder-based model that outputs calibrated probabilities instead of generated text).

Its distinguishing behaviour is **confidence-gated labelling**: it applies a label only when the
model's calibrated confidence clears a threshold chosen on validation data. Otherwise it marks the
issue for human triage. The project delivers:

- an installable GitHub Action,
- a fine-tuned checkpoint published on Hugging Face,
- a reproducible evaluation against public baselines on a published academic benchmark (NLBSE'24),
- a write-up with accuracy, calibration, the coverage–precision trade-off, and CPU latency.

---

## 2. Problem statement

Open-source and internal repositories receive a steady stream of issues. Maintainers spend time
reading each one just to decide what kind of issue it is. Unlabelled issues slow down prioritisation,
search, and contributor onboarding.

Existing automation falls into two groups:

1. **Keyword or rule bots.** They are brittle and miss paraphrases.
2. **LLM-based bots.** They are slow (hundreds of ms to seconds per call), cost money per call,
   return free text that must be parsed, and give no trustworthy confidence. Their stated confidence
   is just generated tokens.

What's missing is a triage tool that is **fast, free to run, structured by construction, and honest
about when it is unsure**, so it can safely act on the easy cases and hand the hard ones to a human.

---

## 3. Solution overview

### 3.1 One-paragraph description

When an issue is opened, a GitHub Actions workflow runs Laya Triage on the runner's CPU. The issue
title and body are cleaned and passed as a structured *state* to a fine-tuned Laya checkpoint,
together with one typed `choice` question ("what kind of issue is this?"). The model returns a
probability for each label in a single forward pass. If the top label's calibrated probability
(`answer_confidence`) is at or above that label's threshold, the Action applies the matching label.
If not, it applies a `triage: needs-human` label and optionally posts a short comment showing the
model's top-2 guesses. A dry-run mode writes decisions to the job summary instead of touching the
issue.

### 3.2 What makes it more than "a classifier"

| Aspect | Typical student classifier | This project |
|---|---|---|
| Output | A label | A label **plus** a calibrated probability used for a decision |
| Decision policy | Always act | Act only above a threshold fitted for a target precision; escalate otherwise |
| Evaluation | Accuracy on a random split | Macro-F1 on a **published benchmark**, calibration (ECE), coverage–precision curve, CPU latency, external baselines |
| Delivery | Notebook | Installable GitHub Action + Hugging Face model + report |
| Model framing | Fixed softmax head | Laya's question-conditioned decision head trained with RLCD (proper-scoring-rule RL) |

---

## 4. Scope

### 4.1 Release map

| Release | Contents | Status |
|---|---|---|
| **v1.0 (MVP, must ship)** | Issue type classification (`bug`/`feature`/`question`), confidence gating, GitHub Action (apply + dry-run), fine-tuned model on HF, evaluation report | **Committed** |
| v1.1 | `needs_info` flag: is the report missing reproduction steps, version, or logs? | Planned, starts only after v1.0 ships |
| v1.2 | ONNX INT8 export for faster CPU inference inside the Action | Optional optimisation |
| v2.0 | Priority (`score`), duplicate suggestion (embedding similarity), `documentation` class | Stretch, not committed |

### 4.2 Explicit non-goals for v1.0

- No pull-request triage (issues only).
- No hosted server, database, or dashboard. Everything runs inside the Action.
- No multilingual support (GitHub issues in the target repos are overwhelmingly English).
- No auto-closing, auto-assigning, or any action besides adding labels and an optional comment.
- No research contribution to Laya itself (no changes to Laya's training algorithm or calibration
  method). We **use** Laya's existing fine-tuning and calibration tooling.

---

## 5. Background: Laya concepts this project depends on

Read this section before touching model code.

### 5.1 What Laya is

Laya (github.com/NandhaKishorM/laya, Apache-2.0) is a "System 1" decision engine. It takes a
**state** (text or a JSON-like dict) and a set of **typed questions**, and returns probabilities for
every question in one forward pass. It never generates text, so there is nothing to parse.

**Question types:**

| Type | Output | Used in this project |
|---|---|---|
| `choice` | Probability per option, top label, confidence | **v1.0**: issue type |
| `noul` | P(true) for a yes/no statement | v1.1: `needs_info` |
| `score` | Distribution over ordinal levels plus expected level | v2.0: priority (stretch) |

### 5.2 Architecture, briefly

- **Sequence packing.** The question, its options (each preceded by a `[MASK]` marker), and the
  state are packed into one sequence: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0
  [MASK] opt1 ... [SEP] <state> [SEP]`.
- **Encoder.** The English checkpoint `convaiinnovations/laya` uses ModernBERT-large, about 421M
  parameters including the head. The encoder is bidirectional.
- **Decision head.** A 2-layer transformer runs on top of the encoder. Hidden states at the `[MASK]`
  positions go through a small MLP, giving one logit per option. A softmax over `logits / T` gives
  the probabilities. `T` is a temperature fitted per question type.
- **Why options are inputs.** Because options are part of the input rather than a fixed output
  layer, labels and their descriptions are part of the prompt. **Changing the label wording changes
  model behaviour.**

### 5.3 Training (RLCD), briefly

The fine-tuning notebook optimises two things together:

- a policy-gradient term over noisy logit samples, rewarded by strictly proper scoring rules
  (spherical and ranked-probability scores);
- a soft cross-entropy term against a target probability distribution ("gold").

After training, the notebook fits **one temperature per question type** on a calibration slice it
carves out of the training data before training starts.

### 5.4 Confidence fields (important for gating)

Each answer exposes two confidence values:

- `confidence` is 1 minus normalised entropy. It describes the shape of the distribution. **Do not
  gate on it.**
- `answer_confidence` is the probability of the reported answer (max p). This is the calibrated
  quantity. **Gate on this.** Laya's built-in `min_confidence=` also reads `answer_confidence` and
  sets `low_confidence: True` on answers below it.

### 5.5 Known Laya limitations relevant to us

| Limitation | Impact on this project | Handling |
|---|---|---|
| Base checkpoints are weak zero-shot on structured decision tasks | Zero-shot accuracy will be modest | Fine-tuning is mandatory (Phase 3) |
| Shipped checkpoints are over-confident | Thresholds on raw confidence are meaningless | Use the notebook's temperature fitting; validate ECE |
| Option-position bias (#131 and related) | Label order can shift predictions | Keep option order fixed in training and inference; optional rotation averaging (§9.6) |
| `noul` can follow its option labels, not the text (#156) | Affects v1.1 `needs_info` | Always give `noul` explicit `criteria` and test the `labels` override |
| Boolean-word keys in `choice` (`yes`/`no`/`true`/`false`) | Model may follow the words | Use semantic keys `bug`/`feature`/`question` only |
| Token budget for many options | Not an issue with 3 options | N/A for v1.0 |
| bf16 vs fp32 numeric differences | Thresholds fitted on GPU may shift on CPU | Fit thresholds from CPU-fp32 inference on the validation set (§10.4) |

---

## 6. Requirements

### 6.1 Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | On `issues: opened`, the Action classifies the issue into exactly one of `bug`, `feature`, `question`. |
| FR-2 | If `answer_confidence >= threshold[label]`, apply the configured label for that class. |
| FR-3 | Otherwise, apply the configured escalation label (default `triage: needs-human`). If `comment_on_escalate` is true, post one comment with the top-2 labels and their probabilities. |
| FR-4 | `dry-run` mode makes no changes to the issue. It writes the decision table to the job summary. |
| FR-5 | Skip, in this order: pull requests (the payload has a `pull_request` key); `issues` events whose action is not `opened` (only when the event carries an action, so backfill issues from the REST API are never skipped for this); authors in `skip_authors` (for example bots); issues that already carry the escalation label ("already escalated", for idempotent backfill); and, with `skip_if_labeled`, issues that already carry any configured type label. Label and author comparisons are case-insensitive. Record the skip reason in the summary. |
| FR-6 | `workflow_dispatch` backfill mode: classify the N most recent open issues, defaulting to dry-run. |
| FR-7 | Every run writes a JSON log line per issue (prediction, probabilities, decision, latency, model revision) to the job summary and as a workflow artifact. |
| FR-8 | Labels, thresholds, mode, and model revision are read from a YAML config file in the target repo (§12.3). Sensible defaults apply if the file is absent. |

### 6.2 Non-functional requirements

| ID | Requirement | Target (verify in Phase 0 / 4) |
|---|---|---|
| NFR-1 | Inference latency per issue, CPU, model already loaded | ≤ 1 s p95 on a GitHub-hosted Linux runner |
| NFR-2 | Total Action wall time with warm cache | ≤ 2 min |
| NFR-3 | Total Action wall time with cold cache (first run) | ≤ 5 min |
| NFR-4 | Cost to the user | $0: public model, standard runner, no paid APIs |
| NFR-5 | Reliability | Fails soft. Any error leaves the issue untouched (or only adds the escalation label) and never fails the user's CI in a way that blocks them. |
| NFR-6 | Security | Least privilege (`issues: write`, `contents: read`). Issue text is treated as untrusted data and never interpolated into shell commands (§7.6). |
| NFR-7 | Reproducibility | Pinned dependency versions, pinned model revision (commit SHA), fixed seeds, committed split manifests. |
| NFR-8 | Determinism | The same issue text and model revision produce the same decision on the same hardware. |

### 6.3 Success criteria for v1.0

These are **targets, not promises**. Report the real numbers either way. An honest miss with
analysis is acceptable; a hidden miss is not.

1. The fine-tuned model beats both the Laya zero-shot and TF-IDF+LR baselines on the headline
   cross-repo macro-F1 (§10.2) on the NLBSE'24 test split.
   **Result (test): MET.** M1 0.8020 vs B1 0.7655 (+0.0364, paired 95% CI [+0.0151, +0.0588]), B2 0.6108
   at 512/192 (+0.1912, [+0.1661, +0.2167]) and 0.6257 at 1024/256 (+0.1763, [+0.1505, +0.2016]).
2. Post-calibration ECE ≤ 0.10 on the test split.
   **Result (test): MET.** ECE 0.0454 at T 2.6968 (0.1465 at T=1).
3. A threshold exists on validation data giving **≥ 90% precision** on auto-labelled issues. Report
   the coverage it achieves on the test split. On validation a threshold meeting a 0.90 bootstrap lower
   bound exists for 1 of 3 labels (bug, lower bound 0.909; feature and question none). On test the bug
   threshold gave precision 0.8747 at coverage 0.2607: below the 0.90 target (not met).
   **Result (test): NOT MET.** 391 of 1,500 issues auto-applied (bug only), precision 0.8747.
4. NFR-1 is met, or the measured latency is documented with the ONNX plan (v1.2).
   **Result (Phase 4): NFR-1 NOT MET on a 2-vCPU runner: p50 864 ms, p95 3.5 s; documented with the v1.2 ONNX
   plan, which satisfies the criterion's second clause.** Sandbox probe (run 37026455422, ubuntu-24.04, 2 logical
   CPUs, torch 1 thread, 60 classifications of the fixtures): p50 863.8 ms, p95 3501.7 ms. For comparison, local
   Apple M4 Pro CPU, fp32, model preloaded: p50 178 ms, p95 436 ms (test run, 8 threads); p95 690 ms at 2 threads
   (Stop A, 100 val issues). Details and caveats in `results/phase4.md`.
5. The Action runs end-to-end on a sandbox repo: 22 authored fixtures created via the gh CLI and 2 further issues opened by hand in the web UI, all as real issues.opened events.
   **Result (Phase 4): MET.** Sandbox `Prasanna-KS-85/laya-triage-sandbox`, action commit `13633af`: 22 issue
   runs (dry-run, 22/22 match the local run), backfill apply 37030197012, idempotent re-run 37030556119, web-UI
   issues 37031205724 and 37031265068 (apply mode, comment template (b)); all runs succeeded.

---

## 7. System architecture

### 7.1 Context diagram

```
                ┌───────────────────────────── GitHub ─────────────────────────────┐
                │                                                                   │
  User opens ──►│  Issue #123  ──(issues.opened event)──►  Actions workflow          │
  an issue      │                                           triage.yml               │
                │                                               │                   │
                │                                               ▼                   │
                │                                  ┌────────────────────────┐       │
                │                                  │  Laya Triage (Python)  │       │
                │                                  │  on ubuntu runner CPU  │       │
                │                                  └─────┬───────────┬──────┘       │
                │     add label / comment (REST API)     │           │              │
                │  ◄─────────────────────────────────────┘           │              │
                └────────────────────────────────────────────────────┼──────────────┘
                                                                     │ download once,
                                                                     │ then actions/cache
                                                         ┌───────────▼────────────┐
                                                         │ Hugging Face Hub        │
                                                         │ <user>/laya-issue-triage│
                                                         │ (pinned revision)       │
                                                         └─────────────────────────┘
```

### 7.2 Runtime component diagram (inside the Action)

```
 GITHUB_EVENT_PATH (event JSON)                     config/triage.yml
          │                                                │
          ▼                                                ▼
 ┌─────────────────┐   ┌────────────────┐   ┌──────────────────────┐
 │ event_loader    │──►│ preprocess     │──►│ classifier           │
 │ - parse payload │   │ - clean text   │   │ - load Laya Agent    │
 │ - skip rules    │   │ - build state  │   │ - QUESTIONS schema   │
 └─────────────────┘   └────────────────┘   │ - predict()          │
                                            └──────────┬───────────┘
                                                       │ TriageResult
                                                       ▼
                                            ┌──────────────────────┐
                                            │ policy (gating)      │
                                            │ - thresholds         │
                                            │ - apply / escalate / │
                                            │   skip               │
                                            └──────────┬───────────┘
                                                       │ Decision
                       ┌───────────────────────────────┼─────────────────────┐
                       ▼                               ▼                     ▼
             ┌──────────────────┐          ┌────────────────────┐  ┌──────────────────┐
             │ github_client    │          │ reporter           │  │ (dry-run: no     │
             │ - add labels     │          │ - job summary (MD) │  │  github writes)  │
             │ - post comment   │          │ - JSONL artifact   │  └──────────────────┘
             │ - retry/backoff  │          └────────────────────┘
             └──────────────────┘
```

### 7.3 Offline (training and evaluation) pipeline

```
 NLBSE'24 dataset (CSV/JSON)
          │
          ▼
 training/build_dataset.py ──► data/processed/
   - dedupe, clean (same preprocess.py as runtime!)      ├─ train.jsonl  (Laya training rows)
   - split: official train → train/val                   ├─ val.jsonl
   - official test kept untouched                        ├─ test.jsonl   (laya-evals format)
   - write split manifests (issue IDs)                   └─ manifests/*.txt
          │
          ├──────────► eval/baselines.py ──► majority, TF-IDF+LR, Laya zero-shot
          │
          ▼
 training/finetune_kaggle.ipynb  (adapted from Laya's 2xT4 notebook)
   - RLCD fine-tune from convaiinnovations/laya
   - temperature fit (notebook's internal calibration slice)
   - push to HF: Prasanna85/laya-issue-triage @ revision SHA
          │
          ▼
 eval/run_eval.py (CPU fp32) ──► results/
   - metrics on val → choose thresholds → config/triage.yml
   - metrics on test (once, after thresholds are frozen)
   - plots: reliability diagram, coverage–precision curve, confusion matrix
```

### 7.4 Sequence of a single triage run

```
GitHub         Workflow            laya_triage                     HF Hub / cache        GitHub API
  │ opened ───► │                       │                                │                    │
  │             │ restore cache ───────────────────────────────────────►│                    │
  │             │ python -m laya_triage │                                │                    │
  │             │──────────────────────►│ load config, read event        │                    │
  │             │                       │ skip? ── yes ─► summary, exit 0 │                    │
  │             │                       │ preprocess(title, body)        │                    │
  │             │                       │ load Agent(revision) ─────────►│ (cache hit)        │
  │             │                       │ predict(state, QUESTIONS)      │                    │
  │             │                       │ policy.decide(result, cfg)     │                    │
  │             │                       │ apply mode: add label / comment ───────────────────►│
  │             │                       │ write summary + JSONL artifact │                    │
  │             │◄──────────── exit 0 ──│                                │                    │
```

### 7.5 Error handling and failure modes

| Failure | Behaviour |
|---|---|
| Model download fails | Log a warning, write the summary, exit 0. No labels. Optionally add the escalation label if `escalate_on_error: true`. |
| Inference raises | Same as above. Record the exception type and traceback frames (file:line:function) in the summary and the JSONL log, never the exception message (it could echo issue text; the same holds for download and tokenizer errors), and never in an issue comment. |
| GitHub API 5xx, 429, secondary rate limit or network error | Retry with exponential backoff, 3 retries (2 s, 4 s, 8 s), then fail soft. |
| GitHub API 403 (permissions) | Exit with a clear message naming the missing permission (`issues: write`). |
| Empty or very short body | Classify on the title alone. The state still has a `body` key with an empty string. |
| Config file invalid | Fail with a validation message listing the bad keys. Do not guess. |
| Label does not exist in repo | Checked against the repository's labels before adding (GitHub's add-labels call would silently create it). Create it with a default colour (opt-in `create_missing_labels: true`). Otherwise warn and skip. |

Misconfiguration (an invalid config, a missing permission, a missing token or a non-https API URL) fails
loudly by design (exit 1); NFR-5 concerns runtime faults, which exit 0.

Action layer (`action.yml`):

| Failure | Behaviour |
|---|---|
| Runner is not Linux X64 | Fail loudly (unsupported by design). |
| `setup-python`, dependency install (pip / CPU torch index) fails or times out | Warning plus a summary line ("no issues were triaged"), exit 0. Torch is retried once with `--extra-index-url` if the CPU index alone fails. |
| Restored dependency venv fails its check (imports, `torch.version.cuda is None`) | Rebuilt; the rebuilt venv is not saved. |
| Cache restore fails | Warning only (cold path). |
| Restored model cache incomplete (`model-cached=false`) | Online load; `HF_HUB_OFFLINE=1` only when the cache hit and `model-cached=true`. |
| Model download fails or is rate limited (R13) | `ClassifierError`, exit 0; the model cache is saved only after a successful load (`model_loaded=true`). |
| Triage step times out (`timeout` 900 s, exit 124 / 137) or ends with another unexpected code | Warning plus a summary line, exit 0. |
| `laya_triage` exits 1 or 2 (invalid config, inputs or permission) | Fail loudly (misconfiguration). |

### 7.6 Security model

- **Permissions:** the workflow declares `permissions: { issues: write, contents: read }` only.
- **Untrusted input:** issue title and body are attacker-controlled. **Never** interpolate
  `${{ github.event.issue.title }}` or `.body` into a `run:` shell step, because that is a known
  script-injection vector. Read the event JSON from `$GITHUB_EVENT_PATH` inside Python.
- **Comments:** the escalation comment contains only fixed text plus label names and numbers. It
  never echoes issue content.
- **Secrets:** only `GITHUB_TOKEN` is needed. The model is public, so no HF token is needed at
  runtime.
- **Supply chain:** pin the model by revision SHA, pin Python dependencies in a lock file, and pin
  third-party actions by version tag (ideally SHA).

### 7.7 Caching strategy

- **Model:** `~/.cache/huggingface` (`HF_HOME` set explicitly), restored and saved with `actions/cache/restore` and
  `actions/cache/save`. Key: `laya-hf-${{ runner.os }}-<owner>--<name>--<revision>`, where repo and revision come
  from the merged config through `python -m laya_triage --print-model` (the same loader as the run; `hashFiles`
  cannot see the action's files or apply the user config). A model change invalidates the cache automatically.
  Saved only after a successful load; on a hit whose snapshot holds every file laya needs, the run is fully
  offline (`HF_HUB_OFFLINE=1`).
- **Dependencies:** a venv at `~/.cache/laya-triage/venv` with the pinned dependencies (CPU-only torch), keyed by
  `runner.os`, `runner.arch`, the exact Python version reported by `setup-python`, and a hash of
  `constraints-linux.txt`. `setup-python`'s `cache: pip` is not used: it needs a dependency file in the consumer's
  workspace. Every restore is checked (imports, CPU-only torch); a failed check rebuilds without saving. The
  action's own code is reinstalled on every run (`--no-deps --no-build-isolation`, offline), so the cached venv
  never runs stale action code. The pins are the Linux lock generated from CI run 37009763300 (`pip freeze --all`,
  Ubuntu 24.04.5, Python 3.11.16).
- **Who can save:** runs triggered by `issues` events restore both caches but cannot save them (observed in the
  sandbox: "cache write denied: token has no writable scopes"; the run still succeeds). Caches are saved by
  `schedule` and `workflow_dispatch` runs. The `warm-cache` input (§12.4) makes such a run only load the model and
  classify one synthetic issue (`python -m laya_triage --warm`: no issues read, no GitHub API calls, no token), so
  the normal save steps run; the consumer example (§12.5) warms twice a week (margin against the 7-day eviction
  and late scheduled starts) and on demand. GitHub may evict caches unused
  for 7 days; the next run is then cold (NFR-3, R13).

---

## 8. Data specification

### 8.1 Sources

| Source | Use | Notes |
|---|---|---|
| **NLBSE'24 Issue Report Classification** (github.com/nlbse2024/issue-report-classification) | **Primary.** Train, validation, and test. | About 3,000 issues from 5 repos (facebook/react, tensorflow/tensorflow, microsoft/vscode, bitcoin/bitcoin, opencv/opencv). One label each: `bug`, `feature`, `question`. Fields: repository, label, title, body. Publishes per-repo baseline P/R/F1 results. No issue IDs/URLs in the data; upstream LICENSE file is empty, so raw data is never committed; obtained by `training/fetch_data.py` pinned to upstream commit `2927bc67eb42db8affd16eaf3e5a6d74f3063961` and verified by SHA-256 (train `18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5`, test `4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85`). **Verify license and split sizes in Phase 1.** |
| NLBSE'23 Issue Report Classification | **Optional (Protocol B only).** Extra training data. | About 1.4M issues with 4 labels including `documentation`. Must be **deduplicated against NLBSE'24 by content hash (title + body), not by URL**, since repos may overlap. |
| Sandbox GitHub repo (own) | End-to-end demo and qualitative evaluation | Issues written by hand during Phase 4. Not used for training. |

### 8.2 Evaluation protocols

- **Protocol A (strict, headline).** Train only on the NLBSE'24 official train split, with no
  resampling or rebalancing, matching the competition rules. Evaluate on the official test split.
  This makes our numbers directly comparable to the published baseline. Pretrained models are
  allowed by the competition rules but may only be fine-tuned on the provided training set;
  Protocol B violates that and is non-comparable.
- **Protocol B (optional).** Protocol A training data plus an NLBSE'23 subsample deduplicated
  against NLBSE'24 by content hash (title + body), not by URL.
  Results are reported **separately** and never mixed with Protocol A numbers.

### 8.3 Splits

Official train has 1,500 rows. The 4 rows at 0-based CSV rows 559, 900, 901 and 1114 have a
`(title, body)` that also appears in the official test file; they are dropped, leaving 1,496.
A row whose (title, body) is duplicated within official train is dropped in all copies; on the real
data this coincides with the test-overlap rule (rows 559, 900, 901, 1114).

| Split | Derived from | Size | Used for |
|---|---|---|---|
| `train` | The 1,496 kept official-train rows minus `val` | ≈ 1,196 | Fine-tuning (the notebook also carves its own calibration slice from this) |
| `val` | 20% of the 1,496 kept rows, stratified by `repo × label`, seed 42, drawn only from rows **not** listed in `results/phase0_sample_ids.txt` (those are forced into `train`) | ≈ 300 | Model selection, **threshold fitting**, ECE check |
| `test` | Official test, all 1,500 rows, untouched | 1,500 | Final metrics, evaluated **once** after thresholds are frozen |

Split membership is written to `data/manifests/{train,val,test}.txt` and committed. The dropped rows
are listed in `data/manifests/dropped_train.txt` (id, content hash, reason).

- **Manifest line format:** `<id>\t<content_sha1>`.
- **ID format:** `nlbse24-<source>-<row:04d>`, where `source` is the CSV the row came from (`train`
  or `test`) and `row` is its 0-based index in that raw CSV (pinned by SHA-256, §8.1). A `val` row
  therefore carries the `train` prefix.
- **`content_sha1`:** SHA-1 over the UTF-8 bytes of the raw `title + "\n" + body`.

### 8.4 Preprocessing (single source: `src/laya_triage/preprocess.py`)

**Invariant:** training, evaluation, and runtime all import the same `preprocess()` function. Never
duplicate cleaning logic.

Steps, in order:

1. Strip HTML comments (`<!-- ... -->`). Issue templates leave large amounts of comment boilerplate.
2. Collapse fenced code blocks and logs longer than `max_code_lines` (default 20) to the first 10 and
   last 5 lines, with a `[... N lines omitted ...]` marker. Logs consume the token budget without
   adding much signal.
3. Replace image markdown `![alt](url)` with `[image: alt]`. Replace bare URLs with `[url]`.
4. Normalise whitespace (collapse 3 or more newlines to 2, strip trailing spaces).
5. Truncate the body to `max_body_chars` (default 6,000) as a safety cap. Token-level truncation is
   handled by Laya's `max_len`.
6. Produce the state: `{"title": <title stripped>, "body": <cleaned body>}`.

### 8.5 Training row format (input to the fine-tuning notebook)

Matches the schema the Laya notebook expects: `state`, `questions`, and `gold` with per-option
probabilities. Hard labels become one-hot distributions. Label smoothing ε is an experiment knob
with default 0.0 (§9.5).

```json
{
  "id": "nlbse24-train-0123",
  "repo": "microsoft/vscode",
  "state": {"title": "Terminal crashes when ...", "body": "Steps to reproduce ..."},
  "questions": {
    "issue_type": {
      "type": "choice",
      "instructions": "You are triaging a newly opened GitHub issue. Using its `title` and `body`, decide which type of issue it is.",
      "criteria": {
        "bug": "reports a defect: a crash, error message, failing build or test, regression, or behaviour that contradicts the documentation",
        "feature": "proposes something new: a new capability, option or API, or an improvement to how existing behaviour works",
        "question": "asks for help: how to use or configure something, why it behaves a certain way, or troubleshooting the author's own setup"
      }
    }
  },
  "gold": {"issue_type": {"probabilities": {"bug": 1.0, "feature": 0.0, "question": 0.0}}}
}
```

IDs are synthetic (§8.3) because the dataset has none.

> Resolved in Phase 3: the upstream notebook's dataset rows store `state`, `questions`, and `gold` as
> JSON strings, and its loading code calls `json.loads` on each. Our rows hold native JSON objects, and
> `training/make_items.py` reads them directly.

### 8.6 Evaluation row format (`laya-evals` JSONL)

```json
{"id": "nlbse24-test-0456",
 "state": {"title": "...", "body": "..."},
 "questions": {"issue_type": { "...same as above..." }},
 "expected": {"issue_type": "bug"},
 "tags": ["repo:microsoft/vscode"]}
```

`laya-evals` ignores extra keys such as `"id"`, so `validate` accepts these rows.

### 8.7 Data quality checks (must pass before Phase 3)

- No ID and no content hash appears in more than one of train/val/test. This is an automated
  assertion.
- train + val + dropped = 1,500 official train rows.
- The Phase 0 sample rows (`results/phase0_sample_ids.txt`) are all in train.
- Class counts per split and per repo are reported in `data/DATA_CARD.md`.
- 30 random training rows are manually reviewed (label looks right, cleaning didn't destroy content).
  Findings go in `DATA_CARD.md`.
- `laya-evals validate data/processed/test.jsonl` passes.

---

## 9. Model specification

### 9.1 Base checkpoint

- `convaiinnovations/laya` (English, ModernBERT-large encoder, about 421M parameters).
- The root checkpoint's native budget is max_len 512 / head_max_len 192; fine-tuning sets 1024/256 as in the Laya notebook (train_ddp.py overrides these).
- The multilingual checkpoint is **not** used, because English issues perform better on the English
  checkpoint (ADR-4).

### 9.2 Question schema (frozen for v1.0)

Defined once in `src/laya_triage/questions.py` and imported everywhere:

```python
"""Frozen issue-type question (PROJECT_SPEC.md §9.2). Changing it requires §16 rule 4."""

ISSUE_TYPE_QUESTION = {
    "issue_type": {
        "type": "choice",
        "instructions": "You are triaging a newly opened GitHub issue. Using its `title` and `body`, decide which type of issue it is.",
        "criteria": {
            "bug": "reports a defect: a crash, error message, failing build or test, regression, or behaviour that contradicts the documentation",
            "feature": "proposes something new: a new capability, option or API, or an improvement to how existing behaviour works",
            "question": "asks for help: how to use or configure something, why it behaves a certain way, or troubleshooting the author's own setup",
        },
    }
}
LABELS = ("bug", "feature", "question")  # order is part of the contract
```

Frozen after the Phase 0 comparison (rule: >=5 macro-F1 points at both 512/192 and 1024/256); original W0 wording and results are in results/phase0.md.

Rules:

- **The option order is part of the contract.** It must be identical in training, evaluation, and
  runtime.
- Wording changes require retraining and a version bump (§16).
- Phase 0 may compare two or three instruction and description wordings zero-shot. The best one is
  frozen **before** Phase 3.

### 9.3 Fine-tuning recipe (starting point: Laya notebook defaults)

| Setting | Value |
|---|---|
| Hardware | Kaggle, GPU T4 ×2, Internet on |
| Epochs | 4 (adjust using val macro-F1) |
| Effective batch | 64 sequences (8 per micro-batch × 2 GPUs × 4 accumulation steps) |
| Learning rates | Encoder 2.5e-5, head 1e-4, AdamW, cosine schedule |
| Memory | fp16 autocast, gradient checkpointing on encoder and head, grad-norm clip 1.0 |
| Sequence budget | `max_len` 1024, `head_max_len` 256 |
| Loss | RLCD policy-gradient term (proper-scoring-rule reward) + soft cross-entropy on `gold` |
| Calibration | The notebook fits one temperature per type on its held-out calibration slice and drops the inherited `temperature_by_options`; that fit is kept in `run_meta.json`. The shipped choice temperature is refit on val by NLL with laya's own fitter (`laya.calibrate.fit_temperature_map` on `records_from_labeled` logits, `results/phase3/refit_temperature.py`); weights are unchanged, so the shipped revision differs from the trained one only in `rl_agent_config.json` `"temperature"`. For R1 the notebook's 119-item fit (4.0188) over-softened (val ECE 0.129; the val-NLL-optimal T is about 2.7). |
| Output | Push to `Prasanna85/laya-issue-triage`. Record the commit SHA. |
| Seed | 42 |

Expected runtime is well under an hour for about 2–3k training rows. The notebook's 4–5 hour figure
is for about 30k questions.

**Phase 3 adaptation (v1.0.6).** `training/finetune_kaggle.ipynb` adapts the Laya notebook at tag
`v0.3.23`, commit `d8a2e59781ca135169a36095056132e273cd9938` (`ae3222b3fcdf424254a2c726d72161f671a86d95`
is the annotated tag object, not a commit). The Apache-2.0 attribution and the list of changes are in its
first cell.

- **Base pinned.** `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, downloaded
  anonymously (`token=False`) with only `rl_agent_config.json`, `model.safetensors`, `tokenizer/*` and
  `encoder/*`, the set `laya.load` needs.
- **Sequences at 1024/256.** `training/make_items.py` builds the training items at `max_len` 1024 /
  `head_max_len` 256, the budget saved in the config and used at inference. Upstream builds them at the
  root config's 512/192 while `train_ddp.py` saves 1024/256, which would cut the 38.0% of our training
  items that are longer than 512 tokens; this is fixed here. The frozen question's head is 112 tokens, so
  `head_max_len` 192 vs 256 does not change our sequences; `max_len` sets the room for the state (399 vs
  911 tokens). The root weights load with `strict=True` at 256 because the head has no positional
  parameters.
- **Pins and items-hash parity.** The notebook's first code cell installs `laya==0.3.23` and
  `transformers==5.18.0` and asserts laya 0.3.23, transformers 5.18.0, tokenizers 0.23.2,
  huggingface_hub 1.33.0 and safetensors 0.8.0 (Kaggle: Python 3.12, torch 2.10.0+cu128, 2× T4).
  `train.jsonl` and `val.jsonl` are checked by SHA-256, size and row count. Before training, the SHA-256
  of the built items must equal the local build recorded in `results/phase3/items_sha256.json` (one hash
  per eps, plus one for the SMOKE subset).
- **Training plan.** 1,196 items → 119 calibration items (upstream rule `min(400, n // 10)`, upstream
  seed 20260922, unchanged) → 1,076 training items → 538 per GPU → 68 micro-batches and 17 optimizer
  steps per epoch → **68 optimizer steps** over 4 epochs (effective batch 64).
- **Deviations from upstream `train_ddp.py`.** Scheduler `T_max` is the true optimizer-step count (68).
  Upstream's floor division gives 64, so its cosine LR rises slightly over the last 4 steps (negligible).
  `torch.manual_seed(SEED + rank)` and `torch.cuda.manual_seed(SEED + rank)` are added (upstream seeds
  only the Python shuffle); GPU kernels stay non-deterministic, so runs are seeded but not bit-for-bit
  reproducible. `model_name` is `laya-issue-triage` (laya does not read it when loading). Settings come
  from a run config, and unused imports are removed. Loss, optimizer, calibration split and temperature
  fitting are unchanged. The calibration seed and all counts are recorded in `run_meta.json`.
- **Output.** `upload_folder` of `model.safetensors`, `rl_agent_config.json`, `tokenizer/*`, `encoder/*`
  and `run_meta.json` (versions, constants, hashes, temperatures; no issue text) to the existing
  repo `Prasanna85/laya-issue-triage` (public since 2026-10-02; model card uploaded). The notebook never creates a repo. `SMOKE = True` is the default:
  50 rows spread over the file, 1 epoch, `GRAD_ACCUM` 1, no push.

### 9.4 Inference settings

- `max_len` at inference **must equal** the value used in training (1024). Read it from the
  checkpoint config, and assert it in a test.
- Device: CPU, fp32 (the GitHub runner). Do not enable bf16 autocast on CPU.
- Load once per Action run. Use `predict_batch` in backfill mode.

### 9.5 Experiments allowed in Phase 3 (keep it to these)

| Knob | Values | Pick by |
|---|---|---|
| Label smoothing ε | 0.0 (default), 0.1 | Val macro-F1, then val ECE |
| Epochs | 3, 4, 6 | Val macro-F1 |
| Instruction wording | The frozen Phase-0 winner only | — |

Log every run in `results/experiments.md`: run ID, settings, val macro-F1, val ECE, HF revision.

### 9.6 Optional: option-order averaging (v1.2+)

Laya supports `option_order`. Asking the same question under all 3 rotations shares one forward pass
and averages out position bias. Evaluate it only if Phase 3 shows systematic confusion tied to option
position. It is not part of the v1.0 contract.

---

## 10. Evaluation specification

### 10.1 Models compared (all on the same test split, Protocol A)

| # | System | Purpose |
|---|---|---|
| B0 | Majority class | Floor |
| B1 | TF-IDF (word 1–2 grams) + logistic regression, `class_weight=None` | Cheap, strong classical baseline. Must be beaten to justify a 421M model. |
| B2 | Laya base checkpoint, zero-shot, same question schema; zero-shot at native 512/192 (secondary run at 1024/256) | Shows the value of fine-tuning |
| B3 | NLBSE'24 published baseline (numbers copied from the competition repo, with citation) | External reference point |
| **M1** | **Laya fine-tuned + calibrated (ours)** | Headline |
| M1-onnx | M1 exported to ONNX INT8 (v1.2, optional) | Latency/accuracy trade-off |

### 10.2 Metrics

| Metric | Definition | Tool |
|---|---|---|
| Accuracy | Fraction correct; reported in `metrics_test.json` | sklearn / `laya-evals` |
| **Cross-repo macro-F1 (headline)** | Arithmetic mean of the 5 per-repo macro-F1 scores on test (the definition behind the published SetFit 0.8270) | sklearn |
| Pooled macro-F1 | Unweighted mean of per-class F1 over all test issues; reported in `metrics_test.json` | sklearn |
| Per-class P/R/F1 | For `bug`, `feature`, `question` | sklearn |
| Per-repo macro-F1 | Same, sliced by repo | sklearn |
| ECE (15 bins) | On `answer_confidence` | `laya-evals` / `laya.common.ece_score` |
| Brier score | Multi-class | numpy |
| Coverage@precision | For each threshold τ, coverage = share of issues with `answer_confidence ≥ τ`, precision = accuracy on those | Custom (`eval/gating.py`) |
| Latency | p50/p95 per issue, CPU, model preloaded, 100 issues | Custom timer |
| Model load time | Cold load from HF cache | Custom timer |

Coverage@precision is reported both as a curve over one global τ and with the per-label τ of the config
(§10.4).

### 10.3 Required plots (`results/figures/`)

1. Reliability diagram, before and after temperature calibration.
2. Coverage–precision curve, with the chosen operating point marked.
3. Confusion matrix (normalised) for M1 and B1.
4. Per-repo macro-F1 bar chart for M1 and B1.

### 10.4 Threshold selection procedure (frozen before touching test)

1. Run M1 on **val** on **CPU, fp32**, the same numerics as the Action.
2. For each label ℓ, the candidate thresholds τ are the distinct `answer_confidence` values among
   val issues predicted as ℓ. S(τ) is the set of those issues with `answer_confidence ≥ τ`. Only
   τ with |S(τ)| ≥ 20 are considered.
3. For each candidate τ, bootstrap the issues in S(τ): 2,000 resamples with replacement, seed 42.
   The 5th percentile of precision across resamples is the lower bound.
4. Choose the smallest τ_ℓ whose lower bound is ≥ 0.90.
5. If no τ qualifies, set τ_ℓ = 1.01 (never auto-apply that label) and document it.
6. Also record the point-estimate threshold (the smallest τ with val precision ≥ 0.90 and
   |S(τ)| ≥ 20) and its coverage, for comparison. The config uses the lower-bound τ_ℓ.
7. Write the thresholds to `config/triage.default.yml` and commit **before** running test. Run test
   once. Report the achieved precision and coverage. Do not re-tune on test.

Candidate thresholds are the distinct val `answer_confidence` values among issues predicted ℓ, and every
candidate's 2,000 resamples come from a fresh `numpy.random.RandomState(42 + i)`, where i is ℓ's index
in `LABELS` (`eval/gating.py`).

The config uses τ_lb. τ_point, the full coverage–precision curve, and B1 under the same rule are
reported on test as pre-declared secondary views.

Lower-bound thresholds are conservative, so coverage will be lower than with point-estimate
thresholds.

### 10.5 Results table (test; filled in Phase 3)

Test split, 1,500 issues, Protocol A. Sources: `results/phase3/test_R1b/metrics_test.json` (M1, and the baselines
recomputed from the committed `results/phase2/predictions_test_*.csv`); B3 copied from `results/phase2.md`.
Details, per-repo scores and paired comparisons: `results/phase3.md` (Test results).

| System | Cross-repo macro-F1 (headline) | Pooled macro-F1 | Acc | F1 bug | F1 feature | F1 question | ECE | Coverage @ val thresholds (test precision) | CPU p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| B0 majority (`question`) | 0.1667 | 0.1667 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | — | — | — |
| B1 TF-IDF+LR | 0.7655 | 0.7666 | 0.7673 | 0.7843 | 0.7925 | 0.7230 | 0.0493 | 0.4187 (0.9156) [s] | 0.29 [a] |
| B2 Laya zero-shot 512/192 (native) | 0.6108 | 0.6160 | 0.6453 | 0.6693 | 0.7757 | 0.4029 | 0.0684 | — | — [b] |
| B2 Laya zero-shot 1024/256 | 0.6257 | 0.6310 | 0.6567 | 0.6773 | 0.7827 | 0.4330 | 0.0647 | — | — [b] |
| B3 SetFit baseline 0.8270 (NLBSE'24 README = `output/results.json`) | 0.8270 | 0.8263 [c] | 0.8267 [c] | 0.8425 [c] | 0.8555 [c] | 0.7809 [c] | — | — | — |
| B3 RoBERTa (`output/roberta/results.json`) | 0.7923 | 0.7926 [c] | 0.7927 [c] | 0.8052 [c] | 0.8064 [c] | 0.7663 [c] | — | — | — |
| B3 fastText (`output/fasttext/results.json`) | 0.7184 | 0.7193 [c] | 0.7193 [c] | 0.7362 [c] | 0.7390 [c] | 0.6827 [c] | — | — | — |
| **M1 ours** (R1b `76ece1fb…`, T 2.6968) | **0.8020** | 0.8023 | 0.8020 | 0.8033 | 0.8358 | 0.7677 | 0.0454 | 0.2607 (0.8747; below the 0.90 target) | 178 |

- [a] B1 latency: `results/phase2/latency_B1.json` (first 100 val issues, local Apple M4 Pro CPU).
  M1 latency: the test run's meta (1,500 issues, 8 threads, model preloaded, same CPU); p95 436 ms. Neither
  is from a GitHub runner.
- [b] B2 latency was not measured in Phase 2 or 3; `results/phase2.md` [b] cites Phase 0 under different
  conditions (W0 wording, uncleaned input).
- [c] **Derived, not published.** NLBSE'24 publishes per-repo, per-class P/R/F1 only; pooled per-class F1,
  pooled macro-F1 and accuracy are recovered exactly from them because every test repo has 100 issues per
  class (`results/phase2.md` [c]). B3 has no probabilities, so no ECE or coverage. A second SetFit result,
  `output/setfit/results.json` (cross-repo 0.8240, provenance unknown), is a footnote only and is neither
  averaged with nor chosen over 0.8270.
- [s] Pre-declared secondary view: B1 under the same rule with its own val thresholds.
- Coverage is the share of all test issues auto-applied at thresholds fitted on val (§10.4), with the
  precision of those issues in parentheses. M1's thresholds are the config's (bug 0.6033, feature and
  question 1.01); B2 has no thresholds. B0's ECE is computed from train priors and carries no information.
- Protocol differences (per-repo vs pooled training, training rows, input text, model selection, copied vs
  run, truncation, latency) are listed in `results/phase2.md` and `results/phase3.md`.

---

## 11. Repository structure

```
laya-triage/
├── PROJECT_SPEC.md               # this document (source of truth)
├── CLAUDE.md                     # short pointer for coding agents (see Appendix A)
├── README.md                     # front door: banner, worked example, results and misses, two ways to use it; generated
├── LICENSE                       # Apache-2.0 (matches Laya)
├── docs/
│   ├── MODEL_CARD.md             # HF model card (README of the model repo), generated; uploaded to the model repo by the owner (2026-10-02)
│   ├── make_readme.py            # builds README.md and the three docs below from their templates and the result files; --check covers all four
│   ├── README.template.md        # README prose with {{placeholders}}; no typed numbers (rule 11), as for every template here
│   ├── DEEP_DIVE.md              # generated: glossary, architecture, training, calibration, evaluation, gating, runtime, security, limitations
│   ├── DEEP_DIVE.template.md
│   ├── USING_THE_ACTION.md       # generated: full workflow, permissions, labels, caches, apply, backfill, comments, config table, troubleshooting
│   ├── USING_THE_ACTION.template.md
│   ├── USING_THE_MODEL.md        # generated: install, the snippet, output, gating, offline use
│   ├── USING_THE_MODEL.template.md
│   ├── snippets/                 # classify_one.py (shown verbatim in README, USING_THE_MODEL.md and the model card) and consumer-minimal.yml
│   ├── architecture.png          # master diagram (shown only in DEEP_DIVE.md); editable source architecture.drawio
│   ├── architecture.drawio
│   └── banner_image.png          # README banner
├── pyproject.toml                # package metadata + pinned deps
├── constraints-linux.txt         # runner pins (CPU-only torch); replaced by the lock-linux freeze
├── action.yml                    # composite GitHub Action definition (Linux X64)
├── config/
│   └── triage.default.yml        # default labels, thresholds (from §10.4), mode
├── src/laya_triage/
│   ├── __init__.py
│   ├── __main__.py               # entry: python -m laya_triage
│   ├── config.py                 # load + validate YAML config (pydantic or dataclasses)
│   ├── questions.py              # ISSUE_TYPE_QUESTION, LABELS (frozen contract)
│   ├── preprocess.py             # preprocess(title, body) -> state   (shared!)
│   ├── classifier.py             # Classifier: load Agent, classify(state) -> TriageResult
│   ├── policy.py                 # decide(result, cfg) -> Decision  (pure function)
│   ├── event_loader.py           # read GITHUB_EVENT_PATH, skip rules
│   ├── github_client.py          # add_labels, comment, retry/backoff
│   └── reporter.py               # job summary markdown + JSONL artifact
├── training/
│   ├── fetch_data.py             # NLBSE'24 at pinned commit + SHA-256 check (never run in CI)
│   ├── nlbse_data.py             # NLBSE'24 raw CSV loading, synthetic IDs, content hashes (§8.3)
│   ├── build_dataset.py          # NLBSE -> data/processed/*.jsonl + manifests
│   ├── make_items.py             # train.jsonl -> training items (stdlib + transformers + laya only)
│   ├── build_notebook.py         # generates finetune_kaggle.ipynb (never edit the notebook by hand)
│   └── finetune_kaggle.ipynb     # adapted Laya notebook (embeds make_items.py byte for byte)
├── eval/
│   ├── baselines.py              # B0, B1, B2
│   ├── run_eval.py               # M1 on val/test, CPU fp32, writes results/*.json
│   ├── gating.py                 # threshold selection + coverage–precision
│   ├── metrics.py                # §10.2 metrics, pure functions (shared by baselines and run_eval)
│   └── plots.py                  # figures in §10.3
├── data/                         # raw/ and processed/ are gitignored
│   ├── manifests/                # committed split ID lists
│   └── DATA_CARD.md
├── results/
│   ├── phase0.md
│   ├── phase4.md                 # Action, CI, sandbox results; FR-5 coverage notes
│   ├── phase3/                   # items_sha256.json: expected training-item hashes (+ generator)
│   ├── phase3/make_model_card.py # builds docs/MODEL_CARD.md from metrics_test.json, metrics_val.json, config
│   ├── phase3/test_report.py     # single test-run report: metrics_test.json, report tables, §10.3 figures
│   ├── experiments.md
│   ├── phase3/val_R1b/, phase3/test_R1b/  # M1 aggregates incl. metrics_val.json / metrics_test.json
│   └── figures/
├── tests/
│   ├── test_preprocess.py
│   ├── test_fetch_data.py        # training/fetch_data.py: download, SHA-256 verify, pinned constants (no network)
│   ├── test_build_dataset.py     # training/build_dataset.py: splits, dedupe, IDs, manifests, row formats
│   ├── test_policy.py
│   ├── conftest.py               # offline_classifier fixture for the slow tests (local HF cache, no network)
│   ├── test_config.py
│   ├── test_event_loader.py
│   ├── test_github_client.py     # mocked HTTP
│   ├── test_classifier.py        # Classifier with a fake agent: checks, mapping, errors without messages
│   ├── test_reporter.py          # JSONL schema and summary; no issue text, no exception messages
│   ├── test_main.py              # python -m laya_triage with fake classifier and client (modes, skips, exits)
│   ├── test_no_shell_execution.py # AST scan of src/ for subprocess / os.system / eval / exec
│   ├── test_eval_metrics.py      # eval/metrics.py on hand-computed fixtures
│   ├── test_run_eval.py          # eval/run_eval.py guards and options (no model, no data files)
│   ├── test_gating.py            # eval/gating.py (§10.4) on hand-computed fixtures
│   ├── test_plots.py             # eval/plots.py on synthetic arrays (Agg backend)
│   ├── test_test_report.py       # results/phase3/test_report.py helpers on synthetic inputs
│   ├── test_posthoc_analysis.py  # results/phase3/posthoc_analysis.py helpers on synthetic inputs
│   ├── test_model_card.py        # docs/MODEL_CARD.md builds from the metrics files and is up to date
│   ├── test_readme.py            # README.md and the three docs build and are up to date; templates have no typed numbers; README structure
│   ├── test_snippets.py          # docs/snippets: constants == config, compiles, workflow rules; @pytest.mark.slow run offline
│   ├── test_make_items.py        # training/make_items.py; tokenizer tests @pytest.mark.slow
│   ├── test_notebook_sync.py     # notebook == make_items.py, pins, no create_repo / token literal
│   ├── test_workflows.py         # action.yml / workflows: no issue content in shells, SHA pins, allow-listed expressions
│   ├── test_constraints.py       # constraints-linux.txt: the Linux lock format and pins
│   ├── test_sandbox.py           # sandbox fixtures, expected_local.json, compare.py, create_issue.py
│   ├── test_model_smoke.py       # @pytest.mark.slow, real model, 3 fixtures
│   └── test_parity_val.py        # @pytest.mark.slow, production path vs Phase 3 val predictions (30 issues)
└── .github/workflows/
    ├── ci.yml                    # ruff + pytest -m "not slow" (no weights); lock-linux freeze job
    └── triage.yml                # dogfood: deferred until this repository is public
sandbox/                          # Phase 4 sandbox kit: authored fixtures (issues.json), expected_local.json,
                                  # make_expected.py, compare.py, create_issue.py, latency_probe.py,
                                  # workflow.example.yml, README.md
```

---

## 12. Interfaces and configuration

### 12.1 Core data types (`src/laya_triage/`)

```python
from dataclasses import dataclass
from typing import Literal

Label = Literal["bug", "feature", "question"]

@dataclass(frozen=True)
class TriageResult:
    issue_number: int
    label: Label                      # argmax
    probabilities: dict[str, float]   # keyed by LABELS, sums to ~1
    answer_confidence: float          # max(probabilities); calibrated
    latency_ms: float
    model_repo: str
    model_revision: str

@dataclass(frozen=True)
class Decision:
    action: Literal["apply", "escalate", "skip"]
    labels_to_add: tuple[str, ...]    # actual GitHub label names from config
    comment: str | None               # fixed-template text, never echoes issue content
    reason: str                       # human-readable, goes to summary
```

### 12.2 Module contracts

| Function | Signature | Purity / side effects |
|---|---|---|
| `preprocess` | `(title: str, body: str \| None, cfg) -> dict` | Pure |
| `Classifier.__init__` | `(repo: str, revision: str, max_len: int, device="cpu")` | Loads the model once |
| `Classifier.classify` | `(state: dict, issue_number: int) -> TriageResult` | Inference only |
| `Classifier.classify_batch` | `(states, issue_numbers) -> list[TriageResult]` | Uses `predict_batch` (`batch_size` 8). Verified against single-issue `predict` (the call the thresholds were fitted on): identical labels and max \|Δp\| 0.0 on 3 synthetic issues and 30 val issues (`tests/test_model_smoke.py`, `tests/test_parity_val.py`, CPU fp32). If the two ever differ by more than the 4-decimal rounding, `classify_batch` must fall back to one `predict` per issue. |
| `decide` | `(result: TriageResult, cfg) -> Decision` | **Pure**, fully unit-tested |
| `should_skip` | `(issue: dict, cfg) -> str \| None` | Pure. Returns the skip reason or None. |
| `GitHubClient.apply` | `(issue_number, decision) -> None` | Network calls, with retry |
| `Reporter.write` | `(results, decisions) -> None` | Writes `$GITHUB_STEP_SUMMARY` and an artifact file |

### 12.3 Config file (`.github/laya-triage.yml` in the target repo; defaults in `config/triage.default.yml`)

```yaml
model:
  repo: "Prasanna85/laya-issue-triage"
  revision: "<commit-sha>"        # pinned; never "main" in production
  max_len: 1024
  backend: torch                  # torch | onnx (v1.2)

labels:                           # model class -> GitHub label name
  bug: "type: bug"
  feature: "type: feature"
  question: "type: question"
  escalate: "triage: needs-human"

gating:
  thresholds:                     # from §10.4; placeholders until Phase 3
    bug: 0.80
    feature: 0.80
    question: 0.85

behaviour:
  mode: dry-run                   # dry-run | apply   (default dry-run for safety)
  comment_on_escalate: false      # comments are opt-in (labels only by default)
  skip_if_labeled: true           # skip if any labels.{bug,feature,question} present
  skip_authors: ["dependabot[bot]", "renovate[bot]"]
  create_missing_labels: false
  escalate_on_error: false

preprocess:
  max_code_lines: 20
  max_body_chars: 6000
```

### 12.4 Action interface (`action.yml`, composite)

| Input | Required | Default | Meaning |
|---|---|---|---|
| `github-token` | no | `${{ github.token }}` | Token for label/comment API calls (needs `issues: write`) |
| `config-path` | no | `.github/laya-triage.yml` | Path to the config file, relative to the workspace |
| `mode` | no | empty (from config) | `dry-run` or `apply`; overrides `behaviour.mode`; backfill defaults to `dry-run` |
| `backfill-count` | no | `0` | If > 0 on `workflow_dispatch`, classify the N most recent open issues; must be a non-negative integer (else exit 2) |
| `warm-cache` | no | `false` | `true`: only validate the config, load the model at the pinned revision and classify one built-in synthetic issue (passed as `--warm`), so the caches are saved (§7.7); no issues are read and no GitHub API call is made. Must be `true` or `false` (else exit 2) |

Linux X64 runners only. Inputs reach the scripts only through `env:`; no `${{ }}` expression appears inside a
`run:` script (enforced by `tests/test_workflows.py`). The triage log is uploaded as the artifact
`laya-triage-log-<run_id>-<run_attempt>` (`if: always()`). The consumer workflow needs `issues: write` and
`contents: read`, and should set a job `timeout-minutes` (20).

### 12.5 Example consumer workflow

```yaml
name: Laya Triage
on:
  issues:
    types: [opened]
  schedule:
    - cron: "17 3 * * 1,4"  # cache warm-up twice a week (§7.7)
  workflow_dispatch:
    inputs:
      backfill-count: { description: "Open issues to classify", default: "20" }
      mode: { description: "dry-run or apply (empty = dry-run for backfill)", default: "" }
      warm-cache: { description: "Only load the model and save the caches", type: boolean, default: false }

permissions:
  issues: write
  contents: read

jobs:
  triage:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1 (reads the config file)
        with:
          persist-credentials: false
      - uses: Prasanna-KS-85/laya-triage@<40-hex commit SHA> # v1.0.0
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          backfill-count: ${{ github.event.inputs.backfill-count || '0' }}
          mode: ${{ github.event.inputs.mode }}
          warm-cache: ${{ github.event_name == 'schedule' || github.event.inputs.warm-cache == 'true' }}
```

`github.event_name` is a fixed event-type string, not attacker-controlled (allow-listed in `tests/test_workflows.py`).
Issues-triggered runs cannot save the caches; the schedule and the `warm-cache` input keep them saved (§7.7, R14).
Pin every action by full commit SHA with the tag as a comment. The Action is tested on ubuntu-24.04; ubuntu-latest moves to Ubuntu 26 on 2026-10-19 and is to be re-tested after that. The sandbox version of this workflow (with runner
diagnostics and the NFR-1 probe) is `sandbox/workflow.example.yml`.

### 12.6 Escalation comment templates (fixed text)

Posted only when `comment_on_escalate: true`. `{label1}`, `{label2}` are the two most probable classes (ties in
`LABELS` order); nothing else is filled in, so the comment never echoes issue content.

(a) The predicted class's threshold is reachable (τ ≤ 1.0) and the confidence is below it:

```
🤖 Laya Triage couldn't classify this issue confidently, so it's been marked for a maintainer.
Top guesses: `{label1}` ({p1:.0%}), `{label2}` ({p2:.0%}).
```

(b) The predicted class's threshold is unreachable (τ > 1.0, e.g. 1.01): auto-labelling of that class is disabled
by configuration:

```
🤖 Laya Triage suggests `{label1}` ({p1:.0%}), but automatic labelling is disabled for this issue type in this repository's configuration, so it's been marked for a maintainer.
Second guess: `{label2}` ({p2:.0%}).
```

FR-2 stays literal (`answer_confidence >= τ` applies), so τ = 1.0 can still apply at confidence 1.0000.

---

## 13. Phased delivery plan

Each phase ends with a **verification gate**. Do not start the next phase until its gate passes and
its deliverables are committed. Time estimates assume part-time work and are guidance only.

### Phase 0: Feasibility and environment (≈ 2–3 days)

**Goal:** prove the stack runs and get first numbers before investing more time.

Tasks:
1. Create the repo with the §11 skeleton, `pyproject.toml`, and Python 3.11 venv. Install the latest
   PyPI `laya` and **record the exact version**.
2. Load `convaiinnovations/laya` on CPU. Record download size, load time, and RAM peak.
3. Run zero-shot on about 100 NLBSE'24 issues with 2–3 candidate wordings of the §9.2 question.
4. Measure CPU latency p50/p95 for those 100 issues, preloaded.
5. Freeze the winning question wording into `questions.py`.

**Deliverables:** `results/phase0.md` (version, sizes, timings, zero-shot accuracy per wording,
chosen wording) and the repo skeleton.

**Gate:** inference works on CPU, latency is recorded, wording is frozen, and model size fits the
Actions cache. If latency > 1 s p95, note it and keep going; v1.2 ONNX is the mitigation.

### Phase 1: Dataset (≈ 3–5 days)

Tasks:
1. Download NLBSE'24. Verify the license, field names, and official split sizes, and record them in
   `DATA_CARD.md`.
2. Implement `training/fetch_data.py`: download NLBSE'24 at the pinned upstream commit (§8.1) and
   check the SHA-256 of both CSVs. Never run in CI.
3. Implement `preprocess.py` with unit tests covering every rule in §8.4.
4. Implement `build_dataset.py`: clean, carve a stratified val split, write training rows (§8.5),
   eval rows (§8.6), and manifests.
5. Write `tests/test_build_dataset.py`.
6. Manually review 30 random training rows; findings go in `DATA_CARD.md`.
7. Run the §8.7 checks.
8. Run laya-evals as `python -m laya.evals_cli` (the launcher script breaks on paths with spaces).

**Deliverables:** `data/manifests/*`, `DATA_CARD.md`, `build_dataset.py`, preprocess tests.

**Gate:** all §8.7 checks pass, `laya-evals validate` passes on val/test files, tests green.

### Phase 2: Baselines (≈ 2 days)

Tasks: implement and run B0, B1, and B2 on val and test. Copy B3 numbers with a citation.

**Deliverables:** `eval/baselines.py`, results table v0 in `results/`.

**Gate:** the table is filled for B0–B3, and the B1 number is plausible against B3 (a sanity check
on our data pipeline).

### Phase 3: Fine-tune and calibrate (≈ 4–6 days)

Tasks:
1. Adapt the Laya Kaggle notebook. Replace its dataset loading with our `train.jsonl` while keeping
   the row schema, and set the HF destination repo. Training items are built by
   `training/make_items.py`, which the notebook embeds byte for byte (§9.3).
2. Train. Push to HF and record the revision SHA in `experiments.md`.
3. Run M1 on val on CPU fp32. Compute metrics and choose thresholds per §10.4. Commit the config.
4. Run M1 on test **once**. Produce all §10.3 plots. Fill in the §10.5 table.
5. Write the HF model card: intended use, data, metrics, limitations.

**Deliverables:** HF model at a pinned revision, `metrics_val.json`, `metrics_test.json` (with the other
aggregates in `results/phase3/val_R1b/` and `results/phase3/test_R1b/`), figures, filled results table,
thresholds in config.

**Gate:** the success criteria in §6.3 (1–3) are evaluated and reported honestly, whether met or not.

### Phase 4: GitHub Action (≈ 4–6 days)

Tasks:
1. Implement `config`, `event_loader`, `classifier`, `policy`, `github_client`, and `reporter`
   per §12.
2. Unit tests: policy (all branches), config validation, skip rules, mocked GitHub client.
3. Write `action.yml`, `constraints-linux.txt` and `ci.yml` (ruff + `pytest -m "not slow"` without weights), plus
   `tests/test_workflows.py` (§7.6). The `lock-linux` CI job's `pip freeze --all` replaces `constraints-linux.txt`
   as the Linux lock file.
4. Create a private sandbox repo and follow `sandbox/README.md`: 22 authored test fixtures created via the gh CLI
   (`sandbox/issues.json`, covering every class, template mismatches, ambiguous, empty, long-log, markdown,
   non-English, prompt-injection-like and emoji cases, and two skip cases). Run in **dry-run**, then **apply**
   (backfill, an idempotency re-run, and two `issues` events), and compare the runner logs with
   `sandbox/expected_local.json` (same label, |Δp| ≤ 1e-3, same decision; near-threshold differences reported as
   borderline). Bot-author and non-`opened` skips are covered by unit tests only (`results/phase4.md`).
5. Measure cold and warm Action wall time (NFR-2, NFR-3), the runner CPU and RAM (Q4), and the per-issue p50 / p95
   with `sandbox/latency_probe.py` (NFR-1).
6. Assert `max_len` from the checkpoint config equals 1024 in `tests/test_model_smoke.py` (§9.4).
7. Parity test (`tests/test_parity_val.py`, slow): for 30 val issues, the raw title/body from
   `data/raw/issues_train.csv` run through the production path (event dict → `preprocess` → `Classifier`) must
   give the label and probabilities (|Δp| ≤ 1e-4) of `results/phase3/val_R1b/predictions_val_M1_R1b.csv`,
   for `classify` and `classify_batch`.
8. Record the sandbox evidence and the gate checklist in `results/phase4.md`.
9. `warm-cache` input and `python -m laya_triage --warm` (§7.7, §12.4, R14), with unit tests (fake classifier) and
   the twice-weekly `schedule` trigger in the consumer and sandbox examples (§12.5); verify a warm-cache run in the
   sandbox saves both caches.

**Deliverables:** working Action tagged `v1.0.0-rc1`, CI green, sandbox screenshots, timing numbers.

**Gate:** every FR-1 to FR-8 is demonstrated on the sandbox repo, and NFR-5/6 are reviewed against
§7.5–7.6.

### Phase 5: Ship v1.0 (≈ 3 days)

Tasks: README (problem, demo GIF, install snippet, results table, limitations), tag `v1.0.0`, publish
the Action (optionally to the Marketplace), finalise the HF model card, and write the LinkedIn post.
The README is a short front door (hard cap 200 lines: banner, 30-second explanation, one worked example from the
sandbox fixtures, results with both misses stated, Path A Action / Path B model, deep-dive links and repository
map); the detail lives in `docs/DEEP_DIVE.md`, `docs/USING_THE_ACTION.md` and `docs/USING_THE_MODEL.md`, all
generated by `docs/make_readme.py`. The Python example is one tested file (`docs/snippets/classify_one.py`) shared
by the README, `USING_THE_MODEL.md` and the model card. Before the repository goes public, revise the banner wording (Q8).

**Deliverables:** a public repo at `v1.0.0`, an HF model, the post.

**Gate:** a stranger could install it from the README alone. Ask one friend to try it.

### Phase 6+ (after v1.0 only)

- **v1.1 `needs_info`:** a `noul` question with explicit `criteria` and a `labels` override (#156).
  Labels come from repos that use `info-needed`-style labels, plus about 200 hand-annotated items for
  eval. This gets its own mini-spec section here before any code.
- **v1.2 ONNX INT8:** use Laya's `scripts/export_onnx.py --quantize` and `ONNXAgent`. Compare
  accuracy, ECE, and latency against torch, and switch `backend` if it is better.
- **v2.0:** priority, duplicates, `documentation` class (would use NLBSE'23).

---

## 14. Architecture decision records

### ADR-1: Laya (encoder decision model) instead of an LLM API
- **Decision:** Laya, fine-tuned.
- **Why:** sub-second CPU inference, $0 per call, structured output by construction, and a
  calibrated probability that enables gating. It also runs inside the Action without external
  services.
- **Trade-off:** it requires fine-tuning and an evaluation pipeline, whereas an LLM works zero-shot.
  Accepted, because this work is the project's core value.

### ADR-2: Three classes for v1.0 (`bug`, `feature`, `question`)
- **Why:** these match NLBSE'24 exactly, so the project gets a published benchmark and an external
  baseline. The `documentation` class is deferred to v2.0.

### ADR-3: Run inside GitHub Actions on CPU, no hosted server
- **Why:** zero infrastructure, zero cost, trivial for users to install, no data leaves GitHub
  except the model download.
- **Trade-off:** model load time on every run (mitigated by caching), and runner CPU latency.
- **Fallback:** if NFR-2 or NFR-3 cannot be met even with ONNX, add an optional `server-url` input
  pointing at a self-hosted `laya-serve` (Docker or a free HF Space). This is not built unless needed.

### ADR-4: English checkpoint as the base
- **Why:** the target data is English, and Laya's own benchmarks show the English checkpoint
  outperforms the multilingual one on English tasks.

### ADR-5: Gate on `answer_confidence` with per-class thresholds fitted for 90% precision
- **Why:** `answer_confidence` is Laya's calibrated quantity. Per-class thresholds handle the fact
  that the `question` class is typically hardest. 90% precision is a defensible maintainer-facing
  bar and is configurable.

### ADR-6: Hard labels become one-hot `gold`, label smoothing as an experiment only
- **Why:** NLBSE provides hard labels, and the notebook expects distributions. Temperature fitting
  handles residual over-confidence, so smoothing is tried only if val ECE is poor.

### ADR-7: Default `mode: dry-run`
- **Why:** a safe default for anyone installing the Action. Users opt into writes explicitly.

### ADR-8: Laya (decision model) instead of a plain fine-tuned classifier
- **Decision:** Laya, fine-tuned, rather than an encoder with a fixed classification head.
- **Why:** labels and their descriptions are input text (§5.2), not positions in a fixed output
  layer; one model answers typed questions (`choice`, `noul`, `score`) with probabilities trained
  with proper scoring rules (§5.3), in one CPU forward pass and with no generated text. The
  probabilities are what the gate needs (ADR-5); the run fits the CPU-only Action (ADR-3); the
  roadmap's follow-up questions (v1.1 `needs_info`, a `noul` question) are meant to be a new
  question and new data rather than a second classifier; and the project set out to explore a
  decision model on a benchmarked task.
- **Evidence:** no same-protocol comparison against a plain fine-tuned encoder exists. The baselines
  are TF-IDF + logistic regression (B1), zero-shot Laya (B2) and the published NLBSE'24 results
  (B3), which use a different protocol (one classifier per repository). Test calibration is
  comparable to B1, not better; zero-shot Laya is weak, so fine-tuning is needed; there is no speed
  advantage. No claim is made that Laya is more accurate or better calibrated than a plain
  classifier. A same-protocol head-to-head is on the docs' Roadmap.
- **Trade-offs:** a newer ecosystem with a smaller community; tooling that relies on pinned versions
  (§9.3); known option-position bias (§5.5); a limited token budget when options are many. For a
  fixed, small label set with no follow-up questions, a plain classifier would be a reasonable choice.
- **Consequence:** exactly three classes (ADR-2). Adding a class is not a configuration setting
  (unknown config keys are rejected and `LABELS` is frozen, §16 rule 4): it needs new labelled data,
  a new frozen question, rebuilt splits, fine-tuning, a fresh temperature, refitted thresholds and
  evaluation under a new pre-declared protocol. A `choice` question is single-label; "needs more
  information" is a separate yes/no question (v1.1).

---

## 15. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Fine-tuned model doesn't beat TF-IDF+LR | Medium | Medium | Report honestly. The calibrated gating plus Action is still the deliverable. Try Protocol B data. |
| R2 | `question` class is weak (common in the literature) | High | Low | Per-class threshold. If needed, never auto-apply `question` and escalate instead. |
| R3 | CPU latency or load time too slow on the runner | Medium | Medium | Caching, then ONNX INT8 (v1.2), then the server fallback (ADR-3). On the runner, install the CPU-only torch wheel (PyTorch CPU index) and fetch only the minimal checkpoint files. |
| R4 | Notebook adaptation breaks (version drift, schema mismatch) | Medium | Medium | Pin `laya` and `transformers` versions from Phase 0. Smoke-train on 50 rows first. |
| R5 | Kaggle session timeout or OOM | Low | Low | The notebook writes a rolling `checkpoint_latest/` per epoch. Small dataset means short runs. |
| R6 | Data leakage between splits or with NLBSE'23 | Low | High | Automated ID-disjointness assertion and URL dedupe (§8.7). |
| R7 | Label noise in NLBSE data | Medium | Low | Manual 30-row audit. Discuss in limitations. |
| R8 | Script injection via issue text | Low | High | §7.6 rules, plus code review of the workflow. |
| R9 | Scope creep (v1.1/v2.0 features before v1.0 ships) | High | High | §4.1 release map is binding, and §16 rule 3. |
| R10 | NLBSE license restricts redistribution | Low | Medium | Do not commit the raw data. Commit only manifests and the build script. Verify in Phase 1. |
| R11 | Balanced data (100 per class per repo, random not temporal split) differs from real repos, so calibration and thresholds may not transfer | TBD | TBD | The sandbox run and a README limitation. |
| R12 | Small val set makes thresholds noisy | TBD | TBD | The 20% split, the bootstrap lower bound, and \|S\| ≥ 20 (§10.4). |
| R13 | Anonymous Hugging Face downloads are rate limited (runner IP ranges are shared) on cold runs: the first run, and after GitHub evicts a cache unused for 7 days | Medium | Low | Warm runs are fully offline (`HF_HUB_OFFLINE=1` on a verified cache hit); the model cache is saved only after a complete load; a failed download fails soft (exit 0, summary says to re-run). An optional `hf-token` input is v1.1. |
| R14 | Runs triggered by `issues` events cannot write caches (observed in the sandbox: "token has no writable scopes"), so caches are only saved by other events, and an issues-only workflow stays cold once its caches are evicted | High (observed) | Medium (cold runs: NFR-3 time, R13 exposure) | `warm-cache` input on `schedule` / `workflow_dispatch` runs (§7.7, §12.4); the §12.5 example warms twice a week. |

---

## 16. Rules for coding agents

These rules apply to Claude, Claude Code, and any other agent or human contributor.

1. **Read this document first.** When in doubt, this document is the authority. If something is
   ambiguous or missing, **stop and ask the owner**. Do not invent requirements.
2. **Work phase by phase.** Only do tasks belonging to the current phase (§13). Each change must
   state which phase and requirement ID (FR-x, NFR-x) it serves.
3. **No scope creep.** Do not implement v1.1/v1.2/v2.0 items while v1.0 is open.
4. **Frozen contracts.** `questions.py` (wording, label keys, option order), the `preprocess()`
   behaviour, and the data split manifests are frozen once their phase gate passes. Changing them
   requires (a) updating this document, (b) a changelog entry below, and (c) for question or
   preprocess changes, retraining and re-evaluating.
5. **One preprocessing path.** Never re-implement text cleaning. Import `preprocess()`.
6. **Never evaluate on test while tuning.** Test is run once per frozen model and config.
7. **Pin everything.** Model by revision SHA, dependencies by version, actions by tag or SHA.
8. **Security rules in §7.6 are mandatory.** Never interpolate issue content into shell commands,
   and never echo issue content in comments.
9. **Tests.** Pure functions (`preprocess`, `decide`, `should_skip`, config validation) need unit
   tests. CI must not download model weights. Model tests are marked `slow`.
10. **Verify incrementally.** After each task, run the relevant tests or commands and report the
    actual output before moving on. Prefer small diffs.
11. **Honest reporting.** Never hard-code or "round up" metrics. All numbers in README and model card
    come from files in `results/`.
12. **Laya gotchas** (§5.5): no boolean-word choice keys, gate on `answer_confidence` not
    `confidence`, the same `max_len` in training and inference, and CPU fp32 for threshold fitting.

### Changelog of this document

| Date | Version | Change | Author |
|---|---|---|---|
| 2026-10-01 | 1.0 | Initial specification | Prasanna + Claude |
| 2026-10-02 | 1.0.1 | Phase 0 clarifications. Env: Python 3.11 (Homebrew); `laya==0.3.23` is the only runtime dep, plus `dev` extra (pytest, ruff) pinned; lock file deferred to Linux/CI in Phase 4; macOS versions recorded in `results/phase0.md`. Skeleton: no placeholder `action.yml`, `config/triage.default.yml`, workflows, notebook, or metrics files until their phase; `.gitkeep` for empty dirs. Phase 0 tasks 2–5: record base checkpoint commit SHA, load at that revision, record download size; NLBSE'24 raw data may be downloaded to `data/raw/` (gitignored) and its license noted, but stop if no official train/test split exists; wording sample = 100 issues from official train only, stratified by label, seed 42, IDs saved to `results/phase0_sample_ids.txt` and forced into `train` in Phase 1; zero-shot input is raw title + body with only the §8.4 step-5 truncation; pick wording by macro-F1 (accuracy secondary), keep §9.2 wording unless another wins by ≥ 5 points or all are near chance; latency on local Apple-silicon CPU fp32 after warm-up, record p50/p95 and thread count, plus one run at `torch.set_num_threads(2)`; Q4 stays open until Phase 4. | Prasanna + Claude |
| 2026-10-02 | 1.0.2 | Phase 0 results. §9.2 wording frozen to W1' (instructions/criteria in `questions.py`): zero-shot on 100 stratified train issues it beat the v1.0 wording by +11.3 macro-F1 points at 512/192 (0.746 vs 0.633) and +10.7 at 1024/256 (0.729 vs 0.622), meeting the ≥ 5-point rule at both settings; §8.5 example updated to match. §9.1/§10.1 B2: root checkpoint's native budget is 512/192 (zero-shot), fine-tuning sets 1024/256 as the Laya notebook does. §17: Q2 resolved (macOS arm64: laya 0.3.23, torch 2.14.1, transformers 5.18.0; Linux pins in Phase 4); Q3 resolved (full snapshot 2.37 GB; `laya.load` needs a minimal 846 MB set, 846,201,702 bytes incl. the cached file listing). Details and caveats in `results/phase0.md`. | Prasanna + Claude |
| 2026-10-02 | 1.0.3 | Phase 1 pre-work (owner decisions: synthetic IDs, content-hash leakage checks, 20% val + bootstrap lower-bound thresholds, cross-repo headline metric). §8.3: drop the 4 official-train rows (559, 900, 901, 1114) whose (title, body) is in test, leaving 1,496; `val` = 20% (≈ 300) stratified by repo × label, seed 42, excluding the Phase 0 sample rows (forced into `train`); `train` ≈ 1,196; `test` = all 1,500 official test rows; manifests `<id>\t<content_sha1>` with IDs `nlbse24-<source>-<row:04d>` (source = train or test) and SHA-1 over raw `title + "\n" + body`; `dropped_train.txt` added. §8.5: example ID `nlbse24-train-0123`, IDs are synthetic. §8.1/§8.2: NLBSE'24 has no IDs/URLs and an empty upstream LICENSE (raw data never committed), fetched by `training/fetch_data.py` at upstream commit `2927bc67…` with SHA-256 checks; NLBSE'23/Protocol B dedupe by content hash, not URL; Protocol A allows pretrained models fine-tuned only on the provided train set, Protocol B is non-comparable. §8.7: no ID or content hash in more than one split; train + val + dropped = 1,500; Phase 0 sample rows in train. §6.3/§10.2/§10.5: headline = cross-repo macro-F1 (mean of 5 per-repo macro-F1 on test, as behind SetFit 0.8270); pooled macro-F1 and accuracy also reported; B3 = SetFit 0.8270. §10.4: thresholds from a bootstrap lower bound (2,000 resamples, seed 42, 5th percentile ≥ 0.90, \|S\| ≥ 20), point-estimate threshold recorded for comparison. §11/§13: `training/fetch_data.py`, `training/nlbse_data.py`; Phase 1 tasks add fetch_data.py, `tests/test_build_dataset.py`, 30-row manual review; gate requires tests green. §15: R3 adds CPU-only torch wheel and minimal checkpoint fetch; new R11 (balanced data may not transfer) and R12 (small val makes thresholds noisy). | Prasanna + Claude |
| 2026-10-02 | 1.0.4 | Phase 1 gate. §8.3: a row whose (title, body) is duplicated within official train is dropped in all copies (on the real data this coincides with the test-overlap rule: rows 559, 900, 901, 1114). §8.6: example row gains `"id": "nlbse24-test-0456"`; `laya-evals` ignores extra keys. §13 Phase 1: run laya-evals as `python -m laya.evals_cli` (the launcher script breaks on paths with spaces). | Prasanna + Claude |
| 2026-10-02 | 1.0.5 | Phase 2. §11: add `eval/metrics.py` (§10.2 metrics as pure functions, shared by `eval/baselines.py` and `eval/run_eval.py`) and `tests/test_eval_metrics.py`. | Prasanna + Claude |
| 2026-10-02 | 1.0.6 | Phase 3a (notebook adaptation). §9.3: adapted from the Laya notebook at `v0.3.23` = commit `d8a2e597…` (`ae3222b…` is the tag object); base pinned at `55cf4c4e…` with the minimal file set, anonymous download; items built by `training/make_items.py` at 1024/256 (upstream builds at the root's 512/192 while saving 1024/256); Kaggle version pins and the items-hash parity rule (`results/phase3/items_sha256.json`); 68 optimizer steps; deviations from upstream `train_ddp.py` (true `T_max` 68 vs upstream 64, torch/CUDA seeding with SEED + rank, calibration seed 20260922 kept, `model_name`). §8.5: JSON-string note resolved. §11: `training/make_items.py`, `results/phase3/`, `tests/test_make_items.py`, `tests/test_notebook_sync.py`. §12.3: repo `Prasanna85/laya-issue-triage`. §13 Phase 3 task 1 names `make_items.py`. §17: Q5 and Q7 resolved. | Prasanna + Claude |
| 2026-10-02 | 1.0.7 | §11: add `training/build_notebook.py`, which generates `training/finetune_kaggle.ipynb`; `tests/test_notebook_sync.py` checks that regenerating gives the committed notebook byte for byte. §7.3 and §9.3: HF repo placeholders replaced with `Prasanna85/laya-issue-triage`. | Prasanna + Claude |
| 2026-10-02 | 1.0.8 | Phase 3c-A (evaluation tooling). §10.4: candidate thresholds are the distinct val `answer_confidence` values among issues predicted ℓ; bootstrap seed `numpy.random.RandomState(42 + index of ℓ in LABELS)`, fresh per candidate. §11: `tests/test_gating.py`, `tests/test_plots.py`. | Prasanna + Claude |
| 2026-10-02 | 1.0.9 | Phase 3c-B Stop A2 (owner decisions after the R1 val run). §9.3 calibration row: the notebook fit stays in `run_meta.json`; the shipped choice temperature is refit on val by NLL with `laya.calibrate` (weights unchanged; the shipped revision differs from R1 only in `rl_agent_config.json` `"temperature"`); R1's 119-item fit (4.0188) over-softened (val ECE 0.129, val-NLL-optimal T ≈ 2.7). §10.4: the config uses τ_lb; τ_point, the full curve and B1 under the same rule are reported on test as pre-declared secondary views. | Prasanna + Claude |
| 2026-10-02 | 1.0.10 | Phase 3c-B Stop A3b (pre-freeze housekeeping). §6.3 criterion 3 is reported per label. §10.2: coverage@precision is reported as a global-τ curve and with the config's per-label τ. §11 and §13 Phase 3: M1 aggregates (incl. `metrics_val.json` / `metrics_test.json`) live in `results/phase3/val_R1b/` and `results/phase3/test_R1b/`. Phase 4 task 6: assert `max_len` 1024 from the checkpoint config in `tests/test_model_smoke.py` (§9.4). | Prasanna + Claude |
| 2026-10-02 | 1.0.11 | Phase 3c-B Stop C (single test run of R1b `76ece1fb…`). §6.3 criterion 3: k filled in, met for 0 of 3 labels on test (bug τ 0.6033: test precision 0.8747 at coverage 0.2607; feature and question τ 1.01). Results in `results/phase3.md` (Test results) and `results/phase3/test_R1b/`. | Prasanna + Claude |
| 2026-10-02 | 1.0.12 | Phase 3 gate. §10.5 filled with the test numbers (B0, B1, B2 both settings, published B3 rows with the derived-values footnote, M1); its coverage column is coverage at the val thresholds with the test precision. §6.3: criterion 3 text states the val and test outcome per label; criteria 1–4 annotated with results (MET / MET / NOT MET / PENDING, Phase 4). §11: `docs/MODEL_CARD.md` and `results/phase3/make_model_card.py`. | Prasanna + Claude |
| 2026-10-02 | 1.0.13 | §11: add `results/phase3/test_report.py`, `tests/test_test_report.py` and `tests/test_model_card.py` to the tree. | Prasanna + Claude |
| 2026-10-02 | 1.0.14 | Phase 4a (owner decisions D1–D6 and rulings). FR-5: skip pull requests, `issues` events whose action is not `opened` (only when an action is present), and issues already carrying the escalation label; case-insensitive comparisons. §7.5: retries are 3 retries (2 s, 4 s, 8 s), also for 429 and network errors; errors are recorded as exception type and frames, never the message; labels are checked before adding; misconfiguration exits 1 by design, NFR-5 concerns runtime faults. §12.2: `classify_batch` uses `predict_batch`, verified equal to `predict` (max \|Δp\| 0.0 on 33 issues). §12.3: `comment_on_escalate` defaults to false. §12.6: two templates, (b) for τ > 1.0. §11: new test files. Phase 4 task 7: parity test. | Prasanna + Claude |
| 2026-10-02 | 1.0.15 | Phase 4b (Action, CI, sandbox kit; owner rulings on the Stop A design). §7.5: action-layer fail-soft table. §7.7: model cache keyed from the merged config via `--print-model`, saved only after a successful load, offline on a verified hit; dependency venv cache keyed by the exact Python version and the pins hash, checked on restore (replaces `cache: pip`). §11: `constraints-linux.txt`, `sandbox/`, `results/phase4.md`, `tests/test_workflows.py`, `tests/test_sandbox.py`; dogfood `triage.yml` deferred. §12.4: inputs (token not required, mode default empty, backfill-count validated), Linux X64, env-only inputs, artifact. §12.5: SHA-pinned example with mode input and timeout. Phase 4 tasks 3-5: lock file, workflow test, sandbox runbook with authored fixtures created via the gh CLI, NFR-1 probe. §15: R13. | Prasanna + Claude |
| 2026-10-02 | 1.0.16 | §6.3 criterion 5: the sandbox run is 22 authored fixtures created via the gh CLI plus 2 issues opened by hand in the web UI, all as real `issues.opened` events. | Prasanna + Claude |
| 2026-10-02 | 1.0.17 | `constraints-linux.txt` is the Linux lock (36 pins) generated from CI run 37009763300 (commit f8267ca), replacing the macOS-derived pins; §7.7 states its source. CI and the sandbox workflow run on `ubuntu-24.04`; §12.5 keeps `ubuntu-latest` for consumers, with a re-test note for Ubuntu 26 (2026-10-19). | Prasanna + Claude |
| 2026-10-02 | 1.0.18 | The cold-install step of `action.yml` records which torch install path succeeded (`cpu-index-only` or `fallback: extra-index-url`) in the step summary and log. | Prasanna + Claude |
| 2026-10-02 | 1.0.19 | Phase 4 sandbox results. §6.3: criterion 4 NFR-1 not met on a 2-vCPU runner (p50 864 ms, p95 3.5 s), documented with the v1.2 ONNX plan; criterion 5 met (run ids). §7.7: issues-triggered runs restore but cannot save caches; `warm-cache` on schedule / dispatch; eviction after 7 days unused. §12.4: `warm-cache` input (`--warm`). §12.5: twice-weekly `schedule` trigger (`17 3 * * 1,4`), `warm-cache` dispatch input and expression; `github.event_name` allow-listed. §11: `sandbox/create_issue.py` (tested in `tests/test_sandbox.py`). §13 Phase 4 tasks 8-9. §15: R14. §17: Q4 resolved. | Prasanna + Claude |
| 2026-10-03 | 1.0.20 | Phase 4 gate passed (`results/phase4.md`): CI run 37058459160 green on `88ad0bf`; warm-cache verified in the sandbox (run 37058789375 saved both caches; issue run 37059668421 restored both); annotated tag `v1.0.0-rc1` on `88ad0bf`. Header notes the gate. §11: the tests tree lists `test_fetch_data.py`, `test_build_dataset.py`, `test_run_eval.py`, `test_posthoc_analysis.py` and `test_constraints.py`. | Prasanna + Claude |
| 2026-10-03 | 1.0.21 | Phase 5, part 1 (README and docs). `README.md` is generated by `docs/make_readme.py` from `docs/README.template.md` and the committed result files (`--check` fails if stale); the template holds no typed numbers (rule 11), enforced by `tests/test_readme.py`. §11: `docs/make_readme.py`, `docs/README.template.md`, `tests/test_readme.py`. `docs/MODEL_CARD.md` gains the GitHub link, through `results/phase3/make_model_card.py`. `action.yml` gains `branding` (icon, colour); no step changes. Appendix A now matches the real `CLAUDE.md` (current phase 5; the `reference/` note). Header notes Phase 5 in progress. | Prasanna + Claude |
| 2026-10-03 | 1.0.22 | README and model-card wording. B3 protocol text now follows `results/phase2.md`: one classifier per repository, each on only that repository's 300 official-train rows (SetFit template), SetFit input raw `title + " " + body`, RoBERTa and fastText setups not inspected; both documents read these facts from `phase2.md` through `b3_training` in `results/phase3/make_model_card.py`. README: the test evaluation was run once and later analyses are post-hoc; gating, S17, steering, validation-vs-test and label-noise wording sharpened. §9.3, §11 and §17 Q7: the Hugging Face model repo is public since 2026-10-02 and its model card is uploaded (was recorded as private / owner to upload). Header updated. | Prasanna + Claude |
| 2026-10-04 | 1.0.23 | Phase 5 README restructure (docs only). `README.md` becomes a short front door (banner, 30-second explanation with a roles table and a five-box Mermaid flow, worked example S01 with S02 and S05, results with both misses, Path A / Path B, deep-dive links and repository map); everything else moves to `docs/DEEP_DIVE.md`, `docs/USING_THE_ACTION.md` and `docs/USING_THE_MODEL.md`, generated by `docs/make_readme.py` from their templates (one `--check` for all four). `docs/snippets/classify_one.py` (imports the package; carries repo, revision, max_len and the bug threshold, checked against the config) is shown verbatim in the README, `USING_THE_MODEL.md` and the model card (new "With the laya-triage package" block in `results/phase3/make_model_card.py`; the standalone `laya.load` block stays); `docs/snippets/consumer-minimal.yml` uses the version tag, with the `git ls-remote` pinning note, accepted by `tests/test_workflows.py` rule (d) for our own action line in that file only. `docs/architecture_section.md` removed (its text is in the deep dive); banner renamed to `docs/banner_image.png`. Wording: the test evaluation was pre-declared and run once; later analyses reused its predictions and are labelled post-hoc. §11, §13 Phase 5, §17 Q8 (banner TODO). | Prasanna + Claude |
| 2026-10-04 | 1.0.24 | Phase 5 docs addition (docs only). §14: ADR-8, Laya instead of a plain fine-tuned classifier, with the explicit statement that no same-protocol comparison exists and the fixed-three-classes consequence. `docs/DEEP_DIVE.md` gains "Why Laya instead of a plain classifier" (after the Laya internals) and "Extending to more classes" (after Limitations); its Roadmap adds a same-protocol head-to-head against a plain fine-tuned encoder. `README.md` gains a short "Why Laya instead of a plain classifier?" note and an "Exactly three classes" limitation, with both deep-dive anchors in its link list. `docs/MODEL_CARD.md` gains a three-classes limitation through `results/phase3/make_model_card.py` (owner re-uploads the card). `tests/test_readme.py` checks the new headings, wording and the absence of accuracy or calibration superiority claims. | Prasanna + Claude |

---

## 17. Open questions and items to verify

| # | Item | Resolve in |
|---|---|---|
| Q1 | Exact NLBSE'24 split sizes, field names, and license | Phase 1 |
| Q2 | Exact latest `laya` PyPI version and compatible `transformers`/`torch` pins | **Resolved (Phase 0):** macOS arm64: laya 0.3.23, torch 2.14.1, transformers 5.18.0; Linux pins decided in Phase 4 |
| Q3 | Checkpoint download size and whether it fits comfortably in the Actions cache | **Resolved (Phase 0):** minimal file set 846,201,702 bytes incl. the cached file listing (`trees/<sha>.json`); full snapshot 2.37 GB |
| Q4 | Actual CPU spec and RAM of the GitHub-hosted runner used (public vs private repo runners differ) | **Resolved (Phase 4):** private-repo `ubuntu-24.04` runner: 2 logical CPUs, Intel Xeon Platinum 8573C, 7,937 MiB RAM, image ubuntu24 20260927.320.1, kernel 6.17.0-1022-azure, Python 3.11.16; torch used 1 thread. Public-repo runners not measured |
| Q5 | Whether the notebook's training script needs changes beyond replacing data loading (for example `max_len` from config) | **Resolved (Phase 3):** yes. Items are built at 1024/256 instead of the root's 512/192, plus a pinned base and versions, torch seeding, the true scheduler `T_max` and `model_name`; see §9.3 |
| Q6 | Whether NLBSE'23 data is needed (only if Protocol A results are weak) | After Phase 3 |
| Q8 | Banner wording (`docs/banner_image.png`): "HIGH ACCURACY", "REAL-WORLD READY" and "PRIORITISE" overstate v1.0 (precision target and NFR-1 missed; priority is v2.0), and the sample card shows `type: bug` and `triage: needs-human` on one issue, which the Action never does | **TODO (owner):** revise before the repository goes public |
| Q7 | Final project and repo name, and HF model repo name | **Resolved (Phase 3):** project `laya-triage` (GitHub `Prasanna-KS-85/laya-triage`), HF model repo `Prasanna85/laya-issue-triage` (public since 2026-10-02; model card uploaded) |

---

## 18. Glossary

| Term | Meaning |
|---|---|
| **State** | The input object Laya reads. Here: `{"title", "body"}` after preprocessing. |
| **Question / criteria** | A typed Laya question. `criteria` maps option keys to descriptions. |
| **`choice` / `noul` / `score`** | Laya's three decision types: pick one, yes/no probability, ordinal level. |
| **`answer_confidence`** | Probability of the chosen answer. Calibrated. Used for gating. |
| **Calibration** | Agreement between predicted probability and observed accuracy. |
| **ECE** | Expected Calibration Error: the binned average gap between confidence and accuracy. Lower is better. |
| **Temperature scaling** | Dividing logits by a fitted scalar T to fix over- or under-confidence. Accuracy is unchanged. |
| **RLCD** | Reinforcement Learning for Calibrated Decisions: Laya's training method using proper-scoring-rule rewards. |
| **Coverage** | Share of issues the bot labels automatically (the rest are escalated). |
| **Gating / escalation** | Acting only above a confidence threshold, otherwise handing off to a human. |
| **NLBSE** | Natural Language-Based Software Engineering workshop, which runs the issue-classification tool competitions. |
| **Protocol A / B** | Strict benchmark-comparable training vs. extra-data training (§8.2). |

---

## 19. References

- Laya repository: https://github.com/NandhaKishorM/laya (README, `docs/finetune.md`,
  `docs/evals.md`, `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`)
- Laya checkpoints: https://huggingface.co/convaiinnovations/laya
- Laya design write-up (architecture and RLCD):
  https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me
- NLBSE'24 issue report classification: https://github.com/nlbse2024/issue-report-classification
- NLBSE'23 tool competition report (1.4M-issue dataset, 4 labels)
- Kallis et al., *Ticket Tagger: Machine Learning Driven Issue Classification*, ICSME 2019
- GitHub Actions security hardening (script injection): GitHub Docs, "Security hardening for GitHub
  Actions"

---

## Appendix A: `CLAUDE.md` (place at repo root)

```markdown
# Agent instructions for laya-triage

The single source of truth for this project is `PROJECT_SPEC.md`. Read it before any change.

- Current phase: **Phase 5** (update this line when a phase gate passes).
- Only work on tasks from the current phase in PROJECT_SPEC.md §13.
- Follow the rules in PROJECT_SPEC.md §16 strictly. If anything is unclear, ask; do not guess.
- Verify every step: run tests or commands and show the output before continuing.
- Prefer small, reviewable diffs. Explain which requirement (FR-x / NFR-x) each change serves.

- `reference/` holds third-party material and is gitignored. Do not read, copy, or commit it unless asked.
```
