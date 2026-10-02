# Laya Triage — Project Specification & System Design

> **Status:** v1.0.4 draft (source of truth) · **Last updated:** 2026-10-02 · **Owner:** Prasanna
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
| FR-5 | Skip issues that already carry any configured type label, and skip authors in `skip_authors` (for example bots). Record the skip reason in the summary. |
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
2. Post-calibration ECE ≤ 0.10 on the test split.
3. A threshold exists on validation data giving **≥ 90% precision** on auto-labelled issues. Report
   the coverage it achieves on the test split.
4. NFR-1 is met, or the measured latency is documented with the ONNX plan (v1.2).
5. The Action runs end-to-end on a sandbox repo for at least 20 real issues opened by hand.

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
   - push to HF: <user>/laya-issue-triage @ revision SHA
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
| Inference raises | Same as above. Record the traceback in the summary, not in an issue comment. |
| GitHub API 5xx or secondary rate limit | Retry with exponential backoff (3 attempts: 2 s, 4 s, 8 s), then fail soft. |
| GitHub API 403 (permissions) | Exit with a clear message naming the missing permission (`issues: write`). |
| Empty or very short body | Classify on the title alone. The state still has a `body` key with an empty string. |
| Config file invalid | Fail with a validation message listing the bad keys. Do not guess. |
| Label does not exist in repo | Create it with a default colour (opt-in `create_missing_labels: true`). Otherwise warn and skip. |

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

- Cache the Hugging Face cache directory (`~/.cache/huggingface`) with `actions/cache`.
- Cache key: `laya-${{ runner.os }}-${model_repo}-${model_revision}`. A model change invalidates the
  cache automatically.
- Also cache pip downloads (`setup-python` with `cache: pip`).

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

> The notebook's dataset rows store `state`, `questions`, and `gold` as JSON strings. The adapter
> cell must `json.dumps` them if it reuses the notebook's loading code unchanged. Verify in Phase 3.

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
| Calibration | Notebook fits one temperature per type on its held-out calibration slice. Inherited `temperature_by_options` is removed. |
| Output | Push to `<hf-user>/laya-issue-triage`. Record the commit SHA. |
| Seed | 42 |

Expected runtime is well under an hour for about 2–3k training rows. The notebook's 4–5 hour figure
is for about 30k questions.

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

Lower-bound thresholds are conservative, so coverage will be lower than with point-estimate
thresholds.

### 10.5 Results table template (fill in after Phase 3)

| System | Cross-repo macro-F1 (headline) | Pooled macro-F1 | Acc | F1 bug | F1 feature | F1 question | ECE | Coverage @ ≥90% prec | CPU p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| B0 majority | | | | | | | — | — | — |
| B1 TF-IDF+LR | | | | | | | | | |
| B2 Laya zero-shot | | | | | | | | | |
| B3 SetFit baseline 0.8270 (NLBSE'24 repo); RoBERTa/fastText results in the upstream `output/` folder may be cited after verification in Phase 2 | | | | | | | — | — | — |
| **M1 ours** | | | | | | | | | |

---

## 11. Repository structure

```
laya-triage/
├── PROJECT_SPEC.md               # this document (source of truth)
├── CLAUDE.md                     # short pointer for coding agents (see Appendix A)
├── README.md                     # user-facing: what it is, install, results, demo GIF
├── LICENSE                       # Apache-2.0 (matches Laya)
├── pyproject.toml                # package metadata + pinned deps
├── action.yml                    # composite GitHub Action definition
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
│   └── finetune_kaggle.ipynb     # adapted Laya notebook
├── eval/
│   ├── baselines.py              # B0, B1, B2
│   ├── run_eval.py               # M1 on val/test, CPU fp32, writes results/*.json
│   ├── gating.py                 # threshold selection + coverage–precision
│   └── plots.py                  # figures in §10.3
├── data/                         # raw/ and processed/ are gitignored
│   ├── manifests/                # committed split ID lists
│   └── DATA_CARD.md
├── results/
│   ├── phase0.md
│   ├── experiments.md
│   ├── metrics_val.json / metrics_test.json
│   └── figures/
├── tests/
│   ├── test_preprocess.py
│   ├── test_policy.py
│   ├── test_config.py
│   ├── test_event_loader.py
│   ├── test_github_client.py     # mocked HTTP
│   └── test_model_smoke.py       # @pytest.mark.slow, real model, 3 fixtures
└── .github/workflows/
    ├── ci.yml                    # ruff + pytest (no model weights)
    └── triage.yml                # dogfood: runs the action on this repo's issues
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
| `Classifier.classify_batch` | `(states, issue_numbers) -> list[TriageResult]` | Uses `predict_batch` |
| `decide` | `(result: TriageResult, cfg) -> Decision` | **Pure**, fully unit-tested |
| `should_skip` | `(issue: dict, cfg) -> str \| None` | Pure. Returns the skip reason or None. |
| `GitHubClient.apply` | `(issue_number, decision) -> None` | Network calls, with retry |
| `Reporter.write` | `(results, decisions) -> None` | Writes `$GITHUB_STEP_SUMMARY` and an artifact file |

### 12.3 Config file (`.github/laya-triage.yml` in the target repo; defaults in `config/triage.default.yml`)

```yaml
model:
  repo: "<hf-user>/laya-issue-triage"
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
  comment_on_escalate: true
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
| `github-token` | yes | `${{ github.token }}` | Token for label/comment API calls |
| `config-path` | no | `.github/laya-triage.yml` | Path to the config file |
| `mode` | no | (from config) | Overrides `behaviour.mode` |
| `backfill-count` | no | `0` | If > 0, classify the N most recent open issues (dispatch mode) |

### 12.5 Example consumer workflow

```yaml
name: Laya Triage
on:
  issues:
    types: [opened]
  workflow_dispatch:
    inputs:
      backfill-count: { description: "Open issues to classify", default: "20" }

permissions:
  issues: write
  contents: read

jobs:
  triage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4            # needed only to read the config file
      - uses: <your-gh-user>/laya-triage@v1
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          backfill-count: ${{ github.event.inputs.backfill-count || '0' }}
```

### 12.6 Escalation comment template (fixed text)

```
🤖 Laya Triage couldn't classify this issue confidently, so it's been marked for a maintainer.
Top guesses: `{label1}` ({p1:.0%}), `{label2}` ({p2:.0%}).
```

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
   the row schema, and set the HF destination repo.
2. Train. Push to HF and record the revision SHA in `experiments.md`.
3. Run M1 on val on CPU fp32. Compute metrics and choose thresholds per §10.4. Commit the config.
4. Run M1 on test **once**. Produce all §10.3 plots. Fill in the §10.5 table.
5. Write the HF model card: intended use, data, metrics, limitations.

**Deliverables:** HF model at a pinned revision, `metrics_val.json`, `metrics_test.json`, figures,
filled results table, thresholds in config.

**Gate:** the success criteria in §6.3 (1–3) are evaluated and reported honestly, whether met or not.

### Phase 4: GitHub Action (≈ 4–6 days)

Tasks:
1. Implement `config`, `event_loader`, `classifier`, `policy`, `github_client`, and `reporter`
   per §12.
2. Unit tests: policy (all branches), config validation, skip rules, mocked GitHub client.
3. Write `action.yml` and `ci.yml` (ruff + pytest without weights).
4. Create a sandbox repo. Run in **dry-run**, then **apply**, on at least 20 hand-written issues
   covering every class plus ambiguous cases.
5. Measure cold and warm Action wall time (NFR-2, NFR-3).

**Deliverables:** working Action tagged `v1.0.0-rc1`, CI green, sandbox screenshots, timing numbers.

**Gate:** every FR-1 to FR-8 is demonstrated on the sandbox repo, and NFR-5/6 are reviewed against
§7.5–7.6.

### Phase 5: Ship v1.0 (≈ 3 days)

Tasks: README (problem, demo GIF, install snippet, results table, limitations), tag `v1.0.0`, publish
the Action (optionally to the Marketplace), finalise the HF model card, and write the LinkedIn post.

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

---

## 17. Open questions and items to verify

| # | Item | Resolve in |
|---|---|---|
| Q1 | Exact NLBSE'24 split sizes, field names, and license | Phase 1 |
| Q2 | Exact latest `laya` PyPI version and compatible `transformers`/`torch` pins | **Resolved (Phase 0):** macOS arm64: laya 0.3.23, torch 2.14.1, transformers 5.18.0; Linux pins decided in Phase 4 |
| Q3 | Checkpoint download size and whether it fits comfortably in the Actions cache | **Resolved (Phase 0):** minimal file set 846,201,702 bytes incl. the cached file listing (`trees/<sha>.json`); full snapshot 2.37 GB |
| Q4 | Actual CPU spec and RAM of the GitHub-hosted runner used (public vs private repo runners differ) | Phase 0 / 4 |
| Q5 | Whether the notebook's training script needs changes beyond replacing data loading (for example `max_len` from config) | Phase 3 |
| Q6 | Whether NLBSE'23 data is needed (only if Protocol A results are weak) | After Phase 3 |
| Q7 | Final project and repo name, and HF model repo name | Phase 0 |

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

- Current phase: **Phase 0** (update this line when a phase gate passes).
- Only work on tasks from the current phase in PROJECT_SPEC.md §13.
- Follow the rules in PROJECT_SPEC.md §16 strictly. If anything is unclear, ask; do not guess.
- Verify every step: run tests or commands and show the output before continuing.
- Prefer small, reviewable diffs. Explain which requirement (FR-x / NFR-x) each change serves.
```
