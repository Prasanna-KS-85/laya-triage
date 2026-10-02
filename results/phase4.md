# Phase 4

## Phase 4a: modules and parity (2026-10-02)

Modules per PROJECT_SPEC.md §12 (`config`, `event_loader`, `classifier`, `policy`, `github_client`, `reporter`,
`__main__`), unit tests without weights or network, and slow tests on the pinned model (local cache, offline):

- `tests/test_parity_val.py`: 30 val issues (2 per repo × label cell), raw title/body through the production path
  (event → `preprocess` → `Classifier`) vs `results/phase3/val_R1b/predictions_val_M1_R1b.csv`: labels 30/30, max
  |Δp| 0.0, for both `classify` and `classify_batch`.
- `tests/test_model_smoke.py`: 3 synthetic issues; checkpoint `max_len` 1024; `classify_batch` = `classify` (|Δp| 0.0).

## Phase 4b: Action, CI and sandbox

`action.yml` (composite, Linux X64), `constraints-linux.txt`, `.github/workflows/ci.yml`, `tests/test_workflows.py`,
and the sandbox kit in `sandbox/` (runbook: `sandbox/README.md`). The sandbox fixtures are authored test fixtures
created via the gh CLI; `sandbox/expected_local.json` holds the local production-path results for them (Apple M4 Pro,
CPU fp32; `classify_batch` vs `classify` max |Δp| 0.0).

### Linux lock (2026-10-02)

`constraints-linux.txt` was rebuilt from the `requirements-linux-lock` artifact of CI run **37009763300** (ci.yml
`lock-linux` job, commit `f8267ca`): `pip freeze --all` on **Ubuntu 24.04.5 LTS x86_64, Python 3.11.16** (the runner
image version is not recorded in the artifact). 36 packages, written as PEP 503-normalised `name==version` lines in
alphabetical order, with `torch==2.14.1` (the freeze lists `2.14.1+cpu`); `tests/test_constraints.py` checks it.

- Checks on the freeze: no `nvidia-*`, no `triton`, no `+cu` versions; torch is `2.14.1+cpu`; laya-triage itself is
  not listed.
- Against the previous macOS-derived pins (11 packages): no version changes; only torch's local label (`2.14.1` ->
  `2.14.1+cpu`, written as `2.14.1`). 25 transitive packages are pinned for the first time.
- For information, against the local macOS evaluation venv: filelock 4.0.8 (macOS) vs 3.32.3 (Linux), fsspec
  2026.9.0 vs 2026.7.0.

CI and the sandbox workflow now run on `ubuntu-24.04`.

- [ ] TODO: The Action is tested on ubuntu-24.04; ubuntu-latest moves to Ubuntu 26 on 2026-10-19 and is to be
  re-tested after that.

### FR-5 coverage that the sandbox cannot show

Two skip rules cannot be produced on the sandbox repository and are covered by unit tests only:

- **Bot authors** (`behaviour.skip_authors`): the fixtures are created by the owner's account.
  Tests: `tests/test_event_loader.py::test_skip_reasons`, `tests/test_main.py::test_each_skip_reason`.
- **Event actions other than `opened`**: the sandbox workflow triggers only on `issues: opened`.
  Tests: `tests/test_event_loader.py::test_event_action_other_than_opened_is_skipped`,
  `tests/test_main.py::test_each_skip_reason`.

### Known limitations observed

- **No language guard.** Non-English input gets a normal prediction: fixture S17 (German) was auto-labelled `bug`
  in the local run (`sandbox/expected_local.json`: bug 0.6172 ≥ threshold 0.6033 → apply), although the model was
  trained and evaluated on English issues only (§4.2). Idea for v1.1: detect the language and escalate non-English
  issues instead of classifying them.
- **Labels in the `opened` event: verified.** S20 and S21 assumed that labels added when the issue is created
  (`gh issue create --label ...`) are present in the `issues.opened` payload. The sandbox run confirmed it: S20 was
  skipped "already labeled" and S21 "already escalated" (Sandbox results).
- **v1.1 hardening idea:** `--no-build-isolation` on the full install.

### Sandbox results (2026-10-02)

Sandbox `Prasanna-KS-85/laya-triage-sandbox` (private), workflow `.github/workflows/triage.yml` from
`sandbox/workflow.example.yml`, action commit `13633af2f8728bc3aa9642fee0464ecfcf3b2f16`, runner `ubuntu-24.04`.
Runbook: `sandbox/README.md`. Every number below is from the runs listed; nothing is rounded up.

**Runner (Q4).** 2 logical CPUs, Intel Xeon Platinum 8573C, 7,937 MiB RAM, image ubuntu24 20260927.320.1, kernel
6.17.0-1022-azure, Python 3.11.16. Torch install path on the cold install: `cpu-index-only`.

**Runs.**

| Run | Trigger | Caches | Wall time | Notes |
|---|---|---|---|---|
| 37024725031 | dispatch, backfill 0 | deps cold | 1m07s | model not loaded (nothing to triage) |
| 37025861693 | issue S01 | both cold | 1m26s | NFR-3; both cache saves refused ("cache write denied: token has no writable scopes"); run succeeded |
| 37026455422 | dispatch, backfill 1 + latency probe | | 2m44s | no mode input: dry-run (FR-6 default); saved both caches |
| 37028030094 | issue S02 | both restored | 44s | NFR-2 |
| 22 issue runs (S01-S22) | issues opened | | | all success |
| 37030197012 | dispatch, backfill 30, apply | | 2m04s | FR-2, FR-3, FR-6 |
| 37030556119 | dispatch, backfill 30, apply (re-run) | | 45s | idempotency |
| 37031205724 | issue #24 (web UI) | | | apply mode, `comment_on_escalate: true`: apply `type: bug` at 0.7836, no comment |
| 37031265068 | issue #25 (web UI) | | | escalate, prediction `question` 0.9597, comment template (b) |
| 37058789375 | dispatch, `warm-cache=true` | both cold (all caches deleted first) | 1m15s | action `88ad0bf`; both caches not found, then both saved (HF 739.96 MiB, venv 256.21 MiB); no issue processed |
| 37059668421 | issue #26 (web UI) | both restored | 56s | action `88ad0bf` (sandbox `2ae7611`); created 20:17:57Z, started 20:18:00Z, completed 20:18:56Z; success; no cache-save attempt in the inspected log; #26 labelled `triage: needs-human` |

The comment on #25 (fixed template (b), §12.6; no issue content): "🤖 Laya Triage suggests `question` (96%), but
automatic labelling is disabled for this issue type in this repository's configuration, so it's been marked for a
maintainer. Second guess: `feature` (2%)."

**compare.py** (runner JSONL logs vs `sandbox/expected_local.json`, tolerance 1e-3):

- dry-run (22 issue runs): 22/22 ok; max |Δp| 1e-4 (S19), 0.0 for every other fixture; 0 borderline.
- backfill apply: 22/22 ok.
- idempotent re-run: 22/22 ok (every record a skip, "already labeled" or "already escalated").
- FR-4 check after the dry-run: no labels beyond those set at creation, 0 comments.
- PR #23 (opened before the backfill): no record, no labels.
- Labels at creation are present in the `issues.opened` payload (the assumption in "Known limitations" is now
  verified): S20 skipped "already labeled", S21 skipped "already escalated".

#### Functional requirements

| ID | Demonstrated on the sandbox | Evidence |
|---|---|---|
| FR-1 | yes | 22 `issues: opened` runs, each issue classified into exactly one of bug / feature / question (S20, S21 skipped by FR-5); dry-run 22/22 match the local production path |
| FR-2 | yes | backfill apply 37030197012 added `type: bug` where bug confidence ≥ 0.6033 (22/22 ok); event #24 (37031205724) applied `type: bug` at 0.7836 |
| FR-3 | yes, template (b) only | backfill apply added `triage: needs-human` to the rest; #25 (37031265068) escalated with one template (b) comment, top-2 labels and numbers only. Template (a) is covered by unit tests only (below) |
| FR-4 | yes | dry-run issue runs: no labels beyond those at creation, 0 comments; decision table in the job summary |
| FR-5 | yes, partly | PR #23 has no record; S20 "already labeled", S21 "already escalated"; idempotent re-run 22/22 skips. Bot authors and non-`opened` actions: unit tests only (below) |
| FR-6 | yes | dispatch backfill 0 (37024725031, nothing to triage), 1 (37026455422), 30 in apply (37030197012) and the re-run (37030556119). Dry-run default: 37026455422 (workflow_dispatch, backfill-count 1, no mode input) showed the "dry-run: no changes were made" banner with #1 decided apply, and #1 carried no label afterwards (only #20 and #21 were labelled before the apply backfill 37030197012); also unit-tested (`tests/test_main.py::test_backfill_defaults_to_dry_run_even_if_config_says_apply`) |
| FR-7 | yes | every run's JSONL log was downloaded as the `laya-triage-log-<run_id>-<run_attempt>` artifact and read by `compare.py`; decision table in each job summary |
| FR-8 | yes | dry-run phase used the defaults (no config file in the sandbox); `.github/laya-triage.yml` with `mode: apply` and `comment_on_escalate: true` took effect for #24 / #25; the default thresholds (question τ 1.01) selected template (b) |

#### Non-functional requirements

| ID | Target | Result |
|---|---|---|
| NFR-1 | ≤ 1 s p95 per issue, model loaded | **NOT MET** on this runner: p50 863.8 ms, p95 3501.7 ms (probe below). Documented with the v1.2 ONNX plan (§6.3 criterion 4) |
| NFR-2 | warm wall time ≤ 2 min | **MET**: 44 s (37028030094, issue S02, both caches restored) |
| NFR-3 | cold wall time ≤ 5 min | **MET**: 1m26s (37025861693, issue S01, both caches cold). A cold dispatch run that also saved both caches and ran the probe took 2m44s (37026455422) |
| NFR-5 | fails soft | Reviewed against §7.5 (gate checklist below). Observed on the sandbox: both cache saves refused on 37025861693, the run still succeeded; 22/22 issue runs and all dispatch runs succeeded |
| NFR-6 | least privilege, no injection | Reviewed against §7.6 (gate checklist below). Observed: the #25 comment is fixed text plus labels and numbers; prompt-injection-like fixture S18 changed the predicted label only (observations) |

**Latency probe (NFR-1, run 37026455422).** n 60, p50 863.8 ms, p95 3501.7 ms, mean 1210.0 ms, max 5487.5 ms, model
load 2.28 s, `torch.get_num_threads()` 1 (2 logical CPUs, likely one physical core with SMT). Caveats:

- The fixture length mix differs from real issues: the 22 fixtures include a long-log fixture (S15, > 6,000
  characters before truncation) and several near-empty ones, so the tail is set by a few long inputs and need not
  match the length distribution of real issues.
- One run on one runner instance; runner hardware and neighbours vary between runs.
- Torch ran single-threaded; the local 2-thread figure (p95 690 ms on an Apple M4 Pro) is not comparable.
- Idea: a length-matched probe (synthetic texts whose token lengths follow the val-split length distribution, so no
  issue text is used) would give a p95 closer to production. Not done in Phase 4.

**Cache finding.** Model cache (Hugging Face) 739.96 MiB, dependency venv 256.18 MiB. Issues-triggered runs restore
both caches but cannot save them (`cache write denied: token has no writable scopes`; `continue-on-error` keeps the
run green); dispatch runs save them. A repository whose workflow only runs on issues would therefore stay cold once
GitHub evicts the caches (unused for 7 days). Mitigation: the `warm-cache` input (`python -m laya_triage --warm`) on
a twice-weekly `schedule` (`17 3 * * 1,4`: margin against the 7-day eviction and late scheduled starts) and on
demand via `workflow_dispatch` (PROJECT_SPEC.md §7.7, §12.4, §12.5, R14). Unit-tested
with a fake classifier and verified in the sandbox (runs 37058789375 and 37059668421, below).

**Covered by unit tests only** (cannot be produced on the sandbox with the fixtures and config used):

- Bot authors (`behaviour.skip_authors`) and event actions other than `opened`: see "FR-5 coverage that the sandbox
  cannot show" above.
- Comment template (a) (reachable τ, confidence below it): comments were enabled only for #24 / #25, and neither
  was a below-threshold bug. Tests: `tests/test_policy.py::test_just_below_threshold_escalates_with_template_a`,
  `tests/test_main.py::test_apply_mode_from_user_config_and_escalation_comment_opt_in`.

**Observations** (fixture texts are our own):

- S17 (German) was auto-labelled `bug` at 0.6172 (≥ 0.6033), as in the local run: no language guard (see "Known
  limitations").
- S18 (text that asks to be labelled a feature) was predicted `feature` at 0.78: issue text can steer the label. The
  impact is limited: the label is chosen from the three classes, no text is executed or echoed, and `feature` is never
  auto-applied with the default thresholds (τ 1.01), so the issue is escalated.
- S11 and S12 give identical results: with an absent and an empty body, both are title-only after preprocessing.

**Tooling note.** `ruff check` is clean, but `ruff format --check .` would reformat 64 of 77 files, including the
frozen `src/laya_triage/preprocess.py`. Handle in Phase 5, excluding the frozen files (`questions.py`,
`preprocess.py`).

## Phase 4 gate (PROJECT_SPEC.md §13 Phase 4)

Gate: every FR-1 to FR-8 is demonstrated on the sandbox repo, and NFR-5/6 are reviewed against §7.5–7.6.

- [x] FR-1 to FR-8 demonstrated on the sandbox (table above), with the documented unit-test-only parts: FR-5 bot
  authors and non-`opened` actions (accepted in §13 Phase 4 task 4), FR-3 template (a).
- [x] NFR-5 reviewed against §7.5:
  - runtime faults exit 0 with a summary line: model load failure, inference error (type and frames only), GitHub
    5xx / 429 / network after retries (unit tests in `tests/test_main.py`, `tests/test_github_client.py`);
  - action layer: install failure, venv check failure, timeouts and unexpected exit codes end in a warning and exit 0
    (`action.yml`); cache restore and save steps are `continue-on-error`, shown live by the refused saves on
    37025861693;
  - misconfiguration fails loudly by design (exit 1 / 2): invalid config, missing permission, invalid inputs
    (including `warm-cache` other than true / false).
- [x] NFR-6 reviewed against §7.6:
  - permissions `issues: write`, `contents: read` only, in every workflow and both examples (`tests/test_workflows.py`
    rule (f));
  - no issue / comment / PR content context and no `${{ }}` inside any `run:` script (rules (a), (b)); every
    expression allow-listed (rule (c)); `github.event_name` added as a fixed event-type string;
  - comments are fixed templates with labels and numbers only (#25);
  - only `GITHUB_TOKEN`, read from the environment; `--warm` needs no token;
  - model pinned by revision SHA, dependencies by the Linux lock, actions by commit SHA (rule (d)); no shell or
    dynamic execution in `src/` (`tests/test_no_shell_execution.py`);
  - `sandbox/create_issue.py` calls gh with an argument list (body on stdin).
- [x] Timings: NFR-2 44 s (met), NFR-3 1m26s (met), NFR-1 p95 3.5 s (not met, documented with the v1.2 ONNX plan);
  Q4 recorded.
- [x] warm-cache verified in the sandbox (sandbox commit `2ae76113b88002b4b4113dd50c196ae59aa652b0`, pinned to
  action commit `88ad0bf3eda4944b5cb243b3da599cc66f404652`):
  - run 37058789375 (workflow_dispatch, `warm-cache=true`, all caches deleted first): both caches not found, then
    both saved (HF 739.96 MiB, venv 256.21 MiB); no issue processed; 1m15s;
  - issue-triggered run 37059668421 (#26, opened by hand; head SHA `2ae7611`, success, 56 s): restored
    `laya-venv-Linux-X64-py3.11.16-0c25539d0c47fb71` and
    `laya-hf-Linux-Prasanna85--laya-issue-triage--76ece1fb0eb8b32bd5d8c509293c1692a2534805`; the inspected log
    excerpt showed no cache-save attempt or failure; #26 received `triage: needs-human`;
  - not yet observed: a run started by the `schedule` trigger.
- [x] CI green: run 37058459160 passed on commit `88ad0bf3eda4944b5cb243b3da599cc66f404652`.
- [x] Tag: annotated tag `v1.0.0-rc1` on commit `88ad0bf3eda4944b5cb243b3da599cc66f404652`, pushed.

**Phase 4 gate: passed.**
