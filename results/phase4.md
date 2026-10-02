# Phase 4

## Phase 4a: modules and parity (2026-10-02)

Modules per PROJECT_SPEC.md §12 (`config`, `event_loader`, `classifier`, `policy`, `github_client`, `reporter`,
`__main__`), unit tests without weights or network, and slow tests on the pinned model (local cache, offline):

- `tests/test_parity_val.py`: 30 val issues (2 per repo × label cell), raw title/body through the production path
  (event → `preprocess` → `Classifier`) vs `results/phase3/val_R1b/predictions_val_M1_R1b.csv`: labels 30/30, max
  |Δp| 0.0, for both `classify` and `classify_batch`.
- `tests/test_model_smoke.py`: 3 synthetic issues; checkpoint `max_len` 1024; `classify_batch` = `classify` (|Δp| 0.0).

## Phase 4b: Action, CI and sandbox (in progress)

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
- **Labels in the `opened` event are unverified.** S20 and S21 assume that labels added when the issue is created
  (`gh issue create --label ...`) are present in the `issues.opened` payload, so the skip rules fire. This is
  unverified until the first sandbox run. Fallback if they are absent: create the issue without labels, apply the
  label afterwards, and check the skip with a backfill run, which lists the labels from the REST API.
- **v1.1 hardening idea:** `--no-build-isolation` on the full install.

### Sandbox results

Pending (owner runs `sandbox/README.md`): dry-run vs local comparison, backfill apply, idempotency re-run, apply-mode
events, cold and warm wall times (NFR-2, NFR-3), runner CPU / RAM (Q4), NFR-1 probe.
