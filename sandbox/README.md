# Sandbox runbook (Phase 4, PROJECT_SPEC.md §13 Phase 4 tasks 4-5)

End-to-end test of the Action on a private sandbox repository with 22 authored test fixtures created via the
gh CLI (`sandbox/issues.json`). Run every command from the root of this repository (`laya-triage`) unless noted.
Nothing here prints issue text; the logs and comparisons hold issue numbers, fixture keys, labels and numbers.

Files:

| File | Purpose |
|---|---|
| `issues.json` | 22 authored fixtures (title, body, labels at creation, expected class and behaviour, rationale) |
| `make_expected.py` | runs the production path locally (offline) and writes `expected_local.json` |
| `expected_local.json` | local labels, probabilities and decisions per fixture (committed) |
| `create_issue.py` | creates fixtures as real issues with `gh issue create` (argument list, body on stdin, optional title suffix) |
| `compare.py` | `map` builds `issue_map.json`; `check` compares runner JSONL logs with `expected_local.json` |
| `latency_probe.py` | NFR-1 probe on the runner (p50 / p95 per issue, model preloaded) |
| `workflow.example.yml` | the sandbox workflow (copy into the sandbox repo) |

## 0. Variables and prerequisites

```bash
SB=Prasanna-KS-85/laya-triage-sandbox            # sandbox repository
SHA=$(git rev-parse HEAD)                         # laya-triage commit under test (pushed to GitHub)
gh auth status                                    # logged in as the owner
```

The laya-triage commit must be pushed. Because laya-triage is private, allow the sandbox to use it: laya-triage →
Settings → Actions → General → Access → "Accessible from repositories owned by the user".

## 1. Create the sandbox repository and the 4 labels

```bash
gh repo create "$SB" --private --add-readme
gh label create "type: bug"           --repo "$SB" --color d73a4a --description "laya-triage: bug"
gh label create "type: feature"       --repo "$SB" --color a2eeef --description "laya-triage: feature"
gh label create "type: question"      --repo "$SB" --color d876e3 --description "laya-triage: question"
gh label create "triage: needs-human" --repo "$SB" --color fbca04 --description "laya-triage: escalated"
```

## 2. Install the workflow (dry-run: the default config, no config file in the sandbox)

The workflow runs on `ubuntu-24.04` (pinned, the image the Linux lock in `constraints-linux.txt` was built on;
ubuntu-latest moves to Ubuntu 26 on 2026-10-19 and is to be re-tested after that).

```bash
WORK=$(mktemp -d); gh repo clone "$SB" "$WORK/sb"
mkdir -p "$WORK/sb/.github/workflows" "$WORK/sb/.github/laya-probe"
sed "s/0000000000000000000000000000000000000000/$SHA/" sandbox/workflow.example.yml \
  > "$WORK/sb/.github/workflows/triage.yml"
cp sandbox/latency_probe.py sandbox/issues.json "$WORK/sb/.github/laya-probe/"
cp config/triage.default.yml "$WORK/sb/.github/laya-probe/triage.default.yml"
git -C "$WORK/sb" add .github && git -C "$WORK/sb" commit -m "Add laya-triage sandbox workflow" && git -C "$WORK/sb" push
```

## 3. Cold run (NFR-3): empty caches, one issue

```bash
gh cache delete --all --repo "$SB" || true
# sandbox/create_issue.py runs gh with an argument list (title as one argument, body on stdin); no shell evaluation
python3 sandbox/create_issue.py --repo "$SB" S01
gh run watch --repo "$SB" "$(gh run list --repo "$SB" --workflow triage.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

## 4. Warm runs (NFR-2) and the other 21 fixtures

```bash
python3 sandbox/create_issue.py --repo "$SB" --sleep 5 $(for i in $(seq -w 2 22); do printf 'S%s ' "$i"; done)
```

The first run after S01 is the first warm run. Wait until all runs finish:
`gh run list --repo "$SB" --workflow triage.yml --limit 30`.

## 5. Map issues to fixtures, download the logs, compare (dry-run)

```bash
gh issue list --repo "$SB" --state all --limit 100 --json number,title > /tmp/gh_issues.json
.venv/bin/python sandbox/compare.py map /tmp/gh_issues.json
rm -rf /tmp/laya-logs && mkdir -p /tmp/laya-logs/dry-run
for id in $(gh run list --repo "$SB" --workflow triage.yml --event issues --limit 30 --json databaseId --jq '.[].databaseId'); do
  gh run download "$id" --repo "$SB" --dir "/tmp/laya-logs/dry-run/$id" || true
done
.venv/bin/python sandbox/compare.py check /tmp/laya-logs/dry-run/*/*/log.jsonl
```

Expected: 22 records; S20 and S21 skipped; the others match `expected_local.json` (same label, |dp| <= 1e-3, same
decision); no GitHub writes in dry-run (no labels or comments on the issues).

## 6. Apply via backfill (FR-6), then the idempotency re-run

```bash
gh workflow run triage.yml --repo "$SB" -f backfill-count=30 -f mode=apply
# wait for it, then:
ID=$(gh run list --repo "$SB" --workflow triage.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$ID" --repo "$SB" --dir /tmp/laya-logs/backfill-apply
.venv/bin/python sandbox/compare.py check /tmp/laya-logs/backfill-apply/*/log.jsonl

gh workflow run triage.yml --repo "$SB" -f backfill-count=30 -f mode=apply   # idempotency
ID=$(gh run list --repo "$SB" --workflow triage.yml --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$ID" --repo "$SB" --dir /tmp/laya-logs/backfill-again
.venv/bin/python sandbox/compare.py check --idempotent /tmp/laya-logs/backfill-again/*/log.jsonl
```

The first backfill applies `type: bug` to the fixtures whose bug confidence reaches 0.6033 and adds
`triage: needs-human` to the rest (S20 and S21 are skipped). The re-run must skip every issue ("already labeled" or
"already escalated"). To also check that pull requests are excluded, open one PR in the sandbox before the first
backfill and confirm it has no record.

## 7. Two issues through the `issues` event in apply mode, with comments (FR-2, FR-3)

```bash
printf 'behaviour:\n  mode: apply\n  comment_on_escalate: true\n' > "$WORK/sb/.github/laya-triage.yml"
git -C "$WORK/sb" add .github && git -C "$WORK/sb" commit -m "apply mode" && git -C "$WORK/sb" push
python3 sandbox/create_issue.py --repo "$SB" --title-suffix " [apply]" S19   # expected: apply type: bug
python3 sandbox/create_issue.py --repo "$SB" --title-suffix " [apply]" S05   # expected: escalate, template (b)
gh issue list --repo "$SB" --state all --limit 100 --json number,title > /tmp/gh_issues.json
.venv/bin/python sandbox/compare.py map /tmp/gh_issues.json
mkdir -p /tmp/laya-logs/apply-events
for id in $(gh run list --repo "$SB" --workflow triage.yml --event issues --limit 2 --json databaseId --jq '.[].databaseId'); do
  gh run download "$id" --repo "$SB" --dir "/tmp/laya-logs/apply-events/$id"
done
.venv/bin/python sandbox/compare.py check /tmp/laya-logs/apply-events/*/*/log.jsonl
```

Look at both issues on GitHub: one label each, and on the escalated one a single fixed-template comment that does
not quote the issue.

## 8. Timing (NFR-2, NFR-3) and the runner (Q4)

```bash
COLD=<run id of S01>; WARM=<run id of S02>
for id in $COLD $WARM; do
  gh run view "$id" --repo "$SB" --json jobs \
    --jq '.jobs[] | {startedAt, completedAt, steps: [.steps[] | {name, startedAt, completedAt}]}'
done
gh workflow run triage.yml --repo "$SB" -f backfill-count=0 -f latency-probe=true   # NFR-1 probe
```

Record in `results/phase4.md`: cold and warm job wall times (completedAt - startedAt) and the per-step split
(install, model cache, Triage), the runner CPU model / logical CPUs / RAM from the "Runner" summary, and the probe's
p50 / p95. Do not edit `expected_local.json` by hand; regenerate it with `make_expected.py` if the model changes.

## Notes

- If a restored model cache is ever incomplete (`model-cached=false` on a cache hit), the run downloads the missing
  files but cannot overwrite the cache entry: delete it with `gh cache delete <key> --repo "$SB"`.
- Bot-author skips and non-`opened` event actions cannot be produced from the sandbox; they are covered by unit tests
  only (tests/test_event_loader.py, tests/test_main.py), as recorded in `results/phase4.md`.
- Issues-triggered runs restore the caches but cannot save them ("cache write denied: token has no writable scopes",
  R14); dispatch and schedule runs save them. To warm the caches without triaging anything:
  `gh workflow run triage.yml --repo "$SB" -f warm-cache=true` (expected: the summary line "Cache warm-up: ...",
  both cache save steps run when the caches were cold, no issue changes).
- Anonymous Hugging Face downloads can be rate limited (R13). Cold runs are the exposed ones; warm runs are offline.
