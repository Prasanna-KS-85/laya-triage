"""Entry point of the Action: python -m laya_triage (PROJECT_SPEC.md §7.4, §7.5, §12.4, FR-1 to FR-8).

Arguments: --config-path (default .github/laya-triage.yml), --default-config (default: config/triage.default.yml
next to the package source), --mode (dry-run | apply; overrides behaviour.mode), --backfill-count (> 0 with a
workflow_dispatch event: the N newest open issues), --log-path (JSONL log, default laya-triage.jsonl),
--print-model (print `model-key=<owner>--<name>--<revision>` and `model-cached=true|false` for the merged config,
in $GITHUB_OUTPUT format, and exit; used by action.yml for the model cache key and offline mode),
--warm (cache warm-up, §7.7: validate the config, load the model at the pinned revision, classify one built-in
synthetic issue, write a fixed summary line and exit 0; no GitHub API calls and no token, the event is not read).
When $GITHUB_OUTPUT is set, a run appends `model_loaded=true|false` (the model loaded in this run; action.yml saves
the model cache only then).
Environment: GITHUB_EVENT_NAME, GITHUB_EVENT_PATH, GITHUB_REPOSITORY, GITHUB_STEP_SUMMARY, GITHUB_TOKEN,
GITHUB_API_URL (https only; default https://api.github.com), GITHUB_WORKSPACE.

Relative paths resolve against $GITHUB_WORKSPACE when it is set, else the working directory; the default config
fallback is located from this file, so neither depends on where the process starts.

Mode: backfill uses --mode if given, else dry-run, regardless of behaviour.mode; an issues event uses --mode,
else behaviour.mode. Dry-run makes no GitHub write calls (and builds no client unless backfill needs to list).

Exit codes: 1 for misconfiguration (invalid config, missing token / repository / non-https API URL in a run that
needs GitHub, or a 403 permission error); 0 otherwise, including model download / inference errors, GitHub
5xx or rate limits after retries, and per-issue errors (§7.5, NFR-5).
"""

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from laya_triage.config import ConfigError, load_config
from laya_triage.event_loader import EventError, issue_from_api, read_event, should_skip
from laya_triage.github_client import GitHubClient, GitHubError, GitHubPermissionError
from laya_triage.policy import Decision, decide, decide_error
from laya_triage.preprocess import preprocess
from laya_triage.reporter import Outcome, Reporter, error_info

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "triage.default.yml"
DEFAULT_API_URL = "https://api.github.com"
# --warm: a fixed synthetic issue (never user text) and fixed summary lines.
WARM_TITLE = "Synthetic warm-up issue: the app crashes when saving a file"
WARM_BODY = "Synthetic text used only to load the model for the cache warm-up. Steps: open, save, crash."
WARM_OK = "Cache warm-up: the model loaded and classified one synthetic issue; no issues were read or changed."


def _mode(value):
    if value in ("", None):
        return None  # an empty Action input means "not given"
    if value not in ("dry-run", "apply"):
        raise argparse.ArgumentTypeError("must be dry-run or apply")
    return value


def _count(value):
    n = int(value or 0)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m laya_triage", description="Label new GitHub issues with Laya.")
    ap.add_argument("--config-path", default=".github/laya-triage.yml")
    ap.add_argument("--default-config", default=None)
    ap.add_argument("--mode", type=_mode, default=None)
    ap.add_argument("--backfill-count", type=_count, default=0)
    ap.add_argument("--log-path", default="laya-triage.jsonl")
    once = ap.add_mutually_exclusive_group()
    once.add_argument("--print-model", action="store_true", help="print model-key / model-cached and exit")
    once.add_argument("--warm", action="store_true", help="load the model, classify one synthetic issue, exit")
    return ap.parse_args(argv)


def resolve(path, env):
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(env.get("GITHUB_WORKSPACE") or os.getcwd()) / p


def default_classifier(cfg):
    from laya_triage.classifier import Classifier

    return Classifier(cfg.model.repo, cfg.model.revision, cfg.model.max_len)


def set_output(env, name, value):
    """Append name=value to $GITHUB_OUTPUT when it is set (values here are fixed tokens, never user text)."""
    path = env.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{name}={value}\n")


def _fail(reporter, message):
    print(f"laya-triage: {message}", file=sys.stderr)
    reporter.notice(message)
    return 1


def main(argv=None, env=None, classifier_factory=None, client_factory=None) -> int:
    env = os.environ if env is None else env
    args = parse_args(argv)
    classifier_factory = classifier_factory or default_classifier
    client_factory = client_factory or GitHubClient
    reporter = Reporter(env.get("GITHUB_STEP_SUMMARY"), resolve(args.log_path, env))
    default_path = resolve(args.default_config, env) if args.default_config else DEFAULT_CONFIG
    try:
        cfg = load_config(default_path, resolve(args.config_path, env))
    except ConfigError as e:
        return _fail(reporter, str(e))
    if args.print_model:
        from laya_triage.classifier import model_cached, model_key

        print(f"model-key={model_key(cfg.model.repo, cfg.model.revision)}")
        print(f"model-cached={str(model_cached(cfg.model.repo, cfg.model.revision, env)).lower()}")
        return 0
    if args.warm:
        return _warm(cfg, env, classifier_factory, reporter)

    event = env.get("GITHUB_EVENT_NAME") or ""
    backfill = event == "workflow_dispatch" and args.backfill_count > 0
    mode = args.mode or ("dry-run" if backfill else cfg.behaviour.mode)
    notices, client = [], None
    if backfill or mode == "apply":
        try:
            client = client_factory(env.get("GITHUB_API_URL") or DEFAULT_API_URL, env.get("GITHUB_REPOSITORY") or "",
                                    env.get("GITHUB_TOKEN") or "")
        except ValueError as e:  # fixed text; never contains the token
            return _fail(reporter, str(e))

    try:
        return _run(cfg, args, env, event, backfill, mode, client, classifier_factory, reporter, notices)
    except Exception as e:  # noqa: BLE001 - last line of defence: fail soft (NFR-5), type name only
        print(f"laya-triage: internal error ({type(e).__name__})", file=sys.stderr)
        reporter.notice(f"internal error ({type(e).__name__})")
        return 0


def _warm(cfg, env, classifier_factory, reporter):
    """--warm: load the model and run one classify so action.yml saves the caches; exit 0 on any runtime fault."""
    try:
        classifier = classifier_factory(cfg)
    except Exception as e:  # noqa: BLE001 - soft failure (§7.5); type name only
        set_output(env, "model_loaded", "false")
        reporter.notice(f"Cache warm-up: model load failed ({type(e).__name__}); nothing was cached (fail-soft).")
        print(f"laya-triage: warm-up: model load failed ({type(e).__name__})", file=sys.stderr)
        return 0
    set_output(env, "model_loaded", "true")
    try:
        classifier.classify(preprocess(WARM_TITLE, WARM_BODY, cfg.preprocess), 0)
    except Exception as e:  # noqa: BLE001
        reporter.notice(f"Cache warm-up: the model loaded but the synthetic classify failed ({type(e).__name__}).")
        print(f"laya-triage: warm-up: classify failed ({type(e).__name__})", file=sys.stderr)
        return 0
    reporter.notice(WARM_OK)
    print(f"laya-triage: {WARM_OK}")
    return 0


def _run(cfg, args, env, event, backfill, mode, client, classifier_factory, reporter, notices):
    issues = []
    try:
        if backfill:
            for it in client.list_open_issues(args.backfill_count):
                try:
                    issues.append(issue_from_api(it))
                except EventError as e:
                    notices.append(f"skipped a malformed issue from the API: {e}")
        else:
            issues = read_event(event, env.get("GITHUB_EVENT_PATH"))
            if event != "issues":
                notices.append(f"event '{event or 'unknown'}' has nothing to triage"
                               + (" (backfill-count is 0)" if event == "workflow_dispatch" else ""))
    except GitHubPermissionError as e:
        return _fail(reporter, str(e))
    except (GitHubError, EventError) as e:
        notices.append(f"could not read issues: {e}")

    # rows: [issue number, TriageResult | None, Decision | None, Outcome]
    rows, todo = [], []
    for issue in issues:
        reason = should_skip(issue, cfg)
        rows.append([issue.number, None, Decision("skip", (), None, reason) if reason else None, Outcome()])
        if not reason:
            todo.append((len(rows) - 1, issue))

    classifier, load_error, batch = None, None, {}
    if todo:
        try:
            classifier = classifier_factory(cfg)
        except Exception as e:  # noqa: BLE001 - soft failure (§7.5)
            load_error = e
    set_output(env, "model_loaded", str(classifier is not None).lower())
    states = {i: preprocess(issue.title, issue.body, cfg.preprocess) for i, issue in todo}
    if classifier is not None and backfill and len(todo) > 1:
        try:  # §9.4: predict_batch in backfill mode; per-issue fallback isolates a failing issue
            batch = dict(zip([i for i, _ in todo],
                             classifier.classify_batch([states[i] for i, _ in todo], [s.number for _, s in todo])))
        except Exception:  # noqa: BLE001
            batch = {}
    for i, issue in todo:
        if load_error is not None:
            rows[i][2], rows[i][3] = decide_error(cfg, "model load failed"), Outcome(error=error_info(load_error))
            continue
        try:
            result = batch.get(i) or classifier.classify(states[i], issue.number)
            rows[i][1], rows[i][2] = result, decide(result, cfg)
        except Exception as e:  # noqa: BLE001 - one failing issue never stops the batch
            rows[i][2], rows[i][3] = decide_error(cfg, "inference failed"), Outcome(error=error_info(e))

    exit_code = 0
    if mode == "apply":
        for row in rows:
            n, _, d, o = row
            if d.action == "skip":
                continue
            try:
                done = client.apply(n, d, cfg.behaviour.create_missing_labels)
                row[3] = replace(o, written=bool(done["labels_added"] or done["comment_posted"]),
                                 labels_added=tuple(done["labels_added"]), comment_posted=done["comment_posted"],
                                 warnings=o.warnings + tuple(done["warnings"]))
            except GitHubPermissionError as e:
                print(f"laya-triage: {e}", file=sys.stderr)
                notices.append(str(e))
                row[3] = replace(o, error=error_info(e))
                exit_code = 1
                break
            except GitHubError as e:
                row[3] = replace(o, error=error_info(e), warnings=o.warnings + (str(e),))

    recs = reporter.write([r[1] for r in rows], [r[2] for r in rows], issue_numbers=[r[0] for r in rows],
                          outcomes=[r[3] for r in rows], mode=mode, event=event or "unknown",
                          model_repo=cfg.model.repo, model_revision=cfg.model.revision, thresholds=cfg.thresholds,
                          notices=notices)
    for r in recs:
        print(f"laya-triage: #{r['issue_number']} {r['action']}: {r['reason']}")
    for n in notices:
        print(f"laya-triage: {n}")
    print(f"laya-triage: {len(recs)} issue(s), mode {mode}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
