"""python -m laya_triage end to end with a fake classifier and a fake GitHub client: no model, no network
(PROJECT_SPEC.md FR-1 to FR-8, §7.5, §12.4; owner decisions D2-D5 and the rulings on exit codes, error text and
working-directory independence). Synthetic issues only."""
import json
import shutil
from pathlib import Path

import pytest

from laya_triage import __main__ as cli
from laya_triage.classifier import ClassifierError, TriageResult
from laya_triage.github_client import GitHubError, GitHubPermissionError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "triage.default.yml"
REV = "76ece1fb0eb8b32bd5d8c509293c1692a2534805"
SENTINEL = "SENTINEL-ISSUE-TEXT-5d1e"
BUG = {"bug": 0.9, "feature": 0.07, "question": 0.03}
LOW_BUG = {"bug": 0.5, "feature": 0.3, "question": 0.2}
FEATURE = {"bug": 0.05, "feature": 0.9, "question": 0.05}


class FakeClassifier:
    """Probabilities per issue number; numbers in `fail` raise with a message carrying the sentinel."""

    def __init__(self, probs, fail=(), batch_fails=False):
        self.probs, self.fail, self.batch_fails = probs, set(fail), batch_fails
        self.calls, self.batch_calls = [], []

    def _one(self, state, n):
        assert set(state) == {"title", "body"}
        if n in self.fail:
            raise RuntimeError(f"inference broke on {SENTINEL}")
        p = self.probs[n]
        label = max(p, key=p.get)
        return TriageResult(n, label, p, p[label], 10.0, "Prasanna85/laya-issue-triage", REV)

    def classify(self, state, n):
        self.calls.append(n)
        return self._one(state, n)

    def classify_batch(self, states, numbers):
        self.batch_calls.append(list(numbers))
        if self.batch_fails or self.fail & set(numbers):
            raise ClassifierError("inference failed", RuntimeError(SENTINEL))
        return [self._one(s, n) for s, n in zip(states, numbers)]


class FakeClient:
    def __init__(self, api_url, repo, token, issues=(), apply_errors=None):
        self.args = (api_url, repo, token)
        self.issues, self.apply_errors = list(issues), dict(apply_errors or {})
        self.applied, self.listed = [], []

    def list_open_issues(self, n):
        self.listed.append(n)
        return self.issues[:n]

    def apply(self, n, decision, create_missing=False):
        if n in self.apply_errors:
            raise self.apply_errors[n]
        self.applied.append((n, decision.action, decision.labels_to_add, decision.comment))
        return {"labels_added": list(decision.labels_to_add), "comment_posted": bool(decision.comment), "warnings": []}


def issue(number, labels=(), author="alice", **extra):
    return {"number": number, "title": f"{SENTINEL} title", "body": f"{SENTINEL} body\n```\nlog\n```",
            "labels": [{"name": x} for x in labels], "user": {"login": author}, **extra}


class Run:
    def __init__(self, tmp_path, monkeypatch, *, event="issues", payload=None, user_config=None, issues=(),
                 apply_errors=None, env=None):
        self.ws = tmp_path / "workspace"
        (self.ws / ".github").mkdir(parents=True)
        if user_config is not None:
            (self.ws / ".github" / "laya-triage.yml").write_text(user_config)
        ev = tmp_path / "event.json"
        ev.write_text(json.dumps(payload if payload is not None else {"action": "opened", "issue": issue(1)}))
        self.summary = tmp_path / "summary.md"
        self.env = {"GITHUB_EVENT_NAME": event, "GITHUB_EVENT_PATH": str(ev), "GITHUB_REPOSITORY": "octo/sandbox",
                    "GITHUB_STEP_SUMMARY": str(self.summary), "GITHUB_TOKEN": "ghs_TOKEN-abc",
                    "GITHUB_API_URL": "https://api.github.test", "GITHUB_WORKSPACE": str(self.ws), **(env or {})}
        self.clients, self.issues, self.apply_errors = [], issues, apply_errors
        self.classifier_builds = 0

    def client_factory(self, *a):
        c = FakeClient(*a, issues=self.issues, apply_errors=self.apply_errors)
        self.clients.append(c)
        return c

    def __call__(self, argv=(), classifier=None, load_error=None, capsys=None):
        def factory(cfg):
            self.classifier_builds += 1
            if load_error:
                raise load_error
            return classifier

        self.code = cli.main(list(argv), self.env, factory, self.client_factory)
        self.records = [json.loads(x) for x in (self.ws / "laya-triage.jsonl").read_text().splitlines()] \
            if (self.ws / "laya-triage.jsonl").exists() else []
        self.md = self.summary.read_text() if self.summary.exists() else ""
        return self.code

    @property
    def writes(self):
        return [a for c in self.clients for a in c.applied]


@pytest.fixture
def run(tmp_path, monkeypatch):
    return lambda **kw: Run(tmp_path, monkeypatch, **kw)


def test_dry_run_makes_zero_github_calls(run):
    r = run()
    clf = FakeClassifier({1: BUG})
    assert r(classifier=clf) == 0
    assert r.clients == [] and clf.calls == [1]  # no client is even constructed in dry-run
    (rec,) = r.records
    assert (rec["action"], rec["mode"], rec["prediction"], rec["written"], rec["labels_to_add"]) == (
        "apply", "dry-run", "bug", False, ["type: bug"])
    assert "dry-run: no changes were made" in r.md


def test_apply_mode_from_argument(run):
    r = run()
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: BUG})) == 0
    assert r.writes == [(1, "apply", ("type: bug",), None)]
    assert r.clients[0].args == ("https://api.github.test", "octo/sandbox", "ghs_TOKEN-abc")
    assert r.records[0]["written"] is True and r.records[0]["labels_added"] == ["type: bug"]


def test_apply_mode_from_user_config_and_escalation_comment_opt_in(run):
    r = run(user_config="behaviour:\n  mode: apply\n  comment_on_escalate: true\n")
    assert r(classifier=FakeClassifier({1: LOW_BUG})) == 0
    ((n, action, labels, comment),) = r.writes
    assert (n, action, labels) == (1, "escalate", ("triage: needs-human",))
    assert comment.startswith("🤖 Laya Triage couldn't classify") and SENTINEL not in comment


def test_default_escalation_has_no_comment(run):
    r = run()
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: FEATURE})) == 0
    assert r.writes == [(1, "escalate", ("triage: needs-human",), None)]  # D2 + feature threshold 1.01


@pytest.mark.parametrize("payload,reason", [
    ({"action": "opened", "issue": issue(1, pull_request={"url": "x"})}, "pull request"),
    ({"action": "edited", "issue": issue(1)}, "action is not opened"),
    ({"action": "opened", "issue": issue(1, author="dependabot[bot]")}, "author skipped"),
    ({"action": "opened", "issue": issue(1, labels=("Triage: Needs-Human",))}, "already escalated"),
    ({"action": "opened", "issue": issue(1, labels=("type: bug",))}, "already labeled"),
])
def test_each_skip_reason(run, payload, reason):
    r = run(payload=payload)
    assert r(["--mode", "apply"], classifier=FakeClassifier({})) == 0
    assert r.classifier_builds == 0 and r.writes == []
    assert (r.records[0]["action"], r.records[0]["reason"]) == ("skip", reason)


@pytest.mark.parametrize("escalate_on_error,expected_writes", [
    (False, []), (True, [(1, "escalate", ("triage: needs-human",), None)])])
def test_model_load_failure_is_soft(run, escalate_on_error, expected_writes, capsys):
    r = run(user_config=f"behaviour:\n  escalate_on_error: {str(escalate_on_error).lower()}\n")
    err = ClassifierError("model load failed", OSError(f"download of {SENTINEL} failed"))
    assert r(["--mode", "apply"], load_error=err) == 0
    assert r.writes == expected_writes
    assert r.records[0]["error"]["type"] == "ClassifierError <- OSError"
    out = capsys.readouterr()
    assert SENTINEL not in r.md + json.dumps(r.records) + out.out + out.err


def test_inference_error_message_never_reaches_summary_jsonl_or_stdout(run, capsys):
    r = run()
    assert r(classifier=FakeClassifier({}, fail={1})) == 0
    out = capsys.readouterr()
    for text in (r.md, json.dumps(r.records), out.out, out.err):
        assert SENTINEL not in text
    assert r.records[0]["error"]["type"] == "RuntimeError" and r.records[0]["action"] == "skip"
    assert any(f.endswith(":_one") for f in r.records[0]["error"]["frames"])


def test_issue_text_never_reaches_outputs_on_success(run, capsys):
    r = run(user_config="behaviour:\n  comment_on_escalate: true\n")
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: LOW_BUG})) == 0
    out = capsys.readouterr()
    assert SENTINEL not in r.md + json.dumps(r.records) + out.out + out.err + json.dumps(r.writes)


def test_invalid_config_exits_1_listing_bad_keys(run, capsys):
    r = run(user_config="behaviour:\n  mode: auto\n  colour: red\ngating:\n  thresholds:\n    bug: 2\n")
    assert r(classifier=FakeClassifier({})) == 1
    err = capsys.readouterr().err
    for key in ("behaviour.mode", "behaviour.colour: unknown key", "gating.thresholds.bug"):
        assert key in err and key in r.md
    assert r.classifier_builds == 0 and r.records == []


def test_permission_error_exits_1_naming_issues_write(run, capsys):
    perm = GitHubPermissionError("POST /repos/octo/sandbox/issues/1/labels: HTTP 403, the workflow token needs "
                                 "the `issues: write` permission", 403)
    r = run(apply_errors={1: perm})
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: BUG})) == 1
    assert "`issues: write`" in capsys.readouterr().err and "`issues: write`" in r.md
    assert r.records[0]["error"]["type"] == "GitHubPermissionError"


def test_github_5xx_after_retries_is_soft(run):
    r = run(apply_errors={1: GitHubError("POST /repos/octo/sandbox/issues/1/labels: HTTP 502 after 4 attempt(s)", 502)})
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: BUG})) == 0
    assert r.records[0]["error"]["type"] == "GitHubError" and "HTTP 502" in r.records[0]["warnings"][0]


@pytest.mark.parametrize("env,message", [({"GITHUB_TOKEN": ""}, "GITHUB_TOKEN is not set"),
                                         ({"GITHUB_API_URL": "http://api.github.test"}, "must be an https URL")])
def test_apply_needs_token_and_https(run, env, message, capsys):
    r = run(env=env)
    r.client_factory = cli.GitHubClient  # the real constructor validates; it is never asked to send anything
    assert r(["--mode", "apply"], classifier=FakeClassifier({1: BUG})) == 1
    assert message in capsys.readouterr().err


def test_backfill_defaults_to_dry_run_even_if_config_says_apply(run):
    r = run(event="workflow_dispatch", user_config="behaviour:\n  mode: apply\n",
            issues=[issue(10), issue(11, labels=("type: question",)), issue(12)])
    clf = FakeClassifier({10: BUG, 12: LOW_BUG})
    assert r(["--backfill-count", "3"], classifier=clf) == 0
    assert r.clients[0].listed == [3] and r.writes == []
    assert [x["mode"] for x in r.records] == ["dry-run"] * 3
    assert [x["action"] for x in r.records] == ["apply", "skip", "escalate"]
    assert clf.batch_calls == [[10, 12]] and clf.calls == []  # §9.4: predict_batch in backfill


def test_backfill_apply_with_explicit_mode_and_failing_issue_continues(run):
    r = run(event="workflow_dispatch", issues=[issue(10), issue(11), issue(12)])
    clf = FakeClassifier({10: BUG, 12: BUG}, fail={11})
    assert r(["--backfill-count", "3", "--mode", "apply"], classifier=clf) == 0
    assert clf.batch_calls == [[10, 11, 12]] and clf.calls == [10, 11, 12]  # batch failed -> per issue
    assert [w[0] for w in r.writes] == [10, 12]
    assert r.records[1]["error"]["type"] == "RuntimeError" and r.records[1]["action"] == "skip"


def test_backfill_issues_without_event_action_are_not_skipped(run):
    r = run(event="workflow_dispatch", issues=[issue(10)])  # REST objects carry no "action"
    assert r(["--backfill-count", "1"], classifier=FakeClassifier({10: BUG})) == 0
    assert r.records[0]["action"] == "apply"


def test_workflow_dispatch_without_backfill_does_nothing(run):
    r = run(event="workflow_dispatch")
    assert r(["--backfill-count", "0", "--mode", ""], classifier=FakeClassifier({})) == 0
    assert r.clients == [] and r.records == [] and "nothing to triage" in r.md


def test_runs_the_same_from_another_working_directory(tmp_path, monkeypatch):
    user = "gating:\n  thresholds:\n    bug: 0.95\n"
    results = []
    for cwd in (ROOT, tmp_path / "elsewhere"):
        cwd.mkdir(exist_ok=True)
        monkeypatch.chdir(cwd)
        sub = tmp_path / f"run{len(results)}"
        sub.mkdir()
        r = Run(sub, monkeypatch, user_config=user)
        assert r(classifier=FakeClassifier({1: BUG})) == 0  # no --default-config: falls back next to the package
        results.append([{k: v for k, v in x.items()} for x in r.records])
        assert (r.ws / "laya-triage.jsonl").exists()  # relative --log-path resolves against GITHUB_WORKSPACE
    assert results[0] == results[1]
    assert results[0][0]["threshold"] == 0.95 and results[0][0]["action"] == "escalate"  # user config was found


def test_explicit_relative_default_config_resolves_against_workspace(tmp_path, monkeypatch):
    r = Run(tmp_path, monkeypatch)
    (r.ws / "cfg").mkdir()
    shutil.copy(DEFAULT, r.ws / "cfg" / "defaults.yml")
    monkeypatch.chdir(tmp_path)
    assert r(["--default-config", "cfg/defaults.yml"], classifier=FakeClassifier({1: BUG})) == 0
    (tmp_path / "other").mkdir()
    r2 = Run(tmp_path / "other", monkeypatch)
    assert r2(["--default-config", "cfg/missing.yml"], classifier=FakeClassifier({1: BUG})) == 1


# ---- --print-model and the model_loaded step output (action.yml) ----

def fake_cache(root, repo="Prasanna85/laya-issue-triage", revision=REV):
    from laya_triage.classifier import REQUIRED_MODEL_FILES

    snap = root / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
    for f in REQUIRED_MODEL_FILES:
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).write_bytes(b"x")


@pytest.mark.parametrize("cached", [False, True])
def test_print_model(run, tmp_path, capsys, cached):
    r = run(env={"HF_HUB_CACHE": str(tmp_path / "hf")})
    if cached:
        fake_cache(tmp_path / "hf")
    assert r(["--print-model"], classifier=FakeClassifier({})) == 0
    out = capsys.readouterr().out
    assert out == (f"model-key=Prasanna85--laya-issue-triage--{REV}\nmodel-cached={str(cached).lower()}\n")
    assert r.classifier_builds == 0 and r.clients == [] and r.records == []


def test_print_model_uses_the_merged_user_config(run, tmp_path, capsys):
    other = "f" * 40
    r = run(user_config=f'model:\n  repo: "someone/other-model"\n  revision: "{other}"\n',
            env={"HF_HUB_CACHE": str(tmp_path / "hf")})
    fake_cache(tmp_path / "hf", "someone/other-model", other)
    assert r(["--print-model"]) == 0
    assert capsys.readouterr().out == f"model-key=someone--other-model--{other}\nmodel-cached=true\n"


def test_print_model_with_invalid_config_exits_1(run, capsys):
    r = run(user_config='model:\n  repo: "bad repo"\n')
    assert r(["--print-model"]) == 1
    captured = capsys.readouterr()
    assert "model-key" not in captured.out and "model.repo" in captured.err


@pytest.mark.parametrize("load_error,expected", [(None, "true"), (ClassifierError("model load failed"), "false")])
def test_model_loaded_output(run, tmp_path, load_error, expected):
    out = tmp_path / "github_output"
    r = run(env={"GITHUB_OUTPUT": str(out)})
    assert r(classifier=FakeClassifier({1: BUG}), load_error=load_error) == 0
    assert out.read_text() == f"model_loaded={expected}\n"


def test_model_loaded_false_when_nothing_needed_the_model(run, tmp_path):
    out = tmp_path / "github_output"
    r = run(payload={"action": "opened", "issue": issue(1, labels=("type: bug",))}, env={"GITHUB_OUTPUT": str(out)})
    assert r(classifier=FakeClassifier({})) == 0
    assert out.read_text() == "model_loaded=false\n" and r.classifier_builds == 0


def test_non_integer_backfill_count_fails_loudly(run):
    r = run(event="workflow_dispatch")
    for bad in ("abc", "-1", "1.5"):
        with pytest.raises(SystemExit) as e:
            r(["--backfill-count", bad])
        assert e.value.code == 2


# ---- --warm: cache warm-up (§7.7, §12.4 warm-cache) ----

class Exploding:
    """A client factory or classifier attribute that must never be touched."""

    def __call__(self, *a, **kw):
        raise AssertionError("not expected in --warm")


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch", "issues"])
def test_warm_loads_classifies_once_and_makes_no_github_calls(run, tmp_path, capsys, event):
    out = tmp_path / "github_output"
    # apply mode in the config and no token: --warm must not need GitHub at all, nor read the event
    r = run(event=event, user_config="behaviour:\n  mode: apply\n",
            env={"GITHUB_OUTPUT": str(out), "GITHUB_TOKEN": "", "GITHUB_EVENT_PATH": str(tmp_path / "missing.json")})
    r.client_factory = Exploding()
    clf = FakeClassifier({0: BUG})
    assert r(["--warm", "--mode", "apply", "--backfill-count", "5"], classifier=clf) == 0
    assert clf.calls == [0] and clf.batch_calls == [] and r.classifier_builds == 1
    assert out.read_text() == "model_loaded=true\n"
    assert r.records == [] and cli.WARM_OK in r.md and cli.WARM_OK in capsys.readouterr().out


def test_warm_state_goes_through_preprocess(run):
    seen = []

    class Recording(FakeClassifier):
        def classify(self, state, n):
            seen.append(state)
            return super().classify(state, n)

    r = run()
    assert r(["--warm"], classifier=Recording({0: BUG})) == 0
    assert seen == [cli.preprocess(cli.WARM_TITLE, cli.WARM_BODY)]


def test_warm_model_load_failure_is_soft(run, tmp_path):
    out = tmp_path / "github_output"
    r = run(env={"GITHUB_OUTPUT": str(out)})
    assert r(["--warm"], load_error=ClassifierError("model load failed", RuntimeError(SENTINEL))) == 0
    assert out.read_text() == "model_loaded=false\n"
    assert "ClassifierError" in r.md and SENTINEL not in r.md and cli.WARM_OK not in r.md


def test_warm_classify_failure_is_soft_and_keeps_model_loaded(run, tmp_path, capsys):
    out = tmp_path / "github_output"
    r = run(env={"GITHUB_OUTPUT": str(out)})
    assert r(["--warm"], classifier=FakeClassifier({0: BUG}, fail={0})) == 0
    captured = capsys.readouterr()
    assert out.read_text() == "model_loaded=true\n" and "RuntimeError" in r.md
    assert SENTINEL not in r.md + captured.out + captured.err and cli.WARM_OK not in r.md


def test_warm_with_invalid_config_exits_1(run, tmp_path):
    out = tmp_path / "github_output"
    r = run(user_config='model:\n  repo: "bad repo"\n', env={"GITHUB_OUTPUT": str(out)})
    assert r(["--warm"], classifier=FakeClassifier({0: BUG})) == 1
    assert r.classifier_builds == 0 and not out.exists()


def test_warm_and_print_model_are_mutually_exclusive(run):
    with pytest.raises(SystemExit) as e:
        run()(["--warm", "--print-model"])
    assert e.value.code == 2
