"""laya_triage.event_loader: event parsing and skip rules (PROJECT_SPEC.md FR-1, FR-5, FR-6; owner decision D3).
Synthetic payloads only."""
import json
from pathlib import Path

import pytest
import yaml

from laya_triage.config import merge, validate
from laya_triage.event_loader import (
    EventError,
    IssueInfo,
    issue_from_api,
    read_event,
    should_skip,
)

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "triage.default.yml"
SENTINEL = "SENTINEL-TITLE-7f3a"


def cfg(**behaviour):
    return validate(merge(yaml.safe_load(DEFAULT.read_text()), {"behaviour": behaviour} if behaviour else {}))


def api_issue(number=7, labels=(), author="alice", body="It crashes.", **extra):
    return {"number": number, "title": "Crash on start", "body": body, "user": {"login": author},
            "labels": [{"name": n} for n in labels], **extra}


def write_event(tmp_path, payload):
    p = tmp_path / "event.json"
    p.write_text(json.dumps(payload))
    return p


def test_read_issues_event(tmp_path):
    path = write_event(tmp_path, {"action": "opened", "issue": api_issue(labels=("x",))})
    (issue,) = read_event("issues", path)
    assert issue == IssueInfo(7, "Crash on start", "It crashes.", ("x",), "alice", False, "opened")


def test_null_body_missing_user_and_string_labels():
    issue = issue_from_api({"number": 3, "title": "t", "body": None, "labels": ["a", {"name": "b"}]})
    assert (issue.body, issue.author, issue.labels, issue.action) == (None, "", ("a", "b"), None)


def test_other_events_have_no_issues(tmp_path):
    assert read_event("workflow_dispatch", None) == []
    assert read_event("push", write_event(tmp_path, {})) == []


@pytest.mark.parametrize("payload", [None, "not json", {"action": "opened"}, {"issue": {"title": "t"}},
                                     {"issue": {"number": 1, "title": 5}}, {"issue": {"number": 1, "title": "t", "labels": [3]}}])
def test_malformed_events_raise_without_issue_text(tmp_path, payload):
    if payload is None:
        with pytest.raises(EventError):
            read_event("issues", None)
        return
    p = tmp_path / "event.json"
    p.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    with pytest.raises(EventError) as e:
        read_event("issues", p)
    assert SENTINEL not in str(e.value)


def test_event_error_never_contains_issue_text(tmp_path):
    p = write_event(tmp_path, {"issue": {"number": 1, "title": SENTINEL, "body": 12}})
    with pytest.raises(EventError) as e:
        read_event("issues", p)
    assert SENTINEL not in str(e.value)


def test_not_skipped_by_default():
    assert should_skip(issue_from_api(api_issue(), "opened"), cfg()) is None


@pytest.mark.parametrize("issue,reason", [
    (api_issue(pull_request={"url": "x"}), "pull request"),
    (api_issue(author="dependabot[bot]"), "author skipped"),
    (api_issue(author="Renovate[Bot]"), "author skipped"),           # case-insensitive
    (api_issue(labels=("triage: needs-human",)), "already escalated"),
    (api_issue(labels=("TRIAGE: Needs-Human", "type: bug")), "already escalated"),
    (api_issue(labels=("type: feature",)), "already labeled"),
    (api_issue(labels=("Type: Question",)), "already labeled"),      # case-insensitive
])
def test_skip_reasons(issue, reason):
    assert should_skip(issue_from_api(issue, "opened"), cfg()) == reason


@pytest.mark.parametrize("action", ["edited", "labeled", "reopened", "closed"])
def test_event_action_other_than_opened_is_skipped(action):
    assert should_skip(issue_from_api(api_issue(), action), cfg()) == "action is not opened"


def test_backfill_issues_have_no_action_and_are_never_skipped_for_it():
    issue = issue_from_api(api_issue())  # REST API object, as in backfill
    assert issue.action is None
    assert should_skip(issue, cfg()) is None


def test_pull_request_rule_comes_first():
    issue = issue_from_api(api_issue(pull_request={}, author="dependabot[bot]"), "edited")
    assert should_skip(issue, cfg()) == "pull request"


def test_skip_if_labeled_off_still_skips_escalated():
    c = cfg(skip_if_labeled=False)
    assert should_skip(issue_from_api(api_issue(labels=("type: bug",)), "opened"), c) is None
    assert should_skip(issue_from_api(api_issue(labels=("triage: needs-human",)), "opened"), c) == "already escalated"


def test_unrelated_labels_do_not_skip():
    assert should_skip(issue_from_api(api_issue(labels=("good first issue", "bug")), "opened"), cfg()) is None
