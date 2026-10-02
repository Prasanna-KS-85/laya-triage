"""Read the triggering event and apply the skip rules (PROJECT_SPEC.md FR-1, FR-5, FR-6, §7.6).

The event JSON is read from $GITHUB_EVENT_PATH inside Python; issue text is never passed through a shell.
`issue_from_api` normalises an issue object (event payload or REST API) into IssueInfo; `should_skip` is pure.

Skip rules, in order (reasons are fixed text):
1. "pull request": the object has a `pull_request` key (§4.2: issues only).
2. "action is not opened": the event action is present and is not "opened". Backfill issues come from the REST
   API and carry no event action, so this rule never applies to them.
3. "author skipped": the author is in behaviour.skip_authors.
4. "already escalated": the issue carries labels.escalate (idempotent backfill).
5. "already labeled": behaviour.skip_if_labeled and the issue carries labels.{bug,feature,question}.
Label and author comparisons are case-insensitive.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from laya_triage.questions import LABELS


class EventError(Exception):
    """The event payload is missing or malformed (fixed text; never contains issue content)."""


@dataclass(frozen=True)
class IssueInfo:
    number: int
    title: str           # untrusted; only ever passed to preprocess()
    body: str | None     # untrusted; only ever passed to preprocess()
    labels: tuple[str, ...]
    author: str
    is_pull_request: bool
    action: str | None   # event action ("opened", ...); None for REST API issues (backfill)


def issue_from_api(issue, action=None) -> IssueInfo:
    if not isinstance(issue, dict) or not isinstance(issue.get("number"), int):
        raise EventError("issue object has no integer 'number'")
    title, body = issue.get("title"), issue.get("body")
    if not isinstance(title, str) or not (body is None or isinstance(body, str)):
        raise EventError(f"issue #{issue['number']}: title must be a string and body a string or null")
    labels = tuple(lb["name"] if isinstance(lb, dict) else lb for lb in issue.get("labels") or ())
    if not all(isinstance(lb, str) for lb in labels):
        raise EventError(f"issue #{issue['number']}: malformed labels")
    author = (issue.get("user") or {}).get("login") or ""
    return IssueInfo(issue["number"], title, body, labels, author, "pull_request" in issue, action)


def read_event(event_name, event_path) -> list[IssueInfo]:
    """The issue of an `issues` event; [] for workflow_dispatch (backfill issues come from the API) and others."""
    if event_name != "issues":
        return []
    if not event_path:
        raise EventError("GITHUB_EVENT_PATH is not set")
    try:
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EventError(f"event payload cannot be read ({type(e).__name__})") from None
    if not isinstance(payload, dict) or "issue" not in payload:
        raise EventError("issues event payload has no 'issue'")
    return [issue_from_api(payload["issue"], payload.get("action"))]


def should_skip(issue: IssueInfo, cfg) -> str | None:
    if issue.is_pull_request:
        return "pull request"
    if issue.action is not None and issue.action != "opened":
        return "action is not opened"
    if issue.author.casefold() in {a.casefold() for a in cfg.behaviour.skip_authors}:
        return "author skipped"
    have = {lb.casefold() for lb in issue.labels}
    if cfg.labels.escalate.casefold() in have:
        return "already escalated"
    if cfg.behaviour.skip_if_labeled and have & {cfg.labels.for_class(c).casefold() for c in LABELS}:
        return "already labeled"
    return None
