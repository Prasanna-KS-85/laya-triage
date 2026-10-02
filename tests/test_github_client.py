"""laya_triage.github_client with an injected fake opener and sleep: never touches the network
(PROJECT_SPEC.md §7.5, §7.6; owner decision D5)."""
import io
import json
import re
import urllib.error
import urllib.parse

import pytest

from laya_triage.github_client import (
    GitHubClient,
    GitHubError,
    GitHubPermissionError,
    GitHubRateLimitError,
)
from laya_triage.policy import Decision

TOKEN = "ghs_SECRET-token-0123456789"
API = "https://api.github.test"
REPO = "octo/sandbox"
PATH_RE = re.compile(r"^/repos/octo/sandbox/(issues|labels|issues/\d+/(labels|comments))$")


class Resp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode() if payload is not None else b""

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError(API + "/x", code, "error", headers or {}, io.BytesIO(body))


class FakeOpener:
    """Returns queued responses (a payload or an exception) and records every request."""

    def __init__(self, *responses, route=None):
        self.queue = list(responses)
        self.route = route
        self.requests = []

    def __call__(self, req, timeout):
        body = json.loads(req.data) if req.data else None
        u = urllib.parse.urlsplit(req.full_url)
        self.requests.append({"method": req.get_method(), "path": u.path, "query": dict(urllib.parse.parse_qsl(u.query)),
                              "headers": dict(req.header_items()), "body": body, "timeout": timeout})
        nxt = self.route(self.requests[-1]) if self.route else self.queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return Resp(nxt)


def client(opener, sleeps=None):
    return GitHubClient(API, REPO, TOKEN, opener=opener, sleep=(sleeps.append if sleeps is not None else lambda s: None))


@pytest.mark.parametrize("api,repo,token", [("http://api.github.com", REPO, TOKEN), ("api.github.com", REPO, TOKEN),
                                            (API, "not a repo", TOKEN), (API, REPO, ""), (API, REPO, None)])
def test_constructor_validation(api, repo, token):
    with pytest.raises(ValueError) as e:
        GitHubClient(api, repo, token, opener=FakeOpener())
    assert TOKEN not in str(e.value)


def test_token_only_in_authorization_header_and_hidden_in_repr():
    op = FakeOpener(None)
    c = client(op)
    c.comment(5, "hi")
    (r,) = op.requests
    assert r["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in r["path"] + json.dumps(r["query"]) + json.dumps(r["body"])
    assert TOKEN not in repr(c)
    assert r["headers"]["X-github-api-version"] == "2022-11-28"


def test_5xx_then_success_backs_off_2_then_4():
    sleeps = []
    op = FakeOpener(http_error(502), http_error(503), None)
    client(op, sleeps).add_labels(3, ["type: bug"])
    assert len(op.requests) == 3 and sleeps == [2, 4]


def test_5xx_gives_up_after_3_retries():
    sleeps = []
    op = FakeOpener(*[http_error(500)] * 4)
    with pytest.raises(GitHubError) as e:
        client(op, sleeps).add_labels(3, ["type: bug"])
    assert len(op.requests) == 4 and sleeps == [2, 4, 8]
    assert not isinstance(e.value, GitHubPermissionError) and e.value.status == 500
    assert TOKEN not in str(e.value)


@pytest.mark.parametrize("make", [
    lambda: http_error(429),
    lambda: http_error(403, headers={"X-RateLimit-Remaining": "0"}),
    lambda: http_error(403, headers={"Retry-After": "60"}),
    lambda: http_error(403, body=b'{"message": "You have exceeded a secondary rate limit."}'),
])
def test_rate_limits_are_retried_then_fail_soft(make):
    sleeps = []
    op = FakeOpener(*(make() for _ in range(4)))  # a fresh response (and body stream) per attempt
    with pytest.raises(GitHubRateLimitError):
        client(op, sleeps).comment(9, "x")
    assert len(op.requests) == 4 and sleeps == [2, 4, 8]


def test_rate_limit_then_success():
    sleeps = []
    op = FakeOpener(http_error(429), None)
    client(op, sleeps).comment(9, "x")
    assert sleeps == [2]


def test_403_permission_is_not_retried_and_names_issues_write():
    sleeps = []
    op = FakeOpener(http_error(403, body=b'{"message": "Resource not accessible by integration"}'))
    with pytest.raises(GitHubPermissionError) as e:
        client(op, sleeps).add_labels(3, ["type: bug"])
    assert len(op.requests) == 1 and sleeps == []
    assert "`issues: write`" in str(e.value) and TOKEN not in str(e.value)
    assert "Resource not accessible" not in str(e.value)  # no response body in messages


def test_4xx_other_is_soft_and_not_retried():
    op = FakeOpener(http_error(422, body=b"validation failed"))
    with pytest.raises(GitHubError) as e:
        client(op).comment(1, "x")
    assert e.value.status == 422 and len(op.requests) == 1 and type(e.value) is GitHubError


def test_network_errors_are_retried():
    sleeps = []
    op = FakeOpener(urllib.error.URLError("dns"), TimeoutError(), None)
    client(op, sleeps).comment(1, "x")
    assert sleeps == [2, 4]
    op2 = FakeOpener(*[urllib.error.URLError(f"down {TOKEN}")] * 4)
    with pytest.raises(GitHubError) as e:
        client(op2).comment(1, "x")
    assert "network error (URLError)" in str(e.value) and TOKEN not in str(e.value)


def _issues_route(n_issues, pr_every=3):
    items = [{"number": 1000 - i, "title": "t", "body": "", "labels": [], "user": {"login": "u"},
              **({"pull_request": {}} if i % pr_every == 0 else {})} for i in range(n_issues)]

    def route(r):
        assert r["path"] == "/repos/octo/sandbox/issues"
        assert (r["query"]["state"], r["query"]["sort"], r["query"]["direction"]) == ("open", "created", "desc")
        page, per = int(r["query"]["page"]), int(r["query"]["per_page"])
        return items[(page - 1) * per: page * per]
    return route


def test_list_open_issues_paginates_and_drops_pull_requests():
    op = FakeOpener(route=_issues_route(250))
    got = client(op).list_open_issues(150)
    assert len(got) == 150 and all("pull_request" not in i for i in got)
    assert [i["number"] for i in got] == sorted((i["number"] for i in got), reverse=True)
    assert [r["query"]["page"] for r in op.requests] == ["1", "2", "3"]


def test_list_open_issues_stops_at_last_page():
    op = FakeOpener(route=_issues_route(30))
    got = client(op).list_open_issues(100)
    assert len(got) == 20 and len(op.requests) == 1
    assert client(FakeOpener()).list_open_issues(0) == []


def labels_route(existing, created=None):
    def route(r):
        if r["method"] == "GET" and r["path"].endswith("/labels"):
            page = int(r["query"]["page"])
            names = [{"name": n} for n in existing]
            return names[(page - 1) * 100: page * 100]
        if r["method"] == "POST" and r["path"] == "/repos/octo/sandbox/labels":
            created.append(r["body"])
            return {"name": r["body"]["name"]}
        return None
    return route


def test_missing_label_is_not_added_without_create():
    op = FakeOpener(route=labels_route(["bug"]))
    done = client(op).apply(4, Decision("apply", ("type: bug",), None, "r"), create_missing=False)
    assert done == {"labels_added": [], "comment_posted": False,
                    "warnings": [("label 'type: bug' does not exist in the repository; not added "
                                  "(create_missing_labels is false)")]}
    assert [r["method"] for r in op.requests] == ["GET"]  # never POSTs labels (GitHub would create them)


def test_missing_label_is_created_when_allowed():
    created = []
    op = FakeOpener(route=labels_route([], created))
    done = client(op).apply(4, Decision("escalate", ("triage: needs-human",), "c", "r"), create_missing=True)
    assert created == [{"name": "triage: needs-human", "color": "ededed"}]
    assert done == {"labels_added": ["triage: needs-human"], "comment_posted": True, "warnings": []}
    assert [(r["method"], r["path"]) for r in op.requests][-2:] == [
        ("POST", "/repos/octo/sandbox/issues/4/labels"), ("POST", "/repos/octo/sandbox/issues/4/comments")]
    assert op.requests[-1]["body"] == {"body": "c"}


def test_existing_label_matches_case_insensitively_and_uses_repo_spelling():
    op = FakeOpener(route=labels_route(["Type: Bug"] + [f"l{i}" for i in range(150)]))
    c = client(op)
    done = c.apply(4, Decision("apply", ("type: bug",), None, "r"))
    assert done["labels_added"] == ["Type: Bug"]
    assert op.requests[-1]["body"] == {"labels": ["Type: Bug"]}
    c.apply(5, Decision("apply", ("type: bug",), None, "r"))  # label list cached: 2 GET pages once
    assert sum(r["method"] == "GET" for r in op.requests) == 2


def test_skip_decision_makes_no_calls():
    op = FakeOpener()
    assert client(op).apply(4, Decision("skip", (), None, "r")) == {"labels_added": [], "comment_posted": False,
                                                                    "warnings": []}
    assert op.requests == []


def test_label_names_never_appear_in_urls():
    weird = "a/b ?#%&: c"
    created = []
    op = FakeOpener(route=labels_route([], created))
    client(op).apply(12, Decision("escalate", (weird,), "x", "r"), create_missing=True)
    for r in op.requests:
        assert PATH_RE.match(r["path"]), r["path"]
        assert weird not in r["path"] and weird not in json.dumps(r["query"])
    assert {"labels": [weird]} in [r["body"] for r in op.requests]


def test_issue_numbers_are_integers_in_paths():
    op = FakeOpener(None)
    client(op).comment(True + 6, "x")
    assert op.requests[0]["path"] == "/repos/octo/sandbox/issues/7/comments"
    with pytest.raises(ValueError):
        client(FakeOpener(None)).comment("7/../../x", "x")
