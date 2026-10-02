"""GitHub REST calls for the Action: list open issues, check / create labels, add labels, comment
(PROJECT_SPEC.md FR-2, FR-3, FR-6, §7.5, §7.6). Stdlib only; the HTTP opener and sleep are injected for tests.

- Retries: 5xx, 429, network errors and 403 rate limits (x-ratelimit-remaining: 0, retry-after, or "secondary
  rate limit") are retried 3 times after 2 s, 4 s, 8 s (4 calls in total), then raise GitHubError /
  GitHubRateLimitError (soft failures).
- Any other 403 raises GitHubPermissionError naming the missing `issues: write` permission (the Action exits 1).
- Labels are checked against the repository's labels (listed once, matched case-insensitively) before they are
  added, because GitHub's add-labels call silently creates missing labels; a missing label is created only with
  create_missing_labels: true. Label names only travel in JSON bodies, never in a URL; if one is ever put in a
  path it must go through `urllib.parse.quote(name, safe="")`.
- The token is used only in the Authorization header. Error messages hold the method, the API path, the status
  and fixed text: never the token, never a response body.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

RETRY_DELAYS_S = (2, 4, 8)
PER_PAGE = 100
DEFAULT_LABEL_COLOR = "ededed"
API_VERSION = "2022-11-28"
USER_AGENT = "laya-triage"
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
PERMISSION_HINT = ("the workflow token needs the `issues: write` permission "
                   "(permissions: { issues: write, contents: read })")


class GitHubError(Exception):
    """A GitHub API call failed (soft failure)."""

    def __init__(self, message, status=None):
        self.status = status
        super().__init__(message)


class GitHubRateLimitError(GitHubError):
    """Still rate limited after the retries (soft failure)."""


class GitHubPermissionError(GitHubError):
    """403 that is not a rate limit: the token lacks a permission (misconfiguration, exit 1)."""


def _is_rate_limited(status, headers, body):
    if status == 429:
        return True
    if status != 403:
        return False
    headers = headers or {}
    return (headers.get("x-ratelimit-remaining") == "0" or headers.get("retry-after") is not None
            or b"rate limit" in body.lower())


class GitHubClient:
    def __init__(self, api_url, repo, token, opener=urllib.request.urlopen, sleep=time.sleep, timeout=30):
        parts = urllib.parse.urlsplit(api_url or "")
        if parts.scheme != "https" or not parts.netloc:
            raise ValueError("GITHUB_API_URL must be an https URL")
        if not isinstance(repo, str) or not REPO_RE.fullmatch(repo):
            raise ValueError("GITHUB_REPOSITORY must look like owner/name")
        if not token:
            raise ValueError("GITHUB_TOKEN is not set")
        self._base = api_url.rstrip("/")
        self._repo = repo
        self._token = token
        self._opener, self._sleep, self._timeout = opener, sleep, timeout
        self._labels = None  # casefolded name -> repository's spelling

    def __repr__(self):
        return f"GitHubClient({self._base!r}, {self._repo!r}, token=<hidden>)"

    # ---- transport ----

    def _request(self, method, path, payload=None, query=None):
        url = self._base + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {self._token}",
                   "X-GitHub-Api-Version": API_VERSION, "User-Agent": USER_AGENT}
        if data is not None:
            headers["Content-Type"] = "application/json"
        what = f"{method} {path}"
        err = None
        for attempt in range(len(RETRY_DELAYS_S) + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with self._opener(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                status, hdrs = e.code, {k.lower(): v for k, v in (e.headers or {}).items()}
                try:
                    body = e.read() or b""
                except Exception:  # noqa: BLE001 - only used to detect a rate limit
                    body = b""
                if _is_rate_limited(status, hdrs, body):
                    err = GitHubRateLimitError(f"{what}: rate limited (HTTP {status}) after {attempt + 1} attempt(s)",
                                               status)
                elif status >= 500:
                    err = GitHubError(f"{what}: HTTP {status} after {attempt + 1} attempt(s)", status)
                elif status == 403:
                    raise GitHubPermissionError(f"{what}: HTTP 403, {PERMISSION_HINT}", status) from None
                else:
                    raise GitHubError(f"{what}: HTTP {status}", status) from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = GitHubError(f"{what}: network error ({type(e).__name__}) after {attempt + 1} attempt(s)")
            if attempt < len(RETRY_DELAYS_S):
                self._sleep(RETRY_DELAYS_S[attempt])
        raise err

    def _paginate(self, path, query, limit=None):
        page = 1
        while True:
            items = self._request("GET", path, query={**query, "per_page": PER_PAGE, "page": page})
            if not isinstance(items, list):
                raise GitHubError(f"GET {path}: unexpected response")
            yield from items
            if len(items) < PER_PAGE:
                return
            page += 1

    # ---- API ----

    def list_open_issues(self, n):
        """The `n` newest open issues (pull requests excluded), as API issue objects."""
        out = []
        if n <= 0:
            return out
        query = {"state": "open", "sort": "created", "direction": "desc"}
        for it in self._paginate(f"/repos/{self._repo}/issues", query):
            if "pull_request" in it:
                continue
            out.append(it)
            if len(out) >= n:
                break
        return out

    def repo_labels(self):
        """{casefolded name: name} of the repository's labels (cached)."""
        if self._labels is None:
            self._labels = {lb["name"].casefold(): lb["name"]
                            for lb in self._paginate(f"/repos/{self._repo}/labels", {})}
        return self._labels

    def create_label(self, name):
        self._request("POST", f"/repos/{self._repo}/labels", {"name": name, "color": DEFAULT_LABEL_COLOR})
        if self._labels is not None:
            self._labels[name.casefold()] = name

    def ensure_labels(self, names, create_missing):
        """(names to add, in the repository's spelling; warnings) for the configured label names."""
        existing, ok, warnings = self.repo_labels(), [], []
        for name in names:
            if name.casefold() in existing:
                ok.append(existing[name.casefold()])
            elif create_missing:
                self.create_label(name)
                ok.append(name)
            else:
                warnings.append(f"label '{name}' does not exist in the repository; not added "
                                "(create_missing_labels is false)")
        return ok, warnings

    def add_labels(self, issue_number, names):
        self._request("POST", f"/repos/{self._repo}/issues/{int(issue_number)}/labels", {"labels": list(names)})

    def comment(self, issue_number, body):
        self._request("POST", f"/repos/{self._repo}/issues/{int(issue_number)}/comments", {"body": body})

    def apply(self, issue_number, decision, create_missing=False):
        """Carry out a Decision: add its labels (existing ones only, unless create_missing) and post its comment.
        Returns {"labels_added": [...], "comment_posted": bool, "warnings": [...]}."""
        done = {"labels_added": [], "comment_posted": False, "warnings": []}
        if decision.action == "skip":
            return done
        if decision.labels_to_add:
            names, done["warnings"] = self.ensure_labels(decision.labels_to_add, create_missing)
            if names:
                self.add_labels(issue_number, names)
                done["labels_added"] = names
        if decision.comment:
            self.comment(issue_number, decision.comment)
            done["comment_posted"] = True
        return done
