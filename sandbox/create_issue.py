"""Create sandbox fixtures as real issues with the gh CLI (PROJECT_SPEC.md §13 Phase 4 task 4; runbook: README.md).

  python sandbox/create_issue.py --repo OWNER/NAME S01 [S02 ...] [--title-suffix " [apply]"] [--sleep 5]

Reads sandbox/issues.json and runs `gh issue create` once per fixture key with an argument list (no shell): the
title (plus the optional suffix) is one argument, the body goes through stdin (`--body-file -`), and each label at
creation is a separate `--label`. Prints the fixture key and the URL gh returns, never the title or body.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "issues.json"


def gh_args(fixture, repo, title_suffix=""):
    """The `gh issue create` argument list for one fixture; the body is passed separately on stdin."""
    args = ["gh", "issue", "create", "--repo", repo, "--title", fixture["title"] + title_suffix, "--body-file", "-"]
    for label in fixture["labels_at_creation"]:
        args += ["--label", label]
    return args


def main(argv=None, run=subprocess.run):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, help="sandbox repository, OWNER/NAME")
    ap.add_argument("keys", nargs="+", help="fixture keys, e.g. S01 S02")
    ap.add_argument("--title-suffix", default="", help='appended to the title, e.g. " [apply]"')
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between issues")
    args = ap.parse_args(argv)
    by_key = {f["key"]: f for f in json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]}
    unknown = [k for k in args.keys if k not in by_key]
    if unknown:
        ap.error(f"unknown fixture key(s): {', '.join(unknown)}")
    for i, key in enumerate(args.keys):
        if i and args.sleep:
            time.sleep(args.sleep)
        f = by_key[key]
        done = run(gh_args(f, args.repo, args.title_suffix), input=f["body"] or "", text=True, capture_output=True)
        if done.returncode != 0:
            print(f"{key}: gh issue create failed (exit {done.returncode})", file=sys.stderr)
            return 1
        print(f"{key}: {done.stdout.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
