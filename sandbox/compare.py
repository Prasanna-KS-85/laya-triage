"""Compare the sandbox runner's JSONL logs with sandbox/expected_local.json (PROJECT_SPEC.md §13 Phase 4 task 4).

  python sandbox/compare.py map GH_ISSUES_JSON     # from `gh issue list --json number,title`; writes issue_map.json
  python sandbox/compare.py check LOG.jsonl [...] [--tol 1e-3] [--idempotent]

`map` matches issues to fixtures by exact title (a title may end in " [apply]" for the extra apply-mode issues) and
writes sandbox/issue_map.json ({issue number: fixture key}). `check` reads every record of the given logs:
- fixtures expected to be skipped (S20, S21): action skip with the same reason;
- classified fixtures: same label as the local run, max |dp| <= tol (macOS arm64 vs Linux x86 fp32), and the same
  action and labels_to_add; a decision that differs while the confidence is within tol of the threshold is reported
  as "borderline", not as a failure;
- --idempotent (a backfill re-run): every record must be a skip with reason "already labeled" or "already escalated".
Prints keys, labels and numbers only. Exit 1 on any failure.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected_local.json"
FIXTURES = HERE / "issues.json"
ISSUE_MAP = HERE / "issue_map.json"
APPLY_SUFFIX = " [apply]"
LABELS = ("bug", "feature", "question")
IDEMPOTENT_REASONS = {"already labeled", "already escalated"}


def make_map(gh_issues, fixtures):
    """{issue number (str): fixture key} by exact title; raises on unknown or duplicate titles."""
    by_title = {f["title"]: f["key"] for f in fixtures}
    if len(by_title) != len(fixtures):
        raise ValueError("fixture titles are not unique")
    out = {}
    for it in gh_issues:
        title = it["title"].removesuffix(APPLY_SUFFIX)
        if title not in by_title:
            raise ValueError(f"issue #{it['number']}: title matches no fixture")
        out[str(it["number"])] = by_title[title]
    return out


def compare_record(rec, exp, tol, idempotent=False):
    """(status, detail) for one JSONL record: status is ok, borderline or FAIL."""
    if idempotent:
        ok = rec["action"] == "skip" and rec["reason"] in IDEMPOTENT_REASONS
        return ("ok" if ok else "FAIL"), f"{rec['action']}: {rec['reason']}"
    if exp["skip_reason"]:
        ok = rec["action"] == "skip" and rec["reason"] == exp["skip_reason"]
        return ("ok" if ok else "FAIL"), f"skip expected ({exp['skip_reason']}), got {rec['action']}: {rec['reason']}"
    if rec["prediction"] is None:
        return "FAIL", f"no prediction ({rec['action']}: {rec['reason']})"
    dp = max(abs(rec["probabilities"][c] - exp["probabilities"][c]) for c in LABELS)
    detail = f"{rec['prediction']} vs {exp['label']}, max |dp| {dp:.1e}"
    if rec["prediction"] != exp["label"] or dp > tol:
        return "FAIL", detail
    same = rec["action"] == exp["decision"]["action"] and rec["labels_to_add"] == exp["decision"]["labels_to_add"]
    if not same:
        near = abs(exp["answer_confidence"] - exp["threshold"]) <= tol
        return ("borderline" if near else "FAIL"), detail + f"; decision {rec['action']} vs {exp['decision']['action']}"
    return "ok", detail + f"; {rec['action']}"


def check(log_paths, expected, issue_map, tol, idempotent):
    rows, seen = [], set()
    for path in log_paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            key = issue_map.get(str(rec["issue_number"]))
            if key is None:
                rows.append((f"#{rec['issue_number']}", "FAIL", "issue not in issue_map.json"))
                continue
            seen.add(key)
            status, detail = compare_record(rec, expected[key], tol, idempotent)
            rows.append((f"{key} #{rec['issue_number']}", status, detail))
    return rows, sorted(set(expected) - seen)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("map")
    m.add_argument("gh_issues")
    c = sub.add_parser("check")
    c.add_argument("logs", nargs="+")
    c.add_argument("--tol", type=float, default=1e-3)
    c.add_argument("--idempotent", action="store_true")
    args = ap.parse_args(argv)
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]
    if args.cmd == "map":
        mapping = make_map(json.loads(Path(args.gh_issues).read_text(encoding="utf-8")), fixtures)
        ISSUE_MAP.write_text(json.dumps(mapping, indent=1, sort_keys=True) + "\n")
        print(f"mapped {len(mapping)} issues -> {ISSUE_MAP.name}")
        return 0
    expected = json.loads(EXPECTED.read_text())["fixtures"]
    rows, unseen = check(args.logs, expected, json.loads(ISSUE_MAP.read_text()), args.tol, args.idempotent)
    for who, status, detail in rows:
        print(f"{status:<10} {who:<10} {detail}")
    fails = sum(s == "FAIL" for _, s, _ in rows)
    border = sum(s == "borderline" for _, s, _ in rows)
    print(f"\n{len(rows)} records: {len(rows) - fails - border} ok, {border} borderline, {fails} failed; "
          f"fixtures without a record: {', '.join(unseen) or 'none'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
