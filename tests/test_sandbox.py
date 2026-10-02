"""sandbox/: authored fixtures, expected_local.json consistency, and compare.py logic on synthetic records
(PROJECT_SPEC.md §13 Phase 4 task 4). No model, no network."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sandbox"))

import compare

from laya_triage.config import load_config
from laya_triage.preprocess import preprocess

FIXTURES = json.loads((ROOT / "sandbox" / "issues.json").read_text(encoding="utf-8"))["fixtures"]
EXPECTED = json.loads((ROOT / "sandbox" / "expected_local.json").read_text())
KEYS = {"key", "title", "body", "labels_at_creation", "expected_class", "expected_behaviour", "rationale"}


def test_fixture_shape():
    assert [f["key"] for f in FIXTURES] == [f"S{i:02d}" for i in range(1, 23)]
    assert all(set(f) == KEYS for f in FIXTURES)
    assert len({f["title"] for f in FIXTURES}) == 22
    by = {f["key"]: f for f in FIXTURES}
    assert by["S11"]["body"] is None and by["S12"]["body"] == ""
    assert len(by["S15"]["body"]) > 6000 and "```" not in by["S15"]["body"]
    assert preprocess(by["S13"]["title"], by["S13"]["body"])["body"] == ""
    assert by["S20"]["labels_at_creation"] == ["type: bug"]
    assert by["S21"]["labels_at_creation"] == ["triage: needs-human"]


def test_fixture_urls_use_example_com_only():
    hosts = {m.group(1) for f in FIXTURES for m in re.finditer(r"https?://([^/\s)]+)", f["body"] or "")}
    assert hosts and all(h == "example.com" or h.endswith(".example.com") for h in hosts)


def test_expected_local_matches_fixtures_and_config():
    cfg = load_config(ROOT / "config" / "triage.default.yml")
    assert EXPECTED["model"]["revision"] == cfg.model.revision and EXPECTED["thresholds"] == cfg.thresholds
    exp = EXPECTED["fixtures"]
    assert list(exp) == [f["key"] for f in FIXTURES]
    assert exp["S20"] == {"skip_reason": "already labeled"} and exp["S21"] == {"skip_reason": "already escalated"}
    for v in exp.values():
        if v["skip_reason"]:
            continue
        p = v["probabilities"]
        assert v["answer_confidence"] == max(p.values()) == p[v["label"]]
        assert (v["decision"]["action"] == "apply") == (v["answer_confidence"] >= cfg.thresholds[v["label"]])
    assert EXPECTED["classify_batch_vs_classify_max_abs_dp"] <= 1e-4


def test_make_map():
    gh = [{"number": 5, "title": FIXTURES[0]["title"]}, {"number": 9, "title": FIXTURES[2]["title"] + " [apply]"}]
    assert compare.make_map(gh, FIXTURES) == {"5": "S01", "9": "S03"}
    with pytest.raises(ValueError, match="#7"):
        compare.make_map([{"number": 7, "title": "something else"}], FIXTURES)


EXP = {"skip_reason": None, "label": "bug", "probabilities": {"bug": 0.62, "feature": 0.2, "question": 0.18},
       "answer_confidence": 0.62, "threshold": 0.6033, "decision": {"action": "apply", "labels_to_add": ["type: bug"]}}


def rec(**kw):
    base = {"issue_number": 1, "action": "apply", "reason": "r", "prediction": "bug",
            "probabilities": {"bug": 0.6205, "feature": 0.1997, "question": 0.1798}, "labels_to_add": ["type: bug"]}
    return {**base, **kw}


def test_compare_record_ok_and_fail():
    assert compare.compare_record(rec(), EXP, 1e-3)[0] == "ok"
    assert compare.compare_record(rec(prediction="feature"), EXP, 1e-3)[0] == "FAIL"
    far = rec(probabilities={"bug": 0.63, "feature": 0.19, "question": 0.18})
    assert compare.compare_record(far, EXP, 1e-3)[0] == "FAIL"
    assert compare.compare_record(rec(prediction=None, action="skip"), EXP, 1e-3)[0] == "FAIL"


def test_compare_record_borderline_only_near_threshold():
    near = {**EXP, "answer_confidence": 0.6038, "probabilities": {"bug": 0.6038, "feature": 0.2, "question": 0.1962}}
    r = rec(action="escalate", labels_to_add=["triage: needs-human"],
            probabilities={"bug": 0.6031, "feature": 0.2005, "question": 0.1964})
    assert compare.compare_record(r, near, 1e-3)[0] == "borderline"
    assert compare.compare_record(rec(action="escalate", labels_to_add=["triage: needs-human"]), EXP, 1e-3)[0] == "FAIL"


def test_compare_record_skips_and_idempotency():
    skip = {"skip_reason": "already labeled"}
    assert compare.compare_record(rec(action="skip", reason="already labeled"), skip, 1e-3)[0] == "ok"
    assert compare.compare_record(rec(), skip, 1e-3)[0] == "FAIL"
    assert compare.compare_record(rec(action="skip", reason="already escalated"), EXP, 1e-3, idempotent=True)[0] == "ok"
    assert compare.compare_record(rec(), EXP, 1e-3, idempotent=True)[0] == "FAIL"


def test_check_reports_unknown_issues_and_unseen_fixtures(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(json.dumps(rec(issue_number=1)) + "\n" + json.dumps(rec(issue_number=99)) + "\n")
    rows, unseen = compare.check([log], {"S01": EXP, "S02": EXP}, {"1": "S01"}, 1e-3, False)
    assert [r[1] for r in rows] == ["ok", "FAIL"] and unseen == ["S02"]
