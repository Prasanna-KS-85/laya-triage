"""results/phase3/test_report.py helpers on synthetic inputs (no data files, no model)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "results" / "phase3"))

import test_report as tr

B, F, Q = "bug", "feature", "question"

CONFIG_YAML = """model:
  revision: "abc"
gating:
  thresholds:
    bug: 0.6033
    feature: 1.01
    question: 1.01
"""

PHASE2_MD = """# Phase 2

## Results table v0 (test)

| System | Cross-repo | Pooled | Acc | F1 bug | F1 feature | F1 question | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 TF-IDF+LR | **0.7655** | 0.7666 | 0.7673 | 0.7843 | 0.7925 | 0.7230 | 0.0493 |
| B3 SetFit (x) | **0.8270** | 0.8263 [c] | 0.8267 [c] | 0.8425 [c] | 0.8555 [c] | 0.7809 [c] | — |
| B3 RoBERTa (y) | 0.7923 | 0.7926 [c] | 0.7927 [c] | 0.8052 [c] | 0.8064 [c] | 0.7663 [c] | — |

| System | Cross-repo | Pooled [c] | Acc [c] | F1 bug [c] | F1 feature [c] | F1 question [c] |
|---|---:|---:|---:|---:|---:|---:|
| SetFit, `output/setfit/results.json` | 0.8240 | 0.8238 | 0.8240 | 0.8328 | 0.8583 | 0.7802 |

### Per-repo subsection (a different table; must not be parsed)

| B3 SetFit (README) | 0.8718 | 0.8644 | 0.8262 | 0.7555 | 0.8173 | 0.8270 |

## Other section

| B3 Ignored | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | — |
"""


def test_config_thresholds():
    assert tr.config_thresholds(CONFIG_YAML) == {B: 0.6033, F: 1.01, Q: 1.01}


def test_config_thresholds_rejects_missing_label():
    with pytest.raises(ValueError):
        tr.config_thresholds(CONFIG_YAML.replace("    question: 1.01\n", ""))


def test_check_declared_rounds_to_4_decimals():
    taus = {B: 0.7697721833977644, F: 0.6885899025760078, Q: 0.8691900491568142}
    assert tr.check_declared(taus, {B: 0.7698, F: 0.6886, Q: 0.8692}, "x") is taus


def test_check_declared_rejects_mismatch():
    with pytest.raises(SystemExit):
        tr.check_declared({B: 0.6034, F: 1.01, Q: 1.01}, {B: 0.6033, F: 1.01, Q: 1.01}, "x")


def test_parse_b3_rows():
    out = tr.parse_b3_rows(PHASE2_MD)
    assert [r["system"] for r in out["published"]] == ["B3 SetFit (x)", "B3 RoBERTa (y)"]
    assert out["published"][0] == {"system": "B3 SetFit (x)", "cross_repo_macro_f1": 0.827, "pooled_macro_f1": 0.8263,
                                   "accuracy": 0.8267, "f1_bug": 0.8425, "f1_feature": 0.8555, "f1_question": 0.7809}
    assert out["footnote"] == [{"system": "SetFit, `output/setfit/results.json`", "cross_repo_macro_f1": 0.824,
                                "pooled_macro_f1": 0.8238, "accuracy": 0.824, "f1_bug": 0.8328,
                                "f1_feature": 0.8583, "f1_question": 0.7802}]


def test_parse_b3_rows_requires_section():
    with pytest.raises(ValueError):
        tr.parse_b3_rows("# nothing here\n| B3 x | 0.1 |\n")


def test_gating_view():
    gold = [B, B, B, F, Q, Q]
    pred = [B, B, F, F, Q, B]
    conf = [0.9, 0.5, 0.95, 0.7, 0.99, 0.95]
    v = tr.gating_view(gold, pred, conf, {B: 0.6, F: 1.01, Q: 0.5})
    a = v["applied"]
    # bug: confs 0.9 (right), 0.5 (below), 0.95 (wrong) -> 2 applied, 1 right; feature never; question 1 right.
    assert a["per_label"][B]["applied"] == 2 and a["per_label"][B]["precision"] == 0.5
    assert a["per_label"][F]["applied"] == 0 and a["per_label"][F]["precision"] is None
    assert a["per_label"][Q]["applied"] == 1 and a["per_label"][Q]["precision"] == 1.0
    assert (a["applied"], a["escalated"], a["coverage"], a["precision"]) == (3, 3, 0.5, 2 / 3)


def _paired(lo, hi):
    return {"bootstrap_xrepo_macro_f1_diff": {"ci95": [lo, hi]}}


def _gate(per_label):
    return {"applied": {"per_label": {lb: {"tau": t, "applied": n, "coverage": n / 100, "precision": p}
                                      for lb, (t, n, p) in per_label.items()}}}


def test_judge_criteria_met_and_missed():
    xrepo = {"M1": 0.85, "B1": 0.77, "B2_512-192": 0.61, "B2_1024-256": 0.63}
    paired = {s: _paired(0.01, 0.2) for s in tr.PAIRED}
    gate = _gate({B: (0.6, 30, 0.9), F: (1.01, 0, None), Q: (0.7, 10, 0.8)})
    c = tr.judge_criteria(xrepo, paired, 0.0999, gate, 450.0)
    assert c["1"]["status"] == "MET" and c["1"]["comparisons"]["B1"]["diff"] == pytest.approx(0.08)
    assert c["2"]["status"] == "MET"
    # bug: precision exactly 0.90 counts; feature: no val threshold; question: precision below 0.90.
    assert [c["3"]["per_label"][lb]["met"] for lb in (B, F, Q)] == [True, False, False]
    assert c["3"]["k"] == 1 and c["3"]["status"] == "met for 1 of 3"
    assert c["4"]["local_p95_within_bound"] is True and c["4"]["status"].startswith("PENDING")


def test_judge_criteria_not_met():
    xrepo = {"M1": 0.70, "B1": 0.77, "B2_512-192": 0.61, "B2_1024-256": 0.63}
    paired = {s: _paired(-0.1, 0.1) for s in tr.PAIRED}
    gate = _gate({B: (1.01, 0, None), F: (1.01, 0, None), Q: (1.01, 0, None)})
    c = tr.judge_criteria(xrepo, paired, 0.1001, gate, 1200.0)
    assert c["1"]["status"] == "NOT MET" and c["1"]["comparisons"]["B1"]["m1_higher"] is False
    assert c["2"]["status"] == "NOT MET"
    assert c["3"]["status"] == "met for 0 of 3"
    assert c["4"]["local_p95_within_bound"] is False


def test_md_table():
    assert tr.md_table(["a", "b"], [[1, "x"], [2.5, None]]) == "| a | b |\n|---|---|\n| 1 | x |\n| 2.5 | None |"
