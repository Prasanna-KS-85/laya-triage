"""results/phase3/posthoc_analysis.py helpers on synthetic inputs (no data files, no model)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "results" / "phase3"))

import gating
import metrics
import posthoc_analysis as ph

B, F, Q = "bug", "feature", "question"


def test_independent_confusion_matches_metrics():
    rs = np.random.RandomState(0)
    gold = [(B, F, Q)[i] for i in rs.randint(0, 3, 200)]
    pred = [(B, F, Q)[i] for i in rs.randint(0, 3, 200)]
    assert ph.independent_confusion(gold, pred) == metrics.confusion_counts(gold, pred)
    assert ph.independent_confusion([B, F, F], [B, F, Q]) == [[1, 0, 0], [0, 1, 1], [0, 0, 0]]


def test_n_correct():
    rows = [{"gold": B, "pred": B, "answer_confidence": 0.9}, {"gold": F, "pred": B, "answer_confidence": 0.8},
            {"gold": Q, "pred": Q, "answer_confidence": 0.7}, {"gold": F, "pred": F, "answer_confidence": 0.2}]
    a = gating.apply_thresholds(rows, {B: 0.5, F: 0.5, Q: 0.5})
    assert ph.n_correct(a) == {"all": 2, B: 1, F: 0, Q: 1}


def test_max_similarity_to_train():
    train = ["alpha beta gamma", "delta epsilon", "alpha beta gamma delta", "zeta eta theta"]
    query = ["alpha beta gamma", "iota kappa", "zeta eta theta"]
    s = ph.max_similarity_to_train(train, query, (1, 2), 1, True)
    assert s[0] == pytest.approx(1.0) and s[2] == pytest.approx(1.0)
    assert s[1] == 0.0  # no shared vocabulary with train
    s2 = ph.max_similarity_to_train(train, ["alpha delta"], (1, 2), 1, False)
    assert 0.0 < s2[0] < 1.0


def test_similarity_summary():
    out = ph.similarity_summary([0.1, 0.5, 0.85, 0.95])
    assert out["n"] == 4 and out["mean"] == pytest.approx(0.6) and out["median"] == pytest.approx(0.675)
    assert out["share_above_0.8"] == 0.5 and out["share_above_0.9"] == 0.25
    assert out["p90"] == pytest.approx(np.percentile([0.1, 0.5, 0.85, 0.95], 90))


def test_tertiles():
    ref = np.arange(1, 10) / 10  # 0.1 .. 0.9
    cuts = ph.tertile_cuts(ref)
    assert cuts == pytest.approx([np.percentile(ref, 100 / 3), np.percentile(ref, 200 / 3)])
    assert list(ph.tertile_of([cuts[0], cuts[0] + 1e-9, cuts[1], 0.95], cuts)) == ["low", "mid", "mid", "high"]


def test_accuracy_by_tertile():
    gold, pred = [B, B, F, F, Q], [B, F, F, F, B]
    out = ph.accuracy_by_tertile(gold, pred, ["low", "low", "mid", "high", "high"])
    assert out == {"low": {"n": 2, "accuracy": 0.5}, "mid": {"n": 1, "accuracy": 1.0}, "high": {"n": 2, "accuracy": 0.5}}
    assert ph.accuracy_by_tertile([B], [B], ["low"])["mid"] == {"n": 0, "accuracy": None}


def test_global_cutoff_uses_gating_rule():
    # 40 correct issues at high confidence, 40 wrong at low confidence: the smallest qualifying τ admits a few
    # wrong issues while the bootstrap lower bound stays >= 0.90.
    conf = np.r_[np.linspace(0.8, 0.99, 40), np.linspace(0.4, 0.6, 40)]
    correct = np.r_[np.ones(40), np.zeros(40)]
    out = ph.global_cutoff(conf, correct)
    fit = gating.select_threshold(conf, correct, ph.GLOBAL_SEED)
    row = next(t for t in fit["table"] if t["tau"] == fit["tau_lb"])
    assert out["tau"] == fit["tau_lb"] and out["tau"] < 0.8
    assert (out["support"], out["precision"], out["lower_bound"]) == (row["support"], row["precision"], row["lower_bound"])
    assert out["coverage"] == row["support"] / 80 and out["lower_bound"] >= 0.90
    assert all(t["lower_bound"] < 0.90 for t in fit["table"] if t["tau"] < out["tau"])


def test_global_cutoff_none():
    conf = np.linspace(0.4, 0.99, 50)
    correct = np.tile([1.0, 0.0], 25)  # 50% precision everywhere
    out = ph.global_cutoff(conf, correct)
    assert out["tau"] is None and out["n_candidates_with_support"] == 31


def test_max_coverage_at_precision():
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 0, 1]
    # cumulative precision by threshold: 1.0 (0.25), 1.0 (0.5), 0.667 (0.75), 0.75 (1.0)
    p = ph.max_coverage_at_precision(conf, correct, 0.75)
    assert p == {"threshold": 0.6, "coverage": 1.0, "precision": 0.75}
    assert ph.max_coverage_at_precision(conf, correct, 0.9)["coverage"] == 0.5
    assert ph.max_coverage_at_precision([0.5, 0.4], [0, 0], 0.5) is None
