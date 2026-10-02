"""eval/gating.py on hand-built fixtures (PROJECT_SPEC.md §10.4). No data files, no model."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import gating as g

B, F, Q = "bug", "feature", "question"
OTHER = {B: F, F: Q, Q: B}


def rows_for(pred, confs, correct):
    """Rows predicted `pred`; a wrong row gets another label as gold."""
    return [{"gold": pred if ok else OTHER[pred], "pred": pred, "answer_confidence": c}
            for c, ok in zip(confs, correct)]


def test_bootstrap_lower_bound_extremes_and_seed():
    assert g.bootstrap_lower_bound([1.0] * 20, seed=42) == 1.0
    assert g.bootstrap_lower_bound([0.0] * 20, seed=42) == 0.0
    x = [1.0] * 17 + [0.0] * 3
    rs = np.random.RandomState(7)
    idx = rs.randint(0, 20, size=(2000, 20))
    expected = float(np.percentile(np.array(x)[idx].mean(axis=1), 5))
    assert g.bootstrap_lower_bound(x, seed=7) == expected


def test_perfect_separation():
    # 25 correct at 0.75..0.99 (distinct), 10 wrong all at 0.40.
    confs = [round(0.75 + 0.01 * i, 2) for i in range(25)] + [0.40] * 10
    res = g.select_threshold(confs, [1] * 25 + [0] * 10, seed=42)
    assert res["tau_lb"] == 0.75 and res["tau_point"] == 0.75
    # Candidates with |S| >= 20: 0.40 (35 issues) and 0.75..0.80 (25..20 issues).
    assert [t["tau"] for t in res["table"]] == [0.40, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80]
    first, second = res["table"][0], res["table"][1]
    assert first["support"] == 35 and first["precision"] == 25 / 35 and first["lower_bound"] < 0.90
    assert second["support"] == 25 and second["precision"] == 1.0 and second["lower_bound"] == 1.0
    assert res["n_predicted"] == 35 and res["n_candidates"] == 26


def test_never_reaching_target_gives_never():
    # Every 4th issue (from the top) is wrong: precision <= 18/23 for every |S| >= 20.
    confs = [1.0 - 0.01 * i for i in range(40)]
    correct = [(i + 1) % 4 != 0 for i in range(40)]
    rows = rows_for(F, confs, correct)
    res = g.fit_thresholds(rows)[F]
    assert res["tau_lb"] == g.NEVER == 1.01
    assert res["tau_point"] is None
    assert max(t["precision"] for t in res["table"]) < 0.90
    applied = g.apply_thresholds(rows, {B: 0.0, F: res["tau_lb"], Q: 0.0})
    assert applied["per_label"][F]["applied"] == 0 and applied["applied"] == 0


@pytest.mark.parametrize(("n", "expected"), [(19, g.NEVER), (20, 0.81)])
def test_support_19_vs_20(n, expected):
    confs = [round(0.81 + 0.01 * i, 2) for i in range(n)]
    res = g.fit_thresholds(rows_for(Q, confs, [1] * n))[Q]
    assert res["tau_lb"] == expected
    if n == 19:
        assert res["table"] == [] and res["tau_point"] is None
    else:
        assert res["table"] == [{"tau": 0.81, "support": 20, "precision": 1.0, "lower_bound": 1.0}]
        assert res["tau_point"] == 0.81


def test_determinism_and_seed_per_label():
    confs = [round(0.5 + 0.002 * i, 3) for i in range(60)]
    correct = [i % 7 != 0 for i in range(60)]
    rows = rows_for(B, confs, correct) + rows_for(F, confs, correct) + rows_for(Q, confs, correct)
    a, b = g.fit_thresholds(rows), g.fit_thresholds(list(rows))
    assert a == b
    assert [a[label]["seed"] for label in (B, F, Q)] == [42, 43, 44]
    # Same data per label, different seeds: the bound at the lowest candidate is the documented one.
    for label, seed in ((B, 42), (F, 43), (Q, 44)):
        assert a[label]["table"][0]["lower_bound"] == g.bootstrap_lower_bound(np.array(correct, float), seed)


def test_non_monotone_precision_takes_smallest_qualifying_tau():
    # Top 20 (0.99..0.80): the 3 highest are wrong -> 17/20 = 0.85 there.
    # Middle 180 (0.790..0.611): all correct.  Bottom 40: wrong, all at 0.30.
    top = [round(0.99 - 0.01 * i, 2) for i in range(20)]
    mid = [round(0.79 - 0.001 * i, 3) for i in range(180)]
    confs = top + mid + [0.30] * 40
    correct = [i >= 3 for i in range(20)] + [True] * 180 + [False] * 40
    res = g.select_threshold(confs, correct, seed=42)
    by_tau = {t["tau"]: t for t in res["table"]}
    assert by_tau[0.30]["precision"] == 197 / 240 and by_tau[0.30]["lower_bound"] < 0.90
    assert by_tau[0.611]["precision"] == 197 / 200 and by_tau[0.611]["lower_bound"] >= 0.90
    assert by_tau[0.80]["precision"] == 17 / 20 and by_tau[0.80]["lower_bound"] < 0.90  # higher tau fails
    assert res["tau_lb"] == 0.611 and res["tau_point"] == 0.611


def test_apply_thresholds_hand_computed():
    rows = (rows_for(B, [0.9, 0.8, 0.6], [1, 0, 1]) + rows_for(F, [0.95, 0.7], [1, 1])
            + rows_for(Q, [0.5, 0.99], [0, 1]))
    res = g.apply_thresholds(rows, {B: 0.75, F: 0.7, Q: 0.9})
    # applied: bug 0.9 (ok), 0.8 (wrong); feature 0.95, 0.7 (ok); question 0.99 (ok) -> 5 of 7, 4 correct
    assert (res["n"], res["applied"], res["escalated"]) == (7, 5, 2)
    assert res["coverage"] == 5 / 7 and res["precision"] == 4 / 5
    assert res["per_label"][B] == {"tau": 0.75, "predicted": 3, "applied": 2, "coverage": 2 / 7, "precision": 0.5}
    assert res["per_label"][F]["coverage"] == 2 / 7 and res["per_label"][F]["precision"] == 1.0
    assert res["per_label"][Q]["coverage"] == 1 / 7 and res["per_label"][Q]["precision"] == 1.0
    assert sum(v["coverage"] for v in res["per_label"].values()) == pytest.approx(res["coverage"])


def test_all_never_applies_nothing():
    rows = rows_for(B, [0.99, 1.0], [1, 1]) + rows_for(Q, [1.0], [0])
    res = g.apply_thresholds(rows, dict.fromkeys((B, F, Q), g.NEVER))
    assert (res["applied"], res["escalated"], res["coverage"], res["precision"]) == (0, 3, 0.0, None)
    assert all(v["applied"] == 0 and v["precision"] is None for v in res["per_label"].values())


def test_apply_thresholds_requires_every_label():
    with pytest.raises(ValueError, match="keys"):
        g.apply_thresholds(rows_for(B, [0.9], [1]), {B: 0.5})


def test_coverage_precision_curve_hand_computed():
    th, cov, prec = g.coverage_precision_curve([0.8, 0.9, 0.5, 0.8], [0, 1, 1, 1])
    assert th.tolist() == [0.9, 0.8, 0.5]
    assert cov.tolist() == [0.25, 0.75, 1.0]
    assert prec.tolist() == pytest.approx([1.0, 2 / 3, 0.75])
