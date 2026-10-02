import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import metrics as m

B, F, Q = "bug", "feature", "question"

# 6 issues. Confusion (rows gold, cols pred): bug [2,1,0], feature [0,1,1], question [1,0,0].
# bug: tp 2 fp 1 fn 1 -> P = R = F1 = 2/3; feature: tp 1 fp 1 fn 1 -> 1/2; question: tp 0 -> 0.
GOLD = [B, B, B, F, F, Q]
PRED = [B, B, F, F, Q, B]


def onehot(label):
    return {c: float(c == label) for c in (B, F, Q)}


def test_accuracy():
    assert m.accuracy(GOLD, PRED) == pytest.approx(3 / 6)


def test_confusion_counts():
    assert m.confusion_counts(GOLD, PRED) == [[2, 1, 0], [0, 1, 1], [1, 0, 0]]


def test_per_class_prf():
    got = m.per_class_prf(GOLD, PRED)
    assert list(got) == [B, F, Q]
    assert got[B] == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3})
    assert got[F] == pytest.approx({"precision": 1 / 2, "recall": 1 / 2, "f1": 1 / 2, "support": 2})
    assert got[Q] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}


def test_macro_f1_averages_over_all_labels():
    assert m.macro_f1(GOLD, PRED) == pytest.approx((2 / 3 + 1 / 2 + 0) / 3)  # 7/18
    # Never-predicted, never-gold class still counts as F1 0 in the average.
    assert m.macro_f1([B, B], [B, B]) == pytest.approx(1 / 3)


# Repo "a": 3 issues, all correct -> macro-F1 1.
# Repo "b": 6 issues, gold 2 per class, all predicted bug -> bug P 1/3 R 1 F1 1/2; others 0 -> 1/6.
# Cross-repo = (1 + 1/6) / 2 = 7/12.
# Pooled: bug tp 3 fp 4 fn 0 -> F1 0.6; feature tp 1 fp 0 fn 2 -> F1 0.5; question same 0.5 -> 8/15.
REPO_GOLD = [B, F, Q] + [B, B, F, F, Q, Q]
REPO_PRED = [B, F, Q] + [B] * 6
REPOS = ["a"] * 3 + ["b"] * 6


def test_per_repo_macro_f1():
    assert m.per_repo_macro_f1(REPO_GOLD, REPO_PRED, REPOS) == pytest.approx({"a": 1.0, "b": 1 / 6})


def test_cross_repo_differs_from_pooled():
    cross = m.cross_repo_macro_f1(REPO_GOLD, REPO_PRED, REPOS)
    pooled = m.pooled_macro_f1(REPO_GOLD, REPO_PRED)
    assert cross == pytest.approx(7 / 12)
    assert pooled == pytest.approx(8 / 15)
    assert cross != pytest.approx(pooled)


def test_cross_repo_is_unweighted_by_repo_size():
    # Repo "b" has twice the issues of "a" but the same weight in the mean.
    weighted = (3 * 1.0 + 6 * (1 / 6)) / 9
    assert m.cross_repo_macro_f1(REPO_GOLD, REPO_PRED, REPOS) != pytest.approx(weighted)


def test_brier():
    # (0.3^2 + 0.2^2 + 0.1^2) = 0.14 and (0.5^2 + 0.5^2 + 0) = 0.5 -> mean 0.32.
    probs = [{B: 0.7, F: 0.2, Q: 0.1}, {B: 0.5, F: 0.5, Q: 0.0}]
    assert m.brier([B, F], probs) == pytest.approx(0.32)
    assert m.brier([B], [onehot(B)]) == 0.0
    assert m.brier([B], [onehot(F)]) == pytest.approx(2.0)


# answer_confidence 0.9 (right), 0.9 (wrong), 0.55 (right), 0.95 (right); 15 bins of width 1/15.
# Bin (0.867, 0.933]: weight 2/4, |0.9 - 0.5| = 0.4 -> 0.2
# Bin (0.533, 0.600]: weight 1/4, |0.55 - 1| = 0.45 -> 0.1125
# Bin (0.933, 1.000]: weight 1/4, |0.95 - 1| = 0.05 -> 0.0125
# ECE = 0.325
ECE_GOLD = [B, F, Q, F]
ECE_PRED = [B, B, Q, F]
ECE_PROBS = [
    {B: 0.9, F: 0.05, Q: 0.05},
    {B: 0.9, F: 0.1, Q: 0.0},
    {B: 0.25, F: 0.2, Q: 0.55},
    {B: 0.05, F: 0.95, Q: 0.0},
]


def test_ece():
    assert m.ece(ECE_GOLD, ECE_PRED, ECE_PROBS) == pytest.approx(0.325)


def test_ece_uses_probability_of_predicted_label():
    # Confidence 1.0 on the predicted label, always right -> 0; always wrong -> 1.
    assert m.ece([B, F], [B, F], [onehot(B), onehot(F)]) == pytest.approx(0.0)
    assert m.ece([F, B], [B, F], [onehot(B), onehot(F)]) == pytest.approx(1.0)


def test_predicted_counts():
    assert m.predicted_counts(PRED) == {B: 3, F: 2, Q: 1}


def test_summarize():
    s = m.summarize(REPO_GOLD, REPO_PRED, [onehot(p) for p in REPO_PRED], REPOS)
    assert s["n"] == 9
    assert s["accuracy"] == pytest.approx(5 / 9)
    assert s["cross_repo_macro_f1"] == pytest.approx(7 / 12)
    assert s["pooled_macro_f1"] == pytest.approx(8 / 15)
    assert s["per_repo_macro_f1"] == pytest.approx({"a": 1.0, "b": 1 / 6})
    assert s["predicted_counts"] == {B: 7, F: 1, Q: 1}
    assert s["confusion"] == {"labels": [B, F, Q], "rows_gold_cols_pred": [[3, 0, 0], [2, 1, 0], [2, 0, 1]]}
    assert s["ece"] == pytest.approx(4 / 9)  # all confidence 1.0, accuracy 5/9
    assert s["brier"] == pytest.approx(2 * 4 / 9)  # each wrong one-hot costs 2


@pytest.mark.parametrize("args", [
    ([B, F], [B]),  # length mismatch
    ([B, "docs"], [B, B]),  # unknown gold label
    ([], []),  # empty
])
def test_rejects_bad_label_inputs(args):
    with pytest.raises(ValueError):
        m.accuracy(*args)


def test_rejects_bad_probabilities():
    with pytest.raises(ValueError):
        m.brier([B], [{B: 1.0, F: 0.0}])  # missing key
    with pytest.raises(ValueError):
        m.brier([B], [{B: 0.7, F: 0.7, Q: 0.0}])  # does not sum to 1
    with pytest.raises(ValueError):
        m.per_repo_macro_f1([B, B], [B, B], ["a"])  # repo length mismatch
