"""Evaluation metrics for baselines and M1 (PROJECT_SPEC.md §10.2). Pure functions, shared by
eval/baselines.py (Phase 2) and eval/run_eval.py (Phase 3).

Inputs are parallel sequences, one entry per issue:
- gold, pred: labels from LABELS
- probs: dicts keyed by exactly LABELS, each summing to 1 within PROB_SUM_TOL
- repos: repository names, e.g. "facebook/react"

F1 averages always run over all of LABELS (a class that is never predicted has F1 0), and a
precision or recall with an empty denominator is 0.
"""

from collections import defaultdict

import numpy as np
from laya.common import ece_score

from laya_triage.questions import LABELS

ECE_BINS = 15
PROB_SUM_TOL = 5e-4  # laya rounds each probability to 4 decimals, so 3 of them can sum to 1 ± 1.5e-4


def _check(gold, pred=None, probs=None, repos=None):
    n = len(gold)
    if n == 0:
        raise ValueError("no issues")
    for name, seq in (("pred", pred), ("probs", probs), ("repos", repos)):
        if seq is not None and len(seq) != n:
            raise ValueError(f"{name} has {len(seq)} entries, gold has {n}")
    for name, seq in (("gold", gold), ("pred", pred)):
        if seq is not None and not set(seq) <= set(LABELS):
            raise ValueError(f"{name} has labels outside {LABELS}: {sorted(set(seq) - set(LABELS))}")
    if probs is not None:
        for p in probs:
            if tuple(sorted(p)) != tuple(sorted(LABELS)):
                raise ValueError(f"probability keys {sorted(p)} != {LABELS}")
            if abs(sum(p.values()) - 1.0) > PROB_SUM_TOL:
                raise ValueError(f"probabilities sum to {sum(p.values())}")


def accuracy(gold, pred):
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def confusion_counts(gold, pred):
    """Rows = gold, columns = predicted, both in LABELS order."""
    _check(gold, pred)
    idx = {c: i for i, c in enumerate(LABELS)}
    m = [[0] * len(LABELS) for _ in LABELS]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    return m


def per_class_prf(gold, pred):
    """{label: {"precision", "recall", "f1", "support"}} for every label in LABELS."""
    _check(gold, pred)
    out = {}
    for c in LABELS:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(gold, pred):
    """Unweighted mean of per-class F1 over LABELS."""
    return sum(v["f1"] for v in per_class_prf(gold, pred).values()) / len(LABELS)


def pooled_macro_f1(gold, pred):
    """Macro-F1 over all issues at once, ignoring repos."""
    return macro_f1(gold, pred)


def per_repo_macro_f1(gold, pred, repos):
    """{repo: macro-F1 on that repo's issues}, repos in sorted order."""
    _check(gold, pred, repos=repos)
    by_repo = defaultdict(list)
    for i, r in enumerate(repos):
        by_repo[r].append(i)
    return {r: macro_f1([gold[i] for i in ix], [pred[i] for i in ix])
            for r, ix in sorted(by_repo.items())}


def cross_repo_macro_f1(gold, pred, repos):
    """Headline (§10.2): arithmetic mean of the per-repo macro-F1 scores."""
    scores = per_repo_macro_f1(gold, pred, repos)
    return sum(scores.values()) / len(scores)


def brier(gold, probs):
    """Multi-class Brier score: mean over issues of sum_k (p_k - 1[gold == k])^2, range [0, 2]."""
    _check(gold, probs=probs)
    return sum(sum((p[c] - (g == c)) ** 2 for c in LABELS) for g, p in zip(gold, probs)) / len(gold)


def ece(gold, pred, probs, bins=ECE_BINS):
    """ECE on answer_confidence = probability of the predicted label (laya.common.ece_score)."""
    _check(gold, pred, probs)
    conf = np.array([p[y] for p, y in zip(probs, pred)], dtype=float)
    correct = np.array([g == y for g, y in zip(gold, pred)], dtype=float)
    return ece_score(conf, correct, bins=bins)


def predicted_counts(pred):
    return {c: sum(p == c for p in pred) for c in LABELS}


def summarize(gold, pred, probs, repos):
    """All §10.2 classification and calibration metrics for one system on one split."""
    _check(gold, pred, probs, repos)
    return {
        "n": len(gold),
        "accuracy": accuracy(gold, pred),
        "cross_repo_macro_f1": cross_repo_macro_f1(gold, pred, repos),
        "pooled_macro_f1": pooled_macro_f1(gold, pred),
        "per_class": per_class_prf(gold, pred),
        "per_repo_macro_f1": per_repo_macro_f1(gold, pred, repos),
        "ece": ece(gold, pred, probs),
        "brier": brier(gold, probs),
        "predicted_counts": predicted_counts(pred),
        "confusion": {"labels": list(LABELS), "rows_gold_cols_pred": confusion_counts(gold, pred)},
    }
