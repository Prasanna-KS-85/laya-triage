"""Per-label confidence thresholds for auto-applying labels (PROJECT_SPEC.md §10.4). Pure functions.

Inputs are rows: mappings with "gold", "pred" (labels from LABELS) and "answer_confidence" (the
probability of the predicted label), one per val issue, from CPU fp32 inference (§10.4 step 1).
`taus` maps every label in LABELS to its threshold.

Definitions:
- An issue is auto-applied when its predicted label l has answer_confidence >= tau_l; otherwise it is
  escalated. tau_l = NEVER (1.01) means label l is never auto-applied.
- coverage = share of ALL issues that are auto-applied. Per label: issues predicted l and
  auto-applied / all issues; overall: all auto-applied issues / all issues (the per-label coverages
  sum to the overall coverage).
- precision = share of auto-applied issues whose prediction is correct, per label and overall; None
  when nothing is auto-applied.

Threshold selection for label l (§10.4 steps 2-6):
- candidates = the distinct answer_confidence values among issues predicted l;
- S(tau) = issues predicted l with answer_confidence >= tau; only candidates with |S(tau)| >= 20 count;
- lower bound = 5th percentile (numpy linear interpolation) of precision over 2,000 resamples of
  S(tau) with replacement. Every candidate's resamples come from a fresh
  numpy.random.RandomState(42 + LABELS.index(l)), so a candidate's bound does not depend on which
  other candidates were evaluated;
- tau_lb = the smallest candidate whose lower bound is >= 0.90, else NEVER (§10.4 step 5);
- tau_point = the smallest candidate with precision >= 0.90 on S(tau), else None (comparison only).
"""

import numpy as np

from laya_triage.questions import LABELS

TARGET_PRECISION = 0.90
MIN_SUPPORT = 20
N_BOOTSTRAP = 2000
LOWER_PERCENTILE = 5
SEED = 42
NEVER = 1.01
# Precisions are ratios of small integers and the percentile interpolates between them, so a bound
# that is 0.90 in exact arithmetic can come out a few ulps below it; compare with this slack.
FLOAT_SLACK = 1e-12


def _label_arrays(rows, label):
    """answer_confidence and correctness of the issues predicted `label`, as float arrays."""
    sel = [r for r in rows if r["pred"] == label]
    conf = np.array([float(r["answer_confidence"]) for r in sel], dtype=float)
    correct = np.array([r["gold"] == r["pred"] for r in sel], dtype=float)
    return conf, correct


def bootstrap_lower_bound(correct, seed, n_boot=N_BOOTSTRAP, percentile=LOWER_PERCENTILE):
    """`percentile`-th percentile of precision over `n_boot` resamples (with replacement) of `correct`."""
    correct = np.asarray(correct, dtype=float)
    if correct.size == 0:
        raise ValueError("empty support set")
    idx = np.random.RandomState(seed).randint(0, correct.size, size=(n_boot, correct.size))
    return float(np.percentile(correct[idx].mean(axis=1), percentile))


def select_threshold(conf, correct, seed):
    """Thresholds for one label from the confidences and correctness of the issues predicted it."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    table = []
    for tau in np.unique(conf):  # ascending, distinct
        in_s = conf >= tau
        support = int(in_s.sum())
        if support < MIN_SUPPORT:
            continue
        table.append({
            "tau": float(tau),
            "support": support,
            "precision": float(correct[in_s].mean()),
            "lower_bound": bootstrap_lower_bound(correct[in_s], seed),
        })
    lb_ok = [t["tau"] for t in table if t["lower_bound"] >= TARGET_PRECISION - FLOAT_SLACK]
    point_ok = [t["tau"] for t in table if t["precision"] >= TARGET_PRECISION - FLOAT_SLACK]
    return {
        "tau_lb": min(lb_ok) if lb_ok else NEVER,
        "tau_point": min(point_ok) if point_ok else None,
        "n_predicted": int(conf.size),
        "n_candidates": int(np.unique(conf).size),
        "seed": seed,
        "table": table,
    }


def fit_thresholds(rows):
    """{label: select_threshold(...)} for every label, seed 42 + label index."""
    return {label: select_threshold(*_label_arrays(rows, label), seed=SEED + i) for i, label in enumerate(LABELS)}


def apply_thresholds(rows, taus):
    """Coverage, precision and escalated count of auto-applying with `taus` (definitions above)."""
    if set(taus) != set(LABELS):
        raise ValueError(f"taus must have exactly the keys {LABELS}, got {sorted(taus)}")
    n = len(rows)
    if n == 0:
        raise ValueError("no issues")
    per_label, applied_all, correct_all = {}, 0, 0
    for label in LABELS:
        conf, correct = _label_arrays(rows, label)
        applied = conf >= taus[label]
        n_app, n_ok = int(applied.sum()), int(correct[applied].sum())
        per_label[label] = {
            "tau": taus[label],
            "predicted": int(conf.size),
            "applied": n_app,
            "coverage": n_app / n,
            "precision": n_ok / n_app if n_app else None,
        }
        applied_all += n_app
        correct_all += n_ok
    return {
        "n": n,
        "applied": applied_all,
        "escalated": n - applied_all,
        "coverage": applied_all / n,
        "precision": correct_all / applied_all if applied_all else None,
        "per_label": per_label,
    }


def coverage_precision_curve(confidences, correct):
    """One global threshold swept over the distinct confidences, highest first.

    Returns (thresholds, coverage, precision) as arrays: at thresholds[i], the issues with
    confidence >= thresholds[i] are applied; coverage is their share of all issues and precision
    the share of them that are correct.
    """
    conf = np.asarray(confidences, dtype=float)
    ok = np.asarray(correct, dtype=float)
    if conf.size == 0 or conf.shape != ok.shape:
        raise ValueError("confidences and correct must be non-empty and the same length")
    order = np.argsort(-conf, kind="stable")
    conf, ok = conf[order], ok[order]
    last = np.r_[conf[1:] != conf[:-1], True]  # last index of each run of equal confidences
    n_applied = np.arange(1, conf.size + 1)[last]
    return conf[last], n_applied / conf.size, np.cumsum(ok)[last] / n_applied
