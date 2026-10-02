"""Figures for PROJECT_SPEC.md §10.3 (Agg backend, no display, deterministic output).

Every plot function takes plain arrays and returns a matplotlib Figure; `save` writes it as PNG.
Metrics come from eval/metrics.py and eval/gating.py, never re-implemented here:
- reliability_diagram: accuracy vs mean answer_confidence in the 15 equal-width bins of
  laya.common.ece_score (first bin [0, 1/15], then (lo, hi]); one or more series, e.g. before and
  after calibration (T=1 vs fitted T);
- coverage_precision_plot: gating.coverage_precision_curve per series, with marked operating points;
- confusion_matrices: confusion counts normalised by gold row, systems side by side;
- per_repo_f1_bars: metrics.per_repo_macro_f1, systems as grouped bars per repo.
"""

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")

import gating
import matplotlib.pyplot as plt
import metrics
import numpy as np

from laya_triage.questions import LABELS

BINS = metrics.ECE_BINS
DPI = 120
# PNG metadata without the matplotlib version, so the bytes depend only on the data.
PNG_METADATA = {"Software": None}


def reliability_bins(conf, correct, bins=BINS):
    """Per bin (laya.common.ece_score edges): count, mean confidence, accuracy (NaN when empty)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    count, mean_conf, acc = np.zeros(bins, dtype=int), np.full(bins, np.nan), np.full(bins, np.nan)
    for i, (lo, hi) in enumerate(pairwise(edges)):
        sel = (conf >= lo if i == 0 else conf > lo) & (conf <= hi)
        count[i] = int(sel.sum())
        if count[i]:
            mean_conf[i], acc[i] = conf[sel].mean(), correct[sel].mean()
    return {"edges": edges, "count": count, "mean_conf": mean_conf, "accuracy": acc}


def normalised_confusion(gold, pred):
    """Confusion counts (rows gold, cols pred, LABELS order) divided by each gold row's total."""
    m = np.array(metrics.confusion_counts(gold, pred), dtype=float)
    totals = m.sum(axis=1, keepdims=True)
    return np.divide(m, totals, out=np.zeros_like(m), where=totals > 0)


def reliability_diagram(series, bins=BINS, title="Reliability (val)"):
    """series: {name: (answer_confidence, correct)}; ECE (laya.common.ece_score) in the legend."""
    from laya.common import ece_score

    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect calibration")
    for name, (conf, correct) in series.items():
        b = reliability_bins(conf, correct, bins)
        ok = b["count"] > 0
        e = ece_score(np.asarray(conf, float), np.asarray(correct, float), bins=bins)
        ax.plot(b["mean_conf"][ok], b["accuracy"][ok], marker="o", label=f"{name} (ECE {e:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="mean answer_confidence in bin", ylabel="accuracy in bin",
           title=f"{title}, {bins} bins")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def coverage_precision_plot(series, operating_points=None, target=gating.TARGET_PRECISION,
                            title="Coverage vs precision (val)"):
    """series: {name: (answer_confidence, correct)} swept with one global threshold.
    operating_points: {label: (coverage, precision)}, e.g. from gating.apply_thresholds."""
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=DPI)
    for name, (conf, correct) in series.items():
        _, cov, prec = gating.coverage_precision_curve(conf, correct)
        ax.step(cov, prec, where="post", label=name)
    for label, (cov, prec) in (operating_points or {}).items():
        if prec is not None:
            ax.plot([cov], [prec], marker="*", markersize=12, linestyle="none", label=f"{label}: {cov:.0%} at {prec:.0%}")
    ax.axhline(target, color="0.5", linestyle=":", linewidth=1, label=f"target precision {target:.0%}")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="coverage (share of all issues auto-applied)",
           ylabel="precision of auto-applied", title=title)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


def confusion_matrices(systems, title="Confusion, normalised by gold (val)"):
    """systems: {name: (gold, pred)}, drawn side by side."""
    fig, axes = plt.subplots(1, len(systems), figsize=(4.2 * len(systems), 4), dpi=DPI, squeeze=False)
    for ax, (name, (gold, pred)) in zip(axes[0], systems.items()):
        m = normalised_confusion(gold, pred)
        ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
        for r in range(len(LABELS)):
            for c in range(len(LABELS)):
                ax.text(c, r, f"{m[r, c]:.2f}", ha="center", va="center", color="white" if m[r, c] > 0.6 else "black")
        ax.set(xticks=range(len(LABELS)), yticks=range(len(LABELS)), xticklabels=LABELS, yticklabels=LABELS,
               xlabel="predicted", ylabel="gold", title=name)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def per_repo_f1_bars(systems, title="Per-repo macro-F1 (val)"):
    """systems: {name: (gold, pred, repos)}; one group of bars per repo, cross-repo mean in the legend."""
    scores = {name: metrics.per_repo_macro_f1(g, p, r) for name, (g, p, r) in systems.items()}
    repos = sorted(next(iter(scores.values())))
    width = 0.8 / len(scores)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=DPI)
    x = np.arange(len(repos))
    for k, (name, s) in enumerate(scores.items()):
        if sorted(s) != repos:
            raise ValueError(f"{name}: repos differ from the other systems")
        ax.bar(x + (k - (len(scores) - 1) / 2) * width, [s[r] for r in repos], width,
               label=f"{name} (cross-repo {np.mean([s[r] for r in repos]):.3f})")
    ax.set(xticks=x, ylim=(0, 1.15), yticks=np.linspace(0, 1, 6), ylabel="macro-F1", title=title)
    ax.set_xticklabels([r.split("/")[-1] for r in repos])
    ax.legend(loc="upper center", ncol=len(scores), fontsize=8)
    fig.tight_layout()
    return fig


def save(fig, path):
    fig.savefig(path, format="png", metadata=PNG_METADATA)
    plt.close(fig)
