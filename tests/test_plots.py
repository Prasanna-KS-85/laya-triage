"""eval/plots.py on synthetic arrays (Agg backend; no data files, no model)."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import matplotlib
import plots
from laya.common import ece_score

B, F, Q = "bug", "feature", "question"
REPOS = ("o/a", "o/b")


def synthetic(seed=0, n=120):
    rs = np.random.RandomState(seed)
    gold = [(B, F, Q)[i % 3] for i in range(n)]
    pred = [g if rs.rand() < 0.7 else (B, F, Q)[rs.randint(3)] for g in gold]
    conf = rs.uniform(0.34, 1.0, size=n)
    correct = np.array([g == p for g, p in zip(gold, pred)], dtype=float)
    repos = [REPOS[i % 2] for i in range(n)]
    return gold, pred, conf, correct, repos


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", metadata=plots.PNG_METADATA)
    matplotlib.pyplot.close(fig)
    return buf.getvalue()


def test_agg_backend():
    assert matplotlib.get_backend().lower() == "agg"


def test_reliability_bins_hand_computed_and_match_laya_ece():
    conf = np.array([0.0, 1 / 15, 0.5, 0.52, 1.0])
    correct = np.array([0, 1, 1, 0, 1], dtype=float)
    b = plots.reliability_bins(conf, correct, bins=15)
    # 0 and 1/15 both fall in the first, closed bin [0, 1/15]; 0.5 and 0.52 in (7/15, 8/15]; 1.0 in the last.
    assert b["count"].tolist() == [2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1]
    assert b["accuracy"][0] == 0.5 and b["accuracy"][7] == 0.5 and b["accuracy"][14] == 1.0
    assert b["mean_conf"][7] == pytest.approx(0.51) and np.isnan(b["mean_conf"][1])
    ok = b["count"] > 0
    ece = float((b["count"][ok] / conf.size * np.abs(b["mean_conf"][ok] - b["accuracy"][ok])).sum())
    assert ece == pytest.approx(ece_score(conf, correct, bins=15))


def test_normalised_confusion_hand_computed():
    m = plots.normalised_confusion([B, B, F, F, Q], [B, F, F, F, B])
    assert m.tolist() == [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


def test_figures_have_expected_structure():
    gold, pred, conf, correct, repos = synthetic()
    fig = plots.reliability_diagram({"T=1": (conf, correct), "fitted T": (conf ** 2, correct)})
    assert len(fig.axes) == 1 and len(fig.axes[0].get_lines()) == 3  # diagonal + 2 series
    fig = plots.coverage_precision_plot({"M1": (conf, correct)}, {"thresholds": (0.4, 0.9)})
    assert len(fig.axes) == 1
    fig = plots.confusion_matrices({"B1": (gold, pred), "B2": (gold, gold)})
    assert [ax.get_title() for ax in fig.axes] == ["B1", "B2"]
    fig = plots.per_repo_f1_bars({"B1": (gold, pred, repos), "B2": (gold, gold, repos)})
    assert len(fig.axes[0].patches) == 4  # 2 systems x 2 repos
    assert [p.get_height() for p in fig.axes[0].patches[2:]] == [1.0, 1.0]
    matplotlib.pyplot.close("all")


def test_png_output_is_deterministic(tmp_path):
    gold, pred, conf, correct, repos = synthetic(seed=1)
    makers = [
        lambda: plots.reliability_diagram({"x": (conf, correct)}),
        lambda: plots.coverage_precision_plot({"x": (conf, correct)}, {"op": (0.5, 0.95)}),
        lambda: plots.confusion_matrices({"a": (gold, pred), "b": (gold, gold)}),
        lambda: plots.per_repo_f1_bars({"a": (gold, pred, repos)}),
    ]
    for make in makers:
        assert png_bytes(make()) == png_bytes(make())
    path = tmp_path / "r.png"
    plots.save(plots.reliability_diagram({"x": (conf, correct)}), path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_per_repo_bars_reject_mismatched_repos():
    gold, pred, _, _, repos = synthetic()
    with pytest.raises(ValueError, match="repos differ"):
        plots.per_repo_f1_bars({"a": (gold, pred, repos), "b": (gold[:2], pred[:2], ["o/c", "o/c"])})
