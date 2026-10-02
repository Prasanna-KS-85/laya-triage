"""laya_triage.policy.decide / decide_error: every branch (PROJECT_SPEC.md FR-2, FR-3, §12.6; owner decision D1 and
the ruling on τ = 1.0: FR-2 literal, template (b) only for τ > 1.0)."""
from pathlib import Path

import pytest
import yaml

from laya_triage.classifier import TriageResult
from laya_triage.config import merge, validate
from laya_triage.policy import (
    TEMPLATE_BELOW_THRESHOLD,
    TEMPLATE_DISABLED,
    Decision,
    decide,
    decide_error,
    top2,
)

DEFAULT = Path(__file__).resolve().parents[1] / "config" / "triage.default.yml"


def cfg(thresholds=None, **behaviour):
    over = {"behaviour": {"comment_on_escalate": True, **behaviour}}
    if thresholds:
        over["gating"] = {"thresholds": thresholds}
    return validate(merge(yaml.safe_load(DEFAULT.read_text()), over))


def result(label="bug", probs=None):
    probs = probs or {"bug": 0.7, "feature": 0.2, "question": 0.1}
    return TriageResult(42, label, probs, probs[label], 12.0, "o/r", "a" * 40)


def test_apply_above_threshold():
    d = decide(result(), cfg({"bug": 0.6}))
    assert d == Decision("apply", ("type: bug",), None, "bug: confidence 0.7000 >= threshold 0.6000")


def test_exact_threshold_applies():
    d = decide(result(probs={"bug": 0.6033, "feature": 0.3, "question": 0.0967}), cfg({"bug": 0.6033}))
    assert d.action == "apply" and d.labels_to_add == ("type: bug",)


def test_just_below_threshold_escalates_with_template_a():
    d = decide(result(probs={"bug": 0.6032, "feature": 0.3, "question": 0.0968}), cfg({"bug": 0.6033}))
    assert d.action == "escalate" and d.labels_to_add == ("triage: needs-human",)
    assert d.comment == ("🤖 Laya Triage couldn't classify this issue confidently, so it's been marked for a maintainer.\n"
                         "Top guesses: `bug` (60%), `feature` (30%).")
    assert d.reason == "bug: confidence 0.6032 < threshold 0.6033"


def test_tau_1_0_below_uses_template_a():
    d = decide(result("feature", {"bug": 0.02, "feature": 0.97, "question": 0.01}), cfg({"feature": 1.0}))
    assert d.action == "escalate"
    assert d.comment == TEMPLATE_BELOW_THRESHOLD.format(label1="feature", p1=0.97, label2="bug", p2=0.02)


def test_tau_1_0_at_conf_1_0_applies():
    d = decide(result("feature", {"bug": 0.0, "feature": 1.0, "question": 0.0}), cfg({"feature": 1.0}))
    assert d.action == "apply" and d.labels_to_add == ("type: feature",)


def test_tau_1_01_never_applies_and_uses_template_b():
    d = decide(result("question", {"bug": 0.0, "feature": 0.0, "question": 1.0}), cfg({"question": 1.01}))
    assert d.action == "escalate" and d.labels_to_add == ("triage: needs-human",)
    assert d.comment == ("🤖 Laya Triage suggests `question` (100%), but automatic labelling is disabled for this issue "
                         "type in this repository's configuration, so it's been marked for a maintainer.\n"
                         "Second guess: `bug` (0%).")
    assert d.reason == "question: auto-labelling disabled (threshold 1.01)"
    assert d.comment == TEMPLATE_DISABLED.format(label1="question", p1=1.0, label2="bug", p2=0.0)


def test_comment_off_still_escalates():
    d = decide(result(probs={"bug": 0.5, "feature": 0.3, "question": 0.2}), cfg({"bug": 0.6}, comment_on_escalate=False))
    assert d == Decision("escalate", ("triage: needs-human",), None, "bug: confidence 0.5000 < threshold 0.6000")


def test_default_config_feature_question_never_apply():
    c = validate(yaml.safe_load(DEFAULT.read_text()))
    for label in ("feature", "question"):
        probs = {"bug": 0.0, "feature": 0.0, "question": 0.0, label: 1.0}
        d = decide(result(label, probs), c)
        assert d.action == "escalate" and d.comment is None  # D2: comments opt-in


def test_custom_label_names_are_used():
    raw = merge(yaml.safe_load(DEFAULT.read_text()), {"labels": {"bug": "kind/bug", "escalate": "needs-triage"},
                                                      "gating": {"thresholds": {"bug": 0.5}}})
    c = validate(raw)
    assert decide(result(), c).labels_to_add == ("kind/bug",)
    low = result(probs={"bug": 0.4, "feature": 0.35, "question": 0.25})
    assert decide(low, c).labels_to_add == ("needs-triage",)


def test_top2_ties_keep_label_order():
    assert top2({"bug": 0.4, "feature": 0.2, "question": 0.4}) == (("bug", 0.4), ("question", 0.4))
    assert top2({"bug": 0.2, "feature": 0.4, "question": 0.4}) == (("feature", 0.4), ("question", 0.4))


@pytest.mark.parametrize("escalate_on_error,expected", [
    (False, Decision("skip", (), None, "inference failed; no label")),
    (True, Decision("escalate", ("triage: needs-human",), None, "inference failed; escalated (escalate_on_error)")),
])
def test_decide_error(escalate_on_error, expected):
    assert decide_error(cfg(escalate_on_error=escalate_on_error), "inference failed") == expected


def test_comments_hold_only_fixed_text_labels_and_numbers():
    for tau in (0.99, 1.01):
        d = decide(result(probs={"bug": 0.5, "feature": 0.3, "question": 0.2}), cfg({"bug": tau}))
        filled = d.comment
        for word in ("bug", "feature", "50%", "30%"):
            filled = filled.replace(word, "")
        template = TEMPLATE_DISABLED if tau > 1.0 else TEMPLATE_BELOW_THRESHOLD
        stripped = template.replace("{label1}", "").replace("{label2}", "").replace("({p1:.0%})", "()").replace(
            "({p2:.0%})", "()")
        assert filled == stripped
