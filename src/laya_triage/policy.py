"""Gating policy: TriageResult + config -> Decision (PROJECT_SPEC.md FR-2, FR-3, §12.1, §12.6). Pure functions.

- FR-2: answer_confidence >= thresholds[label] -> apply the configured label for that class.
- FR-3: otherwise escalate: add the configured escalation label and, if behaviour.comment_on_escalate, post a
  fixed-template comment: template (a) (§12.6) when the threshold is reachable (<= 1.0), template (b) when the
  threshold is > 1.0, i.e. auto-labelling of that class is disabled by configuration.
- Comments and reasons contain only fixed text, class names and numbers; never issue content (§7.6).
"""

from dataclasses import dataclass
from typing import Literal

from laya_triage.questions import LABELS

TEMPLATE_BELOW_THRESHOLD = (
    "🤖 Laya Triage couldn't classify this issue confidently, so it's been marked for a maintainer.\n"
    "Top guesses: `{label1}` ({p1:.0%}), `{label2}` ({p2:.0%})."
)
TEMPLATE_DISABLED = (
    "🤖 Laya Triage suggests `{label1}` ({p1:.0%}), but automatic labelling is disabled for this issue type in this "
    "repository's configuration, so it's been marked for a maintainer.\n"
    "Second guess: `{label2}` ({p2:.0%})."
)


@dataclass(frozen=True)
class Decision:
    action: Literal["apply", "escalate", "skip"]
    labels_to_add: tuple[str, ...]    # actual GitHub label names from config
    comment: str | None               # fixed-template text, never echoes issue content
    reason: str                       # human-readable, goes to summary


def top2(probabilities):
    """The two most probable classes as ((label, p), (label, p)); ties keep LABELS order."""
    ranked = sorted(LABELS, key=lambda c: (-probabilities[c], LABELS.index(c)))
    return (ranked[0], probabilities[ranked[0]]), (ranked[1], probabilities[ranked[1]])


def escalation_comment(probabilities, threshold):
    (l1, p1), (l2, p2) = top2(probabilities)
    template = TEMPLATE_DISABLED if threshold > 1.0 else TEMPLATE_BELOW_THRESHOLD
    return template.format(label1=l1, p1=p1, label2=l2, p2=p2)


def decide(result, cfg) -> Decision:
    label, conf = result.label, result.answer_confidence
    tau = cfg.thresholds[label]
    if conf >= tau:
        return Decision("apply", (cfg.labels.for_class(label),), None,
                        f"{label}: confidence {conf:.4f} >= threshold {tau:.4f}")
    comment = escalation_comment(result.probabilities, tau) if cfg.behaviour.comment_on_escalate else None
    reason = (f"{label}: auto-labelling disabled (threshold {tau:.2f})" if tau > 1.0
              else f"{label}: confidence {conf:.4f} < threshold {tau:.4f}")
    return Decision("escalate", (cfg.labels.escalate,), comment, reason)


def decide_error(cfg, what) -> Decision:
    """Decision when the model could not classify (§7.5): escalation label only if escalate_on_error, no comment."""
    if cfg.behaviour.escalate_on_error:
        return Decision("escalate", (cfg.labels.escalate,), None, f"{what}; escalated (escalate_on_error)")
    return Decision("skip", (), None, f"{what}; no label")
