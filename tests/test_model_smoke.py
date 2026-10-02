"""Slow: the pinned model on 3 hand-written synthetic issues (PROJECT_SPEC.md §9.4, Phase 4 task 6).
Uses the offline_classifier fixture (tests/conftest.py): local HF cache only, skipped when it is missing."""
import pytest

from laya_triage.preprocess import preprocess
from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

pytestmark = pytest.mark.slow

# Hand-written synthetic issues (not from any dataset).
ISSUES = [
    (101, "App crashes with KeyError when opening settings",
     ("Steps to reproduce:\n1. Start the app\n2. Click Settings\n\n```\nTraceback (most recent call last):\n"
      "  File \"app.py\", line 12, in open_settings\nKeyError: 'theme'\n```\n\nExpected: the settings page opens. "
      "Version 2.3.1 on Ubuntu 22.04.")),
    (102, "Add an option to export results as CSV",
     ("It would be useful to export results as CSV next to the existing JSON export, so they can be opened in a "
      "spreadsheet. A new `--format csv` flag would be enough.")),
    (103, "How do I configure a proxy?",
     ("I'm behind a corporate proxy and can't find which setting makes the client use it. Is there an environment "
      "variable or a config key for this? I checked the docs but couldn't find it.")),
]
TOL = 1e-4  # laya rounds probabilities to 4 decimals


@pytest.fixture(scope="module")
def states():
    return [(n, preprocess(t, b)) for n, t, b in ISSUES]


def test_checkpoint_budget_matches_config(offline_classifier, triage_config):
    cfg = offline_classifier.agent.cfg
    assert cfg["max_len"] == 1024 == triage_config.model.max_len  # §9.4
    assert cfg["head_max_len"] == 256
    assert offline_classifier.revision == triage_config.model.revision


def test_results_are_well_formed(offline_classifier, states):
    for n, state in states:
        r = offline_classifier.classify(state, n)
        assert r.issue_number == n and r.label in LABELS and set(r.probabilities) == set(LABELS)
        assert abs(sum(r.probabilities.values()) - 1) <= 3 * 5e-5 + 1e-9
        assert r.answer_confidence == max(r.probabilities.values()) == r.probabilities[r.label]
        assert r.latency_ms > 0
        print(f"#{n}: {r.label} {r.answer_confidence:.4f}")


def test_the_frozen_question_is_asked(offline_classifier, states):
    agent, seen = offline_classifier.agent, []
    original = agent.predict

    def spy(state, questions, *a, **k):
        seen.append(questions)
        return original(state, questions, *a, **k)

    agent.predict = spy
    try:
        offline_classifier.classify(states[0][1], 1)
    finally:
        del agent.predict
    assert seen == [ISSUE_TYPE_QUESTION]
    assert tuple(seen[0]["issue_type"]["criteria"]) == LABELS


def test_classify_batch_agrees_with_classify(offline_classifier, states):
    single = [offline_classifier.classify(s, n) for n, s in states]
    batch = offline_classifier.classify_batch([s for _, s in states], [n for n, _ in states])
    assert [r.issue_number for r in batch] == [n for n, _ in states]
    assert [r.label for r in batch] == [r.label for r in single]
    dev = max(abs(a.probabilities[c] - b.probabilities[c]) for a, b in zip(single, batch) for c in LABELS)
    print(f"classify_batch vs classify on {len(states)} synthetic issues: max |dp| {dev:.1e}")
    assert dev <= TOL + 1e-9
