"""Phase 0 task 3 candidate wordings. Keys and order are fixed: bug, feature, question.

W1' won the comparison and is frozen in src/laya_triage/questions.py (task 5).
"""

W0 = {  # original PROJECT_SPEC.md v1.0 §9.2 wording
    "instructions": "What kind of GitHub issue is described in `title` and `body`?",
    "criteria": {
        "bug": "something is broken, crashes, errors, or behaves differently than documented",
        "feature": "a request for new functionality or an enhancement to existing behaviour",
        "question": "the author asks how to do something or seeks help or clarification",
    },
}

W1_PRIME = {  # W1': explicit triage signals, negations removed at owner's request (2026-10-02)
    "instructions": "You are triaging a newly opened GitHub issue. Using its `title` and `body`, "
                    "decide which type of issue it is.",
    "criteria": {
        "bug": "reports a defect: a crash, error message, failing build or test, regression, or "
               "behaviour that contradicts the documentation",
        "feature": "proposes something new: a new capability, option or API, or an improvement to "
                   "how existing behaviour works",
        "question": "asks for help: how to use or configure something, why it behaves a certain "
                    "way, or troubleshooting the author's own setup",
    },
}

W2 = {  # minimal
    "instructions": "Classify this GitHub issue by its `title` and `body`.",
    "criteria": {
        "bug": "a bug report: something does not work as intended",
        "feature": "a feature request: asks for new or improved functionality",
        "question": "a question: asks for help, guidance, or an explanation",
    },
}

WORDINGS = {"W0": W0, "W1'": W1_PRIME, "W2": W2}


def question(w):
    return {"issue_type": {"type": "choice", **w}}
