"""Frozen issue-type question (PROJECT_SPEC.md §9.2). Changing it requires §16 rule 4."""

ISSUE_TYPE_QUESTION = {
    "issue_type": {
        "type": "choice",
        "instructions": "You are triaging a newly opened GitHub issue. Using its `title` and `body`, decide which type of issue it is.",
        "criteria": {
            "bug": "reports a defect: a crash, error message, failing build or test, regression, or behaviour that contradicts the documentation",
            "feature": "proposes something new: a new capability, option or API, or an improvement to how existing behaviour works",
            "question": "asks for help: how to use or configure something, why it behaves a certain way, or troubleshooting the author's own setup",
        },
    }
}
LABELS = ("bug", "feature", "question")  # order is part of the contract
