"""Classify one issue with the Laya Triage model on CPU (fp32). Install: see docs/USING_THE_MODEL.md."""
# The package brings the frozen question and the shared preprocess(): nothing is retyped.
from laya_triage.classifier import Classifier
from laya_triage.preprocess import preprocess

REPO = "Prasanna85/laya-issue-triage"
REVISION = "76ece1fb0eb8b32bd5d8c509293c1692a2534805"  # pinned: the revision the Action uses
MAX_LEN = 1024  # must equal the training budget
BUG_THRESHOLD = 0.6033  # the Action's default gate; feature and question are never auto-applied

model = Classifier(REPO, REVISION, MAX_LEN)  # downloads once, then loads from the local cache
state = preprocess("App closes when I open the export dialog",
                   "Steps: open a notebook, choose File > Export. The app exits with KeyError: 'last_format'.")
result = model.classify(state, issue_number=0)
print(result.probabilities)  # {'bug': ..., 'feature': ..., 'question': ...}
confident_bug = result.label == "bug" and result.answer_confidence >= BUG_THRESHOLD
print("apply bug" if confident_bug else "escalate to a human")
