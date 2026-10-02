"""Run the production path locally on sandbox/issues.json and write sandbox/expected_local.json (PROJECT_SPEC.md §13
Phase 4 task 4). The runner's JSONL logs are compared against it with sandbox/compare.py.

Run from anywhere: .venv/bin/python sandbox/make_expected.py
Offline (HF_HUB_OFFLINE=1): the pinned model must be in the local Hugging Face cache. For each fixture, the same
path as the Action: issue_from_api -> should_skip -> preprocess -> Classifier.classify -> decide, with the default
config (config/triage.default.yml) and with comment_on_escalate: true (only the comment changes). Also checks that
classify_batch (backfill) equals classify. Writes keys, labels, probabilities and decisions only; prints no text.
"""
import json
import os
import platform
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # before huggingface_hub is imported
os.environ.pop("LAYA_CPU_AMP", None)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laya_triage.classifier import Classifier
from laya_triage.config import load_config, merge, validate
from laya_triage.event_loader import issue_from_api, should_skip
from laya_triage.policy import TEMPLATE_DISABLED, decide
from laya_triage.preprocess import preprocess
from laya_triage.questions import LABELS

FIXTURES = ROOT / "sandbox" / "issues.json"
OUT = ROOT / "sandbox" / "expected_local.json"
DEFAULT_CONFIG = ROOT / "config" / "triage.default.yml"


def template_of(comment):
    if comment is None:
        return None
    return "b" if comment.startswith(TEMPLATE_DISABLED.split("`")[0]) else "a"


def main():
    import torch
    import yaml

    cfg = load_config(DEFAULT_CONFIG)
    cfg_comment = validate(merge(yaml.safe_load(DEFAULT_CONFIG.read_text()), {"behaviour": {"comment_on_escalate": True}}))
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))["fixtures"]
    clf = Classifier(cfg.model.repo, cfg.model.revision, cfg.model.max_len)
    out, states = {}, {}
    for n, f in enumerate(fixtures, 1):
        issue = issue_from_api({"number": n, "title": f["title"], "body": f["body"], "user": {"login": "sandbox-owner"},
                                "labels": [{"name": x} for x in f["labels_at_creation"]]}, "opened")
        skip = should_skip(issue, cfg)
        if skip:
            out[f["key"]] = {"skip_reason": skip}
            continue
        state = preprocess(issue.title, issue.body, cfg.preprocess)
        states[f["key"]] = (n, state)
        r = clf.classify(state, n)
        d, dc = decide(r, cfg), decide(r, cfg_comment)
        out[f["key"]] = {
            "skip_reason": None, "label": r.label, "probabilities": r.probabilities,
            "answer_confidence": r.answer_confidence, "threshold": cfg.thresholds[r.label],
            "decision": {"action": d.action, "labels_to_add": list(d.labels_to_add)},
            "comment_template_if_enabled": template_of(dc.comment),
            "body_chars_after_preprocess": len(state["body"]),
        }
    keys = list(states)
    batch = clf.classify_batch([states[k][1] for k in keys], [states[k][0] for k in keys])
    dev = max(abs(b.probabilities[c] - out[k]["probabilities"][c]) for k, b in zip(keys, batch) for c in LABELS)
    assert [b.label for b in batch] == [out[k]["label"] for k in keys], "classify_batch labels differ from classify"
    import laya
    import transformers

    doc = {
        "generated_by": "sandbox/make_expected.py (production path, CPU fp32, offline)",
        "model": {"repo": cfg.model.repo, "revision": cfg.model.revision, "max_len": cfg.model.max_len},
        "thresholds": cfg.thresholds,
        "platform": {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
                     "torch": torch.__version__, "laya": laya.__version__, "transformers": transformers.__version__,
                     "torch_threads": torch.get_num_threads()},
        "classify_batch_vs_classify_max_abs_dp": dev,
        "fixtures": out,
    }
    OUT.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for k, v in out.items():
        if v["skip_reason"]:
            print(f"{k}: skip ({v['skip_reason']})")
        else:
            print(f"{k}: {v['label']} {v['answer_confidence']:.4f} -> {v['decision']['action']}")
    print(f"classify_batch vs classify: max |dp| {dev:.1e}; wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
