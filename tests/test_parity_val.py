"""Slow: end-to-end parity of the production path with the Phase 3 val predictions (PROJECT_SPEC.md §9.4, §10.4).

For 30 val issues (the first 2 of every repo × label cell in data/manifests/val.txt order), the RAW title and body
are read from data/raw/issues_train.csv (a val id nlbse24-train-<row> is that 0-based CSV row; the content SHA-1
must equal the manifest's), wrapped in a synthetic `issues` event, and run through event_loader -> preprocess ->
Classifier. Each result must match results/phase3/val_R1b/predictions_val_M1_R1b.csv: identical label and
|dp| <= 1e-4 for every probability; classify_batch is held to the same bar. Issue text stays in memory: it is
never printed, and assertion messages hold ids and numbers only. Skipped when the data or the snapshot is missing.
"""
import csv
import json
import sys
from pathlib import Path

import pytest

from laya_triage.event_loader import read_event
from laya_triage.preprocess import preprocess
from laya_triage.questions import LABELS

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "issues_train.csv"
MANIFEST = ROOT / "data" / "manifests" / "val.txt"
PREDICTIONS = ROOT / "results" / "phase3" / "val_R1b" / "predictions_val_M1_R1b.csv"
PER_CELL = 2
TOL = 1e-4


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    if not RAW.is_file():
        pytest.skip("data/raw/issues_train.csv is not present")
    sys.path.insert(0, str(ROOT / "training"))
    import nlbse_data

    assert nlbse_data.sha256_file(RAW) == nlbse_data.SHA256[RAW.name], "raw CSV differs from the pinned file"
    rows = nlbse_data.load_raw(RAW)
    manifest = [line.split("\t") for line in MANIFEST.read_text().splitlines()]
    with open(PREDICTIONS, newline="") as f:
        expected = {r["id"]: r for r in csv.DictReader(f)}
    picked, per_cell = [], {}
    for vid, sha1 in manifest:
        raw = rows[int(vid.rsplit("-", 1)[1])]
        cell = (raw["repo"], raw["label"])
        if per_cell.get(cell, 0) >= PER_CELL:
            continue
        assert nlbse_data.content_hash(raw["title"], raw["body"]) == sha1, f"{vid}: content hash != manifest"
        assert expected[vid]["gold"] == raw["label"], vid
        per_cell[cell] = per_cell.get(cell, 0) + 1
        picked.append((vid, raw))
    assert len(picked) == 15 * PER_CELL == 30
    out, d = [], tmp_path_factory.mktemp("events")
    for k, (vid, raw) in enumerate(picked, 1):
        event = d / f"event{k}.json"
        event.write_text(json.dumps({"action": "opened", "issue": {
            "number": k, "title": raw["title"], "body": raw["body"], "labels": [], "user": {"login": "someone"}}}))
        (issue,) = read_event("issues", event)
        out.append((vid, k, issue, expected[vid]))
        event.unlink()  # raw text never stays on disk
    return out


def _compare(results, cases):
    label_ok, max_dev, bad = 0, 0.0, []
    for r, (vid, _, _, exp) in zip(results, cases):
        dev = max(abs(r.probabilities[c] - float(exp[f"p_{c}"])) for c in LABELS)
        max_dev = max(max_dev, dev)
        label_ok += r.label == exp["pred"]
        if r.label != exp["pred"] or dev > TOL + 1e-9:
            bad.append(f"{vid}: label {r.label} vs {exp['pred']}, max |dp| {dev:.1e}")
    return label_ok, max_dev, bad


def test_production_path_matches_val_predictions(offline_classifier, triage_config, cases):
    states = [preprocess(issue.title, issue.body, triage_config.preprocess) for _, _, issue, _ in cases]
    single = [offline_classifier.classify(s, k) for s, (_, k, _, _) in zip(states, cases)]
    batch = offline_classifier.classify_batch(states, [k for _, k, _, _ in cases])
    s_ok, s_dev, s_bad = _compare(single, cases)
    b_ok, b_dev, b_bad = _compare(batch, cases)
    vs = max(abs(a.probabilities[c] - b.probabilities[c]) for a, b in zip(single, batch) for c in LABELS)
    print(f"\nparity on {len(cases)} val issues: classify labels {s_ok}/{len(cases)}, max |dp| {s_dev:.1e}; "
          f"classify_batch labels {b_ok}/{len(cases)}, max |dp| {b_dev:.1e}; batch vs single max |dp| {vs:.1e}")
    assert not s_bad, s_bad
    assert not b_bad, b_bad
