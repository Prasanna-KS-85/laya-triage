"""training/make_items.py on a tiny synthetic fixture (PROJECT_SPEC.md §8.5, §9.3).

Tests that tokenize are marked slow: they need the base checkpoint's tokenizer in the local HF
cache and are skipped when it is missing (as in CI). Nothing is downloaded.
"""
import copy
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import make_items as mi

BASE_REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


def hub_cache():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


TOKENIZER_DIR = hub_cache() / "models--convaiinnovations--laya/snapshots" / BASE_REV / "tokenizer"


def needs_tokenizer(test):
    skip = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(), reason="tokenizer not in local HF cache")
    return pytest.mark.slow(skip(test))


def row(i, label, body="short body"):
    return {
        "id": f"fixture-{i}",
        "repo": "o/a",
        "state": {"title": f"title {i}", "body": body},
        "questions": copy.deepcopy(mi.ISSUE_TYPE_QUESTION),
        "gold": {"issue_type": {"probabilities": {k: float(k == label) for k in mi.LABELS}}},
    }


FIXTURE = [row(0, "bug"), row(1, "feature"), row(2, "question", body="word " * 3000)]


@pytest.fixture(scope="module")
def tok():
    return mi.load_tokenizer(str(TOKENIZER_DIR))


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_rows_reads_native_objects(tmp_path):
    rows = mi.load_rows(write_jsonl(tmp_path / "t.jsonl", FIXTURE))
    assert [r["id"] for r in rows] == ["fixture-0", "fixture-1", "fixture-2"]
    assert isinstance(rows[0]["state"], dict) and isinstance(rows[0]["questions"], dict)


def test_load_rows_rejects_json_string_fields(tmp_path):
    bad = dict(FIXTURE[0], state=json.dumps(FIXTURE[0]["state"]))  # upstream's string-encoded layout
    with pytest.raises(mi.ItemsError, match="'state'"):
        mi.load_rows(write_jsonl(tmp_path / "t.jsonl", [bad]))


def test_smoke_subset_spreads_over_file():
    rows = list(range(1196))
    sub = mi.smoke_subset(rows, 50)
    assert len(sub) == 50 and sub[:3] == [0, 23, 46]
    with pytest.raises(mi.ItemsError):
        mi.smoke_subset(rows[:10], 50)


def test_items_sha256_rounds_target_and_keeps_order():
    a = {"ids": [1, 2], "markers": [1], "qtype": 0, "label": 0, "target": [1 / 3, 2 / 3]}
    b = dict(a, target=[1 / 3 + 1e-12, 2 / 3])
    c = dict(a, label=1)
    assert mi.items_sha256([a]) == mi.items_sha256([b])
    assert mi.items_sha256([a, c]) != mi.items_sha256([c, a])


@needs_tokenizer
@pytest.mark.parametrize("budget", [(1024, 256), (512, 192)])
def test_build_items(tok, budget):
    max_len, head_max_len = budget
    items = mi.build_items(FIXTURE, tok, max_len, head_max_len)
    assert len(items) == 3
    assert [it["label"] for it in items] == [0, 1, 2]
    for it in items:
        assert set(it) == {"ids", "markers", "qtype", "target", "label"}
        assert len(it["markers"]) == 3 and it["qtype"] == 0
        assert all(it["ids"][m] == tok.mask_token_id for m in it["markers"])
        assert len(it["ids"]) <= max_len
    assert len(items[2]["ids"]) == max_len  # the long state is cut at the budget
    assert items[0]["target"] == [1.0, 0.0, 0.0]


@needs_tokenizer
def test_label_smoothing_and_hash(tok):
    plain = mi.build_items(FIXTURE, tok, 1024, 256)
    smooth = mi.build_items(FIXTURE, tok, 1024, 256, label_smoothing=0.1)
    assert smooth[0]["target"] == pytest.approx([0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3])
    assert all(abs(sum(it["target"]) - 1) < 1e-12 for it in smooth)
    assert [it["label"] for it in smooth] == [it["label"] for it in plain]
    assert mi.items_sha256(plain) == mi.items_sha256(mi.build_items(FIXTURE, tok, 1024, 256))
    assert mi.items_sha256(plain) != mi.items_sha256(smooth)
    summary = mi.summarize(plain)
    assert summary["items"] == 3 and summary["label_counts"] == {"bug": 1, "feature": 1, "question": 1}
    assert summary["token_length"]["max"] == 1024


@needs_tokenizer
def test_build_items_rejects_contract_violations(tok):
    reordered = row(0, "bug")
    crit = reordered["questions"]["issue_type"]["criteria"]
    reordered["questions"]["issue_type"]["criteria"] = {k: crit[k] for k in ("feature", "bug", "question")}
    with pytest.raises(mi.ItemsError, match="fixture-0"):
        mi.build_items([reordered], tok, 1024, 256)
    reworded = row(1, "bug")
    reworded["questions"]["issue_type"]["instructions"] = "something else"
    with pytest.raises(mi.ItemsError, match="question differs"):
        mi.build_items([reworded], tok, 1024, 256)
    bad_gold = row(2, "bug")
    bad_gold["gold"]["issue_type"]["probabilities"] = {"yes": 1.0}
    with pytest.raises(mi.ItemsError, match="gold"):
        mi.build_items([bad_gold], tok, 1024, 256)
