import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import build_dataset as bd
from nlbse_data import COLUMNS, content_hash, make_id

from laya_triage.preprocess import preprocess
from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

REPOS = ("o/a", "o/b")
ID_LINE = re.compile(r"^nlbse24-(train|test)-\d{4}\t[0-9a-f]{40}$")


def fixture_rows():
    """60 train rows (2 repos x 3 labels x 10), plus a within-train duplicate pair (rows 60, 61)."""
    train = [{"repo": repo, "created_at": "2024-01-01", "label": label,
              "title": f"{repo} {label} {j}", "body": f"body {repo} {label} {j}\r\n\r\n\r\nend  "}
             for repo in REPOS for label in LABELS for j in range(10)]
    dup = {"repo": "o/b", "created_at": "2024-01-02", "label": "question", "title": "dup", "body": "same"}
    train += [dict(dup), dict(dup)]
    test = [{"repo": repo, "created_at": "2024-02-01", "label": label,
             "title": f"test {repo} {label}", "body": "t"} for repo in REPOS for label in LABELS]
    test.append(dict(train[3], created_at="2024-02-02"))  # same content as train row 3 (o/a bug 3)
    return train, test


def write_raw(tmp_path, train, test):
    raw = tmp_path / "raw"
    raw.mkdir()
    hashes = {}
    for name, rows in (("issues_train.csv", train), ("issues_test.csv", test)):
        with open(raw / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        hashes[name] = hashlib.sha256((raw / name).read_bytes()).hexdigest()
    return raw, hashes


def run_build(tmp_path, sample_rows=(0, 1, 2), out="out", **kw):
    train, test = fixture_rows()
    raw = tmp_path / "raw"
    if not raw.exists():
        raw, hashes = write_raw(tmp_path, train, test)
    else:
        hashes = {n: hashlib.sha256((raw / n).read_bytes()).hexdigest() for n in ("issues_train.csv", "issues_test.csv")}
    ids = tmp_path / "sample_ids.txt"
    ids.write_text("\n".join(map(str, sample_rows)) + "\n")
    out = tmp_path / out
    kw.setdefault("expected_sha256", hashes)
    splits = bd.build(raw, out / "processed", out / "manifests", ids, real_data_checks=False, **kw)
    return splits, out


def manifest(out, name):
    return (out / "manifests" / f"{name}.txt").read_text().splitlines()


def test_drops_cross_split_and_within_train_duplicates(tmp_path):
    splits, out = run_build(tmp_path)
    dropped = {r.row: reason for r, reason in splits["dropped"]}
    assert dropped == {3: "content also in official test",
                       60: "content duplicated within official train",
                       61: "content duplicated within official train"}
    lines = manifest(out, "dropped_train")
    assert [line.split("\t")[0] for line in lines] == ["nlbse24-train-0003", "nlbse24-train-0060", "nlbse24-train-0061"]
    assert all(len(line.split("\t")) == 3 for line in lines)
    total = sum(len(manifest(out, s)) for s in ("train", "val", "dropped_train"))
    assert total == 62


def test_ids_and_hashes_disjoint_across_splits(tmp_path):
    _, out = run_build(tmp_path)
    ids, hashes = {}, {}
    for s in ("train", "val", "test"):
        for line in manifest(out, s):
            i, h = line.split("\t")
            assert i not in ids
            ids[i] = s
            assert hashes.setdefault(h, s) == s


def test_stratified_val_counts_and_phase0_rows_excluded(tmp_path):
    splits, _ = run_build(tmp_path, sample_rows=(0, 1, 2, 4))
    val = Counter((r.repo, r.label) for r in splits["val"])
    # 20% round half up: every cell has 10 kept rows except o/a bug (row 3 dropped -> 9 -> 1.8 -> 2)
    assert val == {(repo, label): 2 for repo in REPOS for label in LABELS}
    val_rows = {r.row for r in splits["val"]}
    assert not val_rows & {0, 1, 2, 4}
    assert {0, 1, 2, 4} <= {r.row for r in splits["train"]}


def test_val_count_round_half_up():
    assert [bd.val_count(n) for n in (0, 2, 3, 5, 9, 10, 98, 99, 100)] == [0, 0, 1, 1, 2, 2, 20, 20, 20]


def test_cell_that_cannot_supply_val_count_fails(tmp_path):
    # o/a bug keeps rows 0-2 and 4-9 (9 rows, needs 2 val); mark 8 of them as Phase 0 sample rows.
    with pytest.raises(bd.BuildError, match="needs 2 val rows but only 1"):
        run_build(tmp_path, sample_rows=(0, 1, 2, 4, 5, 6, 7, 8))


def test_sample_row_that_would_be_dropped_fails(tmp_path):
    with pytest.raises(bd.BuildError, match="sample rows would be dropped"):
        run_build(tmp_path, sample_rows=(3,))


def test_raw_hash_mismatch_fails(tmp_path):
    bad = {"issues_train.csv": "0" * 64, "issues_test.csv": "0" * 64}
    with pytest.raises(bd.BuildError, match="sha256"):
        run_build(tmp_path, expected_sha256=bad)


def test_real_data_checks_reject_other_data(tmp_path):
    train, test = fixture_rows()
    raw, hashes = write_raw(tmp_path, train, test)
    ids = tmp_path / "ids.txt"
    ids.write_text("0\n")
    with pytest.raises(bd.BuildError, match="real-data check failed"):
        bd.build(raw, tmp_path / "p", tmp_path / "m", ids, expected_sha256=hashes)


def test_check_splits_catches_shared_content_hash():
    a = bd.Record("train", 0, "nlbse24-train-0000", "h" * 40, "o/a", "bug", "t", "b")
    b = bd.Record("test", 0, "nlbse24-test-0000", "h" * 40, "o/a", "bug", "t", "b")
    with pytest.raises(bd.BuildError, match="content hash"):
        bd.check_splits({"train": [a], "val": [], "test": [b], "dropped": []}, 1, set())


def test_deterministic(tmp_path):
    run_build(tmp_path, out="one")
    run_build(tmp_path, out="two")
    files = sorted(p.relative_to(tmp_path / "one") for p in (tmp_path / "one").rglob("*") if p.is_file())
    assert len(files) == 8
    for f in files:
        assert (tmp_path / "one" / f).read_bytes() == (tmp_path / "two" / f).read_bytes(), f


def test_id_and_manifest_formats(tmp_path):
    _, out = run_build(tmp_path)
    for s in ("train", "val", "test"):
        assert all(ID_LINE.match(line) for line in manifest(out, s))
    assert make_id("train", 7) == "nlbse24-train-0007"
    assert make_id("test", 1499) == "nlbse24-test-1499"
    with pytest.raises(ValueError):
        make_id("val", 1)
    assert content_hash("t", "b") == hashlib.sha1(b"t\nb").hexdigest()


def test_row_formats_and_order(tmp_path):
    _, out = run_build(tmp_path)
    rows = [json.loads(line) for line in (out / "processed" / "train.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [line.split("\t")[0] for line in manifest(out, "train")]
    r = rows[0]
    assert list(r) == ["id", "repo", "state", "questions", "gold"]
    assert r["questions"] == ISSUE_TYPE_QUESTION
    assert r["state"] == preprocess("o/a bug 0", "body o/a bug 0\r\n\r\n\r\nend  ")
    assert r["gold"] == {"issue_type": {"probabilities": {"bug": 1.0, "feature": 0.0, "question": 0.0}}}
    for s in ("val", "test"):
        ev = [json.loads(line) for line in (out / "processed" / f"{s}.jsonl").read_text().splitlines()]
        assert [e["id"] for e in ev] == [line.split("\t")[0] for line in manifest(out, s)]
        assert list(ev[0]) == ["id", "state", "questions", "expected", "tags"]
        assert ev[0]["expected"]["issue_type"] in LABELS and ev[0]["tags"][0].startswith("repo:")


def test_data_card_block_replaced(tmp_path):
    card = tmp_path / "CARD.md"
    card.write_text(f"intro\n{bd.CARD_BEGIN}\nold\n{bd.CARD_END}\noutro\n")
    run_build(tmp_path, card_path=card)
    text = card.read_text()
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old" not in text
    assert "| o/a | 7 | 8 | 8 | 23 |" in text  # train: 10 - 2 val per cell; o/a bug 9 kept - 2
    card.write_text("no markers\n")
    with pytest.raises(bd.BuildError, match="marker"):
        run_build(tmp_path, out="again", card_path=card)
