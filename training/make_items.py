"""Training items for the Laya fine-tuning notebook (PROJECT_SPEC.md §8.5, §9.3; Phase 3 task 1).

Standalone: imports only the standard library, transformers and laya (never laya_triage), because
training/finetune_kaggle.ipynb embeds this file byte for byte in a %%writefile cell
(tests/test_notebook_sync.py checks that). An item is the dict the upstream notebook's
build_training_item returns for a `choice` question: ids, markers, qtype, target, label.

Differences from the upstream preprocessing cell:
- Rows hold native JSON objects (§8.5); fields are read directly, never json.loads'ed.
- Sequences are built at the max_len / head_max_len passed in (1024/256 for training), not at the
  root checkpoint's 512/192.
- Optional label smoothing eps: target = (1 - eps) * gold + eps / k (§9.5).
- Nothing is dropped silently: a row that fails a check raises ItemsError naming its id only.

CLI (local, offline; prints counts and hashes only, never issue text):
  .venv/bin/python training/make_items.py data/processed/train.jsonl --tokenizer-dir <dir> \
      [--max-len 1024] [--head-max-len 256] [--label-smoothing 0.0]
<dir> is a checkpoint's tokenizer/ directory. load_tokenizer copies it to a temporary directory
before laya's _fix_tokenizer_config runs, so a Hugging Face cache is never modified.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options
from transformers import AutoTokenizer

# Copy of src/laya_triage/questions.py (frozen, §9.2); tests/test_notebook_sync.py checks equality.
QUESTION_ID = "issue_type"
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
TARGET_DECIMALS = 9
LENGTH_PCTS = (50, 90, 99)


class ItemsError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise ItemsError(msg)


def load_rows(path):
    """Rows of a training JSONL file (§8.5), in file order."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            _check(isinstance(row, dict), f"line {n}: not a JSON object")
            for key, kind in (("id", str), ("state", dict), ("questions", dict), ("gold", dict)):
                _check(isinstance(row.get(key), kind), f"line {n}: {key!r} missing or not a {kind.__name__}")
            rows.append(row)
    return rows


def smoke_subset(rows, n):
    """n rows spread evenly over the file (every len // n-th row). The files are in raw-CSV order,
    which groups labels, so the first n rows can all share one label."""
    _check(0 < n <= len(rows), f"cannot take {n} of {len(rows)} rows")
    return rows[:: len(rows) // n][:n]


def load_tokenizer(tokenizer_dir):
    """AutoTokenizer for a checkpoint's tokenizer/ dir, fixed by laya on a temporary copy."""
    tmp = tempfile.mkdtemp(prefix="laya_tok_")
    try:
        shutil.copytree(tokenizer_dir, os.path.join(tmp, "tokenizer"))
        _fix_tokenizer_config(tmp)
        return AutoTokenizer.from_pretrained(os.path.join(tmp, "tokenizer"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_items(rows, tok, max_len, head_max_len, label_smoothing=0.0):
    """One item per row, in row order. Raises ItemsError rather than dropping a row."""
    _check(0.0 <= label_smoothing < 1.0, "label_smoothing must be in [0, 1)")
    items = []
    for row in rows:
        rid = row["id"]
        _check(row["questions"] == ISSUE_TYPE_QUESTION, f"{rid}: question differs from ISSUE_TYPE_QUESTION")
        q = row["questions"][QUESTION_ID]
        keys = list(q["criteria"].keys())
        _check(tuple(keys) == LABELS, f"{rid}: criteria key order is not LABELS")
        probs = row["gold"].get(QUESTION_ID, {}).get("probabilities")
        _check(isinstance(probs, dict) and set(probs) == set(LABELS), f"{rid}: gold probabilities keys != LABELS")

        # Same target rule as upstream build_training_item (choice), then smoothing.
        target = [float(probs.get(k, 0.0)) for k in keys]
        s = sum(target)
        target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
        k = len(render_options({"t": q["type"], "crit": q["criteria"]}))
        if label_smoothing:
            target = [(1.0 - label_smoothing) * v + label_smoothing / k for v in target]
        label = target.index(max(target))

        seq, markers = build_sequence(
            tok, row["state"], {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}, max_len, head_max_len
        )
        # Upstream silently drops an item whose marker count differs; here it is an error.
        _check(len(markers) == k == len(LABELS), f"{rid}: {len(markers)} markers for {k} options")
        _check(len(seq) <= max_len, f"{rid}: sequence longer than max_len")
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": target, "label": label})
    _check(len(items) == len(rows), f"item count {len(items)} != row count {len(rows)}")
    return items


def items_sha256(items):
    """SHA-256 of the canonical JSON of every item (ids, markers, qtype, label, target), in order."""
    canon = [
        {
            "ids": it["ids"],
            "label": it["label"],
            "markers": it["markers"],
            "qtype": it["qtype"],
            "target": [round(t, TARGET_DECIMALS) for t in it["target"]],
        }
        for it in items
    ]
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def summarize(items):
    """Counts and token-length stats only; nothing that could carry issue text."""
    lengths = [len(it["ids"]) for it in items]
    labels = [it["label"] for it in items]
    return {
        "items": len(items),
        "label_counts": {name: labels.count(i) for i, name in enumerate(LABELS)},
        "token_length": {
            **{f"p{p}": round(_percentile(lengths, p), 1) for p in LENGTH_PCTS},
            "max": max(lengths),
        },
        "share_longer_than_512": round(sum(n > 512 for n in lengths) / len(lengths), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows", help="training JSONL (§8.5)")
    ap.add_argument("--tokenizer-dir", required=True, help="a checkpoint's tokenizer/ directory")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    args = ap.parse_args(argv)
    rows = load_rows(args.rows)
    tok = load_tokenizer(args.tokenizer_dir)
    items = build_items(rows, tok, args.max_len, args.head_max_len, args.label_smoothing)
    out = {
        "rows": len(rows),
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "label_smoothing": args.label_smoothing,
        **summarize(items),
        "sha256": items_sha256(items),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
