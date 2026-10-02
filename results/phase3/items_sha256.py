"""Expected training-item hashes for the Kaggle notebook (PROJECT_SPEC.md §9.3 items-hash parity).

Run from anywhere: .venv/bin/python results/phase3/items_sha256.py
Offline; needs the Laya checkpoint's tokenizer in the local HF cache. Builds items with
training/make_items.py from data/processed/train.jsonl at 1024/256 for each label-smoothing eps,
plus the SMOKE subset (make_items.smoke_subset, 50 rows), and records the SHA-256 and size of train.jsonl and
val.jsonl. Writes items_sha256.json next to this file. Reads issue text but writes only counts and
hashes. training/finetune_kaggle.ipynb embeds these values and asserts them on Kaggle.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "training"))

import laya
import transformers
from huggingface_hub import snapshot_download
from make_items import (
    build_items,
    items_sha256,
    load_rows,
    load_tokenizer,
    smoke_subset,
    summarize,
)

REPO, REV = "convaiinnovations/laya", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
MAX_LEN, HEAD_MAX_LEN = 1024, 256
EPS = (0.0, 0.1)
SMOKE_ROWS = 50


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


snap = Path(snapshot_download(REPO, revision=REV, allow_patterns=["tokenizer/*"]))
tok = load_tokenizer(str(snap / "tokenizer"))
rows = load_rows(ROOT / "data/processed/train.jsonl")

report = {
    "laya": laya.__version__,
    "transformers": transformers.__version__,
    "tokenizer": {"repo": REPO, "revision": REV},
    "max_len": MAX_LEN,
    "head_max_len": HEAD_MAX_LEN,
    "files": {},
    "items": {},
    "smoke": {},
}
for split in ("train", "val"):
    p = ROOT / "data/processed" / f"{split}.jsonl"
    with open(p, encoding="utf-8") as f:
        n_rows = sum(1 for line in f if line.strip())
    report["files"][f"{split}.jsonl"] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "rows": n_rows}
for eps in EPS:
    items = build_items(rows, tok, MAX_LEN, HEAD_MAX_LEN, eps)
    report["items"][str(eps)] = {"rows": len(rows), **summarize(items), "sha256": items_sha256(items)}
    smoke = build_items(smoke_subset(rows, SMOKE_ROWS), tok, MAX_LEN, HEAD_MAX_LEN, eps)
    report["smoke"][str(eps)] = {
        "rows": SMOKE_ROWS, "rule": "smoke_subset", **summarize(smoke), "sha256": items_sha256(smoke)
    }

(HERE / "items_sha256.json").write_text(json.dumps(report, indent=1) + "\n")
print(json.dumps(report, indent=1))
