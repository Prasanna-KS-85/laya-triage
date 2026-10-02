"""NLBSE'24 issue-report-classification data: pinned source, loading, synthetic IDs, content hashes.

See PROJECT_SPEC.md §8.1 and §8.3. Stdlib only.
"""

import csv
import hashlib
from pathlib import Path

UPSTREAM_REPO = "nlbse2024/issue-report-classification"
UPSTREAM_COMMIT = "2927bc67eb42db8affd16eaf3e5a6d74f3063961"
RAW_URL = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/data/{{name}}"

# file name -> sha256 of the raw CSV at UPSTREAM_COMMIT
SHA256 = {
    "issues_train.csv": "18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5",
    "issues_test.csv": "4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85",
}
SOURCES = {"train": "issues_train.csv", "test": "issues_test.csv"}
COLUMNS = ["repo", "created_at", "label", "title", "body"]
LABELS = ("bug", "feature", "question")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path) -> list[dict]:
    """Rows of one raw NLBSE CSV, in file order. Fields are kept exactly as stored."""
    csv.field_size_limit(2**31 - 1)  # bodies reach ~208k chars
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"{path}: expected columns {COLUMNS}, got {reader.fieldnames}")
        rows = list(reader)
    bad = sorted({r["label"] for r in rows} - set(LABELS))
    if bad:
        raise ValueError(f"{path}: unexpected labels {bad}")
    return rows


def make_id(source: str, row: int) -> str:
    """Synthetic ID: nlbse24-<source>-<row:04d>, row = 0-based index in the raw CSV."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    return f"nlbse24-{source}-{row:04d}"


def content_hash(title: str, body: str) -> str:
    """SHA-1 over the UTF-8 bytes of the raw title + "\\n" + body."""
    return hashlib.sha1((title + "\n" + body).encode("utf-8")).hexdigest()
