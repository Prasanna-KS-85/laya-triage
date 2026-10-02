import hashlib
import io
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import fetch_data
import nlbse_data

PAYLOAD = b"repo,created_at,label,title,body\r\n"
GOOD = {"issues_train.csv": hashlib.sha256(PAYLOAD).hexdigest()}


class FakeOpener:
    """Stands in for urllib.request.urlopen; never touches the network."""

    def __init__(self, payload=PAYLOAD):
        self.payload, self.calls = payload, []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        return io.BytesIO(self.payload)


def test_downloads_verifies_and_renames(tmp_path):
    opener = FakeOpener()
    assert fetch_data.fetch(tmp_path, GOOD, opener=opener) == {"issues_train.csv": "downloaded"}
    target = tmp_path / "issues_train.csv"
    assert target.read_bytes() == PAYLOAD
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert [p.name for p in tmp_path.iterdir()] == ["issues_train.csv"]  # no .part left behind
    url, timeout = opener.calls[0]
    assert url == nlbse_data.RAW_URL.format(name="issues_train.csv")
    assert nlbse_data.UPSTREAM_COMMIT in url and timeout == fetch_data.TIMEOUT_S


def test_skips_file_already_present_and_verified(tmp_path):
    (tmp_path / "issues_train.csv").write_bytes(PAYLOAD)
    opener = FakeOpener()
    assert fetch_data.fetch(tmp_path, GOOD, opener=opener) == {"issues_train.csv": "present"}
    assert opener.calls == []


def test_bad_download_leaves_nothing(tmp_path):
    with pytest.raises(ValueError, match="downloaded sha256"):
        fetch_data.fetch(tmp_path, GOOD, opener=FakeOpener(b"tampered"))
    assert list(tmp_path.iterdir()) == []


def test_existing_file_with_wrong_hash_is_not_replaced(tmp_path):
    (tmp_path / "issues_train.csv").write_bytes(b"local edits")
    opener = FakeOpener()
    with pytest.raises(ValueError, match="exists but sha256"):
        fetch_data.fetch(tmp_path, GOOD, opener=opener)
    assert (tmp_path / "issues_train.csv").read_bytes() == b"local edits"
    assert opener.calls == []


def test_pinned_constants():
    assert nlbse_data.UPSTREAM_COMMIT == "2927bc67eb42db8affd16eaf3e5a6d74f3063961"
    assert nlbse_data.SHA256 == {
        "issues_train.csv": "18dc42a30aa33dccadb723ad3baeb164d38bff521496f985ca2791c26b8939f5",
        "issues_test.csv": "4f7d8619d4e5adbea126e548fd8c214449288f3a93bb3bc130c54cd307af7e85",
    }
