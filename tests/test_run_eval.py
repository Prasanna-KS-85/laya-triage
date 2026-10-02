"""eval/run_eval.py guards and options (no model, no data files are read)."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

import run_eval

BASE = ["--model-repo", "convaiinnovations/laya", "--revision", "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
        "--expect-max-len", "512", "--expect-head-max-len", "192", "--tag", "t"]


def test_input_hashes_are_pinned():
    for split in ("val", "test"):
        assert re.fullmatch(r"[0-9a-f]{64}", run_eval.INPUT_SHA256[split]), split


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A throwaway git repo standing in for ROOT, so the real config/ is never touched."""
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.invalid")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README").write_text("x\n")
    git(tmp_path, "add", "README")
    git(tmp_path, "commit", "-q", "-m", "init")
    cfg = tmp_path / "config" / "triage.default.yml"
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    monkeypatch.setattr(run_eval, "THRESHOLD_CONFIG", cfg)
    return tmp_path, cfg


def test_test_split_refused_without_config(fake_repo):
    root, _ = fake_repo
    with pytest.raises(SystemExit, match="does not exist"):
        run_eval.check_test_allowed(root / "predictions_test_t.csv")


def test_test_split_refused_with_uncommitted_or_modified_config(fake_repo):
    root, cfg = fake_repo
    cfg.parent.mkdir()
    cfg.write_text("gating: {}\n")
    with pytest.raises(SystemExit, match="committed and unmodified"):  # untracked
        run_eval.check_test_allowed(root / "predictions_test_t.csv")
    git(root, "add", "config/triage.default.yml")
    git(root, "commit", "-q", "-m", "cfg")
    cfg.write_text("gating: {changed: 1}\n")
    with pytest.raises(SystemExit, match="committed and unmodified"):  # modified
        run_eval.check_test_allowed(root / "predictions_test_t.csv")
    git(root, "checkout", "--", "config/triage.default.yml")
    out = root / "predictions_test_t.csv"
    run_eval.check_test_allowed(out)  # committed and clean: allowed
    out.write_text("id\n")
    with pytest.raises(SystemExit, match="exists"):
        run_eval.check_test_allowed(out)


def test_cli_refuses_test_in_this_repo_without_committed_config():
    if run_eval.THRESHOLD_CONFIG.exists():
        pytest.skip("config/triage.default.yml exists; covered by the fake-repo tests")
    with pytest.raises(SystemExit, match="does not exist"):
        run_eval.main([*BASE, "--split", "test"])


def test_threads_option():
    assert run_eval.parse_args([*BASE, "--split", "val"]).threads is None
    assert run_eval.parse_args([*BASE, "--split", "val", "--threads", "2"]).threads == 2
    with pytest.raises(SystemExit):
        run_eval.parse_args([*BASE, "--split", "val", "--threads", "0"])
