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
REV = "76ece1fb0eb8b32bd5d8c509293c1692a2534805"
GOOD_CFG = f'model:\n  repo: "Prasanna85/laya-issue-triage"\n  revision: "{REV}"\n  max_len: 1024\n'


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


def commit_config(root, cfg, text):
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text)
    git(root, "add", "config/triage.default.yml")
    git(root, "commit", "-q", "-m", "cfg")


def test_test_split_refused_without_config(fake_repo):
    root, _ = fake_repo
    with pytest.raises(SystemExit, match="does not exist"):
        run_eval.check_test_allowed(root / "predictions_test_t.csv", REV, 1024)


def test_test_split_refused_with_uncommitted_or_modified_config(fake_repo):
    root, cfg = fake_repo
    cfg.parent.mkdir()
    cfg.write_text(GOOD_CFG)
    with pytest.raises(SystemExit, match="committed and unmodified"):  # untracked
        run_eval.check_test_allowed(root / "predictions_test_t.csv", REV, 1024)
    commit_config(root, cfg, GOOD_CFG)
    cfg.write_text(GOOD_CFG + "# changed\n")
    with pytest.raises(SystemExit, match="committed and unmodified"):  # modified
        run_eval.check_test_allowed(root / "predictions_test_t.csv", REV, 1024)
    git(root, "checkout", "--", "config/triage.default.yml")
    out = root / "predictions_test_t.csv"
    run_eval.check_test_allowed(out, REV, 1024)  # committed, clean, matching: allowed
    out.write_text("id\n")
    with pytest.raises(SystemExit, match="exists"):
        run_eval.check_test_allowed(out, REV, 1024)


def test_test_split_requires_matching_revision_and_max_len(fake_repo):
    root, cfg = fake_repo
    commit_config(root, cfg, GOOD_CFG)
    out = root / "predictions_test_t.csv"
    run_eval.check_test_allowed(out, REV, 1024)  # matching
    with pytest.raises(SystemExit, match="model.revision"):
        run_eval.check_test_allowed(out, "a704b3eadd185f1fa028576cfa50605b6eada4a4", 1024)
    with pytest.raises(SystemExit, match="model.max_len"):
        run_eval.check_test_allowed(out, REV, 512)


@pytest.mark.parametrize("text", [
    "model:\n  max_len: 1024\n",                      # no revision
    f'model:\n  revision: "{REV}"\n',                 # no max_len
    "gating: {}\n",                                   # no model section
])
def test_test_split_refused_when_config_key_missing(fake_repo, text):
    root, cfg = fake_repo
    commit_config(root, cfg, text)
    with pytest.raises(SystemExit, match="has no model"):
        run_eval.check_test_allowed(root / "predictions_test_t.csv", REV, 1024)


def test_cli_refuses_test_for_the_base_checkpoint_in_this_repo():
    # The base revision and 512 never match the shipped config, so this is refused before any data is read.
    with pytest.raises(SystemExit, match="refusing to run test"):
        run_eval.main([*BASE, "--split", "test"])


def test_threads_option():
    assert run_eval.parse_args([*BASE, "--split", "val"]).threads is None
    assert run_eval.parse_args([*BASE, "--split", "val", "--threads", "2"]).threads == 2
    with pytest.raises(SystemExit):
        run_eval.parse_args([*BASE, "--split", "val", "--threads", "0"])


ROWS = [["nlbse24-test-0001", "o/a", "bug", "bug", "0.9", "0.05", "0.05", "0.9", "0.8", "0.1", "0.1"]]
META = {"tag": "t", "split": "val"}


def leftovers(d):
    return sorted(p.name for p in d.iterdir() if p.name.endswith(".tmp"))


def test_write_outputs_writes_csv_and_meta_without_temp_files(tmp_path):
    out = tmp_path / "predictions_val_t.csv"
    run_eval.write_outputs(out, ROWS, META)
    assert out.read_text().splitlines()[0].split(",") == run_eval.CSV_COLUMNS
    assert out.with_suffix(".meta.json").read_text() == '{\n "tag": "t",\n "split": "val"\n}\n'
    assert leftovers(tmp_path) == []
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run_eval.write_outputs(out, ROWS, META)


class Boom(Exception):
    pass


class BadRow:
    def __iter__(self):
        raise Boom("crash while writing the CSV")


def test_crash_while_writing_leaves_no_files(tmp_path):
    out = tmp_path / "predictions_val_t.csv"
    with pytest.raises(Boom):
        run_eval.write_outputs(out, [*ROWS, BadRow()], META)
    assert list(tmp_path.iterdir()) == []  # no CSV, no meta, no temporary files
    run_eval.write_outputs(out, ROWS, META)  # the identical re-run is not blocked
    assert out.exists()


def test_crash_between_renames_allows_identical_rerun(tmp_path, monkeypatch):
    out = tmp_path / "predictions_val_t.csv"
    real_replace, calls = run_eval.os.replace, []

    def flaky_replace(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise Boom("crash before the CSV is committed")
        real_replace(src, dst)

    monkeypatch.setattr(run_eval.os, "replace", flaky_replace)
    with pytest.raises(Boom):
        run_eval.write_outputs(out, ROWS, META)
    assert not out.exists() and out.with_suffix(".meta.json").exists() and leftovers(tmp_path) == []
    monkeypatch.setattr(run_eval.os, "replace", real_replace)
    run_eval.refuse_if_written(out)  # an orphan meta does not block the re-run
    run_eval.write_outputs(out, ROWS, {**META, "rerun": True})
    assert '"rerun": true' in out.with_suffix(".meta.json").read_text() and out.exists()


def test_test_guard_ignores_orphan_meta(fake_repo):
    root, cfg = fake_repo
    commit_config(root, cfg, GOOD_CFG)
    out = root / "predictions_test_t.csv"
    out.with_suffix(".meta.json").write_text("{}\n")
    run_eval.check_test_allowed(out, REV, 1024)  # only the CSV marks a finished run
