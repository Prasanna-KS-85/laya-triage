"""laya_triage.config: deep merge and validation (PROJECT_SPEC.md §12.3, FR-8; owner decisions D2, D6)."""
import copy
from pathlib import Path

import pytest
import yaml

from laya_triage.config import ConfigError, load_config, merge, validate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "triage.default.yml"


@pytest.fixture
def raw():
    return yaml.safe_load(DEFAULT.read_text())


def problems_of(raw):
    with pytest.raises(ConfigError) as e:
        validate(raw)
    return e.value.problems


def test_default_file_is_valid_and_pinned():
    cfg = load_config(DEFAULT)
    assert cfg.model.repo == "Prasanna85/laya-issue-triage"
    assert cfg.model.revision == "76ece1fb0eb8b32bd5d8c509293c1692a2534805"
    assert cfg.model.max_len == 1024 and cfg.model.backend == "torch"
    assert cfg.thresholds == {"bug": 0.6033, "feature": 1.01, "question": 1.01}
    assert cfg.behaviour.mode == "dry-run"
    assert cfg.behaviour.comment_on_escalate is False  # D2: comments are opt-in
    assert cfg.labels.escalate == "triage: needs-human"
    assert (cfg.preprocess.max_code_lines, cfg.preprocess.max_body_chars) == (20, 6000)


def test_merge_is_deep_and_pure():
    default = {"a": {"x": 1, "y": [1, 2]}, "b": 2}
    user = {"a": {"y": [3]}, "c": 4}
    before = copy.deepcopy(default)
    assert merge(default, user) == {"a": {"x": 1, "y": [3]}, "b": 2, "c": 4}
    assert default == before


def test_user_file_overrides_defaults(tmp_path):
    user = tmp_path / "laya-triage.yml"
    user.write_text("gating:\n  thresholds:\n    question: 0.9\nbehaviour:\n  mode: apply\n  skip_authors: []\n")
    cfg = load_config(DEFAULT, user)
    assert cfg.thresholds == {"bug": 0.6033, "feature": 1.01, "question": 0.9}
    assert cfg.behaviour.mode == "apply" and cfg.behaviour.skip_authors == ()
    assert cfg.labels.bug == "type: bug"  # untouched defaults survive


def test_missing_or_empty_user_file_means_defaults(tmp_path):
    assert load_config(DEFAULT, tmp_path / "absent.yml") == load_config(DEFAULT)
    (tmp_path / "empty.yml").write_text("")
    assert load_config(DEFAULT, tmp_path / "empty.yml") == load_config(DEFAULT)


def test_every_problem_is_listed_together(raw):
    raw["model"]["backend"] = "onnx"
    raw["model"]["revision"] = "main"
    raw["behaviour"]["mode"] = "auto"
    raw["gating"]["thresholds"]["bug"] = 0
    raw["labels"]["feature"] = "TYPE: BUG"
    p = problems_of(raw)
    assert len(p) == 5
    assert any(x.startswith("model.backend") for x in p) and any(x.startswith("model.revision") for x in p)
    assert any(x.startswith("behaviour.mode") for x in p) and any(x.startswith("gating.thresholds.bug") for x in p)
    assert any(x.startswith("labels.feature: duplicates labels.bug") for x in p)


def test_unknown_and_missing_keys(raw):
    raw["extra"] = 1
    raw["behaviour"]["comment_on_escalation"] = True  # typo of a real key
    del raw["preprocess"]["max_body_chars"]
    p = problems_of(raw)
    assert "extra: unknown key" in p
    assert "behaviour.comment_on_escalation: unknown key" in p
    assert "preprocess.max_body_chars: missing" in p


@pytest.mark.parametrize("section,key,value", [
    ("model", "max_len", True),  # bool is not an int
    ("model", "max_len", "1024"),
    ("model", "repo", 5),
    ("behaviour", "comment_on_escalate", "yes"),
    ("behaviour", "skip_authors", "bot"),
    ("behaviour", "skip_authors", ["bot", 3]),
    ("labels", "bug", None),
])
def test_wrong_types(raw, section, key, value):
    raw[section][key] = value
    (p,) = problems_of(raw)
    assert p.startswith(f"{section}.{key}: must be")


def test_threshold_type_and_section_shape(raw):
    raw["gating"]["thresholds"]["bug"] = "0.6"
    raw["gating"]["thresholds"]["feature"] = False
    raw["labels"] = ["type: bug"]
    p = problems_of(raw)
    assert "gating.thresholds.bug: must be a number, got str" in p
    assert "gating.thresholds.feature: must be a number, got bool" in p
    assert "labels: must be a mapping" in p


@pytest.mark.parametrize("value,ok", [(0.0001, True), (0.6033, True), (1, True), (1.0, True), (1.01, True),
                                      (0, False), (-0.1, False), (1.0101, False), (2, False)])
def test_threshold_bounds(raw, value, ok):
    raw["gating"]["thresholds"]["question"] = value
    if ok:
        assert validate(raw).thresholds["question"] == float(value)
    else:
        assert problems_of(raw) == ["gating.thresholds.question: must be in (0, 1.01]"]


@pytest.mark.parametrize("revision", ["main", "76ECE1FB0EB8B32BD5D8C509293C1692A2534805",
                                      "76ece1fb0eb8b32bd5d8c509293c1692a253480", "v1.0.0", ""])
def test_revision_must_be_40_lowercase_hex(raw, revision):
    raw["model"]["revision"] = revision
    assert problems_of(raw) == ["model.revision: must be a 40-character lowercase hex commit SHA"]


def test_label_names_empty_or_duplicate(raw):
    raw["labels"]["question"] = "  "
    raw["labels"]["escalate"] = "Type: Bug"
    p = problems_of(raw)
    assert "labels.question: must not be empty" in p
    assert "labels.escalate: duplicates labels.bug (label names are case-insensitive)" in p


def test_preprocess_limits_are_checked(raw):
    raw["preprocess"]["max_code_lines"] = 5
    (p,) = problems_of(raw)
    assert p.startswith("preprocess: max_code_lines must be >=")


def test_bad_yaml_and_non_mapping(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("behaviour:\n  mode: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML at line"):
        load_config(DEFAULT, bad)
    lst = tmp_path / "list.yml"
    lst.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_config(DEFAULT, lst)
    with pytest.raises(ConfigError, match="cannot be read"):
        load_config(tmp_path / "no-default.yml")


def test_user_cannot_replace_a_section_with_a_scalar(tmp_path):
    user = tmp_path / "u.yml"
    user.write_text("gating: 0.5\n")
    with pytest.raises(ConfigError) as e:
        load_config(DEFAULT, user)
    assert e.value.problems == ["gating: must be a mapping"]
    assert "gating: must be a mapping" in str(e.value)


def test_shape_and_value_problems_are_reported_together(raw):
    raw["behaviour"]["colour"] = "red"
    raw["behaviour"]["mode"] = "auto"
    raw["model"]["max_len"] = "big"
    raw["gating"]["thresholds"]["feature"] = 3
    p = problems_of(raw)
    assert sorted(p) == sorted(["behaviour.colour: unknown key", "model.max_len: must be an integer, got str",
                                "gating.thresholds.feature: must be in (0, 1.01]",
                                "behaviour.mode: must be one of ['dry-run', 'apply']"])
