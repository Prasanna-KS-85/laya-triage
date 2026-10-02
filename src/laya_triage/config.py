"""Triage configuration: defaults deep-merged with the target repo's YAML, then validated (PROJECT_SPEC.md §12.3, FR-8).

`load_config(default_path, user_path)` reads `config/triage.default.yml` and, if it exists, the user's
`.github/laya-triage.yml`; `merge` deep-merges the user mapping over the defaults (lists and scalars are
replaced, mappings are merged); `validate` checks the result against SCHEMA and raises one ConfigError that
lists every problem (unknown keys, missing keys, wrong types, bad values). Nothing is guessed or coerced.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from laya_triage.preprocess import PreprocessConfig
from laya_triage.questions import LABELS

MODES = ("dry-run", "apply")
BACKENDS = ("torch",)  # onnx is v1.2
NEVER_MAX = 1.01  # thresholds live in (0, 1.01]; 1.01 = never auto-apply
REVISION_RE = re.compile(r"[0-9a-f]{40}")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")  # matched in full: no newline can pass
LABEL_KEYS = (*LABELS, "escalate")

# Leaf type tags: "str", "bool", "int", "number", "str_list". Mappings nest.
SCHEMA = {
    "model": {"repo": "str", "revision": "str", "max_len": "int", "backend": "str"},
    "labels": {k: "str" for k in LABEL_KEYS},
    "gating": {"thresholds": {k: "number" for k in LABELS}},
    "behaviour": {"mode": "str", "comment_on_escalate": "bool", "skip_if_labeled": "bool", "skip_authors": "str_list",
                  "create_missing_labels": "bool", "escalate_on_error": "bool"},
    "preprocess": {"max_code_lines": "int", "max_body_chars": "int"},
}


class ConfigError(Exception):
    """Invalid configuration. `problems` lists every bad key with what is wrong."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("invalid laya-triage config:\n" + "\n".join(f"- {p}" for p in self.problems))


@dataclass(frozen=True)
class ModelConfig:
    repo: str
    revision: str
    max_len: int
    backend: str


@dataclass(frozen=True)
class LabelsConfig:
    bug: str
    feature: str
    question: str
    escalate: str

    def for_class(self, label):
        return getattr(self, label)


@dataclass(frozen=True)
class BehaviourConfig:
    mode: str
    comment_on_escalate: bool
    skip_if_labeled: bool
    skip_authors: tuple[str, ...]
    create_missing_labels: bool
    escalate_on_error: bool


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    labels: LabelsConfig
    thresholds: dict  # {label: float} for every label in LABELS
    behaviour: BehaviourConfig
    preprocess: PreprocessConfig


def merge(default, user):
    """Deep merge: mappings merge key by key, anything else in `user` replaces the default. Pure."""
    if not isinstance(default, dict) or not isinstance(user, dict):
        return user
    out = dict(default)
    for k, v in user.items():
        out[k] = merge(default[k], v) if k in default else v
    return out


def _type_ok(tag, v):
    if tag == "str":
        return isinstance(v, str)
    if tag == "bool":
        return isinstance(v, bool)
    if tag == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if tag == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if tag == "str_list":
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    raise ValueError(tag)


_TYPE_NAMES = {"str": "a string", "bool": "true or false", "int": "an integer", "number": "a number",
               "str_list": "a list of strings"}


def _check_shape(raw, schema, prefix, problems, bad):
    """Unknown, missing and wrongly typed keys; `bad` collects the dotted keys whose values cannot be checked."""
    if not isinstance(raw, dict):
        problems.append(f"{prefix.rstrip('.') or 'config'}: must be a mapping")
        bad.add(prefix.rstrip("."))
        return
    for k in raw:
        if k not in schema:
            problems.append(f"{prefix}{k}: unknown key")
    for k, tag in schema.items():
        key = f"{prefix}{k}"
        if k not in raw:
            problems.append(f"{key}: missing")
            bad.add(key)
        elif isinstance(tag, dict):
            _check_shape(raw[k], tag, key + ".", problems, bad)
        elif not _type_ok(tag, raw[k]):
            problems.append(f"{key}: must be {_TYPE_NAMES[tag]}, got {type(raw[k]).__name__}")
            bad.add(key)


def validate(raw):
    """Config from a merged mapping, or ConfigError listing every problem (shape and values). Pure."""
    problems, bad = [], set()
    _check_shape(raw, SCHEMA, "", problems, bad)

    def ok(key):  # the value at `key` exists and has the right type, so its value can be checked
        parts = key.split(".")
        return not any(".".join(parts[:i]) in bad for i in range(1, len(parts) + 1))

    def get(key):
        v = raw
        for part in key.split("."):
            v = v[part]
        return v

    if ok("model.repo") and not REPO_RE.fullmatch(get("model.repo")):
        problems.append("model.repo: must look like owner/name (letters, digits, '_', '.', '-')")
    if ok("model.revision") and not REVISION_RE.fullmatch(get("model.revision")):
        problems.append("model.revision: must be a 40-character lowercase hex commit SHA")
    if ok("model.max_len") and get("model.max_len") <= 0:
        problems.append("model.max_len: must be positive")
    if ok("model.backend") and get("model.backend") not in BACKENDS:
        problems.append(f"model.backend: must be one of {list(BACKENDS)}")
    seen = {}
    for k in LABEL_KEYS:
        if not ok(f"labels.{k}"):
            continue
        name = get(f"labels.{k}")
        if not name.strip():
            problems.append(f"labels.{k}: must not be empty")
            continue
        folded = name.strip().casefold()
        if folded in seen:
            problems.append(f"labels.{k}: duplicates labels.{seen[folded]} (label names are case-insensitive)")
        else:
            seen[folded] = k
    for k in LABELS:
        if ok(f"gating.thresholds.{k}") and not 0 < get(f"gating.thresholds.{k}") <= NEVER_MAX:
            problems.append(f"gating.thresholds.{k}: must be in (0, {NEVER_MAX}]")
    if ok("behaviour.mode") and get("behaviour.mode") not in MODES:
        problems.append(f"behaviour.mode: must be one of {list(MODES)}")
    pcfg = None
    if ok("preprocess.max_code_lines") and ok("preprocess.max_body_chars"):
        try:
            pcfg = PreprocessConfig(max_code_lines=get("preprocess.max_code_lines"),
                                    max_body_chars=get("preprocess.max_body_chars"))
        except ValueError as e:
            problems.append(f"preprocess: {e}")
    if problems:
        raise ConfigError(problems)
    m, lab, th, b = raw["model"], raw["labels"], raw["gating"]["thresholds"], raw["behaviour"]
    return Config(
        model=ModelConfig(m["repo"], m["revision"], m["max_len"], m["backend"]),
        labels=LabelsConfig(*(lab[k] for k in LABEL_KEYS)),
        thresholds={k: float(th[k]) for k in LABELS},
        behaviour=BehaviourConfig(b["mode"], b["comment_on_escalate"], b["skip_if_labeled"], tuple(b["skip_authors"]),
                                  b["create_missing_labels"], b["escalate_on_error"]),
        preprocess=pcfg,
    )


def _read_yaml(path, what):
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError([f"{what} {path}: cannot be read ({type(e).__name__})"]) from None
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError([f"{what} {path}: not valid YAML{where}"]) from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError([f"{what} {path}: must be a YAML mapping"])
    return data


def load_config(default_path, user_path=None):
    """Defaults from `default_path`, deep-merged with `user_path` when that file exists (FR-8)."""
    default = _read_yaml(default_path, "default config")
    user = _read_yaml(user_path, "config") if user_path is not None and Path(user_path).is_file() else {}
    return validate(merge(default, user))
