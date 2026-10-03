"""docs/snippets: the example the README, docs/USING_THE_MODEL.md and the model card show verbatim
(classify_one.py) and the minimal consumer workflow (consumer-minimal.yml). The snippet carries its own constants
because config/triage.default.yml is not installed with the package; they must equal the config. The slow test runs
the snippet against the pinned model from the local Hugging Face cache, offline."""
import ast
import contextlib
import io
import runpy
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SNIPPET = ROOT / "docs" / "snippets" / "classify_one.py"
MINIMAL = ROOT / "docs" / "snippets" / "consumer-minimal.yml"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_workflows import check, load, spec_example


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load((ROOT / "config" / "triage.default.yml").read_text())


def constants():
    tree = ast.parse(SNIPPET.read_text())
    return {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id.isupper()}


def test_snippet_constants_equal_the_config(config):
    assert constants() == {"REPO": config["model"]["repo"], "REVISION": config["model"]["revision"],
                           "MAX_LEN": config["model"]["max_len"], "BUG_THRESHOLD": config["gating"]["thresholds"]["bug"]}


def test_snippet_preprocess_defaults_equal_the_config(config):
    from laya_triage.preprocess import PreprocessConfig

    d = PreprocessConfig()
    assert (d.max_code_lines, d.max_body_chars) == (config["preprocess"]["max_code_lines"],
                                                    config["preprocess"]["max_body_chars"])


def test_snippet_compiles_is_short_and_imports_the_frozen_parts():
    text = SNIPPET.read_text()
    compile(text, str(SNIPPET), "exec")
    assert len(text.splitlines()) <= 25
    assert "from laya_triage.preprocess import preprocess" in text
    assert "from laya_triage.classifier import Classifier" in text
    assert "ISSUE_TYPE_QUESTION" not in text and "def preprocess" not in text  # never retyped


def test_minimal_workflow_passes_the_workflow_rules():
    assert check(MINIMAL.read_text(), "docs/snippets/consumer-minimal.yml") == []


def test_minimal_workflow_agrees_with_the_spec_example():
    mini, spec = load(MINIMAL.read_text()), load(spec_example())
    assert mini["on"] == {"issues": spec["on"]["issues"]}
    assert mini["permissions"] == spec["permissions"]
    for key in ("runs-on", "timeout-minutes"):
        assert mini["jobs"]["triage"][key] == spec["jobs"]["triage"][key]
    m_steps, s_steps = mini["jobs"]["triage"]["steps"], spec["jobs"]["triage"]["steps"]
    assert m_steps[0] == s_steps[0]  # the pinned checkout
    tag = spec_example().split("laya-triage@", 1)[1].split("# ", 1)[1].split()[0]
    assert m_steps[1] == {"uses": f"Prasanna-KS-85/laya-triage@{tag}"}  # dry-run default, github.token default


@pytest.mark.slow
def test_snippet_runs_offline(offline_classifier):
    from laya_triage.questions import LABELS

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        g = runpy.run_path(str(SNIPPET), run_name="__main__")
    lines = out.getvalue().splitlines()
    result = g["result"]
    assert tuple(result.probabilities) == LABELS
    assert abs(sum(result.probabilities.values()) - 1) < 1e-4
    assert result.model_revision == constants()["REVISION"]
    assert lines[0] == str(result.probabilities)
    assert lines[1] in ("apply bug", "escalate to a human")
