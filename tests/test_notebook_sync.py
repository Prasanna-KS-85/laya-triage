"""training/finetune_kaggle.ipynb stays in sync with the repo (PROJECT_SPEC.md §9.3, Phase 3 task 1).

No network, no model weights: the notebook is only read and parsed, never executed.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import make_items

from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

NOTEBOOK = ROOT / "training/finetune_kaggle.ipynb"
MAKE_ITEMS_MAGIC = "%%writefile /kaggle/working/make_items.py\n"
TRAIN_DDP_MAGIC = "%%writefile /kaggle/working/train_ddp.py\n"
PINS = {"laya": "0.3.23", "transformers": "5.18.0", "tokenizers": "0.23.2", "huggingface_hub": "1.33.0",
        "safetensors": "0.8.0"}


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def sources(nb, kind="code"):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == kind]


def cell_starting_with(nb, prefix):
    hits = [s for s in sources(nb) if s.startswith(prefix)]
    assert len(hits) == 1, f"{len(hits)} cells start with {prefix!r}"
    return hits[0]


def test_valid_notebook_without_outputs(nb):
    assert nb["nbformat"] == 4
    assert nb["cells"][0]["cell_type"] == "markdown"
    for c in nb["cells"]:
        assert c["cell_type"] in ("code", "markdown")
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_attribution_first_cell(nb):
    first = sources(nb, "markdown")[0]
    for needle in ("Apache License 2.0", "d8a2e59781ca135169a36095056132e273cd9938",
                   "ae3222b3fcdf424254a2c726d72161f671a86d95", "Changes from upstream"):
        assert needle in first


def test_make_items_cell_is_byte_identical_to_module(nb):
    cell = cell_starting_with(nb, MAKE_ITEMS_MAGIC)
    module = (ROOT / "training/make_items.py").read_text(encoding="utf-8")
    assert cell[len(MAKE_ITEMS_MAGIC):] == module


def test_make_items_question_copy_matches_frozen_contract():
    assert make_items.ISSUE_TYPE_QUESTION == ISSUE_TYPE_QUESTION
    assert make_items.LABELS == LABELS
    assert tuple(make_items.ISSUE_TYPE_QUESTION["issue_type"]["criteria"]) == LABELS


def test_make_items_imports_only_stdlib_transformers_laya():
    src = (ROOT / "training/make_items.py").read_text(encoding="utf-8")
    roots = {m.split(".")[0] for m in re.findall(r"^(?:from|import) ([\w.]+)", src, flags=re.MULTILINE)}
    assert roots <= {"argparse", "hashlib", "json", "os", "shutil", "tempfile", "transformers", "laya"}, roots


def test_train_ddp_cell_compiles_and_passes_ruff_f_rules(nb, tmp_path):
    body = cell_starting_with(nb, TRAIN_DDP_MAGIC)[len(TRAIN_DDP_MAGIC):]
    compile(body, "train_ddp.py", "exec")
    path = tmp_path / "train_ddp.py"
    path.write_text(body, encoding="utf-8")
    res = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache", "--isolated", "--select", "F", str(path)],
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout + res.stderr


def test_train_ddp_keeps_upstream_calibration_and_seeds_torch(nb):
    body = cell_starting_with(nb, TRAIN_DDP_MAGIC)
    for needle in ("CALIB_SEED = 20260922", "CALIB_MAX = 400", 'torch.manual_seed(run["seed"] + rank)',
                   'torch.cuda.manual_seed(run["seed"] + rank)', 'cfg["model_name"] = "laya-issue-triage"',
                   'cfg.pop("temperature_by_options", None)', 'cfg["max_len"] = run["max_len"]'):
        assert needle in body


def test_no_create_repo_and_no_token_literal(nb):
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "create_repo" not in text
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert 'get_secret("HF_TOKEN")' in cell_starting_with(nb, "# 14. Push")


def test_first_code_cell_holds_the_pins(nb):
    first = sources(nb)[0]
    assert first.splitlines()[0] == '!pip install -q "laya==0.3.23" "transformers==5.18.0"'
    for name, version in PINS.items():
        assert f'"{name}": "{version}"' in first
        assert f'assert {name}.__version__ == PINS["{name}"]' in first
    assert '"laya==0.3.23"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_constants_and_expected_hashes_match_results(nb):
    consts = cell_starting_with(nb, "# 3. Constants")
    for needle in ("SMOKE = True\n", 'REPO = "Prasanna85/laya-issue-triage"',
                   'BASE_REV = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"', "MAX_LEN = 1024\n",
                   "HEAD_MAX_LEN = 256\n", "SEED = 42\n", "LABEL_SMOOTHING = 0.0"):
        assert needle in consts
    exp = json.loads((ROOT / "results/phase3/items_sha256.json").read_text(encoding="utf-8"))
    for group in ("items", "smoke"):
        for entry in exp[group].values():
            assert entry["sha256"] in consts
    for name, f in exp["files"].items():
        assert f["sha256"] in consts and str(f["bytes"]) in consts, name


def test_base_download_is_pinned_and_anonymous(nb):
    cell = cell_starting_with(nb, "# 5. Base checkpoint")
    assert "revision=BASE_REV" in cell and "token=False" in cell
    assert '["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]' in cell


def test_regenerating_gives_the_committed_notebook(tmp_path):
    out = tmp_path / "finetune_kaggle.ipynb"
    res = subprocess.run([sys.executable, str(ROOT / "training/build_notebook.py"), "--out", str(out)],
                         capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout + res.stderr
    assert out.read_bytes() == NOTEBOOK.read_bytes(), "run training/build_notebook.py; never edit the notebook by hand"
