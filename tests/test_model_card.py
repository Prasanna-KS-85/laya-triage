"""results/phase3/make_model_card.py builds docs/MODEL_CARD.md from the committed metrics files (no network,
no model weights, no issue data)."""
import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results" / "phase3"))

import make_model_card as mc


@pytest.fixture(scope="module")
def sources():
    return mc.load_sources()


def test_card_builds_from_metrics_files(sources):
    card = mc.build_card(**sources)
    test, cfg = sources["test"], sources["config"]
    assert card.startswith("---\nlicense: apache-2.0\nbase_model: convaiinnovations/laya\nlanguage:\n- en\ntags:\n")
    assert f'REVISION = "{cfg["model"]["revision"]}"' in card
    assert 'device="cpu"' in card and 'HF_HUB_OFFLINE"] = "1"' in card
    m1 = test["systems"]["M1"]["metrics"]
    assert f"**{m1['cross_repo_macro_f1']:.4f}**" in card
    assert f"| {cfg['gating']['thresholds']['bug']:.4f} |" in card
    assert "precision target was missed on test" in card
    for heading in ("## Summary", "## Intended use", "## Not intended use", "## How to load", "## Training data",
                    "## Training procedure", "## Evaluation", "## Gating", "## Limitations", "## Citation"):
        assert f"\n{heading}\n" in card


def test_card_links_to_the_github_repository(sources):
    assert mc.GITHUB_URL == "https://github.com/Prasanna-KS-85/laya-triage"
    assert f"[laya-triage]({mc.GITHUB_URL})" in mc.build_card(**sources)


def test_committed_card_is_up_to_date(sources):
    assert mc.CARD.read_text() == mc.build_card(**sources), "run results/phase3/make_model_card.py"


def test_card_embeds_the_frozen_question(sources):
    from laya_triage.questions import ISSUE_TYPE_QUESTION

    card = mc.build_card(**sources)
    for c, text in ISSUE_TYPE_QUESTION["issue_type"]["criteria"].items():
        assert f'"{c}": "{text}"' in card
    assert ISSUE_TYPE_QUESTION["issue_type"]["instructions"] in card


@pytest.mark.parametrize("path", [("test", "systems", "M1", "metrics", "ece"), ("test", "gating", "config", "applied"),
                                  ("val", "gating", "M1", "per_label", "bug"), ("refit", "T_val"),
                                  ("config", "model", "repo"), ("items", "files")])
def test_missing_key_fails_loudly(sources, path):
    broken = copy.deepcopy(sources)
    d = broken[path[0]]
    for k in path[1:-1]:
        d = d[k]
    del d[path[-1]]
    with pytest.raises(KeyError, match="missing key"):
        mc.build_card(**broken)


def test_inconsistent_threshold_fails(sources):
    broken = copy.deepcopy(sources)
    broken["config"]["gating"]["thresholds"]["bug"] = 0.5
    with pytest.raises(ValueError, match="thresholds"):
        mc.build_card(**broken)


def test_ledger_row():
    md = "| run_id | epochs | notes |\n|---|---|---|\n| R1 | 4 | a \\| b |\n| R2 | 6 | x |\n"
    assert mc.ledger_row(md.replace("a \\| b", "ab"), "R2") == {"run_id": "R2", "epochs": "6", "notes": "x"}
    with pytest.raises(ValueError, match="cells"):
        mc.ledger_row(md, "R1")
    with pytest.raises(ValueError, match="no row"):
        mc.ledger_row(md, "R9")


def test_need():
    assert mc.need({"a": {"0.0": {"b": 1}}}, ("a", "0.0", "b"), "x") == 1
    with pytest.raises(KeyError, match=r"x: missing key 'a.c'"):
        mc.need({"a": {}}, "a.c", "x")


def test_b3_training_facts_come_from_phase2(sources):
    assert mc.b3_training(sources["phase2_md"]) == {"rows_per_classifier": 300}
    with pytest.raises(ValueError, match="pattern not found"):
        mc.b3_training(sources["phase2_md"].replace("were not read", "were read"))
    card = mc.build_card(**sources)
    assert "each on only that repository's 300 official-train rows" in " ".join(card.split())
