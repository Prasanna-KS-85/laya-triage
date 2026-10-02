"""docs/make_readme.py builds README.md from docs/README.template.md and the committed result files (no network, no
model weights, no issue data), and the template holds no typed numbers (PROJECT_SPEC.md §16 rule 11)."""
import copy
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

import make_readme as mr

PLACEHOLDER = re.compile(r"\{\{\s*\w+\s*\}\}")
# Identifiers that contain digits but are not metrics: versions and tags, requirement ids, spec sections, fixture ids,
# system names, hardware and format names, the upstream benchmark's name, list numbering, file paths, the account name.
IDENTIFIERS = re.compile(
    r"\bv\d+(?:\.\d+)*(?:-\w+)?|\bN?FR-\d+|§[\d.]+|\bS\d\d\b|\bphase\d-complete|\b[BM][0-3]\b|NLBSE'24|\bT4\b|\bINT8\b"
    r"|\bfp32\b|\bF1\b|\bp\d\d\b|\bX64\b|\bSHA-256\b|\bApache-2\.0\b|\bubuntu-\d+\.\d+|\bpython3\b|^\d+\. "
    r"|\bphase\d\b|\btest_R1b\b|sha256|Prasanna-KS-85|\bM1_|\bT = 1\b|\brule 11\b", re.MULTILINE)


@pytest.fixture(scope="module")
def sources():
    return mr.load_sources()


@pytest.fixture(scope="module")
def readme(sources):
    return mr.build_readme(**sources)


def test_committed_readme_is_up_to_date(readme):
    assert mr.README.read_text() == readme, "run docs/make_readme.py"


def test_check_flag(capsys):
    mr.main(["--check"])
    assert "up to date" in capsys.readouterr().out


def test_template_has_no_typed_numbers():
    text = mr.TEMPLATE.read_text()
    left = IDENTIFIERS.sub("", PLACEHOLDER.sub("", text))
    bad = [line for line in left.splitlines() if re.search(r"\d", line)]
    assert not bad, f"numbers must be placeholders (rule 11): {bad}"


def test_every_placeholder_is_resolved(readme):
    assert not PLACEHOLDER.search(readme)


def test_unknown_placeholder_fails():
    with pytest.raises(KeyError, match="unknown placeholder"):
        mr.render("a {{nope}} b", {"x": 1})


def test_sections_and_headline_numbers(readme, sources):
    for heading in ("What and why", "Quick start", "Configuration", "How it works", "Results",
                    "Runtime on GitHub-hosted runners", "Limitations", "Reproduce", "How this was built", "Roadmap",
                    "Credits and license"):
        assert f"\n## {heading}\n" in readme
    m1 = sources["test"]["systems"]["M1"]["metrics"]
    assert f"**{m1['cross_repo_macro_f1']:.4f}**" in readme
    assert "precision target was missed on test" in readme
    assert "<!-- Demo: add docs/demo.gif" in readme


def test_quick_start_workflow(readme, sources):
    block = re.search(r"```yaml\nname: Laya Triage\n(.*?)\n```", readme, re.DOTALL).group(0).strip("`yaml\n")
    wf = yaml.safe_load(block)
    assert wf["permissions"] == {"issues": "write", "contents": "read"}
    uses = [s["uses"] for s in wf["jobs"]["triage"]["steps"]]
    assert any(re.fullmatch(r"Prasanna-KS-85/laya-triage@v\d+\.\d+\.\d+", u) for u in uses)
    assert "<" not in block
    assert "warm-cache" in block and "schedule" in block


def test_config_table_covers_every_key(sources):
    table = mr.config_table(sources["config"])
    keys = {k for k, _ in mr.flatten(sources["config"])}
    assert {m.group(1) for m in re.finditer(r"^\| `([\w.]+)` \|", table, re.MULTILINE)} == keys
    broken = copy.deepcopy(sources["config"])
    broken["behaviour"]["brand_new_key"] = True
    with pytest.raises(ValueError, match="undescribed keys.*brand_new_key"):
        mr.config_table(broken)


def test_action_inputs_come_from_action_yml(sources):
    table = mr.action_inputs_table(sources["action"])
    for name in sources["action"]["inputs"]:
        assert f"| `{name}` |" in table


def test_local_links_exist(readme):
    visible = re.sub(r"<!--.*?-->", "", readme, flags=re.DOTALL)  # the demo placeholder is a comment until docs/demo.gif exists
    for target in re.findall(r"\]\((?!https?://|#)([^)\s]+)\)", visible):
        assert (ROOT / target).exists(), target


@pytest.mark.parametrize("path", [("test", "gating", "config", "applied"), ("test", "latency", "test_run"),
                                  ("val", "systems", "M1", "metrics", "cross_repo_macro_f1"),
                                  ("posthoc", "A_near_duplicates"), ("expected_local", "fixtures"),
                                  ("action", "inputs")])
def test_missing_key_fails_loudly(sources, path):
    broken = copy.deepcopy(sources)
    d = broken[path[0]]
    for k in path[1:-1]:
        d = d[k]
    del d[path[-1]]
    with pytest.raises(KeyError, match="missing key|fixtures|A_near_duplicates|inputs"):
        mr.build_readme(**broken)


def test_status_words_follow_the_numbers(sources):
    broken = copy.deepcopy(sources)
    broken["test"]["gating"]["config"]["applied"]["precision"] = 0.95
    assert "precision target was reached on test" in mr.build_readme(**broken)


def test_runtime_roles_are_checked(sources):
    md = sources["phase4_md"]
    with pytest.raises(ValueError, match="is not the cold issue run"):
        mr.runtime(md.replace("both cold", "both restored", 1), sources["spec_md"], sources["test"])
    with pytest.raises(ValueError, match="pattern not found"):
        mr.runtime(md.replace("**Latency probe (NFR-1", "**Latency (NFR-1"), sources["spec_md"], sources["test"])
    rt = mr.runtime(md, sources["phase4_md"] and sources["spec_md"], sources["test"])
    assert rt["nfr1_status"] == "not met" and rt["nfr2_status"] == "met" and rt["nfr3_status"] == "met"


def test_table_rows():
    md = "x\n| Run | A |\n|---|---|\n| 1 | a |\n| 2 | |\n\n| Run | B |\n|---|---|\n| 3 | c |\n"
    assert mr.table_rows(md, "Run", "t") == {"1": {"Run": "1", "A": "a"}, "2": {"Run": "2", "A": ""}}
    with pytest.raises(ValueError, match="cells"):
        mr.table_rows("| Run | A |\n|---|---|\n| 1 | a | b |\n", "Run", "t")
    with pytest.raises(ValueError, match="no table"):
        mr.table_rows(md, "Nope", "t")


def test_consumer_workflow_requires_the_placeholder():
    spec = "### 12.5 Example consumer workflow\n\n```yaml\npermissions:\n  issues: write\n  contents: read\n" \
           "uses: Prasanna-KS-85/laya-triage@v1.0.0\n```\n"
    with pytest.raises(ValueError, match="SHA placeholder"):
        mr.consumer_workflow(spec)


def test_b3_wording_agrees_with_phase2_and_the_model_card(readme, sources):
    import make_model_card as mc

    rows = mc.b3_training(sources["phase2_md"])["rows_per_classifier"]
    card = mc.build_card(**{k: sources[k] for k in ("test", "val", "config", "refit", "items", "experiments_md",
                                                    "spec_md", "phase2_md")})
    for text in (readme, card):
        flat = " ".join(text.split())
        assert f"each on only that repository's {rows} official-train rows" in flat
        assert "RoBERTa and fastText" in flat and "not inspected" in flat
    assert "on all" not in re.search(r"B3 trains one classifier[^.]*\.", " ".join(readme.split())).group(0)


def test_wording_requested_for_results_gating_and_limitations(readme):
    flat = " ".join(readme.split())
    assert "That evaluation was run once, after the thresholds were frozen" in flat
    assert "labelled post-hoc" in flat and "scored once" not in flat
    assert "on test the target was met for 0 of 3 labels (on validation only `bug` had a 0.90 bootstrap lower bound)" in flat
    assert "would be labelled `bug` in apply mode" in flat and "so it would be applied automatically" not in flat
    assert "the label an author can steer towards is `bug`, the only auto-applied label" in flat
    assert "a mistake a maintainer can remove" in flat
    assert "M1 was fine-tuned once" in flat and "not selection among models" in flat
    assert "[data/DATA_CARD.md](data/DATA_CARD.md)" in flat
