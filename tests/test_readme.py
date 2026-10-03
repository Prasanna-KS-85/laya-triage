"""docs/make_readme.py builds README.md, docs/DEEP_DIVE.md, docs/USING_THE_ACTION.md and docs/USING_THE_MODEL.md from
their templates and the committed result files (no network, no model weights, no issue data), and the templates hold
no typed numbers (PROJECT_SPEC.md §16 rule 11)."""
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
# system names, hardware and format names, the upstream benchmark's name, list numbering, file paths, the account name,
# and the README heading "The 30-second explanation".
IDENTIFIERS = re.compile(
    r"\bv\d+(?:\.\d+)*(?:-\w+)?|\bN?FR-\d+|§[\d.]+|\bS\d\d\b|\bphase\d-complete|\b[BM][0-3]\b|NLBSE'24|\bT4\b|\bINT8\b"
    r"|\bfp32\b|\bF1\b|\bp\d\d\b|\bX64\b|\bSHA-256\b|\bApache-2\.0\b|\bubuntu-\d+\.\d+|\bpython3\b|^\d+\. "
    r"|\bphase\d\b|\btest_R1b\b|sha256|Prasanna-KS-85|\bM1_|\bT = 1\b|\brule 11\b|\b30-second\b", re.MULTILINE)
DD, UA, UM = (ROOT / "docs" / f"{n}.md" for n in ("DEEP_DIVE", "USING_THE_ACTION", "USING_THE_MODEL"))
README_HEADINGS = ["The 30-second explanation", "One worked example", "Results and limitations", "Choose how to use it",
                   "Technical deep dive"]
DD_HEADINGS = ["Contents", "Glossary", "Architecture at a glance", "How it works (Laya internals)",
               "Why Laya instead of a plain classifier", "Training and data", "Calibration", "Evaluation",
               "Gating and the missed target", "Runtime and the latency probe", "Deployment and caches", "Security",
               "Reproduce", "Limitations", "Extending to more classes", "Roadmap", "Credits and license",
               "How this was built"]
UA_HEADINGS = ["Add the workflow", "Permissions", "Create the labels", "Warm the caches once",
               "First run: stay in dry-run and read the job summaries", "Backfill", "Switch to apply",
               "Escalation comments", "Action inputs", "Configuration", "Troubleshooting"]
UM_HEADINGS = ["Install", "Classify one issue", "What the output means", "Apply the gate yourself", "Offline use",
               "Limitations"]
GLOSSARY = ["Probability", "Confidence", "Threshold", "Precision", "Coverage", "Escalate", "Dry-run", "Calibration"]


@pytest.fixture(scope="module")
def sources():
    return mr.load_sources()


@pytest.fixture(scope="module")
def docs(sources):
    return mr.build_docs(**sources)


@pytest.fixture(scope="module")
def readme(docs):
    return docs[mr.README]


def h2(text):
    return re.findall(r"^## (.+)$", text, re.MULTILINE)


def flat(text):
    return " ".join(text.split())


def slug(heading):
    """GitHub's anchor for a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_committed_docs_are_up_to_date(docs):
    assert set(docs) == set(mr.OUTPUTS.values())
    for path, text in docs.items():
        assert path.read_text() == text, f"run docs/make_readme.py ({path.name})"


def test_check_flag(capsys):
    mr.main(["--check"])
    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize("template", sorted(mr.OUTPUTS), ids=lambda p: p.name)
def test_template_has_no_typed_numbers(template):
    text = template.read_text()
    left = IDENTIFIERS.sub("", PLACEHOLDER.sub("", text))
    bad = [line for line in left.splitlines() if re.search(r"\d", line)]
    assert not bad, f"numbers must be placeholders (rule 11): {bad}"


def test_every_placeholder_is_resolved(docs):
    for path, text in docs.items():
        assert not PLACEHOLDER.search(text), path.name


def test_unknown_placeholder_fails():
    with pytest.raises(KeyError, match="unknown placeholder"):
        mr.render("a {{nope}} b", {"x": 1})


def test_sections_and_headline_numbers(docs, sources):
    assert h2(docs[mr.README]) == README_HEADINGS
    assert h2(docs[DD]) == DD_HEADINGS
    assert h2(docs[UA]) == UA_HEADINGS
    assert h2(docs[UM]) == UM_HEADINGS
    m1 = sources["test"]["systems"]["M1"]["metrics"]
    for path in (mr.README, DD):
        assert f"**{m1['cross_repo_macro_f1']:.4f}**" in docs[path]
        assert "precision target was missed on test" in docs[path]
    assert "<!-- Demo: add docs/demo.gif" in docs[mr.README]


def test_readme_front_door(readme):
    lines = readme.splitlines()
    assert len(lines) <= 200
    first_image = re.search(r"!\[[^\]]*\]\([^)]*\)|<img\b", readme)
    assert first_image.group(0) == "![Laya Triage](docs/banner_image.png)"
    assert readme.index(first_image.group(0)) < readme.index("\n## ")
    assert (ROOT / "docs" / "banner_image.png").is_file()
    assert re.findall(r"^### (.+)$", readme, re.MULTILINE) == ["Path A: the GitHub Action", "Path B: the model on its own"]
    text = flat(readme)
    assert "Path A, the GitHub Action, needs no code." in text
    assert "for people who want the probabilities in their own Python code" in text
    assert "[docs/USING_THE_ACTION.md](docs/USING_THE_ACTION.md)" in text
    assert "[docs/USING_THE_MODEL.md](docs/USING_THE_MODEL.md)" in text
    assert "[PROJECT_SPEC.md](PROJECT_SPEC.md)" in text or "[`PROJECT_SPEC.md`](PROJECT_SPEC.md)" in text
    assert "dry-run, the default" in text and "changes nothing" in text
    assert "**Confidence-gated GitHub issue triage powered by a fine-tuned Laya decision model.**" in readme
    assert "**No hosted inference server. No per-request API fees. Human review when confidence is insufficient.**" in readme
    assert "> **Project status:** `v" in readme


def test_readme_mermaid_has_five_boxes_and_no_digits(readme):
    (block,) = re.findall(r"```mermaid\n(.*?)\n```", readme, re.DOTALL)
    assert not re.search(r"\d", block)
    boxes = re.findall(r"\b[A-Z]\s*[\[{]([^\]}]+)[\]}]", block)
    assert boxes == ["Issue opened", "Clean text", "Laya model", "Confidence gate", "Apply label or escalate to a human"]


def test_architecture_image_only_in_the_deep_dive(docs):
    assert "architecture.png" not in docs[mr.README]
    assert "[![Laya Triage: master architecture and workflow](architecture.png)](architecture.png)" in docs[DD]
    for path in (UA, UM):
        assert "architecture.png" not in docs[path]


def test_both_misses_are_stated_in_the_readme(readme, sources):
    text = flat(readme)
    rt = mr.runtime(sources["phase4_md"], sources["spec_md"], sources["test"])
    assert "precision target was missed on test" in text
    assert rt["nfr1_status"] == "not met"
    assert (f"the slowest one issue in twenty took {rt['probe_p95_secs']} seconds or longer, against a goal of at most "
            f"{rt['nfr1_goal']}: the speed target was missed ([details](docs/DEEP_DIVE.md#runtime-and-the-latency-probe))") in text
    assert f"took {rt['cold']} on the first run and {rt['warm_range']} on later runs, once caches are restored" in text
    for warm in rt["warm"].split(" and "):
        assert warm in rt["warm_range"]
    assert "only labels the bugs it is most sure about and sends the rest to a human" in text
    assert "p95" not in readme and "NFR-1" not in readme


def test_readme_plain_language_lines(readme, sources, docs):
    text = flat(readme)
    acc = sources["test"]["systems"]["M1"]["metrics"]["accuracy"]
    assert f"On this benchmark the fine-tuned model picks the right type for about {100 * acc:.0f}% of issues." in text
    assert ("Unfamiliar terms (precision, threshold, calibration) are explained in the "
            "[glossary](docs/DEEP_DIVE.md#glossary).") in text
    assert "**Step three.** When you trust what dry-run shows, add `.github/laya-triage.yml`:\n\n```yaml\nbehaviour:\n  mode: apply\n```" in readme
    assert ("In apply mode the labels must already exist in your repository; the guide has the commands to create them."
            ) in text
    labels = sources["config"]["labels"]
    for name in labels.values():
        assert f'gh label create "{name}"' in docs[UA]


def test_worked_example_values_come_from_expected_local(readme, sources):
    fx = sources["expected_local"]["fixtures"]
    s01 = fx["S01"]
    text = flat(readme)
    probs = ", ".join(f"`{lab}` {f'**{p:.4f}**' if lab == 'bug' else f'{p:.4f}'}" for lab, p in s01["probabilities"].items())
    assert f"One forward pass gives one probability per label: {probs}." in text
    assert f"the `bug` threshold is {s01['threshold']:.4f}" in text
    assert "<summary>Show the issue text</summary>" in readme
    for key in ("S02", "S05"):
        assert f"| {fx[key]['answer_confidence']:.4f} |" in readme
    broken = copy.deepcopy(sources)
    broken["expected_local"]["fixtures"]["S02"]["probabilities"]["bug"] = 0.9
    broken["expected_local"]["fixtures"]["S02"]["answer_confidence"] = 0.9
    with pytest.raises(ValueError, match="S02 no longer shows the decision"):
        mr.build_docs(**broken)


def test_glossary_comes_first_with_eight_terms(docs):
    dd = docs[DD]
    section = dd.split("\n## Glossary\n", 1)[1].split("\n## ", 1)[0]
    assert re.findall(r"^- \*\*([\w-]+)\.\*\*", section, re.MULTILINE) == GLOSSARY


def test_snippets_are_shared_verbatim(docs, sources):
    import make_model_card as mc

    snippet = sources["snippets"]["snippet_classify_one"]
    card = mc.build_card(**{k: sources[k] for k in ("test", "val", "config", "refit", "items", "experiments_md",
                                                    "spec_md", "phase2_md", "classify_snippet")})
    for text in (docs[mr.README], docs[UM], card):
        assert f"```python\n{snippet}\n```" in text
    assert f"```yaml\n{sources['snippets']['snippet_consumer_minimal']}\n```" in docs[mr.README]
    assert "git ls-remote https://github.com/Prasanna-KS-85/laya-triage 'refs/tags/v" in docs[mr.README]


def test_quick_start_workflow(docs):
    ua = docs[UA]
    block = re.search(r"```yaml\nname: Laya Triage\n(.*?)\n```", ua, re.DOTALL).group(0).strip("`yaml\n")
    wf = yaml.safe_load(block)
    assert wf["permissions"] == {"issues": "write", "contents": "read"}
    uses = [s["uses"] for s in wf["jobs"]["triage"]["steps"]]
    assert any(re.fullmatch(r"Prasanna-KS-85/laya-triage@v\d+\.\d+\.\d+", u) for u in uses)
    assert "<" not in block
    assert "warm-cache" in block and "schedule" in block


def test_config_table_covers_every_key(sources, docs):
    table = mr.config_table(sources["config"])
    keys = {k for k, _ in mr.flatten(sources["config"])}
    assert {m.group(1) for m in re.finditer(r"^\| `([\w.]+)` \|", table, re.MULTILINE)} == keys
    assert table in docs[UA]
    broken = copy.deepcopy(sources["config"])
    broken["behaviour"]["brand_new_key"] = True
    with pytest.raises(ValueError, match="undescribed keys.*brand_new_key"):
        mr.config_table(broken)


def test_action_inputs_come_from_action_yml(sources, docs):
    table = mr.action_inputs_table(sources["action"])
    for name in sources["action"]["inputs"]:
        assert f"| `{name}` |" in table
    assert table in docs[UA]


def test_local_links_exist(docs):
    anchors = {p: {slug(h) for h in re.findall(r"^#+ (.+)$", t, re.MULTILINE)} for p, t in docs.items()}
    for path, text in docs.items():
        visible = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # the demo placeholder is a comment until docs/demo.gif exists
        visible = re.sub(r"```.*?```", "", visible, flags=re.DOTALL)
        for target in re.findall(r"\]\((?!https?://)([^)\s]+)\)", visible):
            file, _, anchor = target.partition("#")
            dest = (path.parent / file).resolve() if file else path
            assert dest.exists(), f"{path.name}: {target}"
            if anchor and dest in anchors:
                assert anchor in anchors[dest], f"{path.name}: {target}"


@pytest.mark.parametrize("path", [("test", "gating", "config", "applied"), ("test", "latency", "test_run"),
                                  ("val", "systems", "M1", "metrics", "cross_repo_macro_f1"),
                                  ("posthoc", "A_near_duplicates"), ("expected_local", "fixtures"),
                                  ("action", "inputs"), ("issues", "fixtures")])
def test_missing_key_fails_loudly(sources, path):
    broken = copy.deepcopy(sources)
    d = broken[path[0]]
    for k in path[1:-1]:
        d = d[k]
    del d[path[-1]]
    with pytest.raises(KeyError, match="missing key|fixtures|A_near_duplicates|inputs"):
        mr.build_docs(**broken)


def test_status_words_follow_the_numbers(sources):
    broken = copy.deepcopy(sources)
    broken["test"]["gating"]["config"]["applied"]["precision"] = 0.95
    built = mr.build_docs(**broken)
    for path in (mr.README, DD):
        assert "precision target was reached on test" in built[path]


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


def test_b3_wording_agrees_with_phase2_and_the_model_card(docs, sources):
    import make_model_card as mc

    rows = mc.b3_training(sources["phase2_md"])["rows_per_classifier"]
    card = mc.build_card(**{k: sources[k] for k in ("test", "val", "config", "refit", "items", "experiments_md",
                                                    "spec_md", "phase2_md", "classify_snippet")})
    deep_dive = docs[DD]
    for text in (deep_dive, card):
        flat_text = " ".join(text.split())
        assert f"each on only that repository's {rows} official-train rows" in flat_text
        assert "RoBERTa and fastText" in flat_text and "not inspected" in flat_text
    assert "on all" not in re.search(r"B3 trains one classifier[^.]*\.", " ".join(deep_dive.split())).group(0)


def test_wording_requested_for_results_gating_and_limitations(docs):
    text = flat(docs[DD])
    assert "That evaluation was run once, after the thresholds were frozen" in text
    assert "labelled post-hoc" in text
    assert "on test the target was met for 0 of 3 labels (on validation only `bug` had a 0.90 bootstrap lower bound)" in text
    assert "would be labelled `bug` in apply mode" in text and "so it would be applied automatically" not in text
    assert "the label an author can steer towards is `bug`, the only auto-applied label" in text
    assert "a mistake a maintainer can remove" in text
    assert "M1 was fine-tuned once" in text and "not selection among models" in text
    assert "[data/DATA_CARD.md](../data/DATA_CARD.md)" in text
    readme = flat(docs[mr.README])
    assert "The test evaluation was pre-declared and run once" in readme
    assert "later exploratory analyses reused its predictions and are labelled post-hoc" in readme


def test_architecture_source_has_the_corrected_wording():
    # docs/architecture.png is exported from this file and text tests cannot read images, so guard the source.
    text = flat((ROOT / "docs" / "architecture.drawio").read_text())
    assert "scored once" not in text and "scored exactly once" not in text
    assert "pre-declared" in text


def test_no_scored_once_wording(docs):
    import make_model_card as mc

    for path in (*docs, mc.CARD):
        text = flat(docs.get(path) or path.read_text())
        assert "scored once" not in text and "scored exactly once" not in text, path.name


def test_why_laya_and_more_classes(docs):
    readme = flat(docs[mr.README])
    assert "**Why Laya instead of a plain classifier?**" in readme
    assert "we did not run a same-protocol head-to-head against one, so we make no accuracy claim" in readme
    assert ("- **Exactly three classes.** Adding a class is not a setting: it needs new labelled data, retraining and "
            "recalibrating ([details](docs/DEEP_DIVE.md#extending-to-more-classes)).") in docs[mr.README]
    for anchor in ("why-laya-instead-of-a-plain-classifier", "extending-to-more-classes"):
        assert f"(docs/DEEP_DIVE.md#{anchor})" in readme.split("## Technical deep dive", 1)[1]
    dd = h2(docs[DD])
    assert dd.index("How it works (Laya internals)") + 1 == dd.index("Why Laya instead of a plain classifier")
    assert dd.index("Limitations") + 1 == dd.index("Extending to more classes")
    assert "a same-protocol head-to-head against a plain fine-tuned encoder" in flat(docs[DD].split("\n## Roadmap\n")[1])
    for path in (mr.README, DD):
        text = flat(docs[path]).lower()
        assert "more accurate" not in text and "better calibrated" not in text, path.name
