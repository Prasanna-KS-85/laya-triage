"""docs/report/make_report.py builds docs/report/REPORT.md, the v1.0 technical report, from its chapter templates and
the committed files (no network, no model weights, no issue data). These tests check that the committed report is up
to date, that the templates hold no typed numbers and every prose-only value is registered with its source text
(PROJECT_SPEC.md §16 rule 11), the wording rules for the results, that no benchmark issue text appears, and the
tags snapshot. The PDF rendering test is `slow` (it needs the `report` extra)."""
import copy
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "report"))

import make_report as mrp

# Identifiers that contain digits but are not metrics. The README's list (tests/test_readme.py) plus the report's own:
# phase names and tags, decision-record, risk, question, run and wording ids, the other benchmark year, SHA-1.
IDENTIFIERS = re.compile(
    r"\bv\d+(?:\.\d+)*(?:-\w+)?|\bN?FR-\d+|§[\d.]+|\bS\d\d\b|\bphase\d[\w-]*|\b[BM][0-3]\b|NLBSE'2[34]|\bT4\b|\bINT8\b"
    r"|\bfp32\b|\bF1\b|\bp\d\d\b|\bX64\b|\bSHA-256\b|\bSHA-1\b|\bApache-2\.0\b|\bubuntu-\d+\.\d+|\bpython3\b|^\d+\. "
    r"|\btest_R1b\b|sha256|Prasanna-KS-85|\bM1_|\bT = 1\b|\brule 11\b|\bPhase \d\b|\bADR-\d\b|\bR1[0-4]?b?\b|\bQ\d\b"
    r"|\bW[0-2]'?|\bnlbse24-|\b[\w/.-]*_R1b\b", re.MULTILINE)
PLACEHOLDER = re.compile(r"\{\{(?:fact:\w+|include:[^}]*|\s*\w+\s*)\}\}|\{\{caption:|\{\{figure:\w+\|")


@pytest.fixture(scope="module")
def sources():
    return mrp.load_sources()


@pytest.fixture(scope="module")
def report(sources):
    return mrp.build(copy.deepcopy(sources))


@pytest.fixture(scope="module")
def facts():
    return yaml.safe_load(mrp.FACTS.read_text())["facts"]


def flat(text):
    return " ".join(text.split())


def prose(text):
    """The report without fenced code blocks and HTML lines (captions, figures, contents)."""
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("<"))


def units(text):
    """Sentences of the prose, with every table row and list item as its own unit."""
    out = []
    for para in prose(text).split("\n\n"):
        lines = para.splitlines()
        if lines and all(line.startswith("|") for line in lines):
            out += lines
            continue
        items, cur = [], []
        for line in lines:
            if re.match(r"^\s*(?:- |\d+\. )", line) and cur:
                items.append(" ".join(cur))
                cur = []
            cur.append(line.strip())
        items.append(" ".join(cur))
        for item in items:
            out += [s for s in re.split(r"(?<=[.!?])\s+", item) if s]
    return out


def sections(text):
    """{heading: body} for every heading of the report (any level), body up to the next heading."""
    parts = re.split(r"^(#{1,3} .+)$", prose(text), flags=re.MULTILINE)
    return {parts[i].lstrip("#").strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def section(text, prefix):
    hits = [body for head, body in sections(text).items() if head.startswith(prefix)]
    assert len(hits) == 1, f"one section starting {prefix!r} expected, found {len(hits)}"
    return flat(hits[0])


# --------------------------------------------------------------------------------------------- build and rule 11

def test_committed_report_is_up_to_date(report):
    assert mrp.REPORT.read_text() == report, "run docs/report/make_report.py"


def test_check_flag(capsys):
    mrp.main(["--check"])
    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(p.name for p in mrp.CHAPTERS.glob("*.md")))
def test_template_has_no_typed_numbers(name):
    text = (mrp.CHAPTERS / name).read_text()
    left = IDENTIFIERS.sub("", PLACEHOLDER.sub("", text))
    bad = [line for line in left.splitlines() if re.search(r"\d", line)]
    assert not bad, f"numbers must be placeholders or registered facts (rule 11): {bad}"


def test_every_placeholder_is_resolved(report):
    assert not re.search(r"(?<!\$)\{\{", report)
    assert "<!-- include" not in report and "<!-- alias" not in report and "\x00" not in report


def test_unknown_placeholder_and_fact_fail(sources):
    broken = copy.deepcopy(sources)
    broken["chapters"]["99_extra.md"] = "# Extra\n\n{{no_such_value}}\n"
    with pytest.raises(KeyError, match="unknown placeholder"):
        mrp.build(broken)
    broken["chapters"]["99_extra.md"] = "# Extra\n\n{{fact:no_such_fact}}\n"
    with pytest.raises(KeyError, match="unknown fact"):
        mrp.build(broken)


def test_tables_need_captions(sources):
    broken = copy.deepcopy(sources)
    broken["chapters"]["99_extra.md"] = "# Extra\n\n| a | b |\n|---|---|\n| x | y |\n"
    with pytest.raises(ValueError, match="table without a caption"):
        mrp.build(broken)


def test_structure(report):
    chapters = re.findall(r"^# (\d+) ", report, re.MULTILINE)
    assert chapters == [str(i) for i in range(1, 16)]
    assert re.findall(r"^# Appendix ([A-G]):", report, re.MULTILINE) == list("ABCDEFG")
    assert len(re.findall(r'<figure id="figure-', report)) == 5
    for path in re.findall(r'<img src="\.\./\.\./([^"]+)"', report):
        assert (ROOT / path).is_file(), path
    assert not any("banner" in src for src in re.findall(r'<img src="([^"]+)"', report))  # Q8: it overstates v1.0


def test_included_sections_are_word_for_word(sources, report):
    built = mrp.mr.build_docs(**sources["mr"])
    dd = built[mrp.DOC_NAMES["DEEP_DIVE"]]
    for anchor in ("limitations", "security", "how-this-was-built"):
        body = mrp.include("DEEP_DIVE", {"DEEP_DIVE": dd}, anchor, None, None).split("\n", 1)[1]
        body = re.sub(r"\{\{xref:[^}]*\}\}", "", body)
        for line in body.splitlines():
            if line.strip():
                assert re.sub(r" \(Section [\w.]+\)", "", line) in re.sub(r" \(Section [\w.]+\)", "", report), line


def test_status_words_follow_the_numbers(sources):
    broken = copy.deepcopy(sources)
    broken["mr"]["test"]["gating"]["config"]["applied"]["precision"] = 0.95
    assert "**The precision target for auto-applied labels was reached.**" in mrp.build(broken)


# --------------------------------------------------------------------------------------------- facts registry

def test_facts_are_quoted_from_their_sources(facts):
    keys = [f["key"] for f in facts]
    assert len(keys) == len(set(keys))
    for f in facts:
        src = Path(f["source"])
        assert not src.parts[0] in ("reference",) and src.parts[:2] not in (("data", "raw"), ("data", "processed"))
        text = flat((ROOT / src).read_text())
        assert flat(f["quote"]) in text, f"{f['key']}: quote no longer in {src}"
        assert str(f["value"]) in flat(f["quote"]), f"{f['key']}: value not inside its quote"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_fact_sources_are_tracked(facts):
    for src in sorted({f["source"] for f in facts}):
        r = subprocess.run(["git", "ls-files", "--error-unmatch", src], cwd=ROOT, capture_output=True, check=False)
        if src == "results/phase5.md" and r.returncode:
            pytest.skip("results/phase5.md is new and not committed yet")
        assert r.returncode == 0, f"{src} is not tracked by git"


def test_every_fact_is_used(facts):
    used = set(re.findall(r'facts_values"\]\["(\w+)"\]', (ROOT / "docs" / "report" / "make_report.py").read_text()))
    for p in mrp.CHAPTERS.glob("*.md"):
        used |= set(re.findall(r"\{\{fact:(\w+)\}\}", p.read_text()))
    assert used == {f["key"] for f in facts}


def test_a_changed_source_text_is_caught(facts):
    f = next(x for x in facts if x["key"] == "author")
    assert flat(f["quote"]) not in flat((ROOT / f["source"]).read_text().replace("Prasannakumar KS", "Someone Else"))


# --------------------------------------------------------------------------------------------- no benchmark text

def test_generator_reads_no_raw_or_processed_data():
    for rel in mrp.SOURCES.values():
        assert not rel.startswith(("data/raw", "data/processed", "reference"))
    code = (ROOT / "docs" / "report" / "make_report.py").read_text()
    assert "data/raw" not in code.split("SOURCES = {", 1)[1].split("}", 1)[0]


def test_no_duplicate_pair_titles_from_phase0(report):
    # results/phase0.md lists the train/test duplicate pairs with their titles; none may reach the report.
    rows = mrp.table(mrp.block((ROOT / "results" / "phase0.md").read_text(), "### Duplicate", "phase0.md"), "Pair", "phase0.md")
    for r in rows:
        title = r["Title"].strip("`").rstrip("…").strip()
        if len(title) >= 5:
            assert title not in report


def test_fixture_bodies_do_not_appear(report):
    for f in mrp.load_sources()["mr"]["issues"]["fixtures"]:
        for line in (f["body"] or "").splitlines():
            if len(line.strip()) >= 40:
                assert line.strip() not in report, f["key"]


@pytest.mark.skipif(not (ROOT / "data" / "raw" / "issues_test.csv").is_file(), reason="raw NLBSE'24 data not present")
def test_no_nlbse_titles_local_only(report):
    csv.field_size_limit(sys.maxsize)
    for name in ("issues_train.csv", "issues_test.csv"):
        with open(ROOT / "data" / "raw" / name, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                title = (row.get("title") or "").strip()
                if len(title) >= 25:
                    assert title not in report


# --------------------------------------------------------------------------------------------- wording rules

def test_test_run_wording(report):
    text = flat(report)
    assert "pre-declared and run once" in text
    assert "reused its predictions and are labelled post-hoc" in text
    assert "scored once" not in text and "scored exactly once" not in text


@pytest.mark.parametrize("prefix", ["Abstract", "9.7 Verdict", "12 Limitations"])
def test_both_misses_are_stated(report, prefix):
    body = section(report, prefix)
    assert re.search(r"precision target for auto-applied labels (?:was|were) (?:missed|not met)", body) or \
        ("precision target" in body and "missed" in body), prefix
    assert re.search(r"(?:per-issue )?latency target", body) and re.search(r"not met|missed", body), prefix


def test_precision_and_coverage_are_kept_apart(sources, report):
    v = mrp.mr.build_values(**sources["mr"])
    for u in units(report):
        if u.startswith("|"):
            continue
        low = u.lower()
        if v["gate_prec_pct"] in u or v["gate_prec"] in u:
            assert "precision" in low, u
        if v["gate_cov_pct"] in u or v["gate_cov"] in u:
            assert "coverage" in low or "auto-label" in low, u
    text = flat(report)
    for bad in (f"{v['gate_prec_pct']} of", f"precision of {v['gate_cov_pct']}", f"coverage of {v['gate_prec_pct']}"):
        assert bad not in text, bad


def test_latency_figure_is_model_call_latency(sources, report):
    rt = mrp.mr.runtime(sources["mr"]["phase4_md"], sources["mr"]["spec_md"], sources["mr"]["test"])
    figures = (rt["probe_p95"], f"{rt['probe_p95_secs']} seconds")
    hits = [u for u in units(report) if any(f in u for f in figures)]
    assert hits
    for u in hits:
        low = u.lower()
        assert "probe" in low, u
        assert any(q in low for q in ("per issue", "per-issue", "model call", "model-call", "model already loaded",
                                      "model preloaded")), u
        assert not any(w in low for w in ("job", "wall", "workflow run", "action run")), u


SETFIT_CAVEATS = ("different protocol", "another protocol", "other protocol", "not a like-for-like")


def paragraphs(text):
    """Blank-line separated blocks outside fenced code (a table is one block)."""
    return [p for p in re.split(r"\n\s*\n", re.sub(r"```.*?```", "", text, flags=re.DOTALL)) if p.strip()]


@pytest.mark.parametrize("doc", ["docs/report/REPORT.md", "README.md", "docs/DEEP_DIVE.md"])
def test_setfit_always_carries_the_protocol_caveat(report, doc):
    text = report if doc == "docs/report/REPORT.md" else (ROOT / doc).read_text()
    for para in paragraphs(text):
        if "SetFit" in para:
            assert any(c in para for c in SETFIT_CAVEATS), flat(para)[:200]


def test_no_superiority_claims(report):
    low = flat(report).lower()
    for phrase in ("more accurate", "better calibrated", "well calibrated", "well-calibrated", "perfectly calibrated"):
        assert phrase not in low, phrase


def test_calibration_claims_are_specific(sources, report):
    v = mrp.mr.build_values(**sources["mr"])
    body = section(report, "9.3 Calibration")
    t1 = mrp.f4(sources["mr"]["test"]["systems"]["M1"]["uncalibrated_T1"]["ece"])
    assert f"from {t1}" in body and v["m1_ece"] in body
    assert "predicted labels are unchanged" in body
    assert "no calibration advantage" in body


def test_warm_up_wording(report):
    text = flat(report)
    assert "verified by manual dispatch" in text
    assert "A schedule is configured" in text or "a schedule is configured" in text
    p4 = flat((ROOT / "results" / "phase4.md").read_text())
    p5 = flat((ROOT / "results" / "phase5.md").read_text())
    if "not yet observed: a run started by the `schedule` trigger" in p4 and "schedule" not in p5:
        claims = re.compile(r"schedul\w*(?: warm-up| warm-cache)? runs? (?:has|have|was|were) (?:been )?"
                            r"(?:observed|seen|recorded)|observed (?:a )?scheduled", re.IGNORECASE)
        assert not claims.search(text)


# --------------------------------------------------------------------------------------------- tags snapshot

def _local_tags():
    if shutil.which("git") is None:
        return []
    r = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True, check=False)
    return r.stdout.split() if r.returncode == 0 else []


@pytest.mark.skipif(not _local_tags(), reason="no git tags in this checkout (CI)")
def test_tags_snapshot_matches_git():
    snap = yaml.safe_load(mrp.TAGS.read_text())["tags"]
    assert snap == mrp.snapshot_tags()


def test_tags_snapshot_names_the_release():
    tags = {t["tag"]: t for t in yaml.safe_load(mrp.TAGS.read_text())["tags"]}
    assert "v1.0.0" in tags and tags["v1.0.0"]["annotated"]
    release = next(f for f in yaml.safe_load(mrp.FACTS.read_text())["facts"] if f["key"] == "release_commit")
    assert tags["v1.0.0"]["commit"] == release["value"]


# --------------------------------------------------------------------------------------------- PDF (slow)

@pytest.mark.slow
def test_pdf_renders(tmp_path, report):
    pytest.importorskip("weasyprint")
    pytest.importorskip("markdown_it")
    doc = mrp.render_pdf(report, tmp_path / "report.pdf")
    assert (tmp_path / "report.pdf").stat().st_size > 0
    pages = mrp.anchor_pages(doc)
    chapters = [mrp.slug(h) for h in re.findall(r"^# (\d+ .+|Appendix .+)$", report, re.MULTILINE)]
    starts = [pages[a] for a in chapters]
    assert starts == sorted(starts) and starts[0] > 1
    for fid in re.findall(r'<figure id="([^"]+)"', report):
        assert fid in pages, fid
    assert len(doc.pages) >= starts[-1]


def test_report_images_resolve_relative_to_the_report_folder():
    text = mrp.REPORT.read_text()
    srcs = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', text) + re.findall(r"(?<!`)!\[[^\]]*\]\(([^)\s]+)\)", prose(text))
    assert srcs
    for src in srcs:
        assert not src.startswith(("http", "/")), src
        assert (mrp.REPORT_DIR / src).resolve().is_file(), src
