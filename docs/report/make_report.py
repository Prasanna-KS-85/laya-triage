"""Build docs/report/REPORT.md, the v1.0 technical report, from the chapter templates in docs/report/chapters/ and the
committed files of this repository, and optionally render it to build/report/laya-triage-report.pdf
(PROJECT_SPEC.md §13 Phase 5, §16 rule 11). Never touches the network, the model or the data.

Run from anywhere:
  .venv/bin/python docs/report/make_report.py            # write REPORT.md
  .venv/bin/python docs/report/make_report.py --check    # exit 1 if REPORT.md is out of date
  .venv/bin/python docs/report/make_report.py --pdf      # also write build/report/*.html and *.pdf (report extra)
  .venv/bin/python docs/report/make_report.py --snapshot-tags   # rewrite docs/report/tags.yml from local git

Where the numbers come from:
- every value docs/make_readme.py builds for README.md and the three docs (it reads metrics_test.json,
  metrics_val.json, the config, action.yml, posthoc.json, the sandbox files, phase3.md, phase4.md and the spec);
- the generated docs themselves: `{{include:DOC#anchor}}` copies a section of DEEP_DIVE, USING_THE_ACTION or
  USING_THE_MODEL word for word (links become plain text or section references), so the report cannot drift from
  them;
- JSON / YAML / CSV result files and markdown tables, read here (SOURCES);
- docs/report/facts.yml for values that exist only in prose: each entry names its source file and the exact text it
  appears in, and tests/test_tech_report.py fails when that text is gone;
- docs/report/tags.yml, a snapshot of the local git tags (tag, peeled commit, creator date).
The chapter templates hold prose and placeholders only: no typed numbers (tests/test_tech_report.py). Chapter, section,
table and figure numbers are assigned here. A missing key, row or pattern raises naming the file.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "docs"), str(ROOT / "results" / "phase3"), str(ROOT / "eval")]

import make_model_card as mmc
import make_readme as mr
from make_model_card import f4, need

from laya_triage.questions import LABELS

REPORT_DIR = ROOT / "docs" / "report"
CHAPTERS = REPORT_DIR / "chapters"
REPORT = REPORT_DIR / "REPORT.md"
FACTS = REPORT_DIR / "facts.yml"
TAGS = REPORT_DIR / "tags.yml"
CSS = REPORT_DIR / "report.css"
BUILD = ROOT / "build" / "report"
PDF = BUILD / "laya-triage-report.pdf"
HTML_OUT = BUILD / "laya-triage-report.html"
# Report-only sources, all committed; nothing under data/raw, data/processed or reference/ is ever read.
SOURCES = {
    "phase0_summary": "results/phase0/summary.json",
    "token_lengths": "results/phase1/token_lengths.json",
    "b1_config": "results/phase2/b1_config.json",
    "b1_grid": "results/phase2/b1_grid_val.csv",
    "val_r1": "results/phase3/val_R1/metrics_val.json",
    "phase0_md": "results/phase0.md",
    "phase5_md": "results/phase5.md",
    "data_card": "data/DATA_CARD.md",
    "pyproject": "pyproject.toml",
    "questions_py": "src/laya_triage/questions.py",
    "config_text": "config/triage.default.yml",
    "workflow_tests": "tests/test_workflows.py",
    "ci": ".github/workflows/ci.yml",
}
MANIFESTS = ("train", "val", "test", "dropped_train")
DOC_NAMES = {"DEEP_DIVE": ROOT / "docs" / "DEEP_DIVE.md", "USING_THE_ACTION": ROOT / "docs" / "USING_THE_ACTION.md",
             "USING_THE_MODEL": ROOT / "docs" / "USING_THE_MODEL.md"}
FIGURES = {"architecture": "docs/architecture.png"}  # plus the four §10.3 figures from metrics_test.json
SHORT_REPO = {"bitcoin/bitcoin": "bitcoin", "facebook/react": "react", "microsoft/vscode": "vscode",
              "opencv/opencv": "opencv", "tensorflow/tensorflow": "tensorflow"}
SPEC = "PROJECT_SPEC.md"


# ----------------------------------------------------------------------------------------------- markdown helpers

def slug(heading):
    """GitHub's anchor for a heading (same rule as tests/test_readme.py)."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def cells(line):
    """Cells of a markdown table row; escaped pipes (`\\|`) stay inside their cell."""
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def tables(md, first_header, src):
    """Every markdown table in `md` whose header row starts with `first_header`, as lists of {header: cell}."""
    out, header, rows = [], None, None
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            c = cells(line)
            if header is None:
                header, rows = c, []
            elif not set(line) <= set("|-: "):
                if len(c) != len(header):
                    raise ValueError(f"{src}: row {c[0]!r} has {len(c)} cells, header has {len(header)}")
                rows.append(dict(zip(header, c)))
        else:
            if header is not None and header[0] == first_header and rows:
                out.append(rows)
            header = None
    if not out:
        raise ValueError(f"{src}: no table with header starting {first_header!r}")
    return out


def table(md, first_header, src):
    return tables(md, first_header, src)[0]


def block(md, heading_prefix, src):
    """Text under the heading line that starts with `heading_prefix`, up to the next heading of the same or a higher
    level (fenced code blocks are skipped when looking for headings)."""
    lines, out, level, fence = md.splitlines(), None, None, False
    for line in lines:
        if line.startswith("```"):
            fence = not fence
        if out is None:
            if not fence and line.startswith(heading_prefix):
                level, out = len(line) - len(line.lstrip("#")), []
            continue
        m = re.match(r"^(#{1,6}) ", line)
        if not fence and m and len(m.group(1)) <= level:
            break
        out.append(line)
    if out is None:
        raise ValueError(f"{src}: heading not found: {heading_prefix!r}")
    return "\n".join(out).strip("\n")


def numbered_items(md):
    """Items of the numbered lists in `md` (continuation lines joined), in order."""
    items, open_ = [], False
    for line in md.splitlines():
        if re.match(r"^\d+\. ", line):
            items.append(re.sub(r"^\d+\. ", "", line).strip())
            open_ = True
        elif open_ and line.startswith("   ") and line.strip():
            items[-1] += " " + line.strip()
        elif not line.strip():
            open_ = False
    return items


def bullets(md):
    out = []
    for line in md.splitlines():
        if line.startswith("- "):
            out.append(line[2:].strip())
        elif out and line.startswith("  ") and line.strip():
            out[-1] += " " + line.strip()
    return out


def flat(text):
    return " ".join(text.split())


def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    esc = [[str(c).replace("|", "\\|") for c in r] for r in rows]
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"] +
                     ["| " + " | ".join(r) + " |" for r in esc])


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}%"


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z§`(*])", flat(text), maxsplit=1)[0]


def shorten(text, limit=150):
    """At most `limit` characters, cut at a word boundary, with an ellipsis when cut."""
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def strip_md(text):
    return text.replace("**", "")


# ------------------------------------------------------------------------------------------------ sources

def load_sources():
    s = {"mr": mr.load_sources()}
    for k, rel in SOURCES.items():
        text = (ROOT / rel).read_text()
        if rel.endswith(".json"):
            s[k] = json.loads(text)
        elif rel.endswith(".csv"):
            s[k] = list(csv.DictReader(io.StringIO(text)))
        elif k == "pyproject":
            s[k] = pyproject(text)
        else:
            s[k] = text
    s["manifests"] = {m: (ROOT / "data" / "manifests" / f"{m}.txt").read_bytes() for m in MANIFESTS}
    s["facts"] = yaml.safe_load(FACTS.read_text())
    s["tags"] = yaml.safe_load(TAGS.read_text())
    s["chapters"] = {p.name: p.read_text() for p in sorted(CHAPTERS.glob("*.md"))}
    return s


def fact_values(facts):
    out = {}
    for f in facts["facts"]:
        if f["key"] in out:
            raise ValueError(f"docs/report/facts.yml: duplicate key {f['key']}")
        out[f["key"]] = str(f["value"])
    return out


def snapshot_tags():
    """docs/report/tags.yml from local git (read-only): every tag, its peeled commit and its creator date."""
    fmt = "%(refname:short)|%(objecttype)|%(objectname)|%(*objectname)|%(creatordate:iso-strict)"
    out = subprocess.run(["git", "for-each-ref", "refs/tags", "--sort=creatordate", f"--format={fmt}"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    tags = []
    for line in out.splitlines():
        name, kind, obj, peeled, date = line.split("|")
        tags.append({"tag": name, "annotated": kind == "tag", "commit": peeled or obj, "date": date})
    return tags


def write_tags(tags):
    head = ("# Snapshot of git metadata: local tags, peeled commit and creator date, written once by\n"
            "# `docs/report/make_report.py --snapshot-tags` (read-only git for-each-ref). tests/test_tech_report.py checks it\n"
            "# against local git when tags are present (skipped in CI, where the checkout has no tags).\n")
    TAGS.write_text(head + yaml.safe_dump({"tags": tags}, sort_keys=False, allow_unicode=True))


# ------------------------------------------------------------------------------------------------ report values

def spec_values(spec):
    """Structured pieces of PROJECT_SPEC.md: requirement tables, lists, the phase plan and the changelog."""
    out = {}
    fr = table(block(spec, "### 6.1 ", SPEC), "ID", SPEC)
    nfr = table(block(spec, "### 6.2 ", SPEC), "ID", SPEC)
    crit_items = numbered_items(block(spec, "### 6.3 ", SPEC))
    if len(crit_items) != 5:
        raise ValueError(f"{SPEC} §6.3: expected 5 criteria, found {len(crit_items)}")
    out["fr"], out["nfr"], out["criteria"] = fr, nfr, crit_items
    out["non_goals"] = "\n".join(f"- {b}" for b in bullets(block(spec, "### 4.2 ", SPEC)))
    out["release_map"] = table(block(spec, "### 4.1 ", SPEC), "Release", SPEC)
    out["question_types"] = table(block(spec, "### 5.1 ", SPEC), "Type", SPEC)
    out["packing"] = mr.grab(r"packed into one sequence: `([^`]+)`", flat(block(spec, "### 5.2 ", SPEC)), SPEC).group(1)
    out["recipe"] = {r["Setting"]: r["Value"] for r in table(block(spec, "### 9.3 ", SPEC), "Setting", SPEC)}
    out["systems"] = table(block(spec, "### 10.1 ", SPEC), "#", SPEC)
    out["metrics"] = table(block(spec, "### 10.2 ", SPEC), "Metric", SPEC)
    out["threshold_steps"] = numbered_items(block(spec, "### 10.4 ", SPEC))[:7]
    out["preprocess_steps"] = numbered_items(block(spec, "### 8.4 ", SPEC))
    fail = block(spec, "### 7.5 ", SPEC)
    runtime, action = fail.split("Action layer (`action.yml`):")
    out["fail_runtime"], out["fail_action"] = table(runtime, "Failure", SPEC), table(action, "Failure", SPEC)
    out["misconfig"] = flat(mr.grab(r"\n(Misconfiguration \(an invalid config.*?)\n\n", fail, SPEC, re.DOTALL).group(1))
    out["modules"] = table(block(spec, "### 12.2 ", SPEC), "Function", SPEC)
    out["sequence"] = mr.grab(r"```\n(.*?)\n```", block(spec, "### 7.4 ", SPEC), SPEC, re.DOTALL).group(1)
    out["glossary"] = table(block(spec, "## 18. ", SPEC), "Term", SPEC)
    out["references"] = bullets(block(spec, "## 19. ", SPEC))
    rules = numbered_items(block(spec, "## 16. ", SPEC))
    out["rules"] = rules
    phases = []
    plan = block(spec, "## 13. ", SPEC)
    for m in re.finditer(r"^### (Phase (\d)\+?): ([^\n(]+?)\s*(?:\(([^)]*)\))?$", plan, re.MULTILINE):
        body = block(plan, m.group(0), SPEC)
        gate = re.search(r"\*\*Gate:\*\* (.*?)(?:\n\n|$)", body, re.DOTALL)
        goal = re.search(r"\*\*Goal:\*\* (.*?)(?:\n\n|$)", body, re.DOTALL)
        phases.append({"phase": m.group(1), "n": int(m.group(2)), "title": m.group(3).strip(),
                       "estimate": (m.group(4) or "").strip(), "gate": flat(gate.group(1)) if gate else "",
                       "goal": flat(goal.group(1)) if goal else ""})
    out["phases"] = [p for p in phases if not p["phase"].endswith("+")]
    log = block(spec, "### Changelog of this document", SPEC)
    out["changelog"] = table(log, "Date", SPEC)
    return out


def pyproject(text):
    """Version, dependencies and extras of pyproject.toml (read with regular expressions: tomllib needs 3.11)."""
    def deps(body):
        return re.findall(r'"([^"]+)"', body)

    project = mr.grab(r"(?ms)^\[project\]\n(.*?)^\[", text, "pyproject.toml").group(1)
    extras = mr.grab(r"(?ms)^\[project\.optional-dependencies\]\n(.*?)^\[", text, "pyproject.toml").group(1)
    return {"project": {"version": mr.grab(r'(?m)^version = "([^"]+)"', project, "pyproject.toml").group(1),
                        "dependencies": deps(mr.grab(r"(?ms)^dependencies = \[(.*?)\]", project, "pyproject.toml").group(1)),
                        "optional-dependencies": {k: deps(v) for k, v in re.findall(r"(?ms)^(\w+) = \[(.*?)\]", extras)}}}


def pins(constraints):
    return [tuple(line.split("==")) for line in constraints.splitlines() if line and not line.startswith("#")]


def report_values(s):
    m = s["mr"]
    test, val, spec, config = m["test"], m["val"], m["spec_md"], m["config"]
    t, v = "metrics_test.json", "metrics_val.json"
    F = mmc.facts(test, val, config, m["refit"], m["items"], m["experiments_md"], spec)
    sp = spec_values(spec)
    rt = mr.runtime(m["phase4_md"], spec, test)
    sysm = {k: need(test, f"systems.{k}.metrics", t) for k in ("M1", "B0", "B1", "B2_512-192", "B2_1024-256")}
    names = {"M1": "M1 (this model)", "B0": "B0 majority class", **dict(mmc.BASELINES)}
    out = {}

    # front matter
    out["version"] = s["pyproject"]["project"]["version"]
    out["n_tags"] = len(s["tags"]["tags"])

    # ch1 / ch2
    out["non_goals"] = sp["non_goals"]
    out["release_map_table"] = md_table(["Release", "Contents", "Status"],
                                        [[r["Release"], r["Contents"], r["Status"]] for r in sp["release_map"]])
    out["question_types_table"] = md_table(["Type", "Output", "Use in this project"],
                                           [[r["Type"], r["Output"], r["Used in this project"]] for r in sp["question_types"]])
    out["packing"] = f"`{sp['packing']}`"
    out["t1_ece"] = f4(need(test, "systems.M1.uncalibrated_T1.ece", t))
    out["t1_brier"] = f4(need(test, "systems.M1.uncalibrated_T1.brier", t))
    out["b1_ece"] = f4(sysm["B1"]["ece"])
    b3 = {r["system"].split()[1]: r for r in need(test, "B3_copied_from_phase2_md.published", t)}
    out["roberta_xrepo"] = f4(b3["RoBERTa"]["cross_repo_macro_f1"])
    out["fasttext_xrepo"] = f4(b3["fastText"]["cross_repo_macro_f1"])

    # ch3 requirements
    p4 = m["phase4_md"]
    p4_fr = {r["ID"]: r for r in table(block(p4, "#### Functional requirements", "results/phase4.md"), "ID", "phase4.md")}
    p4_nfr = {r["ID"]: r for r in table(block(p4, "#### Non-functional requirements", "results/phase4.md"), "ID", "phase4.md")}
    out["fr_table"] = md_table(["ID", "Requirement", "On the sandbox"],
                               [[r["ID"], r["Requirement"], p4_fr[r["ID"]]["Demonstrated on the sandbox"]] for r in sp["fr"]])

    def nfr_result(i):
        r = p4_nfr.get(i)
        if r is None:
            return "Not measured on the sandbox; see the text after the table"
        return re.sub(r"\s*\((?:probe below|observations|gate checklist below)\)", "", r["Result"])

    out["nfr_table"] = md_table(["ID", "Requirement", "Target", "Result"],
                                [[r["ID"], r["Requirement"], r["Target (verify in Phase 0 / 4)"], nfr_result(r["ID"])]
                                 for r in sp["nfr"]])
    crit = need(test, "criteria_6_3", t)
    p3 = m["phase3_md"]
    protocol = block(p3, "## Test protocol (pre-declared)", "results/phase3.md")
    proto_items = numbered_items(protocol)
    if len(proto_items) != 13:
        raise ValueError(f"results/phase3.md: expected 9 protocol items and 4 judging rules, found {len(proto_items)}")
    judging = proto_items[9:]
    sandbox_rule = flat(mr.grab(r"compare the runner logs with\s+`sandbox/expected_local\.json` \(([^)]*)\)", spec, SPEC).group(1))
    second_clause = s["facts_values"]["crit4_second_clause"]
    statuses = [
        "Met" if need(crit, "1.status", t) == "MET" else "Not met",
        "Met" if need(crit, "2.status", t) == "MET" else "Not met",
        f"Not met: {need(crit, '3.status', t)} labels" if need(crit, "3.k", t) < len(LABELS) else "Met",
        (f"NFR-1 {rt['nfr1_status']} on the runner (latency probe: {rt['probe_p95']} per issue at p95, model "
         f"preloaded); documented with the v1.2 ONNX plan, {second_clause}"),
        "Met" if s["facts_values"]["crit5_status"] == "MET" else "Not met",
    ]
    judged = [*judging, f"End-to-end sandbox run compared with the local production path ({sandbox_rule})"]
    out["criteria_table"] = md_table(["#", "Criterion", "How it was judged", "Result"],
                                     [[str(i + 1), first_sentence(c.replace("**", "")), j, st]
                                      for i, (c, j, st) in enumerate(zip(sp["criteria"], judged, statuses))])

    # ch4 process
    tags = {x["tag"]: x for x in s["tags"]["tags"]}
    rows = []
    for ph in sp["phases"]:
        tag = f"phase{ph['n']}-complete" if ph["n"] < 5 else "v1.0.0"
        tg = tags.get(tag)
        rows.append([ph["phase"], ph["title"], ph["gate"] or "—",
                     f"`{tag}` ({tg['date'][:10]})" if tg else "no tag"])
    out["phases_table"] = md_table(["Phase", "Title", "Gate", "Tag (date)"], rows)
    out["rule_frozen"] = sp["rules"][3]
    out["rule_test"] = sp["rules"][5]
    out["rule_honest"] = sp["rules"][10]
    out["n_rules"] = len(sp["rules"])
    log = sp["changelog"]
    out["n_changelog"] = len(log)
    out["changelog_first"], out["changelog_last"] = log[0]["Version"], log[-1]["Version"]
    out["protocol_primary"] = "\n".join(f"{n}. {i}" for n, i in enumerate(proto_items[:4], 1))
    out["protocol_secondary"] = "\n".join(f"{n}. {i}" for n, i in enumerate(proto_items[4:9], 5))
    out["protocol_judging"] = "\n".join(f"- {i}" for i in judging)
    out["protocol_rerun"] = flat(mr.grab(r"\n(If the run crashes.*?)\n\n", protocol, "results/phase3.md", re.DOTALL).group(1))
    out["posthoc_rule"] = flat(mr.grab(r"(\*\*This is not part of the pre-declared protocol\.\*\*.*?)\n\n",
                                       p3, "results/phase3.md", re.DOTALL).group(1))
    out["rc1_commit"] = tags["v1.0.0-rc1"]["commit"][:7]
    out["prefreeze_commit"] = tags["phase3-prefreeze"]["commit"][:7] if "phase3-prefreeze" in tags else "—"
    out["test_run_date"] = mr.grab(r"## Test results \((\d{4}-\d\d-\d\d)\)", p3, "results/phase3.md").group(1)

    # ch5 architecture
    out["sequence_diagram"] = sp["sequence"]
    out["fail_runtime_table"] = md_table(["Failure", "Behaviour"], [[r["Failure"], r["Behaviour"]] for r in sp["fail_runtime"]])
    out["fail_action_table"] = md_table(["Failure", "Behaviour"], [[r["Failure"], r["Behaviour"]] for r in sp["fail_action"]])
    out["misconfig"] = sp["misconfig"]
    out["module_table"] = md_table(["Function", "Signature", "Purity / side effects"],
                                   [[r["Function"], r["Signature"], r["Purity / side effects"]] for r in sp["modules"]])

    # ch6 data
    dc = s["data_card"]
    src = table(dc, "Item", "data/DATA_CARD.md")
    out["data_source_table"] = md_table(["Item", "Value"], [[r["Item"], r["Value"]] for r in src])
    dropped = table(dc, "id", "data/DATA_CARD.md")
    out["dropped_table"] = md_table(["Synthetic id", "Content SHA-1", "Reason"],
                                    [[f"`{r['id']}`", r["content_sha1"], r["reason"]] for r in dropped])
    split_rows = []
    for name, tb in zip(("train", "val", "test"), tables(dc, "repo", "data/DATA_CARD.md")):
        tot = next(r for r in tb if r["repo"] == "**total**")
        per = sorted({r[lab] for r in tb if r["repo"] != "**total**" for lab in LABELS}, key=int)
        split_rows.append([name, *(tot[lab] for lab in LABELS), tot["total"], f"{per[0]}–{per[-1]}" if len(per) > 1 else per[0]])
    out["split_table"] = md_table(["Split", *LABELS, "Total", "Per repository × label cell"], split_rows,
                                  ["---", "---:", "---:", "---:", "---:", "---"])
    man_rows = []
    for name, data in s["manifests"].items():
        man_rows.append([f"`data/manifests/{name}.txt`", f"{len(data.splitlines()):,}", f"`{hashlib.sha256(data).hexdigest()}`"])
    out["manifest_table"] = md_table(["File", "Lines", "SHA-256"], man_rows, ["---", "---:", "---"])
    out["preprocess_steps"] = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(sp["preprocess_steps"]))
    hits = table(dc, "step / property", "data/DATA_CARD.md")
    out["hit_table"] = md_table(["Step or property", "Bodies", "Share"], [[r["step / property"], r["bodies"], r["share"]] for r in hits],
                                ["---", "---:", "---:"])
    tl = s["token_lengths"]
    tl_rows = []
    for name, d in tl["splits"].items():
        p = d["state_tokens_percentiles"]
        tl_rows.append([name, f"{d['rows']:,}", *(f"{p[k]:g}" for k in p), str(d["state_tokens_max"]),
                        pct(d["share_truncated_at_512/192"]), pct(d["share_truncated_at_1024/256"])])
    pkeys = list(next(iter(tl["splits"].values()))["state_tokens_percentiles"])
    out["token_table"] = md_table(["Split", "Rows", *pkeys, "max", "Truncated at 512/192", "Truncated at 1024/256"], tl_rows,
                                  ["---"] + ["---:"] * (len(pkeys) + 4))
    out["state_room_native"] = tl["state_room"]["512/192"]
    out["state_room_prod"] = tl["state_room"]["1024/256"]

    # ch7 model development
    ph0 = s["phase0_summary"]
    w_rows = []
    for run, r in ph0["runs"].items():
        w, setting = run.split("@")
        w_rows.append([w, setting, f4(r["macro_f1"]), f"{r['accuracy']:.2f}", *(f4(r["f1_per_class"][lab]) for lab in LABELS),
                       pct(r["share_truncated"], 0)])
    out["wording_table"] = md_table(["Wording", "Setting", "Macro-F1", "Accuracy", *(f"F1 {lab}" for lab in LABELS), "Truncated"],
                                    w_rows, ["---", "---"] + ["---:"] * 6)
    out["phase0_n"] = ph0["meta"]["n_issues"]
    gains = {st: ph0["runs"][f"W1'@{st}"]["macro_f1"] - ph0["runs"][f"W0@{st}"]["macro_f1"] for st in ("512/192", "1024/256")}
    out["w1_gain_native"] = f"{100 * gains['512/192']:.1f}"
    out["w1_gain_prod"] = f"{100 * gains['1024/256']:.1f}"
    rec = sp["recipe"]
    r1 = mmc.ledger_row(m["experiments_md"], "R1")
    losses = mr.grab(r"mean epoch loss ([\d. /]+);", r1["notes"], "results/experiments.md").group(1).strip()
    out["train_table"] = md_table(["Setting", "Value"], [
        ["Base checkpoint", f"`convaiinnovations/laya` at `{F['base_revision']}`"],
        ["Hardware", f"Kaggle, {F['n_gpus']}× T4 (Python {F['kaggle_python']}, torch {F['kaggle_torch']})"],
        ["Training items", f"{F['n_train']:,} rows, of which {F['n_calibration']} are held out by the notebook as its calibration slice"],
        ["Epochs / optimizer steps", f"{F['epochs']} / {F['steps']}"],
        ["Effective batch", rec["Effective batch"]],
        ["Learning rates", rec["Learning rates"]],
        ["Memory", rec["Memory"]],
        ["Sequence budget", f"`max_len` {F['max_len']}, `head_max_len` {F['head_max_len']}"],
        ["Loss", rec["Loss"]],
        ["Label smoothing ε", F["eps"]],
        ["Seed", str(F["seed"])],
        ["Mean epoch loss", losses],
        ["Training time", f"{F['train_seconds']:.0f} s"],
        ["Trained revision (R1)", f"`{F['trained_revision']}`"],
    ])
    out["items_laya"], out["items_transformers"] = F["items_laya"], F["items_transformers"]
    out["items_sha"] = need(m["items"], ("items", F["eps"], "sha256"), "items_sha256.json")
    out["share_longer_512"] = pct(need(m["items"], ("items", F["eps"], "share_longer_than_512"), "items_sha256.json"))
    out["trained_revision"] = F["trained_revision"]
    out["n_calibration"] = F["n_calibration"]
    out["kaggle_gpus"] = F["n_gpus"]
    refit = m["refit"]
    out["temperature_table"] = md_table(["Temperature", "Val NLL", "Val ECE", "Val Brier", "Val accuracy"],
                                        [[f"{r['T']:.4f} ({'notebook fit' if i == 0 else 'validation refit, shipped'})",
                                          f4(r["nll"]), f4(r["ece"]), f4(r["brier"]), f4(r["accuracy"])]
                                         for i, r in enumerate(need(refit, "table", "refit_summary.json"))],
                                        ["---", "---:", "---:", "---:", "---:"])
    out["refit_grid_T"] = need(refit, "grid_check.T_min_nll", "refit_summary.json")
    out["refit_fitter"] = need(refit, "fitter", "refit_summary.json").split(" (")[0]
    out["r1_val_ece"] = f4(need(refit, "table", "refit_summary.json")[0]["ece"])
    vr1 = s["val_r1"]["systems"]["M1"]
    out["val_r1_t1_ece"] = f4(vr1["uncalibrated_T1"]["ece"])
    vg = need(val, "gating", v)
    th_rows = []
    for sysname in ("M1", "B1"):
        for lab in LABELS:
            g = need(vg, f"{sysname}.per_label.{lab}", v)
            lb = g["at_tau_lb"]
            tp = g["at_tau_point"]
            th_rows.append([sysname, lab, str(g["n_predicted"]),
                            "never (1.01)" if g["tau_lb"] >= 1 else f"{g['tau_lb']:.4f}",
                            str(lb["support"]) if lb else "—", f4(lb["precision"]) if lb else "—", f4(lb["lower_bound"]) if lb else "—",
                            f"{g['tau_point']:.4f}", f4(tp["precision"]), f4(tp["lower_bound"])])
    out["val_threshold_table"] = md_table(["System", "Label", "Predicted", "τ_lb", "|S|", "Precision", "Lower bound",
                                           "τ_point", "Precision", "Lower bound"], th_rows,
                                          ["---", "---"] + ["---:"] * 8)
    out["threshold_steps"] = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(sp["threshold_steps"]))
    vlb = need(vg, "M1.tau_lb.applied", v)
    out["val_gate_cov"], out["val_gate_prec"] = f4(vlb["coverage"]), f4(vlb["precision"])

    # ch8 method
    out["systems_table"] = md_table(["#", "System", "Purpose"], [[r["#"], r["System"], r["Purpose"]] for r in sp["systems"]])
    out["metrics_table"] = md_table(["Metric", "Definition"], [[r["Metric"], r["Definition"]] for r in sp["metrics"]])
    out["b1_C"] = s["b1_config"]["classifier"]["C"]
    out["b1_min_df"] = s["b1_config"]["vectorizer"]["min_df"]
    out["b1_grid_n"] = len(s["b1_grid"])

    # ch9 results
    mc_rows = []
    for b, name in mmc.BASELINES:
        p = need(test, f"paired_M1_vs.{b}", t)
        mcn, bs = p["mcnemar_exact"], p["bootstrap_xrepo_macro_f1_diff"]
        mc_rows.append([f"M1 vs {name}", str(mcn["a_right_b_wrong"]), str(mcn["a_wrong_b_right"]), f"{mcn['p_value']:.2g}",
                        pct(bs["share_resamples_a_better"], 2)])
    out["mcnemar_table"] = md_table(["Comparison", "M1 right, other wrong", "M1 wrong, other right", "Exact McNemar p",
                                     "Resamples with M1 ahead"], mc_rows, ["---", "---:", "---:", "---:", "---:"])
    sec_rows = []
    for key, name in (("config", "M1, shipped thresholds (τ_lb)"), ("M1_tau_point", "M1, τ_point (secondary)"),
                      ("B1_same_rule", "B1, same rule, its own thresholds (secondary)")):
        g = need(test, f"gating.{key}.applied", t)
        per = "; ".join(f"{lab} {g['per_label'][lab]['applied']}" + (f" at {f4(g['per_label'][lab]['precision'])}"
                                                                    if g["per_label"][lab]["precision"] is not None else "")
                        for lab in LABELS)
        sec_rows.append([name, f"{g['applied']:,}", f4(g["coverage"]), f4(g["precision"]), per])
    out["secondary_gate_table"] = md_table(["Rule", "Applied", "Coverage", "Precision", "Per label: applied at precision"],
                                           sec_rows, ["---", "---:", "---:", "---:", "---"])
    pc_rows = []
    for k in ("M1", "B1", "B2_512-192", "B2_1024-256"):
        pc = sysm[k]["per_class"]
        pc_rows.append([names[k], *(f"{f4(pc[lab]['precision'])} / {f4(pc[lab]['recall'])} / {f4(pc[lab]['f1'])}" for lab in LABELS)])
    out["per_class_table"] = md_table(["System", *(f"{lab} P / R / F1" for lab in LABELS)], pc_rows)
    conf_rows = []
    for k in ("M1", "B1"):
        cm = sysm[k]["confusion"]["rows_gold_cols_pred"]
        for lab, row in zip(LABELS, cm):
            conf_rows.append([k, lab, *(str(x) for x in row)])
    out["confusion_table"] = md_table(["System", "Gold", *(f"Predicted {lab}" for lab in LABELS)], conf_rows,
                                      ["---", "---", "---:", "---:", "---:"])
    out["m1_question_recall"] = f4(sysm["M1"]["per_class"]["question"]["recall"])
    out["m1_bug_recall"] = f4(sysm["M1"]["per_class"]["bug"]["recall"])
    out["m1_pred_counts"] = ", ".join(f"{lab} {sysm['M1']['predicted_counts'][lab]}" for lab in LABELS)
    repos = list(sysm["M1"]["per_repo_macro_f1"])
    published = table(m["phase2_md"].split("### Per-repo macro-F1 (test), ours vs published", 1)[1], "System", "results/phase2.md")
    rp_rows = [[names[k], *(f4(sysm[k]["per_repo_macro_f1"][r]) for r in repos), f4(sysm[k]["cross_repo_macro_f1"])]
               for k in ("M1", "B1", "B2_512-192", "B2_1024-256")]
    for r in published:
        if r["System"].startswith("B3"):
            rp_rows.append([f"{r['System']} (published, other protocol)", *(r[SHORT_REPO[x]] for x in repos), r["Cross-repo"].strip("*")])
    out["per_repo_table"] = md_table(["System", *(SHORT_REPO[r] for r in repos), "Cross-repo"], rp_rows,
                                     ["---"] + ["---:"] * (len(repos) + 1))
    margins = {r: sysm["M1"]["per_repo_macro_f1"][r] - sysm["B1"]["per_repo_macro_f1"][r] for r in repos}
    out["n_repos_m1_ahead"] = sum(d > 0 for d in margins.values())
    small = sorted(margins, key=margins.get)[:2]
    out["smallest_margins"] = " and ".join(f"{SHORT_REPO[r]} ({100 * margins[r]:+.1f} points)" for r in small)
    weakest = min(repos, key=lambda r: sysm["M1"]["per_repo_macro_f1"][r])
    out["m1_weakest_repo"] = f"{SHORT_REPO[weakest]} ({f4(sysm['M1']['per_repo_macro_f1'][weakest])})"
    gap_rows = []
    for k in ("M1", "B1", "B2_512-192", "B2_1024-256"):
        vx = need(val, f"systems.{k}.metrics.cross_repo_macro_f1", v)
        tx = sysm[k]["cross_repo_macro_f1"]
        gap_rows.append([names[k], f4(vx), f4(tx), f"{100 * (tx - vx):+.1f}"])
    out["m1_gap_points"] = f"{100 * (need(val, 'systems.M1.metrics.cross_repo_macro_f1', v) - sysm['M1']['cross_repo_macro_f1']):.1f}"
    out["gap_table"] = md_table(["System", "Validation", "Test", "Change (points)"], gap_rows, ["---", "---:", "---:", "---:"])
    ph = m["posthoc"]
    nd = need(ph, "A_near_duplicates", "posthoc.json")
    nd_rows = [[sp_, str(d["n"]), f4(d["mean"]), f4(d["median"]), f4(d["p90"]), pct(d["share_above_0.8"]), pct(d["share_above_0.9"])]
               for sp_, d in nd["summary"].items()]
    out["near_dup_table"] = md_table(["Split", "n", "Mean", "Median", "p90", "Share > 0.8", "Share > 0.9"], nd_rows,
                                     ["---"] + ["---:"] * 6)
    tert_rows = []
    for sp_, d in nd["accuracy_by_tertile"].items():
        for k in ("M1", "B1"):
            tert_rows.append([sp_, k, *(f"{d[k][b]['n']} / {f4(d[k][b]['accuracy'])}" for b in ("low", "mid", "high")),
                              f4(d["overall"][k])])
    out["tertile_table"] = md_table(["Split", "System", "Low (n / accuracy)", "Mid (n / accuracy)", "High (n / accuracy)",
                                     "Overall"], tert_rows, ["---", "---", "---:", "---:", "---:", "---:"])
    out["tertile_cuts"] = " and ".join(f4(x) for x in nd["tertile_cuts_from_val"])
    gc = need(ph, "B_global_cutoff", "posthoc.json")
    gc_rows = [[k, f"{d['val']['tau']:.4f}", str(d["val"]["support"]), f4(d["val"]["coverage"]), f4(d["val"]["precision"]),
                f4(d["val"]["lower_bound"]), f"{d['test']['applied']:,}", f4(d["test"]["coverage"]), f4(d["test"]["precision"])]
               for k, d in gc.items()]
    out["global_cut_table"] = md_table(["System", "τ_global", "Val |S|", "Val coverage", "Val precision", "Val bound",
                                        "Test applied", "Test coverage", "Test precision"], gc_rows,
                                       ["---"] + ["---:"] * 8)
    out["gc_m1_prec"], out["gc_b1_prec"] = f4(gc["M1"]["test"]["precision"]), f4(gc["B1"]["test"]["precision"])
    cs = need(ph, "C_sanity", "posthoc.json")
    out["sanity_correct"] = cs["M1_test_tau_lb"]["correct"]["all"]
    out["sanity_applied"] = cs["M1_test_tau_lb"]["applied"]
    out["val_b1_xrepo"] = f4(need(val, "systems.B1.metrics.cross_repo_macro_f1", v))
    gate = need(test, "gating.config.applied", t)
    if cs["M1_test_tau_lb"]["applied"] != gate["applied"]:
        raise ValueError("posthoc.json C_sanity applied != metrics_test.json gating.config.applied")
    out["gate_wrong"] = gate["applied"] - cs["M1_test_tau_lb"]["correct"]["all"]
    out["m1_brier"] = f4(sysm["M1"]["brier"])

    # ch10 deployment
    lock = pins(m["constraints"])
    out["n_pins"] = len(lock)
    doc = s["workflow_tests"].split('"""')[1]
    rules = re.findall(r"^\(([a-f])\) (.*?)(?=^\([a-f]\) |\Z)", doc, re.MULTILINE | re.DOTALL)
    if len(rules) != 6:
        raise ValueError(f"tests/test_workflows.py: expected rules (a)-(f), found {len(rules)}")
    out["workflow_rules"] = "\n".join(f"- ({a}) {flat(b)}" for a, b in rules)
    ci = yaml.safe_load(s["ci"])
    out["ci_jobs"] = ", ".join(f"`{j}`" for j in ci["jobs"])
    runs = table(p4, "Run", "results/phase4.md")
    out["runs_table"] = md_table(["Run", "Trigger", "Caches", "Wall time", "Notes"],
                                 [[r["Run"], r["Trigger"], r["Caches"], r["Wall time"], r["Notes"]] for r in runs])
    issues = need(m["issues"], "fixtures", "sandbox/issues.json")
    out["n_fixtures"] = len(issues)
    out["n_skip_fixtures"] = sum(1 for f in issues if f["labels_at_creation"])
    wf = yaml.safe_load(mr.consumer_workflow(spec)[0])
    out["cron"] = wf.get("on", wf.get(True))["schedule"][0]["cron"]  # PyYAML reads the key `on` as True
    out["run_warm_cache"], out["run_warm_issue"] = mr.RUN_WARM_CACHE, mr.RUN_WARM_ISSUE_2
    out["run_probe"] = mr.RUN_COLD_DISPATCH
    lat_rows = []
    for key, d in ph0["latency"].items():
        if key.startswith("W0@1024/256"):
            lat_rows.append([f"Phase 0, base checkpoint, {d['n']} train issues", key.split("=")[1], f"{d['p50_ms']:.0f}", f"{d['p95_ms']:.0f}"])
    for d in need(test, "latency.stop_a_100_val_issues_R1", t).values():
        lat_rows.append([f"Phase 3 Stop A, R1, first {d['n']} val issues", str(d["threads"]), f"{d['p50']:.0f}", f"{d['p95']:.0f}"])
    tr = need(test, "latency.test_run", t)
    lat_rows.append([f"Phase 3 test run, M1, {tr['n']:,} test issues", str(tr["torch_threads"]), f"{tr['p50']:.0f}", f"{tr['p95']:.0f}"])
    out["local_latency_table"] = md_table(["Measurement (model preloaded)", "Threads", "p50 ms", "p95 ms"], lat_rows,
                                          ["---", "---:", "---:", "---:"])
    out["b1_p50_ms"] = f"{need(test, 'latency.B1.p50_ms', t):.2f}"
    out["load_s_test"] = need(test, "latency.test_run.load_s", t)

    # ch11 / appendices
    out["pin_table"] = md_table(["Package", "Version", "Package", "Version"],
                                [[f"`{a[0]}`", a[1], *((f"`{b[0]}`", b[1]) if b else ("", ""))]
                                 for a, b in zip(lock[::2], lock[1::2] + [None] * (len(lock) % 2))])
    pp = s["pyproject"]["project"]
    out["pyproject_table"] = md_table(["Group", "Pins"], [["runtime", ", ".join(f"`{d}`" for d in pp["dependencies"])]] +
                                      [[f"`{k}` extra", ", ".join(f"`{d}`" for d in v_)] for k, v_ in pp["optional-dependencies"].items()])
    meta = need(test, "systems.M1.meta", t)
    out["hash_table"] = md_table(["What", "Identifier"], [
        ["Shipped model (R1b)", f"`{F['repo']}` at `{F['revision']}`"],
        ["Trained weights (R1)", f"`{F['repo']}` at `{F['trained_revision']}`"],
        ["Base checkpoint", f"`convaiinnovations/laya` at `{F['base_revision']}`"],
        ["Laya notebook source", f"tag `{s['mr_values']['laya_tag']}`, commit `{s['mr_values']['laya_commit']}`"],
        ["NLBSE'24 upstream", f"commit `{s['mr_values']['nlbse_commit']}`"],
        ["Test rows (`test.jsonl`)", f"SHA-256 `{meta['input']['sha256']}`"],
        ["Training items (eps " + F["eps"] + ")", f"SHA-256 `{out['items_sha']}`"],
        ["`train.jsonl`", f"SHA-256 `{need(m['items'], ('files', 'train.jsonl', 'sha256'), 'items_sha256.json')}`"],
        ["`val.jsonl`", f"SHA-256 `{need(m['items'], ('files', 'val.jsonl', 'sha256'), 'items_sha256.json')}`"],
        ["Linux lock", f"`constraints-linux.txt`, SHA-256 `{hashlib.sha256(m['constraints'].encode()).hexdigest()}`"],
    ])
    out["eval_versions"] = ", ".join(f"{k} {v_}" for k, v_ in F["eval_versions"].items())
    out["questions_py"] = s["questions_py"].rstrip("\n")
    out["config_yaml"] = s["config_text"].rstrip("\n")
    out["val_full_table"] = full_metrics_table(val, ("M1", "B1", "B2_512-192", "B2_1024-256"), names, v)
    out["test_full_table"] = full_metrics_table(test, ("M1", "B0", "B1", "B2_512-192", "B2_1024-256"), names, t)
    fx = need(m["expected_local"], "fixtures", "expected_local.json")
    f_rows = []
    for f in issues:
        e = fx[f["key"]]
        if e["skip_reason"]:
            dec = f"skip ({e['skip_reason']})"
        else:
            dec = f"{e['label']} {f4(e['answer_confidence'])} → {e['decision']['action']}"
        f_rows.append([f["key"], f["title"], f["expected_class"], dec])
    out["fixture_table"] = md_table(["Key", "Title (our own fixture)", "Expected class", "Local result → decision"], f_rows)
    out["tags_table"] = md_table(["Tag", "Commit", "Created"], [[f"`{x['tag']}`", f"`{x['commit'][:12]}`", x["date"]]
                                                                 for x in s["tags"]["tags"]])
    out["changelog_table"] = md_table(["Version", "Date", "Summary"],
                                      [[r["Version"], r["Date"], shorten(first_sentence(r["Change"]))] for r in log])
    out["spec_glossary_table"] = md_table(["Term", "Meaning"], [[r["Term"].replace("**", ""), r["Meaning"]] for r in sp["glossary"]])
    out["references"] = "\n".join(f"- {r}" for r in sp["references"])
    return out


def full_metrics_table(metrics, systems, names, src):
    rows = []
    for k in systems:
        mm = need(metrics, f"systems.{k}.metrics", src)
        pcnt = mm["predicted_counts"]
        rows.append([names[k], f4(mm["accuracy"]), f4(mm["cross_repo_macro_f1"]), f4(mm["pooled_macro_f1"]),
                     *(f4(mm["per_class"][lab]["f1"]) for lab in LABELS), f4(mm["ece"]), f4(mm["brier"]),
                     " / ".join(str(pcnt[lab]) for lab in LABELS)])
    return md_table(["System", "Accuracy", "Cross-repo macro-F1", "Pooled macro-F1", *(f"F1 {lab}" for lab in LABELS),
                     "ECE", "Brier", "Predicted " + " / ".join(LABELS)], rows, ["---"] + ["---:"] * 8 + ["---"])


# ------------------------------------------------------------------------------------------------ includes

LINK = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(([^)\s]+)\)")


def include(doc_name, docs, anchor, select, tables_captions):
    """One section (heading excluded) of a generated doc, or one paragraph of it, with links made printable."""
    text = docs[doc_name]
    heading = next((ln for ln in text.splitlines() if re.match(r"^#{2,4} ", ln) and slug(ln.lstrip("#")) == anchor), None)
    if heading is None:
        raise ValueError(f"{doc_name}: no heading with anchor #{anchor}")
    body = block(text, heading, doc_name)
    if select:
        paras = [p for p in body.split("\n\n") if strip_md(p).startswith(select)]
        if len(paras) != 1:
            raise ValueError(f"{doc_name}#{anchor}: {len(paras)} paragraphs start with {select!r}")
        body = paras[0]
    body = printable_links(body, doc_name)
    if tables_captions is not None:
        body = caption_tables(body, tables_captions, f"{doc_name}#{anchor}")
    return f"<!-- include {doc_name}#{anchor} -->\n{body}"


def printable_links(md, doc_name):
    """Images become numbered figures; links to sections of the generated docs become cross-references; other
    relative links keep their text only. Code blocks are left alone."""
    parts = re.split(r"(```.*?```)", md, flags=re.DOTALL)
    base = DOC_NAMES[doc_name].parent

    def image(m):
        path = (base / m.group(3)).resolve().relative_to(ROOT)
        return f"{{{{figure:@{path}|{m.group(2)}}}}}"

    def link(m):
        bang, label, target = m.groups()
        if bang:
            return image(m)
        if target.startswith("http"):
            return m.group(0)
        file, _, anchor = target.partition("#")
        if re.fullmatch(r"!\[.*\]\(.*\)", label):  # a linked image
            return image(LINK.match(label))
        dest = (base / file).resolve() if file else DOC_NAMES[doc_name]
        names = {p.resolve(): n for n, p in DOC_NAMES.items()}
        if anchor and dest in names:
            return f"{label}{{{{xref:{names[dest]}#{anchor}}}}}"
        return label

    for i in range(0, len(parts), 2):
        parts[i] = LINK.sub(link, parts[i])
    return "".join(parts)


def caption_tables(md, captions, src):
    caps = [c.strip() for c in captions.split(";")]
    out, n, prev = [], 0, ""
    for line in md.splitlines():
        if line.startswith("|") and not prev.startswith("|"):
            if n >= len(caps):
                raise ValueError(f"{src}: more tables than captions")
            out += [f"{{{{caption:{caps[n]}}}}}", ""]
            n += 1
        out.append(line)
        prev = line
    if n != len(caps):
        raise ValueError(f"{src}: {n} tables, {len(caps)} captions")
    return "\n".join(out)


INCLUDE = re.compile(r"\{\{include:(\w+)#([\w-]+)(?:~([^|}]+))?(?:\|tables=([^}]*))?\}\}")
FACT = re.compile(r"\{\{fact:(\w+)\}\}")


# ------------------------------------------------------------------------------------------------ numbering

def esc_inline(text):
    """Plain caption text to HTML: backticks become <code>."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(text, quote=False).replace("&#x27;", "'"))


def number(md, figures):
    """Number chapters, sections, tables and figures; resolve cross-references; build the contents and the lists."""
    out, fence = [], False
    chap, sec, app, label = 0, 0, 0, None
    toc, lof, lot, tab_n, fig_n, includes, ids = [], [], [], 0, 0, {}, set()
    pending_caption = None
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("```"):
            fence = not fence
        if fence or line.startswith("```"):
            out.append(line)
            continue
        m = re.match(r"^(#{1,3}) (.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2)
            if title.endswith(" {-}"):
                title, num = title[:-4], None
            elif level == 1 and title.endswith(" {appendix}"):
                app, sec, tab_n, fig_n = app + 1, 0, 0, 0
                label = chr(ord("A") + app - 1)
                title, num = f"Appendix {label}: {title[:-11]}", ""
            elif level == 1:
                chap, sec, tab_n, fig_n = chap + 1, 0, 0, 0
                label = str(chap)
                num = label
            elif level == 2 and label is not None:
                sec += 1
                num = f"{label}.{sec}"
            else:
                num = None
            text = f"{num} {title}" if num else title
            anchor = slug(text)
            if anchor in ids:
                raise ValueError(f"duplicate heading anchor #{anchor}")
            ids.add(anchor)
            if level == 1 or (level == 2 and num):
                toc.append((level, text, anchor))
            out.append(f"{m.group(1)} {text}")
            current = (num or title, anchor)
            continue
        if "<!-- include " in line or "<!-- alias " in line:
            for name in re.findall(r"<!-- (?:include|alias) (\S+) -->", line):
                includes[name] = current
            line = re.sub(r"\s*<!-- (?:include|alias) \S+ -->", "", line)
            if not line.strip():
                continue
        cm = re.fullmatch(r"\{\{caption:(.+)\}\}", line.strip())
        if cm:
            pending_caption = cm.group(1)
            continue
        fm = re.fullmatch(r"\{\{figure:([^|]+)\|(.+)\}\}", line.strip())
        if fm:
            key, cap = fm.groups()
            path = key[1:] if key.startswith("@") else figures[key]
            if not (ROOT / path).is_file():
                raise ValueError(f"figure {path} does not exist")
            fig_n += 1
            fid = f"figure-{label}-{fig_n}".lower()
            lof.append((f"Figure {label}.{fig_n}", cap, fid))
            out += [f'<figure id="{fid}">', f'<img src="{Path("../..") / path}" alt="{html.escape(cap)}">',
                    f"<figcaption><strong>Figure {label}.{fig_n}.</strong> {esc_inline(cap)}</figcaption>", "</figure>"]
            continue
        if line.startswith("|") and not (out and out[-1].startswith("|")):
            if pending_caption is None:
                raise ValueError(f"table without a caption near: {line[:80]!r}")
            tab_n += 1
            tid = f"table-{label}-{tab_n}".lower()
            lot.append((f"Table {label}.{tab_n}", pending_caption, tid))
            if out and out[-1] == "":
                out.pop()
            out += ["", f'<p class="caption" id="{tid}"><strong>Table {label}.{tab_n}.</strong> {esc_inline(pending_caption)}</p>', ""]
            pending_caption = None
        out.append(line)
    if pending_caption is not None:
        raise ValueError(f"caption without a table: {pending_caption!r}")
    text = "\n".join(out)

    def xref(m):
        hit = includes.get(m.group(1))
        return f" (Section {hit[0]})" if hit else ""

    text = re.sub(r"\{\{xref:([\w-]+#[\w-]+)\}\}", xref, text)
    toc_html = ['<nav class="toc">', "<ul>"]
    for level, t, a in toc:
        if t == "Contents":
            continue
        toc_html.append(f'<li class="toc-{level}"><a href="#{a}">{esc_inline(t)}</a></li>')
    toc_html += ["</ul>", "</nav>"]

    def listing(items, cls):
        return "\n".join([f'<nav class="{cls}">', "<ul>"] +
                         [f'<li><a href="#{a}">{n}. {esc_inline(c)}</a></li>' for n, c, a in items] + ["</ul>", "</nav>"])

    text = (text.replace("\x00toc\x00", "\n".join(toc_html)).replace("\x00lof\x00", listing(lof, "lof"))
            .replace("\x00lot\x00", listing(lot, "lot")))
    return text, {"figures": lof, "tables": lot, "toc": toc}


# ------------------------------------------------------------------------------------------------ build

def build(s):
    """REPORT.md text from the loaded sources."""
    m = s["mr"]
    built = mr.build_docs(**m)
    docs = {n: built[p] for n, p in DOC_NAMES.items()}
    s["mr_values"] = values = mr.build_values(**m)
    s["facts_values"] = facts = fact_values(s["facts"])
    values = {**values, **report_values(s)}
    values.update({"toc": "\x00toc\x00", "list_of_figures": "\x00lof\x00", "list_of_tables": "\x00lot\x00"})
    figures = {**FIGURES, **need(m["test"], "figures", "metrics_test.json")}
    parts = []
    for name, text in s["chapters"].items():
        text = INCLUDE.sub(lambda x: include(x.group(1), docs, x.group(2), x.group(3), x.group(4)), text)

        def fact(x, name=name):
            if x.group(1) not in facts:
                raise KeyError(f"{name}: unknown fact {x.group(1)}")
            return facts[x.group(1)]

        text = FACT.sub(fact, text)
        parts.append(mr.render(text, values).strip("\n"))
    body, _ = number("\n\n".join(parts), figures)
    head = ("<!-- Generated by docs/report/make_report.py from docs/report/chapters/*.md and the committed files it "
            "names; do not edit by hand. -->\n\n")
    return head + body + "\n"


def to_html(md_text):
    """REPORT.md to a standalone HTML page (markdown-it-py, `report` extra)."""
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    tokens = md.parse(md_text)
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            tok.attrSet("id", slug(tokens[i + 1].content))
            if tok.tag == "h1":
                title = tokens[i + 1].content
                later_appendix = title.startswith("Appendix ") and not title.startswith("Appendix A:")
                first = re.match(r"^1 |^Appendix A:", title) is not None
                numbered = re.match(r"^\d+ |^Appendix ", title) is not None
                tok.attrSet("class", ("appendix" if later_appendix else "chapter first" if first else "chapter")
                            if numbered else "front")
    body = md.renderer.render(tokens, md.options, {})
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>Laya Triage: technical report</title><style>{CSS.read_text()}</style></head>"
            f"<body>{body}</body></html>\n")


def render_pdf(md_text, out=PDF):
    from weasyprint import HTML

    out.parent.mkdir(parents=True, exist_ok=True)
    page = to_html(md_text)
    (out.parent / HTML_OUT.name).write_text(page)
    doc = HTML(string=page, base_url=str(REPORT_DIR) + "/").render()
    doc.write_pdf(out)
    return doc


def anchor_pages(doc):
    """{anchor id: 1-based page number} for every anchor in a rendered WeasyPrint document."""
    pages = {}
    for n, page in enumerate(doc.pages, 1):
        for a in page.anchors:
            pages.setdefault(a, n)
    return pages


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/report/REPORT.md is out of date")
    ap.add_argument("--pdf", action="store_true", help="also render build/report/laya-triage-report.pdf")
    ap.add_argument("--snapshot-tags", action="store_true", help="rewrite docs/report/tags.yml from local git")
    args = ap.parse_args(argv)
    if args.snapshot_tags:
        write_tags(snapshot_tags())
        print(f"wrote {TAGS.relative_to(ROOT)}")
        return
    text = build(load_sources())
    if args.check:
        if not REPORT.exists() or REPORT.read_text() != text:
            sys.exit("out of date: docs/report/REPORT.md; run docs/report/make_report.py")
        print("up to date: docs/report/REPORT.md")
        return
    REPORT.write_text(text)
    print(f"wrote {REPORT.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    if args.pdf:
        doc = render_pdf(text)
        print(f"wrote {PDF.relative_to(ROOT)} ({len(doc.pages)} pages, {PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
