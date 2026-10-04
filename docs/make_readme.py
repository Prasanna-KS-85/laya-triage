"""Build README.md, docs/DEEP_DIVE.md, docs/USING_THE_ACTION.md and docs/USING_THE_MODEL.md from their templates
(docs/*.template.md) and the committed result files (PROJECT_SPEC.md §13 Phase 5, §16 rule 11).
Never touches the network, the model or the data.

Run from anywhere: .venv/bin/python docs/make_readme.py [--check]
The templates hold prose and `{{name}}` placeholders only; tests/test_readme.py fails if one contains a number that
is not a placeholder (version tags, FR-/NFR- ids, § references and fixture ids excepted). Links in a template are
relative to its output file. Every number comes from:
- results/phase3/test_R1b/metrics_test.json: test metrics, paired comparisons, gating, latency, B3 rows, figure path;
- results/phase3/val_R1b/metrics_val.json: val metrics and the §10.4 fit;
- config/triage.default.yml: the configuration table, labels, shipped thresholds;
- action.yml: the Action inputs table;
- results/phase3/posthoc/posthoc.json: the near-duplicate check;
- sandbox/expected_local.json: fixtures S01, S02, S05 (the worked example, re-decided here with policy.decide), S17
  and S18 (numbers only); sandbox/issues.json: the authored text of S01 (our own fixture, never dataset text);
- docs/snippets/classify_one.py and docs/snippets/consumer-minimal.yml: inserted verbatim (their constants are
  checked against config/triage.default.yml by tests/test_snippets.py);
- constraints-linux.txt: the torch pin for the CPU-only install in docs/USING_THE_MODEL.md;
- results/phase4.md: sandbox run table (wall times) and the NFR-1 latency probe;
- results/phase3.md: the pre-declared test command;
- PROJECT_SPEC.md: the consumer workflow (§12.5), NFR targets (§6.2), runner (§17 Q4), pinned upstream commits (§8.1,
  §9.3); and everything docs/make_model_card.py already reads (results/phase3/make_model_card.py `facts`).
A missing key, row or pattern raises (KeyError / ValueError) naming the file. Statuses (met / not met) are computed
from the numbers, never typed. --check exits 1 if any output differs from a fresh build instead of writing it.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "results" / "phase3"))
sys.path.insert(0, str(ROOT / "eval"))

import gating
import make_model_card as mmc
from make_model_card import CFG, f4, need

from laya_triage import policy
from laya_triage.classifier import TriageResult
from laya_triage.config import validate
from laya_triage.preprocess import CODE_HEAD_LINES, CODE_TAIL_LINES
from laya_triage.questions import LABELS

DOCS = ROOT / "docs"
README = ROOT / "README.md"
TEMPLATE = DOCS / "README.template.md"
OUTPUTS = {  # template -> generated file; links inside a template are relative to its output
    TEMPLATE: README,
    DOCS / "DEEP_DIVE.template.md": DOCS / "DEEP_DIVE.md",
    DOCS / "USING_THE_ACTION.template.md": DOCS / "USING_THE_ACTION.md",
    DOCS / "USING_THE_MODEL.template.md": DOCS / "USING_THE_MODEL.md",
}
SNIPPETS = {"snippet_classify_one": DOCS / "snippets" / "classify_one.py",
            "snippet_consumer_minimal": DOCS / "snippets" / "consumer-minimal.yml"}
# The worked example: fixture -> the decision it must show (apply, bug below its threshold, never-applied question).
WORKED = {"S01": "apply", "S02": "escalate", "S05": "escalate"}
SPEC = "PROJECT_SPEC.md"
P4 = "results/phase4.md"
EXTRA_SOURCES = {
    "template": TEMPLATE,
    "action": ROOT / "action.yml",
    "posthoc": ROOT / "results" / "phase3" / "posthoc" / "posthoc.json",
    "expected_local": ROOT / "sandbox" / "expected_local.json",
    "phase3_md": ROOT / "results" / "phase3.md",
    "phase4_md": ROOT / "results" / "phase4.md",
    "issues": ROOT / "sandbox" / "issues.json",
    "constraints": ROOT / "constraints-linux.txt",
}
# Run ids of the sandbox runs the README cites; each row's role is checked against its cells (see `runtime`).
RUN_COLD_ISSUE, RUN_WARM_ISSUE, RUN_WARM_ISSUE_2 = "37025861693", "37028030094", "37059668421"
RUN_WARM_CACHE, RUN_COLD_DISPATCH = "37058789375", "37026455422"
DESCRIPTIONS = {  # meaning of each config key; `{never}`, `{head}`, `{tail}` are filled from the code
    "model.repo": "Hugging Face repository of the fine-tuned model.",
    "model.revision": "Commit SHA of the model; pinned, never a branch name. It is also the model-cache key, so a new revision means a cold download.",
    "model.max_len": "Token budget for the state. It must equal the value the model was trained with.",
    "model.backend": "Inference backend. Only `torch` is supported in v1.0; `onnx` is planned for v1.2.",
    "labels.bug": "GitHub label applied to a confident `bug` prediction.",
    "labels.feature": "GitHub label applied to a confident `feature` prediction.",
    "labels.question": "GitHub label applied to a confident `question` prediction.",
    "labels.escalate": "Label for issues the Action will not label itself: low confidence, or a class whose threshold is {never} or more.",
    "gating.thresholds.bug": "Minimum calibrated `answer_confidence` to auto-apply `bug`. Fitted on validation data (§10.4).",
    "gating.thresholds.feature": "Same for `feature`. A value of {never} or more means never auto-apply: such issues are always escalated.",
    "gating.thresholds.question": "Same for `question`. A value of {never} or more means never auto-apply: such issues are always escalated.",
    "behaviour.mode": "`dry-run` writes decisions to the job summary and changes nothing; `apply` adds labels. The `mode` input of the Action overrides it, and backfill runs are dry-run unless the input says `apply`.",
    "behaviour.comment_on_escalate": "Post one fixed-text comment with the top two guesses when escalating. It never repeats issue text.",
    "behaviour.skip_if_labeled": "Skip issues that already carry any of the three type labels.",
    "behaviour.skip_authors": "Skip issues opened by these authors (case-insensitive), for example bots.",
    "behaviour.create_missing_labels": "Create labels that do not exist in the repository (default colour). Otherwise a missing label is skipped with a warning.",
    "behaviour.escalate_on_error": "If the model cannot be loaded or inference fails, add the escalation label instead of leaving the issue untouched.",
    "preprocess.max_code_lines": "Fenced code blocks and logs longer than this are collapsed to their first {head} and last {tail} lines.",
    "preprocess.max_body_chars": "Safety cap on the body length in characters, applied after cleaning.",
}


def grab(pattern, text, src, flags=0):
    m = re.search(pattern, text, flags)
    if not m:
        raise ValueError(f"{src}: pattern not found: {pattern}")
    return m


def versus(gap):
    """'2.5 points below' / '1.0 points above' for a difference in F1 (fractions)."""
    return f"{abs(100 * gap):.1f} points {'above' if gap > 0 else 'below'}"


def pct(x):
    return f"{100 * x:.1f}%"


def md_cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_rows(md, first_header, src):
    """{first cell: {header: cell}} for the markdown table whose header row starts with `first_header`."""
    header, rows = None, {}
    for line in md.splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = md_cells(line)
        if header is None:
            header = cells if cells[0] == first_header else None
        elif not set(line) <= set("|-: "):
            if len(cells) != len(header):
                raise ValueError(f"{src}: row {cells[0]!r} has {len(cells)} cells, header has {len(header)}")
            rows[cells[0]] = dict(zip(header, cells))
    if not rows:
        raise ValueError(f"{src}: no table with header starting {first_header!r}")
    return rows


def consumer_workflow(spec_md):
    """The §12.5 example, with the SHA placeholder of the Action line replaced by the release tag; also the tag."""
    block = grab(r"### 12\.5 Example consumer workflow\s+```yaml\n(.*?)\n```", spec_md, SPEC, re.DOTALL).group(1)
    block, n = re.subn(r"(uses: Prasanna-KS-85/laya-triage)@<[^>]*> # (v[\d.]+)", r"\1@\2", block)
    if n != 1:
        raise ValueError(f"{SPEC} §12.5: expected one Action line with a SHA placeholder, found {n}")
    wf = yaml.safe_load(block)
    if wf["permissions"] != {"issues": "write", "contents": "read"}:
        raise ValueError(f"{SPEC} §12.5: unexpected permissions {wf['permissions']}")
    return block, grab(r"laya-triage@(v[\d.]+)", block, SPEC).group(1)


def flatten(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def config_table(config):
    flat = dict(flatten(config))
    missing, extra = sorted(set(flat) - set(DESCRIPTIONS)), sorted(set(DESCRIPTIONS) - set(flat))
    if missing or extra:
        raise ValueError(f"{CFG} vs DESCRIPTIONS in docs/make_readme.py: undescribed keys {missing}, stale keys {extra}")
    fill = {"never": gating.NEVER, "head": CODE_HEAD_LINES, "tail": CODE_TAIL_LINES}
    rows = [f"| `{k}` | `{json.dumps(v)}` | {DESCRIPTIONS[k].format(**fill)} |" for k, v in flat.items()]
    return "| Key | Default | Meaning |\n|---|---|---|\n" + "\n".join(rows)


def action_inputs_table(action):
    rows = []
    for name, spec in need(action, "inputs", "action.yml").items():
        desc = " ".join(need(spec, "description", "action.yml").split())
        rows.append(f"| `{name}` | `{json.dumps(need(spec, 'default', 'action.yml'))}` | {desc} |")
    return "| Input | Default | Meaning |\n|---|---|---|\n" + "\n".join(rows)


def runtime(phase4_md, spec_md, test):
    """Sandbox wall times, the NFR-1 probe and the NFR targets, with every run's role checked against its cells."""
    runs = table_rows(phase4_md, "Run", P4)

    def run(run_id, role, wall_ok):
        r = runs.get(run_id)
        if r is None or not wall_ok(r):
            raise ValueError(f"{P4}: run {run_id} is not the {role} run any more: {r}")
        return r["Wall time"]

    cold = run(RUN_COLD_ISSUE, "cold issue", lambda r: "cold" in r["Caches"] and "issue" in r["Trigger"])
    warm = [run(RUN_WARM_ISSUE, "warm issue", lambda r: "restored" in r["Caches"] and "issue" in r["Trigger"]),
            run(RUN_WARM_ISSUE_2, "warm issue", lambda r: "restored" in r["Caches"] and "issue" in r["Trigger"])]
    warm_cache = run(RUN_WARM_CACHE, "warm-cache", lambda r: "warm-cache=true" in r["Trigger"])
    cold_dispatch = run(RUN_COLD_DISPATCH, "cold dispatch + probe", lambda r: "latency probe" in r["Trigger"])
    pr = grab(r"\*\*Latency probe \(NFR-1, run (\d+)\)\.\*\* n (\d+), p50 ([\d.]+) ms, p95 ([\d.]+) ms, mean ([\d.]+) ms, "
              r"max ([\d.]+) ms, model\s+load ([\d.]+) s, `torch\.get_num_threads\(\)` (\d+)", phase4_md, P4)
    if pr.group(1) != RUN_COLD_DISPATCH:
        raise ValueError(f"{P4}: latency probe run {pr.group(1)} != {RUN_COLD_DISPATCH}")
    n, p50, p95, threads = int(pr.group(2)), float(pr.group(3)), float(pr.group(4)), int(pr.group(8))
    targets = {m.group(1): m.group(2) for m in re.finditer(r"\| NFR-([123]) \|[^|]*\|\s*≤ ([\d.]+ (?:s|min))", spec_md)}
    if set(targets) != {"1", "2", "3"}:
        raise ValueError(f"{SPEC} §6.2: NFR-1..3 targets not found: {targets}")

    def secs(text):  # "1m26s" / "44s" -> seconds
        m = grab(r"^(?:(\d+)m)?(\d+)s$", text, P4)
        return 60 * int(m.group(1) or 0) + int(m.group(2))

    def within(target, seconds):
        value, unit = target.split()
        return seconds <= float(value) * (60 if unit == "min" else 1)

    goal, goal_unit = targets["1"].split()
    if goal_unit != "s":
        raise ValueError(f"{SPEC} §6.2: NFR-1 target {targets['1']!r} is not in seconds")
    warm_sorted = sorted(warm, key=secs)
    q4 = grab(r"(\d+) logical CPUs, ([^,]+), ([\d,]+) MiB RAM", spec_md, f"{SPEC} Q4")
    lat = need(test, "latency.test_run", "metrics_test.json")
    return {
        "cold": cold, "warm": " and ".join(warm),
        "warm_cache": warm_cache, "cold_dispatch": cold_dispatch,
        "probe_n": n, "probe_p50": f"{p50:.0f} ms", "probe_p95": f"{p95 / 1000:.1f} s", "probe_threads": threads,
        "nfr1": targets["1"], "nfr2": targets["2"], "nfr3": targets["3"],
        "probe_p95_secs": f"{p95 / 1000:.1f}", "nfr1_goal": f"{goal} second{'' if float(goal) == 1 else 's'}",
        "warm_range": f"{warm_sorted[0]} to {warm_sorted[-1]}" if len(set(warm)) > 1 else warm[0],
        "nfr1_status": "met" if within(targets["1"], p95 / 1000) else "not met",
        "latency_status": "met" if within(targets["1"], p95 / 1000) else "missed",
        "nfr2_status": "met" if all(within(targets["2"], secs(w)) for w in warm) else "not met",
        "nfr3_status": "met" if within(targets["3"], secs(cold)) else "not met",
        "runner_cpus": q4.group(1), "runner_cpu_model": q4.group(2),
        "runner_ram_mib": q4.group(3),
        "local_hw": need(test, "latency.hardware", "metrics_test.json").split(";")[0],
        "local_p50": f"{lat['p50']:.0f} ms", "local_p95": f"{lat['p95']:.0f} ms", "local_threads": lat["torch_threads"],
    }


def worked_example(expected_local, issues, config, phase4_md):
    """S01, S02 and S05 from expected_local.json, each decided again by policy.decide with the default config (comments
    on, to render the escalation templates); raises if a decision or a threshold no longer matches. S01's title and
    body come from sandbox/issues.json, our own authored fixture text."""
    fx = need(expected_local, "fixtures", "expected_local.json")
    texts = {f["key"]: f for f in need(issues, "fixtures", "sandbox/issues.json")}
    raw = json.loads(json.dumps(config))
    raw["behaviour"]["comment_on_escalate"] = True
    cfg = validate(raw)
    out = {}
    for key, action in WORKED.items():
        f = need(fx, key, "expected_local.json")
        result = TriageResult(0, f["label"], f["probabilities"], f["answer_confidence"], 0.0, "", "")
        d = policy.decide(result, cfg)
        if (d.action, f["decision"]["action"], list(d.labels_to_add), f["threshold"]) != (
                action, action, f["decision"]["labels_to_add"], cfg.thresholds[f["label"]]):
            raise ValueError(f"sandbox/expected_local.json: {key} no longer shows the decision the docs describe")
        k = key.lower()
        out.update({f"{k}_{lab}": f4(f["probabilities"][lab]) for lab in LABELS})
        out.update({f"{k}_label": f["label"], f"{k}_conf": f4(f["answer_confidence"]),
                    f"{k}_tau": f"{f['threshold']:.4f}" if f["threshold"] < gating.NEVER else f"never ({f['threshold']})",
                    f"{k}_title": need(texts, key, "sandbox/issues.json")["title"],
                    f"{k}_outcome_label": d.labels_to_add[0], f"{k}_comment": d.comment or ""})
    out["s01_body"] = texts["S01"]["body"]
    runs = grab(r"- dry-run \((\d+) issue runs\): (\d+)/(\d+) ok", phase4_md, P4)
    apply_ok = grab(r"- backfill apply: (\d+)/(\d+) ok", phase4_md, P4)
    if len({runs.group(1), runs.group(2), runs.group(3), apply_ok.group(1), apply_ok.group(2)}) != 1:
        raise ValueError(f"{P4}: the sandbox runs no longer all match expected_local.json")
    out["sandbox_runs"] = runs.group(1)
    return out


def torch_pin(constraints):
    return grab(r"^torch==([\w.]+)$", constraints, "constraints-linux.txt", re.MULTILINE).group(1)


def coincidence_note(point_coverage, shipped_precision):
    """A warning, emitted only while the coverage at the point-estimate thresholds and the precision at the shipped
    thresholds round to the same four decimals (they are different quantities)."""
    if f4(point_coverage) != f4(shipped_precision):
        return ""
    return (f"A coincidence of the numbers needs a warning: the coverage of the τ_point rule ({f4(point_coverage)}) "
            f"equals the precision of the shipped thresholds ({f4(shipped_precision)}) to four decimals. They are "
            "different quantities: the first is the share of all test issues the τ_point rule would label, the second "
            "the share of correct labels among the issues the shipped thresholds did label.")


def short_results_table(test):
    """README table: M1, B1 and the native zero-shot B2 setting, 512/192 (the deep dive shows both B2 settings)."""
    t = "metrics_test.json"
    native = "B2_512-192"
    rows = [("**Our model (fine-tuned Laya)**", "M1"), ("TF-IDF + logistic regression", "B1"),
            ("Zero-shot Laya (no fine-tuning)", native)]
    out = ["| System | Cross-repo macro-F1 | Calibration error (ECE) |", "|---|---:|---:|"]
    for name, key in rows:
        m = need(test, f"systems.{key}.metrics", t)
        f1 = f4(need(m, "cross_repo_macro_f1", t))
        out.append(f"| {name} | {f'**{f1}**' if key == 'M1' else f1} | {f4(need(m, 'ece', t))} |")
    return "\n".join(out)


def build_values(test, val, config, refit, items, experiments_md, spec_md, phase2_md, template, action, posthoc,
                 expected_local, phase3_md, phase4_md, issues, constraints, snippets, classify_snippet=None,
                 templates=None):
    t, v = "metrics_test.json", "metrics_val.json"
    F = mmc.facts(test, val, config, refit, items, experiments_md, spec_md)
    sysm = {s: need(test, f"systems.{s}.metrics", t) for s in ("M1", "B0", *(b for b, _ in mmc.BASELINES))}
    m1 = sysm["M1"]
    t1 = need(test, "systems.M1.uncalibrated_T1", t)
    paired = {b: need(test, f"paired_M1_vs.{b}", t) for b, _ in mmc.BASELINES}
    gate = need(test, "gating.config.applied", t)
    gate_point = need(test, "gating.M1_tau_point.applied", t)
    b1_gate = need(test, "gating.B1_same_rule.applied", t)
    b3 = need(test, "B3_copied_from_phase2_md.published", t)
    crit = need(test, "criteria_6_3", t)
    lb = {r["system"].split()[1]: r for r in b3}
    if {"SetFit", "RoBERTa"} - set(lb):
        raise ValueError(f"{t}: B3 rows SetFit and RoBERTa expected, found {sorted(lb)}")
    repos = list(need(m1, "per_repo_macro_f1", t))
    pc = need(m1, "per_class", t)
    bug_lb = need(val, "gating.M1.per_label.bug.at_tau_lb", v)
    bug_gate = need(gate, "per_label.bug", t)
    tr = F["train_label_counts"]
    supports = {need(pc, f"{lab}.support", t) for lab in LABELS}
    per_cell = next(iter(supports)) // len(repos)
    block, tag = consumer_workflow(spec_md)
    b3t = mmc.b3_training(phase2_md)
    reachable = [f"`{lab}`" for lab in LABELS if F["taus"][lab] < gating.NEVER]
    if len(reachable) != 1:  # the Limitations wording ("the only auto-applied label") assumes exactly one
        raise ValueError(f"{CFG}: expected exactly one auto-applied label, found {reachable}")
    grab(r"no §9\.5 experiments were run", phase3_md, "results/phase3.md")  # M1 is the single fine-tuning run
    grab(r"\| R1b \|.*? \| R1 weights \+ val-refit temperature\.", experiments_md, "results/experiments.md")
    val_ok = [f"`{lab}`" for lab in LABELS if need(val, f"gating.M1.per_label.{lab}", v).get("at_tau_lb")]
    rt = runtime(phase4_md, spec_md, test)

    def row(name, m, ece=True):
        c = need(m, "per_class", t)
        return (f"| {name} | {f4(need(m, 'cross_repo_macro_f1', t))} | {f4(need(m, 'pooled_macro_f1', t))} | "
                f"{f4(need(m, 'accuracy', t))} | " + " | ".join(f4(need(c, f'{lab}.f1', t)) for lab in LABELS) +
                f" | {f4(need(m, 'ece', t)) if ece else '—'} |")

    results = [row("**M1 (this model)**", m1), row("B0 majority class", sysm["B0"], ece=False),
               *(row(name, sysm[b]) for b, name in mmc.BASELINES)]
    for r in b3:
        results.append(f"| B3 {r['system'].split()[1]} (NLBSE'24, published, other protocol) | {f4(need(r, 'cross_repo_macro_f1', t))} | " +
                       " | ".join(f"{f4(need(r, k, t))} [c]" for k in ("pooled_macro_f1", "accuracy", "f1_bug", "f1_feature",
                                                                        "f1_question")) + " | — |")
    paired_rows = []
    for b, name in mmc.BASELINES:
        bs, mc = need(paired[b], "bootstrap_xrepo_macro_f1_diff", t), need(paired[b], "mcnemar_exact", t)
        ci = need(bs, "ci95", t)
        paired_rows.append(f"| M1 vs {name} | {need(bs, 'point', t):+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                           f"{need(mc, 'p_value', t):.2g} |")
    cis = [need(paired[b], "bootstrap_xrepo_macro_f1_diff.ci95", t) for b, _ in mmc.BASELINES]
    n_boot, seed = (need(paired["B1"], f"bootstrap_xrepo_macro_f1_diff.{k}", t) for k in ("n_boot", "seed"))
    ci_key = grab(r"\b(ci\d+)\b", " ".join(need(paired["B1"], "bootstrap_xrepo_macro_f1_diff", t)), t).group(1)
    gate_rows = []
    for lab in LABELS:
        g = need(gate, f"per_label.{lab}", t)
        tau = F["taus"][lab]
        gate_rows.append(f"| {lab} | {f'never ({gating.NEVER})' if tau >= gating.NEVER else f'{tau:.4f}'} | "
                         f"{f4(bug_lb['lower_bound']) if lab == 'bug' else f'none ≥ {gating.TARGET_PRECISION:.2f}'} | "
                         f"{need(g, 'applied', t)} | {f4(need(g, 'coverage', t))} | {f4(need(g, 'precision', t))} |")
    nd = need(posthoc, "A_near_duplicates", "posthoc.json")
    cut = grab(r"share_above_([\d.]+)", " ".join(nd["summary"]["val"]), "posthoc.json").group(1)
    acc = nd["accuracy_by_tertile"]
    drops = ", ".join(f"{100 * (acc['val']['M1'][k]['accuracy'] - acc['test']['M1'][k]['accuracy']):.1f}"
                      for k in ("low", "mid", "high"))
    fx = need(expected_local, "fixtures", "expected_local.json")
    s17, s18 = fx["S17"], fx["S18"]
    if (s17["label"], s17["decision"]["action"], s18["label"], s18["decision"]["action"]) != ("bug", "apply", "feature", "escalate"):
        raise ValueError("sandbox/expected_local.json: S17 / S18 no longer show the behaviour the README describes")
    fig = need(test, "figures.coverage_precision", t)
    if not (ROOT / fig).is_file():
        raise ValueError(f"{t}: figure {fig} does not exist")
    test_cmd = grab(r"```\n(\.venv/bin/python eval/run_eval\.py [^\n]*--split test[^\n]*)\n```", phase3_md,
                    "results/phase3.md").group(1)
    laya_commit = grab(r"tag\s+`(v[\d.]+)`, commit `([0-9a-f]{40})`", spec_md, f"{SPEC} §9.3")
    nlbse_commit = grab(r"upstream commit `([0-9a-f]{40})`", spec_md, f"{SPEC} §8.1").group(1)
    labels = need(config, "labels", CFG)
    ci_text = "all three intervals exclude 0" if all(lo > 0 or hi < 0 for lo, hi in cis) else "not every interval excludes 0"
    beats = "beats B1 and both B2 settings" if need(crit, "1.status", t) == "MET" else "does not beat B1 and both B2 settings"
    setfit_gap = need(m1, "cross_repo_macro_f1", t) - need(lb["SetFit"], "cross_repo_macro_f1", t)
    roberta_gap = need(m1, "cross_repo_macro_f1", t) - need(lb["RoBERTa"], "cross_repo_macro_f1", t)
    met3 = need(crit, "3.k", t)
    return {
        "github_url": mmc.GITHUB_URL, "laya_url": "https://github.com/NandhaKishorM/laya",
        "hf_url": f"https://huggingface.co/{F['repo']}", "model_repo": F["repo"], "revision": F["revision"],
        "tag": tag, "consumer_workflow": block, "action_inputs_table": action_inputs_table(action),
        "config_table": config_table(config), "escalate_label": need(labels, "escalate", CFG),
        "label_cmds": "\n".join(f'gh label create "{labels[k]}"' for k in (*LABELS, "escalate")),
        "mode": need(config, "behaviour.mode", CFG),
        "thresholds_text": ", ".join(f"`{lab}` {F['taus'][lab]}" for lab in LABELS), "never": gating.NEVER,
        "labels_list": ", ".join(f"`{lab}`" for lab in LABELS),
        "T_notebook": f"{F['T_notebook']:.4f}", "T_val": F["T_val"], "n_train": f"{F['n_train']:,}",
        "n_val": F["n_val"], "n_test": f"{F['n_test']:,}", "n_official": f"{F['n_official']:,}", "n_dropped": F["n_dropped"],
        "n_repos": len(repos), "per_cell": per_cell, "max_len": F["max_len"], "head_max_len": F["head_max_len"],
        "epochs": F["epochs"], "kaggle_gpus": F["n_gpus"], "share_truncated": pct(F["share_truncated"]),
        "train_counts": f"bug {tr['bug']}, feature {tr['feature']}, question {tr['question']}",
        "test_xrepo": f4(m1["cross_repo_macro_f1"]), "b1_xrepo": f4(sysm["B1"]["cross_repo_macro_f1"]),
        "setfit_xrepo": f4(lb["SetFit"]["cross_repo_macro_f1"]), "m1_acc": f4(m1["accuracy"]), "m1_ece": f4(m1["ece"]),
        "setfit_cmp": versus(setfit_gap), "roberta_cmp": versus(roberta_gap),
        "results_table": "\n".join(results), "paired_table": "\n".join(paired_rows), "ci_text": ci_text, "beats": beats,
        "n_boot": f"{n_boot:,}", "seed": seed, "ci_level": ci_key[2:], "question_f1": f4(pc["question"]["f1"]),
        "calib_table": (f"| Calibrated (T = {F['T_applied']}, shipped) | {f4(m1['ece'])} | {f4(m1['brier'])} |\n"
                        f"| Uncalibrated (T = 1) | {f4(t1['ece'])} | {f4(t1['brier'])} |"),
        "ece_bins": mmc.ECE_BINS, "T_applied": F["T_applied"], "ece_limit": f"{float(need(crit, '2.criterion', t).split('<=')[-1]):.2f}",
        "ece_status": need(crit, "2.status", t).lower(), "n_labels": len(LABELS),
        "ubuntu_switch_date": grab(r"ubuntu-latest moves to Ubuntu \d+ on (\d{4}-\d\d-\d\d)", spec_md, f"{SPEC} §12.5").group(1),
        "gating_table": "\n".join(gate_rows), "gate_applied": gate["applied"], "gate_n": f"{gate['n']:,}",
        "gate_cov": f4(gate["coverage"]), "gate_prec": f4(gate["precision"]), "gate_cov_pct": pct(gate["coverage"]),
        "gate_prec_pct": pct(gate["precision"]), "target": f"{gating.TARGET_PRECISION:.2f}",
        "target_status": "missed" if gate["precision"] < gating.TARGET_PRECISION else "reached",
        "bug_val_prec": f4(bug_lb["precision"]), "bug_val_lb": f4(bug_lb["lower_bound"]), "bug_val_n": bug_lb["support"],
        "bug_test_prec": f4(bug_gate["precision"]), "bug_prec_loss": f"{100 * (bug_lb['precision'] - bug_gate['precision']):.1f}",
        "bug_applied": bug_gate["applied"], "met_k": met3, "b3_rows_per_clf": b3t["rows_per_classifier"], "auto_label": reachable[0],
        "val_bound_labels": (f"only {val_ok[0]} had" if len(val_ok) == 1 else
                             (f"{', '.join(val_ok)} had" if val_ok else "no label had")), "point_cov": f4(gate_point["coverage"]),
        "coincidence_note": coincidence_note(gate_point["coverage"], gate["precision"]),
        "point_prec": f4(gate_point["precision"]), "b1_gate_cov": f4(b1_gate["coverage"]), "b1_gate_prec": f4(b1_gate["precision"]),
        "b1_gate_applied": b1_gate["applied"], "figure_path": fig, "val_xrepo": f4(need(val, "systems.M1.metrics.cross_repo_macro_f1", v)),
        "near_cut": cut, "near_val_share": pct(nd["summary"]["val"][f"share_above_{cut}"]),
        "near_test_share": pct(nd["summary"]["test"][f"share_above_{cut}"]), "tertile_drops": drops,
        "s17_conf": f4(s17["answer_confidence"]), "s17_tau": s17["threshold"], "s18_conf": f4(s18["answer_confidence"]),
        "s18_tau": s18["threshold"], "test_command": test_cmd,
        "laya_version": F["eval_versions"]["laya"], "laya_tag": laya_commit.group(1),
        "laya_commit": laya_commit.group(2), "nlbse_commit": nlbse_commit, "laya_citation": mmc.LAYA_CITATION,
        "nlbse_bibtex": mmc.NLBSE24_BIBTEX, "eval_python": F["eval_versions"]["python"], **rt,
        "short_results_table": short_results_table(test), "torch_version": torch_pin(constraints),
        "m1_acc_pct": f"{100 * m1['accuracy']:.0f}%", "target_pct": f"{100 * gating.TARGET_PRECISION:.0f}%",
        "b2_native_xrepo": f4(sysm["B2_512-192"]["cross_repo_macro_f1"]),
        "ls_remote": f"git ls-remote {mmc.GITHUB_URL} 'refs/tags/{tag}^{{}}'",
        **worked_example(expected_local, issues, config, phase4_md), **snippets,
    }


def render(template, values):
    """Substitute every `{{name}}`; an unknown placeholder raises."""
    def sub(m):
        name = m.group(1)
        if name not in values:
            raise KeyError(f"docs/README.template.md: unknown placeholder {{{{{name}}}}}")
        return str(values[name])

    return re.sub(r"\{\{\s*(\w+)\s*\}\}", sub, template)


def build_docs(**sources):
    """{output path: text} for every generated document, all from one set of values."""
    values = build_values(**sources)
    return {OUTPUTS[t]: render(text, values) for t, text in sources["templates"].items()}


def build_readme(**sources):
    return build_docs(**sources)[README]


def load_sources():
    s = mmc.load_sources()
    s["action"] = yaml.safe_load(EXTRA_SOURCES["action"].read_text())
    for k in ("posthoc", "expected_local", "issues"):
        s[k] = json.loads(EXTRA_SOURCES[k].read_text())
    for k in ("template", "phase3_md", "phase4_md", "constraints"):
        s[k] = EXTRA_SOURCES[k].read_text()
    s["templates"] = {t: t.read_text() for t in OUTPUTS}
    s["snippets"] = {k: p.read_text().rstrip("\n") for k, p in SNIPPETS.items()}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if any generated document is out of date")
    args = ap.parse_args(argv)
    docs = build_docs(**load_sources())
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, text in docs.items() if not p.exists() or p.read_text() != text]
        if stale:
            sys.exit(f"out of date: {', '.join(stale)}; run docs/make_readme.py")
        print(f"up to date: {', '.join(str(p.relative_to(ROOT)) for p in docs)}")
        return
    for path, text in docs.items():
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
