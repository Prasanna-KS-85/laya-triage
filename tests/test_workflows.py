"""Static checks of action.yml and the workflows (PROJECT_SPEC.md §7.6, R8; NFR-6). No network.

Rules (each one is shown to fire on a synthetic bad file below):
(a) no context that can carry issue, comment or pull-request content anywhere in the file (github.event.issue,
    .comment, .pull_request, .review, .discussion, .head_commit, .commits, .release, .pages, github.head_ref,
    and any github.event.*.title / .body);
(b) no `${{ ... }}` inside a `run:` script: values reach scripts only through `env:`;
(c) every expression elsewhere (env, with, if, key, defaults, ...) references only allow-listed contexts
    (github.event_name is allowed: a fixed event-type string, not attacker-controlled);
(d) every `uses:` is pinned to a 40-hex commit SHA with a `# v...` tag comment (local `./` actions excepted); the
    only exemption is the documented placeholder SHA for Prasanna-KS-85/laya-triage in sandbox/workflow.example.yml;
(e) every `run:` step of the composite action declares `shell: bash`;
(f) every workflow declares top-level permissions limited to contents: read and issues: write.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "action.yml", *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
         ROOT / "sandbox" / "workflow.example.yml"]
EXAMPLE = "sandbox/workflow.example.yml"
SELF = "Prasanna-KS-85/laya-triage"
PLACEHOLDER_SHA = "0" * 40

FORBIDDEN_CONTEXT = re.compile(
    r"github\.event\.(issue|comment|pull_request|review|review_comment|discussion|head_commit|commits|release|pages)"
    r"\b|github\.head_ref\b|github\.event\.[\w.\-]*\.(title|body)\b")
EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
ALLOWED = [re.compile(p) for p in (
    r"inputs\.[\w-]+", r"github\.event\.inputs\.[\w-]+", r"steps\.[\w-]+\.outputs\.[\w-]+", r"steps\.[\w-]+\.outcome",
    r"runner\.(os|arch|temp)", r"github\.(run_id|run_attempt|token|event_name)", r"secrets\.GITHUB_TOKEN")]
FUNCTIONS = {"always", "success", "failure", "cancelled", "contains", "startsWith", "endsWith", "format"}
LITERALS = {"true", "false", "null"}
USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)\s*(#.*)?$")
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")
PERMISSIONS = {"contents": {"read"}, "issues": {"read", "write"}}


def load(text):
    data = yaml.safe_load(text)
    if isinstance(data, dict) and True in data and "on" not in data:  # YAML 1.1 reads the key `on` as True
        data["on"] = data.pop(True)
    return data


def walk(node, path=()):
    """(path, key, string value) for every string in the document."""
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str):
                yield path, k, v
            else:
                yield from walk(v, (*path, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str):
                yield path, i, v
            else:
                yield from walk(v, (*path, i))


def references(expression):
    """Context references in an expression: dotted names that are not functions or literals (strings removed)."""
    expression = re.sub(r"'(?:[^']|'')*'", "''", expression)
    refs = []
    for m in re.finditer(r"[A-Za-z_][\w-]*(?:\.[\w*-]+)*(\s*\()?", expression):
        name = m.group(0).rstrip("( ").strip()
        if m.group(1) or name in LITERALS:
            if m.group(1) and name not in FUNCTIONS:
                refs.append(name + "()")
            continue
        refs.append(name)
    return refs


def check(text, name):
    """Violations of rules (a)-(f) in one YAML file; `name` is its repo-relative path."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if FORBIDDEN_CONTEXT.search(line):
            out.append(f"(a) {name}:{i}: issue / PR content context")
        m = USES.match(line)
        if m and not m.group(1).startswith("./"):
            ref, comment = m.group(1), m.group(2) or ""
            exempt = name == EXAMPLE and ref == f"{SELF}@{PLACEHOLDER_SHA}" and comment.startswith("#")
            placeholder = ref.endswith("@" + PLACEHOLDER_SHA)  # never a real pin outside the one exemption
            if not exempt and (placeholder or not PINNED.match(ref) or not re.match(r"#\s*v\d", comment)):
                out.append(f"(d) {name}:{i}: uses not pinned to a commit SHA with a # vX comment: {ref}")
    data = load(text)
    for path, key, value in walk(data):
        where = ".".join(map(str, (*path, key)))
        if key == "run":
            if "${{" in value:
                out.append(f"(b) {name}: expression inside run: at {where}")
            continue
        exprs = [value] if key == "if" else EXPR.findall(value)
        for e in exprs:
            for ref in references(e):
                if not any(p.fullmatch(ref) for p in ALLOWED):
                    out.append(f"(c) {name}: {ref} not allow-listed at {where}")
    if isinstance(data, dict) and (data.get("runs") or {}).get("using") == "composite":
        for i, step in enumerate(data["runs"].get("steps") or []):
            if "run" in step and step.get("shell") != "bash":
                out.append(f"(e) {name}: composite step {i} has no shell: bash")
    if isinstance(data, dict) and "jobs" in data:
        perms = data.get("permissions")
        if not isinstance(perms, dict) or any(k not in PERMISSIONS or v not in PERMISSIONS[k] for k, v in perms.items()):
            out.append(f"(f) {name}: top-level permissions missing or broader than contents: read / issues: write")
    return out


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_real_files_pass(path):
    assert path.is_file(), path
    assert check(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))) == []


def test_action_steps_use_only_pinned_actions():
    data = load((ROOT / "action.yml").read_text())
    uses = {s["uses"].split("@")[0] for s in data["runs"]["steps"] if "uses" in s}
    assert uses == {"actions/setup-python", "actions/cache/restore", "actions/cache/save", "actions/upload-artifact"}


GOOD_WF = """name: t
on: [push]
permissions:
  contents: read
jobs:
  j:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - run: echo ok
        env:
          X: ${{ inputs.mode }}
"""

BAD = {
    "(a)": GOOD_WF.replace("X: ${{ inputs.mode }}", "X: ${{ github.event.issue.title }}"),
    "(a) comment": GOOD_WF.replace("X: ${{ inputs.mode }}", "X: ${{ github.event.comment.id }}"),
    "(a) head_ref": GOOD_WF.replace("X: ${{ inputs.mode }}", "X: ${{ github.head_ref }}"),
    "(a) any title": GOOD_WF.replace("X: ${{ inputs.mode }}", "X: ${{ github.event.foo.body }}"),
    "(b)": GOOD_WF.replace("- run: echo ok", "- run: echo ${{ inputs.mode }}"),
    "(b) block": GOOD_WF.replace("- run: echo ok", "- run: |\n          echo hi\n          echo ${{ runner.os }}"),
    "(c) env": GOOD_WF.replace("X: ${{ inputs.mode }}", "X: ${{ github.actor }}"),
    "(c) if": GOOD_WF.replace("- run: echo ok", "- if: github.actor == 'octocat'\n        run: echo ok"),
    "(c) with": GOOD_WF.replace("# v7.0.1", "# v7.0.1\n        with:\n          ref: ${{ github.sha }}"),
    "(d) tag": GOOD_WF.replace("@3d3c42e5aac5ba805825da76410c181273ba90b1", "@v7"),
    "(d) no comment": GOOD_WF.replace(" # v7.0.1", ""),
    "(d) placeholder outside example": GOOD_WF.replace("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
                                                       f"{SELF}@{PLACEHOLDER_SHA}"),
    "(f) missing": GOOD_WF.replace("permissions:\n  contents: read\n", ""),
    "(f) broad": GOOD_WF.replace("contents: read", "contents: write"),
    "(f) other scope": GOOD_WF.replace("contents: read", "contents: read\n  actions: write"),
}


def test_good_synthetic_workflow_passes():
    assert check(GOOD_WF, "x.yml") == []


@pytest.mark.parametrize("case", sorted(BAD))
def test_each_rule_fires_on_a_bad_workflow(case):
    rule = case.split()[0]
    found = check(BAD[case], "x.yml")
    assert any(v.startswith(rule) for v in found), found  # (a) cases also trip (c): defence in depth


def test_composite_run_without_shell_fires():
    text = "name: a\nruns:\n  using: composite\n  steps:\n    - run: echo hi\n"
    assert check(text, "action.yml") == ["(e) action.yml: composite step 0 has no shell: bash"]


def test_placeholder_is_exempt_only_for_self_in_the_example():
    line = f"      - uses: {SELF}@{PLACEHOLDER_SHA} # replace with the commit SHA under test"
    text = GOOD_WF.replace("      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1", line)
    assert check(text, EXAMPLE) == []
    assert check(text, ".github/workflows/x.yml")[0].startswith("(d)")
    other = text.replace(SELF, "someone/else")
    assert check(other, EXAMPLE)[0].startswith("(d)")


def test_references_parser():
    assert references("always() && steps.x.outputs.ok == 'true'") == ["steps.x.outputs.ok"]
    assert references(" github.event.inputs.backfill-count || '0' ") == ["github.event.inputs.backfill-count"]
    assert references("toJSON(github)") == ["toJSON()", "github"]


WARM = "${{ github.event_name == 'schedule' || github.event.inputs.warm-cache == 'true' }}"


def test_event_name_is_allow_listed():
    assert references(WARM[3:-2]) == ["github.event_name", "github.event.inputs.warm-cache"]
    text = GOOD_WF.replace("- run: echo ok", "- if: github.event_name == 'schedule'\n        run: echo ok")
    assert check(text.replace("X: ${{ inputs.mode }}", f"X: {WARM}"), "x.yml") == []
    assert check(GOOD_WF.replace("- run: echo ok", "- run: echo ${{ github.event_name }}"), "x.yml")[0].startswith("(b)")


def spec_example():
    """The YAML block of PROJECT_SPEC.md §12.5 (the consumer workflow)."""
    spec = (ROOT / "PROJECT_SPEC.md").read_text(encoding="utf-8")
    section = spec.split("### 12.5 Example consumer workflow", 1)[1].split("\n### ", 1)[0]
    return re.search(r"```yaml\n(.*?)```", section, re.DOTALL).group(1)


def test_spec_consumer_example_passes_rules_other_than_pinning():
    # (d) is excluded: the spec shows `@<40-hex commit SHA>` as a placeholder for the release pin.
    assert [v for v in check(spec_example(), "PROJECT_SPEC.md §12.5") if not v.startswith("(d)")] == []


@pytest.mark.parametrize("name", ["spec", "sandbox"])
def test_examples_warm_the_cache_on_schedule_and_dispatch(name):
    text = spec_example() if name == "spec" else (ROOT / EXAMPLE).read_text(encoding="utf-8")
    data = load(text)
    assert data["on"]["schedule"] and data["on"]["workflow_dispatch"]["inputs"]["warm-cache"]["type"] == "boolean"
    (step,) = [s for s in data["jobs"]["triage"]["steps"] if s.get("uses", "").startswith(SELF + "@")]
    assert step["with"]["warm-cache"] == WARM


def test_action_warm_cache_input_reaches_the_script_only_through_env():
    data = load((ROOT / "action.yml").read_text())
    assert data["inputs"]["warm-cache"]["default"] == "false"
    (triage,) = [s for s in data["runs"]["steps"] if s.get("name") == "Triage"]
    assert triage["env"]["INPUT_WARM_CACHE"] == "${{ inputs.warm-cache }}"
    assert 'true) warm=(--warm)' in triage["run"] and '"${warm[@]}"' in triage["run"]
