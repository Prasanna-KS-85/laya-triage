"""No shell or dynamic code execution anywhere in src/ (PROJECT_SPEC.md §7.6, NFR-6): an AST scan."""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "laya_triage"
FORBIDDEN_MODULES = {"subprocess", "pty", "commands"}
FORBIDDEN_OS = {"system", "popen", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
                "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
                "posix_spawn", "posix_spawnp", "startfile"}
FORBIDDEN_BUILTINS = {"eval", "exec", "__import__"}


def violations(source, name="<src>"):
    out = []
    for node in ast.walk(ast.parse(source, name)):
        if isinstance(node, ast.Import):
            out += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                out.append(f"from {node.module} import ...")
            if node.module == "os":
                out += [f"from os import {a.name}" for a in node.names if a.name in FORBIDDEN_OS]
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os" \
                and node.attr in FORBIDDEN_OS:
            out.append(f"os.{node.attr}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_BUILTINS:
            out.append(node.id)
    return out


def test_scanner_catches_each_form():
    src = ("import subprocess\nfrom subprocess import run\nimport os\nos.system('x')\nfrom os import popen\n"
           "eval('1')\nexec('x=1')\nf = os.execvp\n")
    assert sorted(violations(src)) == sorted(["import subprocess", "from subprocess import ...", "os.system",
                                              "from os import popen", "eval", "exec", "os.execvp"])


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_src_has_no_shell_or_dynamic_execution(path):
    assert violations(path.read_text(encoding="utf-8"), str(path)) == []


def test_token_is_only_read_from_the_environment():
    hits = {p.name: p.read_text().count("GITHUB_TOKEN") for p in SRC.glob("*.py") if "GITHUB_TOKEN" in p.read_text()}
    assert set(hits) <= {"__main__.py", "github_client.py"}
    main = (SRC / "__main__.py").read_text()
    assert 'env.get("GITHUB_TOKEN")' in main and "--token" not in main
