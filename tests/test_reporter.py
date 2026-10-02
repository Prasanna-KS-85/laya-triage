"""laya_triage.reporter: JSONL schema and job summary hold no issue text and no exception messages
(PROJECT_SPEC.md FR-4, FR-7, §7.5, §7.6)."""
import json

from laya_triage.classifier import ClassifierError, TriageResult
from laya_triage.policy import Decision
from laya_triage.reporter import Outcome, Reporter, error_info

SENTINEL = "SENTINEL-MSG-91c2"
REV = "76ece1fb0eb8b32bd5d8c509293c1692a2534805"
TH = {"bug": 0.6033, "feature": 1.01, "question": 1.01}
KEYS = {"schema", "issue_number", "event", "mode", "action", "reason", "prediction", "probabilities",
        "answer_confidence", "threshold", "labels_to_add", "comment", "written", "labels_added", "comment_posted",
        "warnings", "latency_ms", "model_repo", "model_revision", "error"}


def boom():
    raise RuntimeError(f"tokenizer choked on {SENTINEL}")


def caught(fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return e


def write(tmp_path, mode="apply"):
    r = TriageResult(5, "bug", {"bug": 0.7, "feature": 0.2, "question": 0.1}, 0.7, 120.04, "o/r", REV)
    err = caught(boom)
    rep = Reporter(tmp_path / "summary.md", tmp_path / "logs" / "log.jsonl")
    recs = rep.write(
        [r, None, None],
        [Decision("apply", ("type: bug",), None, "bug: confidence 0.7000 >= threshold 0.6033"),
         Decision("skip", (), None, "already labeled"),
         Decision("skip", (), None, "inference failed; no label")],
        issue_numbers=[5, 6, 7],
        outcomes=[Outcome(written=True, labels_added=("type: bug",)), Outcome(), Outcome(error=error_info(err))],
        mode=mode, event="issues", model_repo="o/r", model_revision=REV, thresholds=TH, notices=["a | b"])
    return rep, recs


def test_jsonl_schema(tmp_path):
    _, recs = write(tmp_path)
    lines = [json.loads(x) for x in (tmp_path / "logs" / "log.jsonl").read_text().splitlines()]
    assert lines == recs and len(lines) == 3 and all(set(x) == KEYS for x in lines)
    a, s, e = lines
    assert (a["action"], a["prediction"], a["threshold"], a["latency_ms"], a["written"]) == ("apply", "bug", 0.6033,
                                                                                               120.0, True)
    assert a["probabilities"] == {"bug": 0.7, "feature": 0.2, "question": 0.1} and a["model_revision"] == REV
    assert (s["action"], s["prediction"], s["reason"], s["error"]) == ("skip", None, "already labeled", None)
    assert e["error"]["type"] == "RuntimeError" and all(f.count(":") >= 2 for f in e["error"]["frames"])
    assert any(f.endswith(":boom") for f in e["error"]["frames"])


def test_summary_layout(tmp_path):
    write(tmp_path, mode="dry-run")
    md = (tmp_path / "summary.md").read_text()
    assert md.startswith("## Laya Triage (dry-run: no changes were made)\n")
    assert f"`o/r` @ `{REV[:12]}`" in md and "bug 0.6033, feature 1.01 (never), question 1.01 (never)" in md
    assert "| #5 | bug | 0.7000 | 0.6033 | apply | type: bug | bug: confidence 0.7000 >= threshold 0.6033 |" in md
    assert "| #6 | — | — | — | skip | — | already labeled |" in md
    assert "> a \\| b" in md  # pipes escaped
    assert "### Errors" in md and "#7: `RuntimeError` at " in md


def test_exception_message_never_reaches_outputs(tmp_path, capsys):
    write(tmp_path)
    for f in (tmp_path / "summary.md", tmp_path / "logs" / "log.jsonl"):
        assert SENTINEL not in f.read_text()
    out = capsys.readouterr()
    assert SENTINEL not in out.out + out.err


def test_classifier_error_reports_its_cause_type_and_frames_only():
    e = caught(lambda: _wrap())
    info = error_info(e)
    assert info.type == "ClassifierError <- RuntimeError"
    assert any(f.endswith(":boom") for f in info.frames)
    assert SENTINEL not in str(e) and SENTINEL not in json.dumps(info.__dict__)


def _wrap():
    try:
        boom()
    except Exception as e:  # noqa: BLE001
        raise ClassifierError("inference failed", e) from None


def test_summary_is_appended_and_notice(tmp_path):
    p = tmp_path / "summary.md"
    p.write_text("previous step\n")
    rep = Reporter(p, None)
    rep.notice("invalid laya-triage config:\n- behaviour.mode: must be one of ['dry-run', 'apply']")
    assert p.read_text() == ("previous step\n## Laya Triage\n\n> invalid laya-triage config:\n"
                             "> - behaviour.mode: must be one of ['dry-run', 'apply']\n\n")


def test_no_paths_means_no_files(tmp_path):
    rep = Reporter(None, None)
    assert rep.write([], [], issue_numbers=[], outcomes=[], mode="dry-run", event="issues", model_repo="o/r",
                     model_revision=REV, thresholds=TH) == []
    rep.notice("x")
    assert list(tmp_path.iterdir()) == []
