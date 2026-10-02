"""Job summary (markdown) and JSONL log, one line per issue (PROJECT_SPEC.md FR-4, FR-7, §7.5, §7.6).

Both contain only fixed text, label names, numbers, issue numbers and the model revision: never an issue title
or body, and never an exception message (errors are recorded as the exception type and traceback frames).

JSONL schema (one object per issue, schema 1):
  schema, issue_number, event, mode ("dry-run" | "apply"), action ("apply" | "escalate" | "skip"), reason,
  prediction (label | null), probabilities ({label: p} | null), answer_confidence, threshold, labels_to_add,
  comment (bool: the decision has a comment), written (bool: GitHub write calls were made), labels_added,
  comment_posted, warnings, latency_ms, model_repo, model_revision, error ({"type", "frames"} | null)
"""

import json
from dataclasses import dataclass
from pathlib import Path

from laya_triage.classifier import ClassifierError, frames_of

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ErrorInfo:
    type: str
    frames: tuple[str, ...]


def error_info(exc) -> ErrorInfo:
    """Type name and frames of an exception (for ClassifierError: of its cause); never the message."""
    if isinstance(exc, ClassifierError) and exc.cause_type:
        return ErrorInfo(f"ClassifierError <- {exc.cause_type}", exc.frames)
    return ErrorInfo(type(exc).__name__, frames_of(exc))


@dataclass(frozen=True)
class Outcome:
    written: bool = False
    labels_added: tuple[str, ...] = ()
    comment_posted: bool = False
    warnings: tuple[str, ...] = ()
    error: ErrorInfo | None = None


def _cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


class Reporter:
    def __init__(self, summary_path, log_path):
        self.summary_path = Path(summary_path) if summary_path else None
        self.log_path = Path(log_path) if log_path else None

    def records(self, results, decisions, *, issue_numbers, outcomes, mode, event, model_repo, model_revision,
                thresholds):
        out = []
        for n, r, d, o in zip(issue_numbers, results, decisions, outcomes, strict=True):
            out.append({
                "schema": SCHEMA_VERSION, "issue_number": int(n), "event": event, "mode": mode,
                "action": d.action, "reason": d.reason,
                "prediction": r.label if r else None,
                "probabilities": dict(r.probabilities) if r else None,
                "answer_confidence": r.answer_confidence if r else None,
                "threshold": thresholds[r.label] if r else None,
                "labels_to_add": list(d.labels_to_add), "comment": d.comment is not None,
                "written": o.written, "labels_added": list(o.labels_added), "comment_posted": o.comment_posted,
                "warnings": list(o.warnings),
                "latency_ms": round(r.latency_ms, 1) if r else None,
                "model_repo": model_repo, "model_revision": model_revision,
                "error": {"type": o.error.type, "frames": list(o.error.frames)} if o.error else None,
            })
        return out

    def summary(self, records, *, mode, model_repo, model_revision, thresholds, notices=()):
        head = "dry-run: no changes were made" if mode == "dry-run" else "apply"
        th = ", ".join(f"{k} {v:.4f}" if v <= 1.0 else f"{k} {v:.2f} (never)" for k, v in thresholds.items())
        lines = [f"## Laya Triage ({head})", "",
                 f"Model `{model_repo}` @ `{model_revision[:12]}`; thresholds: {th}.", ""]
        lines += [f"> {_cell(n)}" for n in notices] + ([""] if notices else [])
        if records:
            lines += ["| Issue | Prediction | Confidence | Threshold | Decision | Labels | Reason |",
                      "|---|---|---:|---:|---|---|---|"]
            for r in records:
                conf = "—" if r["answer_confidence"] is None else f"{r['answer_confidence']:.4f}"
                tau = "—" if r["threshold"] is None else f"{r['threshold']:.4f}"
                labels = ", ".join(r["labels_added"] if mode == "apply" else r["labels_to_add"]) or "—"
                lines.append(f"| #{r['issue_number']} | {r['prediction'] or '—'} | {conf} | {tau} | {r['action']} | "
                             f"{_cell(labels)} | {_cell(r['reason'])} |")
        else:
            lines.append("No issues to triage.")
        warns = [(r["issue_number"], w) for r in records for w in r["warnings"]]
        if warns:
            lines += ["", "### Warnings", ""] + [f"- #{n}: {_cell(w)}" for n, w in warns]
        errs = [r for r in records if r["error"]]
        if errs:
            lines += ["", "### Errors", ""]
            for r in errs:
                frames = "; ".join(r["error"]["frames"][-6:]) or "no frames"
                lines.append(f"- #{r['issue_number']}: `{r['error']['type']}` at {_cell(frames)}")
        return "\n".join(lines) + "\n"

    def notice(self, text):
        """Append a run-level notice (fixed text: config problems, permission errors) to the summary, if set."""
        if self.summary_path is not None:
            with open(self.summary_path, "a", encoding="utf-8") as f:
                f.write("## Laya Triage\n\n" + "".join(f"> {line}\n" for line in str(text).splitlines()) + "\n")

    def write(self, results, decisions, *, issue_numbers, outcomes, mode, event, model_repo, model_revision,
              thresholds, notices=()):
        """Append the summary to $GITHUB_STEP_SUMMARY (if set) and write the JSONL log; returns the records."""
        recs = self.records(results, decisions, issue_numbers=issue_numbers, outcomes=outcomes, mode=mode,
                            event=event, model_repo=model_repo, model_revision=model_revision, thresholds=thresholds)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs), encoding="utf-8")
        if self.summary_path is not None:
            with open(self.summary_path, "a", encoding="utf-8") as f:
                f.write(self.summary(recs, mode=mode, model_repo=model_repo, model_revision=model_revision,
                                     thresholds=thresholds, notices=notices))
        return recs
