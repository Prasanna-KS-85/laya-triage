"""The fine-tuned Laya model behind the Action (PROJECT_SPEC.md §9.4, §12.1-§12.2).

`Classifier` loads the pinned revision once on CPU in fp32 (autocast off), checks that the checkpoint's
`max_len` equals the config's (§9.4), and asks the frozen ISSUE_TYPE_QUESTION. `classify` uses one `predict()`
per issue, the call the thresholds were fitted on; `classify_batch` uses `predict_batch` (verified against
`predict` by tests/test_model_smoke.py and tests/test_parity_val.py).

Errors become ClassifierError with fixed text, the original exception's type name and its traceback frames
(file:line:function). The original message is never kept: model, download and tokenizer errors could echo
issue text (§7.6). Networking is not forced off here, so a cold cache can download the revision (NFR-3).
"""

import os
import time
import traceback
from dataclasses import dataclass
from typing import Literal

from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

Label = Literal["bug", "feature", "question"]
QID = next(iter(ISSUE_TYPE_QUESTION))
BATCH_SIZE = 8


@dataclass(frozen=True)
class TriageResult:
    issue_number: int
    label: Label                      # argmax
    probabilities: dict[str, float]   # keyed by LABELS, sums to ~1
    answer_confidence: float          # max(probabilities); calibrated
    latency_ms: float
    model_repo: str
    model_revision: str


def frames_of(exc):
    """Traceback frames of `exc` as 'file:line:function' strings (no source text, no message)."""
    return tuple(f"{_short_path(f.filename)}:{f.lineno}:{f.name}" for f in traceback.extract_tb(exc.__traceback__))


def _short_path(filename):
    """Path relative to site-packages or the package root when possible, else the base name."""
    for marker in ("site-packages/", "src/"):
        if marker in filename:
            return filename.split(marker, 1)[1]
    return os.path.basename(filename)


class ClassifierError(Exception):
    """Model load or inference failed. Carries the cause's type name and frames, never its message."""

    def __init__(self, what, cause=None):
        self.cause_type = type(cause).__name__ if cause is not None else None
        self.frames = frames_of(cause) if cause is not None else ()
        super().__init__(what + (f" ({self.cause_type})" if self.cause_type else ""))


class Classifier:
    def __init__(self, repo: str, revision: str, max_len: int, device: str = "cpu", *, loader=None):
        if device != "cpu":
            raise ClassifierError(f"device {device!r} is not supported; the Action runs on CPU")
        os.environ.pop("LAYA_CPU_AMP", None)  # CPU autocast off: fp32, as the thresholds were fitted (§9.4)
        try:
            import torch

            if loader is None:
                import laya

                loader = laya.load
            agent = loader(repo, revision=revision, device=device)
        except Exception as e:  # noqa: BLE001 - any load or download failure is a soft failure (§7.5)
            raise ClassifierError("model load failed", e) from None
        problems = []
        if getattr(agent, "revision", None) != revision:
            problems.append("loaded revision differs from the pinned revision")
        if str(getattr(agent, "device", "")) != "cpu" or getattr(agent, "amp_enabled", None) is not False:
            problems.append("model is not on CPU with autocast off")
        if getattr(agent, "dtype", None) != torch.float32:
            problems.append("model is not fp32")
        got = (getattr(agent, "cfg", None) or {}).get("max_len")
        if got != max_len:
            problems.append(f"checkpoint max_len {got} != config max_len {max_len}")
        if problems:
            raise ClassifierError("model check failed: " + "; ".join(problems))
        self.agent, self.repo, self.revision, self.max_len = agent, repo, revision, max_len

    def _result(self, out, issue_number, latency_ms):
        a = out["answers"][QID]
        probs = {c: float(a["probabilities"][c]) for c in LABELS}
        label, conf = a["choice"], float(a["answer_confidence"])
        if label not in LABELS or conf != probs[label] or conf != max(probs.values()):
            raise ClassifierError("model output is inconsistent (choice, probabilities, answer_confidence)")
        return TriageResult(int(issue_number), label, probs, conf, latency_ms, self.repo, self.revision)

    def classify(self, state: dict, issue_number: int) -> TriageResult:
        try:
            t = time.perf_counter()
            out = self.agent.predict(state, ISSUE_TYPE_QUESTION)
            ms = (time.perf_counter() - t) * 1000
            return self._result(out, issue_number, ms)
        except ClassifierError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ClassifierError("inference failed", e) from None

    def classify_batch(self, states, issue_numbers) -> list[TriageResult]:
        states, issue_numbers = list(states), list(issue_numbers)
        if len(states) != len(issue_numbers):
            raise ValueError("states and issue_numbers differ in length")
        if not states:
            return []
        try:
            t = time.perf_counter()
            outs = self.agent.predict_batch(states, ISSUE_TYPE_QUESTION, batch_size=BATCH_SIZE)
            ms = (time.perf_counter() - t) * 1000 / len(states)  # mean per issue
            if len(outs) != len(states):
                raise ClassifierError("predict_batch returned a different number of results")
            return [self._result(o, n, ms) for o, n in zip(outs, issue_numbers)]
        except ClassifierError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ClassifierError("inference failed", e) from None
