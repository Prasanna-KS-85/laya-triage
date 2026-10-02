"""laya_triage.classifier with a fake laya agent (no weights, no network): model checks, result mapping, and errors
that carry the cause's type and frames but never its message (PROJECT_SPEC.md §7.5, §7.6, §9.4)."""
import pytest
import torch

from laya_triage.classifier import (
    REQUIRED_MODEL_FILES,
    Classifier,
    ClassifierError,
    TriageResult,
    hf_hub_cache,
    model_cached,
    model_key,
)
from laya_triage.questions import ISSUE_TYPE_QUESTION, LABELS

REPO, REV = "o/r", "a" * 40
SENTINEL = "SENTINEL-HF-ERROR-3b7d"


class FakeAgent:
    def __init__(self, probs=None, choice=None, conf=None, max_len=1024, revision=REV, amp=False, dtype=torch.float32,
                 fail=None):
        self.cfg = {"max_len": max_len, "head_max_len": 256}
        self.revision, self.device, self.amp_enabled, self.dtype = revision, "cpu", amp, dtype
        self.probs = probs or {"bug": 0.7, "feature": 0.2, "question": 0.1}
        self.choice = choice or max(self.probs, key=self.probs.get)
        self.conf = self.probs[self.choice] if conf is None else conf
        self.fail, self.questions = fail, []

    def _out(self):
        return {"answers": {"issue_type": {"choice": self.choice, "probabilities": dict(self.probs),
                                           "answer_confidence": self.conf}}}

    def predict(self, state, questions):
        self.questions.append(questions)
        if self.fail:
            raise self.fail
        return self._out()

    def predict_batch(self, states, questions, batch_size=None):
        if self.fail:
            raise self.fail
        return [self._out() for _ in states]


def make(agent, max_len=1024):
    return Classifier(REPO, REV, max_len, loader=lambda repo, revision, device: agent)


def test_classify_maps_laya_output(monkeypatch):
    monkeypatch.setenv("LAYA_CPU_AMP", "1")
    agent = FakeAgent()
    r = make(agent).classify({"title": "t", "body": ""}, 9)
    assert r == TriageResult(9, "bug", {"bug": 0.7, "feature": 0.2, "question": 0.1}, 0.7, r.latency_ms, REPO, REV)
    assert agent.questions == [ISSUE_TYPE_QUESTION]
    import os
    assert "LAYA_CPU_AMP" not in os.environ  # fp32 on CPU, autocast off


def test_classify_batch_keeps_order_and_numbers():
    out = make(FakeAgent()).classify_batch([{"title": "a", "body": ""}, {"title": "b", "body": ""}], [3, 4])
    assert [r.issue_number for r in out] == [3, 4] and all(r.label == "bug" for r in out)
    assert make(FakeAgent()).classify_batch([], []) == []
    with pytest.raises(ValueError):
        make(FakeAgent()).classify_batch([{}], [1, 2])


@pytest.mark.parametrize("kwargs,message", [
    ({"max_len": 512}, "checkpoint max_len 512 != config max_len 1024"),
    ({"revision": "b" * 40}, "loaded revision differs"),
    ({"amp": True}, "not on CPU with autocast off"),
    ({"dtype": torch.float16}, "not fp32"),
])
def test_model_checks(kwargs, message):
    with pytest.raises(ClassifierError, match=message):
        make(FakeAgent(**kwargs))


def test_non_cpu_device_rejected():
    with pytest.raises(ClassifierError, match="not supported"):
        Classifier(REPO, REV, 1024, device="cuda", loader=lambda *a, **k: FakeAgent())


@pytest.mark.parametrize("exc", [OSError(f"could not download {SENTINEL}"), ValueError(f"tokenizer: {SENTINEL}")])
def test_load_errors_keep_type_and_frames_not_message(exc):
    def loader(repo, revision, device):
        raise exc

    with pytest.raises(ClassifierError) as e:
        Classifier(REPO, REV, 1024, loader=loader)
    err = e.value
    assert str(err) == f"model load failed ({type(exc).__name__})" and err.cause_type == type(exc).__name__
    assert any(f.endswith(":loader") for f in err.frames)
    assert SENTINEL not in str(err) + repr(err) + " ".join(err.frames)
    assert err.__cause__ is None and err.__suppress_context__  # the original message is not chained


def test_inference_error_keeps_type_and_frames_not_message():
    with pytest.raises(ClassifierError) as e:
        make(FakeAgent(fail=RuntimeError(f"bad input {SENTINEL}"))).classify({"title": "t", "body": ""}, 1)
    assert str(e.value) == "inference failed (RuntimeError)" and SENTINEL not in repr(e.value)
    with pytest.raises(ClassifierError) as e:
        make(FakeAgent(fail=RuntimeError(SENTINEL))).classify_batch([{"title": "t", "body": ""}], [1])
    assert SENTINEL not in str(e.value)


@pytest.mark.parametrize("kwargs", [{"choice": "maybe", "conf": 0.7}, {"conf": 0.5}, {"choice": "feature"}])
def test_inconsistent_output_is_an_error(kwargs):
    with pytest.raises(ClassifierError, match="inconsistent"):
        make(FakeAgent(**kwargs)).classify({"title": "t", "body": ""}, 1)


def test_labels_constant():
    assert LABELS == ("bug", "feature", "question")


# ---- model cache helpers (used by --print-model) ----


def test_hf_hub_cache_resolution(tmp_path):
    from pathlib import Path

    assert hf_hub_cache({"HF_HUB_CACHE": str(tmp_path / "c")}) == tmp_path / "c"
    assert hf_hub_cache({"HF_HOME": str(tmp_path / "h")}) == tmp_path / "h" / "hub"
    assert hf_hub_cache({"XDG_CACHE_HOME": str(tmp_path / "x")}) == tmp_path / "x" / "huggingface" / "hub"
    assert hf_hub_cache({}) == Path.home() / ".cache" / "huggingface" / "hub"


def test_hf_hub_cache_agrees_with_huggingface_hub():
    import os

    import huggingface_hub.constants as c

    assert str(hf_hub_cache(os.environ)) == os.path.expanduser(c.HF_HUB_CACHE)


def fake_snapshot(root, files, empty=()):
    snap = root / "models--o--r" / "snapshots" / REV
    blobs = root / "models--o--r" / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        blob = blobs / f"b{i}"
        blob.write_bytes(b"" if f in empty else b"x")
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).symlink_to(blob)
    return snap


def test_model_cached(tmp_path):
    env = {"HF_HUB_CACHE": str(tmp_path)}
    assert model_cached("o/r", REV, env) is False  # nothing cached
    fake_snapshot(tmp_path, REQUIRED_MODEL_FILES)
    assert model_cached("o/r", REV, env) is True
    assert model_cached("o/r", "b" * 40, env) is False  # another revision


@pytest.mark.parametrize("missing", REQUIRED_MODEL_FILES)
def test_model_not_cached_if_a_required_file_is_missing(tmp_path, missing):
    fake_snapshot(tmp_path, [f for f in REQUIRED_MODEL_FILES if f != missing])
    assert model_cached("o/r", REV, {"HF_HUB_CACHE": str(tmp_path)}) is False


def test_model_not_cached_if_a_file_is_empty_or_a_dangling_link(tmp_path):
    snap = fake_snapshot(tmp_path, REQUIRED_MODEL_FILES, empty=("model.safetensors",))
    env = {"HF_HUB_CACHE": str(tmp_path)}
    assert model_cached("o/r", REV, env) is False
    (snap / "model.safetensors").resolve().write_bytes(b"w")
    assert model_cached("o/r", REV, env) is True
    (snap / "encoder" / "config.json").resolve().unlink()  # dangling symlink (interrupted download)
    assert model_cached("o/r", REV, env) is False


def test_model_key():
    assert model_key("Prasanna85/laya-issue-triage", REV) == f"Prasanna85--laya-issue-triage--{REV}"
