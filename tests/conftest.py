"""Shared fixture for the slow model tests: the pinned revision from the local Hugging Face cache, strictly offline.

Skipped when the snapshot is not cached (CI never downloads weights, PROJECT_SPEC.md §16 rule 9). While the fixture
is alive, Hugging Face offline mode is on and socket connections raise, so a test cannot reach the network.
"""
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "triage.default.yml"


def snapshot_dir(repo, revision):
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


@pytest.fixture(scope="session")
def triage_config():
    from laya_triage.config import load_config

    return load_config(DEFAULT_CONFIG)


@pytest.fixture(scope="session")
def offline_classifier(triage_config):
    m = triage_config.model
    if not snapshot_dir(m.repo, m.revision).is_dir():
        pytest.skip(f"{m.repo}@{m.revision[:12]} is not in the local Hugging Face cache")

    def no_network(*args, **kwargs):
        raise OSError("network access is disabled in the slow tests")

    with pytest.MonkeyPatch.context() as mp:
        import huggingface_hub.constants

        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("TRANSFORMERS_OFFLINE", "1")
        mp.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        mp.setattr(socket.socket, "connect", no_network)
        mp.setattr(socket, "create_connection", no_network)
        from laya_triage.classifier import Classifier

        yield Classifier(m.repo, m.revision, m.max_len)
