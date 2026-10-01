# Phase 0 results

Spec: PROJECT_SPEC.md §13 Phase 0. Resolves Q2 (versions) for macOS; Q3, Q4, Q7 still open.

## Task 1: repo skeleton + environment (done 2026-10-02)

### Environment (macOS arm64)

| Item | Value |
|---|---|
| Machine | Apple silicon, arm64, 24 GiB RAM |
| OS | macOS 27.0 (`macOS-27.0-arm64-arm-64bit`) |
| Python | 3.11.17 (Homebrew `python@3.11`), venv at `.venv` |
| torch default threads | 8 (MPS available but not used; spec requires CPU fp32) |

### Package versions (macOS arm64)

These are the versions pip resolved on macOS arm64. They are **not** a lock file. The lock file
will be generated on Linux/CI in Phase 4, where torch wheels differ.

| Package | Version | How it got there |
|---|---|---|
| laya | **0.3.23** | pinned in `pyproject.toml` (`laya==0.3.23`) |
| torch | **2.14.1** | resolved from `torch>=2.0.0` (laya requirement) |
| transformers | **5.18.0** | resolved from `transformers>=4.48.0` (laya requirement) |
| huggingface_hub | 1.33.0 | resolved |
| tokenizers | 0.23.2 | resolved |
| safetensors | 0.8.0 | resolved |
| numpy | 2.4.6 | resolved |
| pytest | 9.1.1 | pinned in `dev` extra |
| ruff | 0.16.10 | pinned in `dev` extra |

Verification:

```
$ .venv/bin/python -I -c "import laya; print(laya.__version__)"
0.3.23
$ .venv/bin/python -m pip check
No broken requirements found.
```

### Package source check: PyPI `laya` 0.3.23 vs github.com/NandhaKishorM/laya

Compared PyPI JSON metadata for 0.3.23 against `pyproject.toml` at git tag `v0.3.23`
(`ae3222b3fcdf424254a2c726d72161f671a86d95`). `pyproject.toml` on `main`
(`4aa6761be8173de4ce6d92c31b3e40b6eaf59a7c` at time of check) is byte-identical to the tag.

| Field | PyPI 0.3.23 | GitHub `v0.3.23` pyproject | Match |
|---|---|---|---|
| name / version | laya / 0.3.23 | laya / 0.3.23 | yes |
| author | Convai Innovations | Convai Innovations | yes |
| license | Apache-2.0 | Apache-2.0 | yes |
| requires-python | >=3.10 | >=3.10 | yes |
| core dependencies | torch>=2.0.0, transformers>=4.48.0, safetensors>=0.4.0, huggingface_hub>=0.20.0, numpy>=1.20.0 | same | yes |
| project URLs | Homepage: huggingface.co/convaiinnovations/laya; Demo: huggingface.co/spaces/convaiinnovations/laya-demo | same | yes |

**Mismatches: none.** Note: neither PyPI nor the upstream `pyproject.toml` links back to the GitHub
repo; both point to Hugging Face. That is consistent, not a mismatch.

PyPI artifact hashes (sha256), uploaded 2026-10-01:

- `laya-0.3.23-py3-none-any.whl` `30247fd93dec16b131d8483b1621db198600e90c777ad7d9992e48fc118db677`
- `laya-0.3.23.tar.gz` `5812dfd7bc27032a0b969b7ee2de655460e4d925ad2d55f97a869cf15ad0deba`

### Install notes

- Downloads from `files.pythonhosted.org` ran at about 41 KiB/s on this network, and pip stalled
  twice mid-download on the torch wheel (127 MB). The torch wheel was fetched with resumable
  `curl`, verified against PyPI's sha256
  (`b6074b130fd26bc50f5d171f07dfed70db22dd0c354ab34767729249f9e0f042`), and installed from that
  local file. All other packages came from pip's index or cache as usual.
- Disk free (home volume): 25 GiB before install, 24 GiB after.

### Skeleton deviations from §11 (agreed with owner)

Not created yet, because placeholders could be mistaken for real content: `action.yml`,
`config/triage.default.yml`, `.github/workflows/ci.yml`, `.github/workflows/triage.yml`,
`training/finetune_kaggle.ipynb`, `results/metrics_val.json`, `results/metrics_test.json`. Empty
folders hold a `.gitkeep`.

## Task 2: model load on CPU

Not started.

## Task 3: zero-shot wording comparison

Not started.

## Task 4: CPU latency

Not started.

## Task 5: frozen wording

Not started.
