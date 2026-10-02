# Phase 3 run ledger

One row per fine-tuning run of `training/finetune_kaggle.ipynb` with `SMOKE = False` (PROJECT_SPEC.md §9.5).
`items_sha256` must equal the value for the run's eps in `results/phase3/items_sha256.json`. Val metrics come
from CPU fp32 inference (§10.4), never from the notebook's GPU smoke predict. `HF revision` is the commit oid the
push cell prints.

| run_id | date | base_rev | items_sha256 | epochs | eps | grad_accum | seed | val macro-F1 (CPU) | val ECE | HF revision | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
