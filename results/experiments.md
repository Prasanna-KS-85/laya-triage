# Phase 3 run ledger

One row per fine-tuning run of `training/finetune_kaggle.ipynb` with `SMOKE = False` (PROJECT_SPEC.md §9.5).
`items_sha256` must equal the value for the run's eps in `results/phase3/items_sha256.json`. Val metrics come
from CPU fp32 inference (§10.4), never from the notebook's GPU smoke predict. `HF revision` is the commit oid the
push cell prints.

| run_id | date | base_rev | items_sha256 | epochs | eps | grad_accum | seed | val macro-F1 (CPU) | val ECE | HF revision | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 2026-10-02 | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | `e1ad1d4d2810d90a307af53c0e1be821ccb72684b0fbd2f93f9ebd0e495e88ab` | 4 | 0.0 | 4 | 42 | 0.8691 cross-repo / 0.8699 pooled (n = 300) | 0.1288 (fitted T 4.0188; T=1: 0.1043) | `a704b3eadd185f1fa028576cfa50605b6eada4a4` | Notebook at commit `a2451b3`, `SMOKE = False`. Kaggle 2× T4: cumulative epoch times 170.2 / 347.2 / 524.1 / 701.3 s; mean epoch loss 0.7001 / 0.4066 / 0.2311 / 0.2242; 68 optimizer steps (T_max 68); train_seconds 712.0; fitted choice T 4.0188 on 119 calibration items. GPU smoke predict 26/30 val rows (diagnostic only). Snapshot verified 25/25 (`results/phase3/verify_snapshot.py`). Val details: `results/phase3/val_R1/metrics_val.json`, `results/phase3.md`. Test not run. |
| R1b | 2026-10-02 | `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` | `e1ad1d4d2810d90a307af53c0e1be821ccb72684b0fbd2f93f9ebd0e495e88ab` | 4 | 0.0 | 4 | 42 | 0.8691 cross-repo / 0.8699 pooled (labels identical to R1 on 300/300) | 0.0699 (in-sample: T 2.6968 fitted on val) | TBD (filled after upload) | R1 weights + val-refit temperature. Choice temperature refit on the 300 val issues with `laya.calibrate.fit_temperature_map` (NLL, LBFGS, clamp [0.5, 5.0]); grid check 2.70. Only `rl_agent_config.json` `temperature[0]` changes (4.0188 → 2.6968); file at `results/phase3/refit/rl_agent_config.json`. Test not run. |
