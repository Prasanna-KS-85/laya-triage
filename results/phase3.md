# Phase 3

## Environment (local evaluation, 2026-10-02)

Apple M4 Pro (12 cores), macOS 27.0 arm64, CPU fp32 (`LAYA_CPU_AMP` unset), 8 torch threads (default).

| Package | Version |
|---|---|
| Python | 3.11.17 |
| laya | 0.3.23 |
| torch | 2.14.1 |
| transformers | 5.18.0 |
| tokenizers | 0.23.2 |
| huggingface_hub | 1.33.0 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 (new in the `eval` extra) |
| pillow | 12.3.0 (matplotlib dependency) |
| pytest / ruff | 9.1.1 / 0.16.10 |

## Evaluation tooling check on the base checkpoint (val, not a result)

`eval/run_eval.py --model-repo convaiinnovations/laya --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
--split val --expect-max-len 512 --expect-head-max-len 192 --tag dryrun_base`, written to a temp directory
(not committed), compared with `results/phase2/predictions_val_B2_512-192.csv`:

- ids, repos and gold identical; predicted labels identical on 300/300; max |Δp| = 0.0.
- Uncalibrated (T=1) probabilities: softmax of the raw option logits from `laya.calibrate.records_from_labeled`.
  On every issue, softmax(logits / 1.7602) reproduced laya's returned probabilities within their 4-decimal
  rounding (max deviation 5.0e-5) with the same label. 1.7602 is the base config's `choice:3-5` bucket.
- 39.3% of states truncated at 512/192 (as in Phase 2). Latency per `predict()`, model preloaded, 3 warm-up
  calls: p50 182 ms, p95 234 ms (plus a second forward per issue for the T=1 logits, not timed). Wall time 104 s.
