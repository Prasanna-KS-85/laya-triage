Test, n = 1500 (M1 = R1b `76ece1fb0eb8b32bd5d8c509293c1692a2534805`; baselines from committed Phase 2 predictions; B3 copied from results/phase2.md)

§10.5 results table (test)

| System | Cross-repo macro-F1 | Pooled macro-F1 | Acc | F1 bug | F1 feature | F1 question | ECE | Brier | Coverage @ config τ (precision of auto-applied) | CPU p50 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 majority | 0.1667 | 0.1667 | 0.3333 | 0.0000 | 0.0000 | 0.5000 | — | — | — | — |
| B1 TF-IDF+LR | 0.7655 | 0.7666 | 0.7673 | 0.7843 | 0.7925 | 0.7230 | 0.0493 | 0.3273 | 0.4187 (prec 0.9156) [s7] | 0.29 [a] |
| B2 zero-shot 512/192 (native) | 0.6108 | 0.6160 | 0.6453 | 0.6693 | 0.7757 | 0.4029 | 0.0684 | 0.4967 | — | — [b] |
| B2 zero-shot 1024/256 | 0.6257 | 0.6310 | 0.6567 | 0.6773 | 0.7827 | 0.4330 | 0.0647 | 0.4877 | — | — [b] |
| B3 SetFit baseline (NLBSE'24 README = `output/results.json`) | 0.8270 | 0.8263 [c] | 0.8267 [c] | 0.8425 [c] | 0.8555 [c] | 0.7809 [c] | — | — | — | — |
| B3 RoBERTa (`output/roberta/results.json`) | 0.7923 | 0.7926 [c] | 0.7927 [c] | 0.8052 [c] | 0.8064 [c] | 0.7663 [c] | — | — | — | — |
| B3 fastText (`output/fasttext/results.json`) | 0.7184 | 0.7193 [c] | 0.7193 [c] | 0.7362 [c] | 0.7390 [c] | 0.6827 [c] | — | — | — | — |
| **M1 R1b (ours)** | 0.8020 | 0.8023 | 0.8020 | 0.8033 | 0.8358 | 0.7677 | 0.0454 | 0.2890 | 0.2607 (prec 0.8747) | 178 |

Footnote row (provenance unknown; not averaged or chosen): SetFit, `output/setfit/results.json`: cross-repo 0.8240, pooled 0.8238 [c], acc 0.8240 [c], F1 bug/feature/question 0.8328 / 0.8583 / 0.7802 [c].

- [a] B1 latency from results/phase2/latency_B1.json (first 100 val issues, local CPU).
- [b] B2 latency not measured in Phase 2 or 3 (results/phase2.md cites Phase 0 under different conditions).
- [c] Derived, not published (results/phase2.md [c]): pooled per-class values recovered from the published per-repo P/R/F1; ECE, Brier and coverage unavailable for B3.
- [s7] Secondary view 7: B1 under the same rule with its own val τ_lb.
- B0's ECE/Brier are computed from train priors and carry no information (see metrics_test.json).

M1 per-class (test)

| Label | Precision | Recall | F1 | Support | Predicted |
|---|---|---|---|---|---|
| bug | 0.8326 | 0.7760 | 0.8033 | 500 | 466 |
| feature | 0.8317 | 0.8400 | 0.8358 | 500 | 505 |
| question | 0.7467 | 0.7900 | 0.7677 | 500 | 529 |

Per-repo macro-F1 (test)

| System | bitcoin | react | vscode | opencv | tensorflow | Cross-repo |
|---|---|---|---|---|---|---|
| M1 R1b (ours) | 0.7645 | 0.8445 | 0.7581 | 0.8002 | 0.8425 | 0.8020 |
| B1 TF-IDF+LR | 0.7489 | 0.7903 | 0.7490 | 0.7383 | 0.8012 | 0.7655 |
| B2 zero-shot 512/192 (native) | 0.6213 | 0.7081 | 0.5053 | 0.5358 | 0.6833 | 0.6108 |
| B2 zero-shot 1024/256 | 0.6298 | 0.7358 | 0.5106 | 0.5622 | 0.6901 | 0.6257 |
| B0 majority | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 | 0.1667 |

Confusion counts (rows gold, cols predicted; bug/feature/question)

| System | Confusion |
|---|---|
| M1 R1b (ours) | `[[388, 33, 79], [25, 420, 55], [53, 52, 395]]` |
| B1 TF-IDF+LR | `[[380, 55, 65], [25, 420, 55], [64, 85, 351]]` |

M1 calibration (test, 15 bins)

| Probabilities | ECE | Brier |
|---|---|---|
| calibrated (T = 2.6968) | 0.0454 | 0.2890 |
| uncalibrated (T = 1) | 0.1465 | 0.3301 |

Paired comparisons (test; bootstrap 2,000 resamples of issues, seed 42; exact McNemar)

| Comparison | Δ cross-repo macro-F1 | 95% CI | M1 better in | M1 right / other wrong | M1 wrong / other right | McNemar p |
|---|---|---|---|---|---|---|
| M1 vs B1 TF-IDF+LR | +0.0364 | [+0.0151, +0.0588] | 100.0% | 170 | 118 | 0.00259 |
| M1 vs B2 zero-shot 512/192 (native) | +0.1912 | [+0.1661, +0.2167] | 100.0% | 339 | 104 | 3.98e-30 |
| M1 vs B2 zero-shot 1024/256 | +0.1763 | [+0.1505, +0.2016] | 100.0% | 325 | 107 | 1.25e-26 |

Gating at the config thresholds (primary item 2)

| Label | τ | Predicted | Applied | Coverage | Precision |
|---|---|---|---|---|---|
| bug | 0.6033 | 466 | 391 | 0.2607 | 0.8747 |
| feature | 1.01 (never) | 505 | 0 | 0.0000 | — |
| question | 1.01 (never) | 529 | 0 | 0.0000 | — |
| **overall** |  | 1500 | 391 applied / 1109 escalated | **0.2607** | **0.8747** |

Secondary 5: M1 at val τ_point

| Label | τ | Predicted | Applied | Coverage | Precision |
|---|---|---|---|---|---|
| bug | 0.5027 | 466 | 438 | 0.2920 | 0.8539 |
| feature | 0.7320 | 505 | 444 | 0.2960 | 0.8806 |
| question | 0.6088 | 529 | 430 | 0.2867 | 0.8116 |
| **overall** |  | 1500 | 1312 applied / 188 escalated | **0.8747** | **0.8491** |

Secondary 7: B1 at its own val τ_lb

| Label | τ | Predicted | Applied | Coverage | Precision |
|---|---|---|---|---|---|
| bug | 0.7698 | 469 | 210 | 0.1400 | 0.9476 |
| feature | 0.6886 | 560 | 323 | 0.2153 | 0.8762 |
| question | 0.8692 | 471 | 95 | 0.0633 | 0.9789 |
| **overall** |  | 1500 | 628 applied / 872 escalated | **0.4187** | **0.9156** |

Secondary 6: M1 coverage–precision curve, one global τ (1137 points in metrics_test.json; rows = first point reaching each coverage)

| Coverage ≥ | τ | Coverage | Precision |
|---|---|---|---|
| 0.05 | 0.9953 | 0.0520 | 0.9744 |
| 0.10 | 0.9926 | 0.1000 | 0.9667 |
| 0.20 | 0.9802 | 0.2000 | 0.9500 |
| 0.30 | 0.9575 | 0.3000 | 0.9489 |
| 0.40 | 0.9398 | 0.4000 | 0.9467 |
| 0.50 | 0.9170 | 0.5000 | 0.9333 |
| 0.60 | 0.8661 | 0.6007 | 0.9134 |
| 0.70 | 0.7856 | 0.7000 | 0.8905 |
| 0.80 | 0.6708 | 0.8000 | 0.8658 |
| 0.90 | 0.5535 | 0.9000 | 0.8422 |
| 1.00 | 0.3546 | 1.0000 | 0.8020 |

Config operating point: coverage 0.2607, precision 0.8747.

Secondary 8: latency (local Apple M4 Pro CPU, fp32, model preloaded; not a GitHub runner)

| Measurement | Threads | n | p50 ms | p95 ms | mean ms | max ms |
|---|---|---|---|---|---|---|
| Test run, M1 R1b (meta) | 8 | 1500 | 178 | 436 | 215 | 509 |
| Stop A, R1, first 100 val issues (default) | 8 | 100 | 223 | 465 | 248 | 519 |
| Stop A, R1, first 100 val issues (2) | 2 | 100 | 288 | 690 | 327 | 756 |

Test-run load 2.6 s, wall 648.5 s. Cold load (Stop A, fresh process, default threads): import laya 0.47 s, laya.load 3.14 s, peak RSS after load 2863 MiB.

Figures: results/figures/reliability_M1_R1b_test.png, results/figures/coverage_precision_M1_R1b_test.png, results/figures/confusion_M1_R1b_vs_B1_test.png, results/figures/per_repo_f1_M1_R1b_vs_B1_test.png

§6.3 criteria 1-4 (test)

| # | Criterion | Numbers | Status |
|---|---|---|---|
| 1 | M1 cross-repo macro-F1 > B1 and > B2 (both settings), paired CIs reported | M1 0.8020; vs B1 TF-IDF+LR +0.0364 [+0.0151, +0.0588]; vs B2 zero-shot 512/192 (native) +0.1912 [+0.1661, +0.2167]; vs B2 zero-shot 1024/256 +0.1763 [+0.1505, +0.2016] | MET |
| 2 | calibrated ECE on test <= 0.1 | ECE 0.0454 | MET |
| 3 | val thresholds with >= 90% precision; test coverage and precision per label | bug: τ 0.6033, applied 391, coverage 0.2607, precision 0.8747 -> not met; feature: τ 1.01 (no val threshold), applied 0, coverage 0.0000, precision — -> not met; question: τ 1.01 (no val threshold), applied 0, coverage 0.0000, precision — -> not met | met for 0 of 3 |
| 4 | NFR-1: p95 <= 1000 ms per issue on a GitHub-hosted Linux runner, or documented with the v1.2 ONNX plan | local p95 436 ms (≤ 1000 ms locally) | PENDING (runner not measured; Phase 4) |
