Post-hoc exploratory analysis (not pre-declared). Aggregates only.

C. Sanity (test)

- M1 confusion [[388, 33, 79], [25, 420, 55], [53, 52, 395]]; B1 confusion [[380, 55, 65], [25, 420, 55], [64, 85, 351]] (metrics.confusion_counts == independent count == metrics_test.json).
- Gold feature (500): M1 and B1 predict the same label on 429, both `feature` on 386.
- M1 at val tau_lb: applied 391 / 1500, correct {'all': 342, 'bug': 342, 'feature': 0, 'question': 0}
- M1 at val tau_point: applied 1312 / 1500, correct {'all': 1114, 'bug': 374, 'feature': 391, 'question': 349}

A. Max TF-IDF cosine similarity to any train issue

| Split | n | mean | median | p90 | share > 0.8 | share > 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| val | 300 | 0.3739 | 0.2839 | 0.7574 | 0.0833 | 0.0500 |
| test | 1500 | 0.3840 | 0.3206 | 0.7799 | 0.0953 | 0.0553 |

Tertile cut points (val distribution): 0.2015, 0.4309

| Split | System | low: n / acc | mid: n / acc | high: n / acc | overall acc |
|---|---|---|---|---|---:|
| val | M1 | 100 / 0.8500 | 100 / 0.8500 | 100 / 0.9100 | 0.8700 |
| val | B1 | 100 / 0.6600 | 100 / 0.7900 | 100 / 0.8300 | 0.7600 |
| test | M1 | 449 / 0.7639 | 534 / 0.7697 | 517 / 0.8685 | 0.8020 |
| test | B1 | 449 / 0.7171 | 534 / 0.7491 | 517 / 0.8298 | 0.7673 |

B. One global cutoff (rule pre-stated; fitted on val, evaluated once on test)

| System | val τ | val \|S\| | val coverage | val precision | val bound | test applied | test coverage | test precision | per label applied / precision (bug, feature, question) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M1 | 0.7625 | 215 | 0.7167 | 0.9302 | 0.9023 | 1089 | 0.7260 | 0.8797 | 306 / 0.9183; 440 / 0.8818; 343 / 0.8426 |
| B1 | 0.6905 | 155 | 0.5167 | 0.9355 | 0.9032 | 829 | 0.5527 | 0.8999 | 275 / 0.9200; 320 / 0.8750; 234 / 0.9103 |

Descriptive only (reads the test curve; optimistic):

- M1: largest test coverage with precision ≥ 0.90: 0.6433 (precision 0.9005, τ 0.8342)
- M1: largest test coverage with precision ≥ B1's pre-declared test precision 0.9156: 0.5847 (precision 0.9156, τ 0.8730)
- B1: largest test coverage with precision ≥ 0.90: 0.5500 (precision 0.9006, τ 0.6927)
