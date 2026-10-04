# Requirements and success criteria

The requirements below are copied by the generator from §6 of the specification; the sandbox column and the results come from `results/phase4.md`, `results/phase3/test_R1b/metrics_test.json` and `results/phase5.md`.

## Functional requirements

{{caption:Functional requirements (PROJECT_SPEC.md §6.1) and how each was shown on the sandbox repository (results/phase4.md)}}

{{fr_table}}

Two parts of FR-5 (bot authors and event actions other than `opened`) and comment template (a) of FR-3 could not be produced on the sandbox with the fixtures and configuration used; they are covered by unit tests only, as recorded in `results/phase4.md`.

## Non-functional requirements

{{caption:Non-functional requirements (PROJECT_SPEC.md §6.2) with the sandbox results recorded in results/phase4.md}}

{{nfr_table}}

The three requirements not measured on the sandbox are properties of the design:

- **NFR-4 (cost).** The model repository is public (anonymous access to the pinned revision returned {{fact:anonymous_access}}), the Action runs on a standard hosted runner, and no paid API is called. The cost of the runner minutes themselves depends on the user's plan and was not measured.
- **NFR-7 (reproducibility).** The model is pinned by revision, the Python dependencies by a Linux lock generated in CI, the third-party actions by commit SHA, the data by upstream commit and SHA-256, and the splits by committed manifests; seeds are fixed. Appendix A lists every pin and hash.
- **NFR-8 (determinism).** In Phase 0, changing the number of CPU threads did not change any prediction ({{fact:thread_determinism}}). The production path reproduced the validation predictions exactly in the parity test ({{fact:parity_val}}), and the sandbox runner reproduced the local results within the comparison tolerance (chapter ten). GPU training is seeded but not bit-for-bit reproducible.

## Success criteria and how each was judged

The specification sets five success criteria for v1.0 and calls them targets, not promises: an honest miss with analysis is acceptable, a hidden miss is not. Criteria one to four were judged by rules written down before the test run (`results/phase3.md`, "Test protocol (pre-declared)"); criterion five by the sandbox runbook. The verdicts are discussed in chapter nine.

{{caption:Success criteria for v1.0 (PROJECT_SPEC.md §6.3), the rule each was judged by, and the result}}

{{criteria_table}}
