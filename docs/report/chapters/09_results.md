# Results

All numbers in this chapter come from the single test run of {{test_run_date}} and the files derived from it: `results/phase3/test_R1b/metrics_test.json` for the pre-declared items, and `results/phase3/posthoc/posthoc.json` for the post-hoc analysis, which is marked as such.

## Headline results

{{include:DEEP_DIVE#evaluation|tables=Test results of all systems on the NLBSE'24 test split (results/phase3/test_R1b/metrics_test.json, B3 copied from results/phase2.md);Paired comparisons of M1 with each baseline on the test split}}

## Paired comparisons

The paired tests count the issues on which the two systems disagree. M1 is right and the baseline wrong on more issues than the reverse in every comparison, and in almost every bootstrap resample M1 has the higher cross-repo macro-F1:

{{caption:Discordant issues, exact McNemar p-values and the share of bootstrap resamples in which M1 is ahead (test)}}

{{mcnemar_table}}

The margin over B1 is the one that matters for the design: it is the cheapest strong baseline, and the specification required M1 to beat it to justify a model of this size. The interval of that difference excludes zero but is wide, so the size of the advantage is uncertain. The margin over B2, the same checkpoint without fine-tuning, shows that fine-tuning is necessary.

## Calibration

{{include:DEEP_DIVE#calibration|tables=Calibration of M1 on the test split before and after temperature scaling}}

Temperature scaling changed the expected calibration error on test from {{t1_ece}} at T = 1 to {{m1_ece}} at the shipped temperature, and the Brier score from {{t1_brier}} to {{m1_brier}}. The predicted labels are unchanged, as temperature scaling cannot change them, so accuracy and every F1 value are the same before and after. The criterion of an ECE of at most {{ece_limit}} is met. The TF-IDF baseline has a test ECE of {{b1_ece}}, which is comparable; this project shows no calibration advantage of M1 over B1.

{{figure:reliability|Reliability diagram of M1 on the test split, before (T = 1) and after temperature scaling}}

## Gating and the missed precision target

{{include:DEEP_DIVE#gating-and-the-missed-target|tables=Gating of M1 at the shipped thresholds on the test split}}

The pre-declared secondary views put the shipped operating point in context:

{{caption:Gating on test under the shipped thresholds and the two pre-declared secondary rules}}

{{secondary_gate_table}}

Read together, these numbers say that the gate works in the direction it should (precision on the auto-applied subset is well above the overall accuracy of {{m1_acc}}), but the thresholds fitted on {{n_val}} validation issues did not carry the precision they promised to the test split. As pre-declared, the miss is reported and the thresholds were not re-tuned on test. On test, {{gate_wrong}} of the {{gate_applied}} automatically applied `bug` labels were wrong. A user who switches to apply mode should expect errors of this kind on data like the benchmark, and should check the dry-run output on their own repository first.

## Per-class and per-repository results

{{caption:Per-class precision, recall and F1 on the test split}}

{{per_class_table}}

`question` is the weakest class for every system. M1 has recall {{m1_question_recall}} on `question` and {{m1_bug_recall}} on `bug`; its predicted counts are {{m1_pred_counts}}. The confusion counts show where the errors go:

{{caption:Confusion counts on the test split (rows: gold label; columns: predicted label)}}

{{confusion_table}}

{{figure:confusion|Normalised confusion matrices of M1 and B1 on the test split}}

By repository, M1 is ahead of B1 in {{n_repos_m1_ahead}} of {{n_repos}} repositories; the smallest margins are {{smallest_margins}}, and M1's weakest repository is {{m1_weakest_repo}}. The published B3 rows are included for reference only: they come from a different protocol (one classifier per repository, trained on that repository's rows, copied and not re-run), so they are not a like-for-like comparison with M1 or B1.

{{caption:Macro-F1 per repository on the test split}}

{{per_repo_table}}

{{figure:per_repo_f1|Macro-F1 per repository of M1 and B1 on the test split}}

## Validation-to-test gap and near-duplicate analysis

M1's cross-repo macro-F1 fell from {{val_xrepo}} on validation to {{test_xrepo}} on test, while B1 went from {{val_b1_xrepo}} to {{b1_xrepo}}:

{{caption:Cross-repo macro-F1 on validation and on test}}

{{gap_table}}

The validation numbers therefore overstated the test performance of M1, and the thresholds and the temperature were fitted on that same split. Because M1 was fine-tuned once, the drop is not caused by choosing the best of several models on validation. The rest of this section is post-hoc: it was run after the test numbers were known, it reused the committed predictions, and it changes no criterion.

**Near-duplicates to training issues (post-hoc).** One possible explanation is that validation issues resemble training issues more closely than test issues do. For every validation and test issue, the highest TF-IDF cosine similarity to any training issue was computed:

{{caption:Highest TF-IDF cosine similarity of each issue to the training issues (post-hoc; results/phase3/posthoc/posthoc.json)}}

{{near_dup_table}}

{{caption:Accuracy by similarity tertile, with cut points {{tertile_cuts}} taken from the validation distribution (post-hoc)}}

{{tertile_table}}

Test is, if anything, slightly closer to the training data than validation, and M1's accuracy fell in every tertile (by {{tertile_drops}} points). Near-duplicate overlap does not explain the gap. The analysis does not identify the cause; sampling noise on {{n_val}} validation issues is one candidate it cannot rule out.

**One global cutoff (post-hoc).** A second question was whether a single threshold over all labels, chosen by the same bootstrap rule on validation, would have met the target. The rule was stated before it was computed and evaluated on test once:

{{caption:One global threshold fitted on validation and evaluated once on test (post-hoc)}}

{{global_cut_table}}

Both systems fall below the target on test with their own global cutoff: M1 reaches a precision of {{gc_m1_prec}} and B1 {{gc_b1_prec}}. A sanity recomputation confirmed the gating counts of the main run ({{sanity_correct}} correct of {{sanity_applied}} auto-applied issues at the shipped thresholds) and the confusion matrices.

## Verdict on each success criterion

The criteria and the rules they were judged by are listed in chapter three. The verdicts:

- **Criterion one, met.** M1 beats the TF-IDF baseline and the base checkpoint at both budgets on the headline metric, and all three paired intervals exclude zero.
- **Criterion two, met.** The test ECE after temperature scaling is {{m1_ece}}, below the limit of {{ece_limit}}.
- **Criterion three, not met.** The precision target for auto-applied labels was {{target_status}} on test: the auto-applied `bug` labels had a precision of {{gate_prec}}, below {{target}}, at a coverage of {{gate_cov}}. On validation only `bug` had a qualifying threshold; on test the target was met for {{met_k}} of {{n_labels}} labels.
- **Criterion four, met only through its second clause.** The per-issue latency target (NFR-1) was {{nfr1_status}}: the latency probe on the GitHub-hosted runner measured {{probe_p95}} per issue at p95 with the model already loaded, against a target of at most {{nfr1}}. The criterion allows the miss to be documented with the ONNX plan of v1.2, which chapter ten and chapter fourteen do.
- **Criterion five, met.** The Action ran end to end on the sandbox repository, and every run agreed with the local production path (chapter ten).
