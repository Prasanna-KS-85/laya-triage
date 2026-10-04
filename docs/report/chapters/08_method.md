# Evaluation methodology

All systems are compared under Protocol A of the specification (§8.2): trained only on the official NLBSE'24 training data, with no resampling, and evaluated on the official test split.

## Baselines

{{caption:Systems compared (PROJECT_SPEC.md §10.1)}}

{{systems_table}}

B0 predicts the most frequent training class for every issue and gives the floor. B1 is a TF-IDF model over word unigrams and bigrams with logistic regression; its hyperparameters were chosen on the validation split from a grid of {{b1_grid_n}} configurations (selected: C {{b1_C}}, minimum document frequency {{b1_min_df}}, sublinear term frequency) and frozen before the test run. B2 is the base Laya checkpoint with the same frozen question and no fine-tuning, at its native budget and at the production budget. B3 is copied from the competition repository with a citation and was not re-run; it has no probabilities, so it has no ECE and no coverage. The ONNX row is a plan for v1.2 and was not evaluated.

## Metrics

{{caption:Metrics (PROJECT_SPEC.md §10.2)}}

{{metrics_table}}

The SetFit value in the table only identifies the definition of the headline metric; that number was obtained under a different protocol (chapter two) and is not a like-for-like comparison.

Coverage and precision of the gate are reported in two ways: at the per-label thresholds of the shipped configuration, which is the operating point a user gets, and as a curve over one global threshold, which describes the trade-off but must not be used to pick a threshold because it reads the test split.

## The pre-declared test protocol

Chapter four quotes the protocol in full. In short: the test evaluation was run once, by one fixed command, after the thresholds and the configuration were committed; the baselines were not re-run, because their test predictions had been committed in Phase 2; and every item of the report, primary and secondary, was named in advance. A run that crashed could be repeated once with the identical command; the run did not crash and was not repeated.

## Statistical tests

Two paired tests compare the fine-tuned model with each baseline on the same test issues.

- **Paired bootstrap of the headline metric.** The test issues are resampled with replacement {{n_boot}} times (seed {{seed}}); in each resample the cross-repo macro-F1 of both systems is computed, and the {{ci_level}}% confidence interval of the difference is read from the percentiles of the resampled differences. An interval that excludes zero means the difference is unlikely to be an artefact of which issues happened to be in the test split.
- **Exact McNemar test.** Each issue is classified as right or wrong by each system. The test looks only at the issues where the two systems disagree and asks whether one system wins those more often than chance would allow. It tests accuracy, not macro-F1, and complements the bootstrap.

The bootstrap lower bound in the threshold rule (chapter seven) serves a different purpose: it guards against an optimistic precision estimate from a small validation split, which the specification lists as risk R12.

## Rules for post-hoc analysis

After the single test run, some questions were still open, the largest being why validation overstated the test result. They were examined by an exploratory analysis whose status is stated at its head in `results/phase3.md`:

> {{posthoc_rule}}

The analysis reused the committed predictions and ran no model. Its results are in chapter nine, in the section on the validation-to-test gap, and each table there is marked as post-hoc. None of them changes a success criterion, a threshold or the configuration.
