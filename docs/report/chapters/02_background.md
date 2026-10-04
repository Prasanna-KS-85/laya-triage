# Background

## Laya, a decision model

<!-- alias DEEP_DIVE#how-it-works-laya-internals -->
Laya ([{{laya_url}}]({{laya_url}}), Apache-2.0) is a non-autoregressive decision engine. It takes a *state*, here the cleaned title and body of an issue, and one or more *typed questions*, and returns probabilities for every question in one forward pass. It never generates text, so there is nothing to parse and the output is always one of the options that were asked about. Laya offers three question types; this project uses the first and plans the other two:

{{caption:Laya question types and their use in this project (PROJECT_SPEC.md §5.1)}}

{{question_types_table}}

**Sequence packing.** The question, its options and the state are packed into one input sequence, with a mask marker in front of each option: {{packing}}. Because the options and their descriptions are part of the input rather than positions in a fixed output layer, changing the wording or the order of the options changes the model's behaviour. For this reason the question is a frozen contract (Appendix B).

**Encoder and decision head.** The English checkpoint `convaiinnovations/laya` uses a bidirectional ModernBERT-large encoder, {{fact:encoder_size}} ({{fact:encoder_params_exact}} parameters were counted when the checkpoint was loaded in Phase 0). A {{fact:head_layers}} transformer head runs on top of the encoder; the hidden state at each option's mask position goes through a small network that gives one score (logit) per option, and a softmax over the logits divided by a temperature `T` gives the probabilities.

**Two confidence fields.** Each answer carries `confidence`, which is one minus the normalised entropy and only describes the shape of the distribution, and `answer_confidence`, the probability of the chosen option. Only `answer_confidence` is meant to be calibrated (next sections), and it is the only quantity the gate uses.

## Training with RLCD

Laya's fine-tuning recipe, RLCD (Reinforcement Learning for Calibrated Decisions), optimises two terms together (specification §5.3). The first is a policy-gradient term over noisy samples of the logits, rewarded by strictly proper scoring rules ({{fact:rlcd_scores}}): such a reward is highest, on average, when the reported probabilities match the true frequencies, so it discourages over-confidence. The second is a soft cross-entropy term against a target distribution; in this project the targets are one-hot, because the benchmark gives hard labels. After training, the notebook fits one temperature per question type on a calibration slice it holds out from the training data before training starts.

## Calibration, temperature scaling and ECE

A classifier is *calibrated* when its probabilities can be taken at face value: among issues given a confidence of about seven in ten, about seven in ten should be right. Calibration matters here because the gate compares a probability with a threshold; if the probabilities were systematically too high, a threshold chosen on one data set would admit too many wrong labels.

*Temperature scaling* divides every logit by one fitted number `T` before the softmax. A value above one softens the probabilities and a value below one sharpens them. Because the order of the logits does not change, the predicted label never changes; only the confidence does. `T` is fitted by minimising the negative log-likelihood on held-out labelled data.

The *expected calibration error* (ECE) measures the remaining gap. The issues are sorted into {{ece_bins}} equal-width bins by `answer_confidence`; in each bin the absolute difference between the mean confidence and the accuracy is taken, and ECE is the average of these differences weighted by the share of issues in each bin. Lower is better, and zero means that confidence and accuracy agree in every bin. The *Brier score* is reported alongside it: the mean squared distance between the probability vector and the one-hot truth, which mixes calibration with accuracy.

## Selective classification: coverage and precision

A system that may abstain is judged by two numbers, not one. *Coverage* is the share of all issues it labels by itself; *precision* is the share of those labels that are correct. Raising a threshold usually raises precision and lowers coverage. The two are different quantities and are never interchangeable in this report: a precision of a given value says nothing about how many issues were labelled.

Laya Triage uses one threshold per label. For each label, the threshold is the smallest value at which a bootstrap lower bound of the validation precision reaches the target of {{target}}, among candidate thresholds that keep at least a minimum number of validation issues (chapter seven gives the exact procedure). A label for which no threshold qualifies gets a threshold above one, which means it is never applied automatically. Using the lower bound instead of the point estimate is deliberately conservative: it gives up coverage to make the precision target more likely to hold on new data.

## The NLBSE'24 benchmark

The NLBSE'24 tool competition on issue report classification provides {{n_official}} labelled training issues and {{n_test}} test issues from {{n_repos}} large repositories, with exactly {{per_cell}} issues per class in each repository of the test file. Each issue has a repository, a label, a title and a body, but no issue number or URL. The competition's headline metric, also used here, is the *cross-repo macro-F1*: the macro-F1 over the three classes is computed in each repository, and the {{n_repos}} values are averaged. The pooled macro-F1, over all test issues at once, is reported as well.

The competition repository publishes baseline results (B3 in this report): SetFit with cross-repo macro-F1 {{setfit_xrepo}}, RoBERTa with {{roberta_xrepo}} and fastText with {{fasttext_xrepo}}. These baselines were trained under a different protocol: one classifier per repository, each on only that repository's {{b3_rows_per_clf}} official-train rows, on raw text, with their numbers copied rather than re-run. The model in this report is one classifier across all repositories, trained on fewer rows after a validation split was held out. The published numbers are therefore a reference point, not a like-for-like comparison (chapter nine lists every difference).

## Why Laya instead of a plain classifier

The following passage is taken from the deep dive. It explains the choice of Laya and, as importantly, what the evidence of this project does and does not support.

{{include:DEEP_DIVE#why-laya-instead-of-a-plain-classifier|tables=What the evidence says about Laya against a plain classifier}}
