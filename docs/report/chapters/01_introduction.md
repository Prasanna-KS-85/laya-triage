# Introduction

## The issue-triage problem

Open-source and internal repositories receive a steady stream of issues. Before anyone can prioritise, search or route an issue, a maintainer has to read it and decide what kind of issue it is: a defect report, a request for something new, or a request for help. That first decision is repetitive and easy for an experienced maintainer, but it takes time, and issues that stay unlabelled slow down prioritisation, search and the onboarding of new contributors. The specification of this project (`PROJECT_SPEC.md`, §2) states the problem in these terms, and this report follows it.

The task studied here is the narrowest useful version of that decision: put each newly opened issue into exactly one of {{n_labels}} classes, {{labels_list}}, and do it only when the decision can be trusted. An issue the tool is unsure about should go to a person, not receive a guess.

## Why keyword and LLM bots fall short

Existing automation falls into two groups (§2 of the specification).

- **Keyword and rule bots** match words such as "crash" or "feature request". They are cheap, but brittle: they miss paraphrases, and issue templates put the same words into every issue whatever its real content.
- **Bots built on large language models** read the issue and generate an answer. The specification lists four drawbacks that motivated this project: such calls are slow ({{fact:llm_bot_latency}}), they cost money per call, they return free text that must be parsed, and the confidence they state is itself generated text rather than a probability that can be checked. This report does not measure any language-model bot; the drawbacks are the design motivation, not a result.

What is missing is a triage tool that is fast, free to run, structured by construction, and honest about when it is unsure, so that it can act on the easy cases and hand the hard ones to a human.

## Goal and contributions

The goal of v1.0 is an installable GitHub Action that classifies new issues with a fine-tuned Laya decision model, gates every label on a calibrated probability, and reports its accuracy, calibration, coverage and speed on a published benchmark. The project contributes:

- **A fine-tuned model** for the {{n_labels}}-class task, published on Hugging Face at a pinned revision ([`{{model_repo}}`]({{hf_url}})), with a temperature refitted on validation data.
- **A confidence gate** with one threshold per label, each chosen on validation data as the smallest value whose bootstrap lower bound of precision reaches the target; a label that cannot reach it is never applied automatically.
- **A pre-declared evaluation** on the NLBSE'24 test split against a majority-class floor, a TF-IDF baseline, the base checkpoint without fine-tuning and the published competition results, with paired statistical tests.
- **A composite GitHub Action** for Linux runners with a dry-run default, fail-soft behaviour, caching, a pinned dependency lock and an end-to-end validation on a sandbox repository.
- **Generated documentation:** the README, the deep dive, the usage guides, the model card and this report are built from the committed result files, so a number cannot be typed in by hand.

## Scope and non-goals

The release map of the specification (§4.1) fixes what v1.0 contains and what comes later:

{{caption:Release map (PROJECT_SPEC.md §4.1)}}

{{release_map_table}}

The explicit non-goals for v1.0 (§4.2) are:

{{non_goals}}

## Structure of this report

Chapter two explains the background: Laya, its training method, calibration, selective classification and the benchmark. Chapters three and four give the requirements and the development process. Chapter five describes the architecture, chapter six the data and chapter seven how the model was developed. Chapter eight sets out the evaluation method and chapter nine the results, including both missed targets. Chapter ten covers the deployment as a GitHub Action, and chapter eleven how to use it. The report closes with limitations, lessons learned, future work and a conclusion. The appendices hold the reproducibility guide, the frozen question and default configuration, the full metric tables, the sandbox fixtures, the project timeline, a glossary and the references.

Wherever this report repeats a passage of the generated documentation (`docs/DEEP_DIVE.md`, `docs/USING_THE_ACTION.md`, `docs/USING_THE_MODEL.md`), the passage is copied word for word by the generator, with links turned into plain text or section references, so the two cannot drift apart. The rest of the text is written for this report.
