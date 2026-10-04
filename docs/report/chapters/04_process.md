# Development process

## A specification as the single source of truth

The project was built from a written specification, `PROJECT_SPEC.md`, which every human and coding agent had to follow. It holds the requirements, the architecture, the data and model specifications, the evaluation protocol, the phase plan, the decision records, the risks and {{n_rules}} rules for contributors. Where code and specification disagreed, the specification won until it was deliberately changed. The coding agent (Claude Code) worked from a short `CLAUDE.md` that points to the specification and names the current phase. The deep dive summarises the approach:

{{include:DEEP_DIVE#how-this-was-built}}

## Phases and gates

Work went in phases. Each phase had a goal, a list of tasks and a verification gate, and the next phase could start only when the gate had passed and its deliverables were committed. The table lists the phases from §13 of the specification with the tag that marks the end of each one; the tag dates come from the snapshot of git metadata in `docs/report/tags.yml`.

{{caption:Phases, their gates and the tags that close them (PROJECT_SPEC.md §13; tag dates from docs/report/tags.yml)}}

{{phases_table}}

The gate of Phase 5, an install by someone other than the author, had not been recorded when this report was generated: `results/phase5.md` lists "{{fact:pending}}" as still pending.

## Frozen contracts

Some artefacts may not change once their phase gate has passed, because the model or the evaluation depends on them. Rule four of the specification states:

> {{rule_frozen}}

The frozen contracts are the question (its wording, its label keys and the option order, in `src/laya_triage/questions.py`), the behaviour of the shared `preprocess()` function, and the data split manifests. Two further rules protect the evaluation. Rule six:

> {{rule_test}}

Rule eleven:

> {{rule_honest}}

In this project rule eleven is enforced mechanically: the README, the deep dive, the usage guides, the model card and this report are generated from the committed result files, and tests fail if a template contains a typed number or if a generated document is out of date.

## Tags and the changelog

Every change to the specification is recorded in its changelog, from version {{changelog_first}} to version {{changelog_last}} ({{n_changelog}} entries, Appendix E). Phase ends and releases are marked by git tags ({{n_tags}} in the snapshot). The release candidate `v1.0.0-rc1` closed Phase 4, and the annotated tag `v1.0.0` marks the release described in this report; `results/phase5.md` records that it points to `{{fact:release_commit}}`.

## The pre-registered test protocol

The test split was protected by a protocol written down in `results/phase3.md` on the day of the test run, before any test prediction of the fine-tuned model existed. The thresholds and the configuration were committed and tagged (`phase3-prefreeze`, commit `{{prefreeze_commit}}`) first. The protocol fixed the exact command:

```
{{test_command}}
```

It also fixed the crash rule: "{{protocol_rerun}}" The evaluation script refuses to score the test split unless the configuration is committed and unmodified, its model revision and token budget match the command line, no test predictions exist yet for the tag, and the test file has its pinned SHA-256.

The primary items to report were:

{{protocol_primary}}

The pre-declared secondary views were:

{{protocol_secondary}}

The success criteria were to be judged as follows ("item" refers to the numbered items above):

{{protocol_judging}}

The test evaluation was pre-declared and run once; later analyses reused its predictions and are labelled post-hoc. Chapter eight describes those analyses and the rule they followed.
