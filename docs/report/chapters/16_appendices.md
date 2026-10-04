# Reproducibility guide {appendix}

## Commands

The commands of the deep dive, from data to test:

{{include:DEEP_DIVE#reproduce}}

To rebuild this report: `.venv/bin/python docs/report/make_report.py` (Markdown) and, with the `report` extra installed, `.venv/bin/python docs/report/make_report.py --pdf` (PDF in `build/report/`). `--check` fails when `docs/report/REPORT.md` is out of date.

## Pinned versions

The evaluation used {{eval_versions}}. The runner environment is the Linux lock:

{{caption:The Linux lock, constraints-linux.txt}}

{{pin_table}}

{{caption:Direct pins in pyproject.toml}}

{{pyproject_table}}

## Hashes and revisions

{{caption:Pinned revisions and hashes}}

{{hash_table}}

# Question schema and default configuration {appendix}

## The frozen question

`src/laya_triage/questions.py`, verbatim:

```python
{{questions_py}}
```

## The default configuration

`config/triage.default.yml`, verbatim:

```yaml
{{config_yaml}}
```

# Full metric tables {appendix}

## Validation

The validation numbers of M1 are in-sample for the temperature and the thresholds, which were fitted on the same issues.

{{caption:All systems on the validation split (results/phase3/val_R1b/metrics_val.json)}}

{{val_full_table}}

## Test

{{caption:All systems on the test split (results/phase3/test_R1b/metrics_test.json)}}

{{test_full_table}}

B0 always predicts `question`; its probabilities are the training priors, so its ECE and Brier score carry no information. The published B3 rows are in chapter nine; they come from a different protocol and are not a like-for-like comparison. The per-class, per-repository and confusion tables are in chapter nine as well.

# Sandbox fixtures and run records {appendix}

## Fixtures

The fixtures are written for this project; none is taken from the benchmark. The table gives each fixture's key, its title, the class its author intended, and the local result of the production path (`sandbox/expected_local.json`) with the decision of the default configuration. The bodies are in `sandbox/issues.json`.

{{caption:Sandbox fixtures (sandbox/issues.json, sandbox/expected_local.json)}}

{{fixture_table}}

## Run records

The sandbox runs are listed in chapter ten. The release smoke run was run {{fact:smoke_run}} on issue {{fact:smoke_issue}} (`results/phase5.md`).

# Project timeline {appendix}

## Tags

A snapshot of git metadata (`docs/report/tags.yml`, written once from the local repository; times are those of the tag creator):

{{caption:Git tags of the project (snapshot of git metadata)}}

{{tags_table}}

## Specification changelog

{{caption:Changelog of PROJECT_SPEC.md (first sentence of each entry)}}

{{changelog_table}}

# Glossary {appendix}

## Plain-language terms

{{include:DEEP_DIVE#glossary}}

## Terms of the specification

{{caption:Glossary of PROJECT_SPEC.md §18}}

{{spec_glossary_table}}

# References and credits {appendix}

## References

{{references}}

## Credits and licence

{{include:DEEP_DIVE#credits-and-license}}
