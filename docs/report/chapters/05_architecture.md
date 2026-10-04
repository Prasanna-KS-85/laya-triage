# System architecture

## The master diagram

{{figure:architecture|Laya Triage: master architecture and workflow (docs/architecture.png; editable source docs/architecture.drawio)}}

{{include:DEEP_DIVE#architecture-at-a-glance~How to read the diagram}}

{{include:DEEP_DIVE#architecture-at-a-glance~The figures inside the image}}

The figure was drawn by hand and is the only part of this report that the generator cannot check; where it shows a number, the tables of this report take precedence.

## Build-time pipeline

{{include:DEEP_DIVE#band-a-build-time-once-per-model-version}}

## Run-time pipeline

{{include:DEEP_DIVE#band-b-run-time-every-new-issue}}

The sequence of one triage run, from the specification (§7.4):

```
{{sequence_diagram}}
```

## Module design

The run-time package `src/laya_triage/` is split so that every decision is a pure function that can be unit-tested without the model or the network, and the side effects sit at the edges. The contracts are fixed in §12.2 of the specification:

{{caption:Module contracts of the run-time package (PROJECT_SPEC.md §12.2)}}

{{module_table}}

`preprocess()` is imported by the training, evaluation and run-time code alike, so the model sees the same cleaned state in all three. The escalation comment is chosen from two fixed templates, and only label names and percentages are filled in.

## Security model

{{include:DEEP_DIVE#security}}

These rules are enforced by tests rather than by review alone. `tests/test_workflows.py` checks `action.yml`, the CI workflow and the example workflows against six rules (chapter ten), and `tests/test_no_shell_execution.py` scans the package for shell or dynamic code execution.

## Failure modes and fail-soft behaviour

The Action must never block a user's pipeline because of a runtime fault (NFR-5), but it must fail loudly when it is misconfigured. The specification defines both layers:

{{caption:Run-time failure modes inside the Python package (PROJECT_SPEC.md §7.5)}}

{{fail_runtime_table}}

{{caption:Failure modes of the composite Action layer (PROJECT_SPEC.md §7.5)}}

{{fail_action_table}}

{{misconfig}}
