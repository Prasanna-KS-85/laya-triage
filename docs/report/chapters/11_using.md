# Using it

There are two ways to use the work. Path A, the GitHub Action, needs no code. Path B, the model on its own, is for people who want the probabilities in their own Python code. This chapter repeats the essential parts of the two usage guides word for word; the guides themselves (`docs/USING_THE_ACTION.md`, `docs/USING_THE_MODEL.md`) also cover backfill, applying the gate in your own code, and troubleshooting.

## Path A: the GitHub Action

{{include:USING_THE_ACTION#add-the-workflow}}

**Permissions.** {{include:USING_THE_ACTION#permissions}}

**Warm the caches once.** {{include:USING_THE_ACTION#warm-the-caches-once~In the Actions tab}}

**First run: stay in dry-run.** The default mode changes nothing on GitHub and writes each decision to the run's job summary. The guide recommends collecting a few dozen dry-run decisions, and checking the `bug` labels the Action would apply, before switching to apply mode.

**Switch to apply.** {{include:USING_THE_ACTION#switch-to-apply}}

**Escalation comments.** {{include:USING_THE_ACTION#escalation-comments}}

## Path B: the model on its own

{{include:USING_THE_MODEL#install}}

{{include:USING_THE_MODEL#classify-one-issue}}

{{include:USING_THE_MODEL#what-the-output-means}}

## Configuration reference

The Action reads its configuration from `.github/laya-triage.yml` in the user's repository, deep-merged over the defaults in `config/triage.default.yml` (Appendix B). An invalid file fails the run with a list of the bad keys.

{{caption:Inputs of the Action (action.yml)}}

{{action_inputs_table}}

{{caption:Configuration keys and their defaults (config/triage.default.yml)}}

{{config_table}}
