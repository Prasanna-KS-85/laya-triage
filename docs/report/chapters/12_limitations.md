# Limitations

Two targets of v1.0 were missed, and they come first:

- **The precision target for auto-applied labels was {{target_status}}.** At the shipped thresholds the auto-applied `bug` labels had a precision of {{gate_prec}} on test, against a target of {{target}}, at a coverage of {{gate_cov}} of the test issues; and `feature` and `question` are never applied automatically.
- **The per-issue latency target was {{nfr1_status}}.** The latency probe on a GitHub-hosted runner measured {{probe_p95}} per issue at p95, model already loaded, against a target of at most {{nfr1}}. Whole Action runs met their wall-time targets, which include the model load and the cache restore.

The remaining limitations are those of the deep dive:

{{include:DEEP_DIVE#limitations}}

Three further limitations concern this report and the evidence behind it:

- **No same-protocol comparison with a plain classifier.** There is no fine-tuned plain encoder trained and evaluated under the same protocol, so no claim is made about Laya's accuracy or calibration against one, in either direction.
- **One training run.** M1 is a single fine-tuning run; the variance across seeds or runs is unknown.
- **One runner, one probe.** The runner figures come from one private-repository runner instance, and the latency probe used the fixtures, not a sample of real issues.
