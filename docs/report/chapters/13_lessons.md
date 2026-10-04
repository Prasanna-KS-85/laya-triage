# Lessons learned

**Write the protocol before the numbers exist.** Committing the thresholds and the test protocol before the test run made both misses visible and impossible to tune away. The precision miss would have been easy to hide by choosing a threshold on the test curve; the pre-declared rule made that a reported miss instead, and the post-hoc analyses could be read for what they are.

**A small calibration set gives a noisy temperature.** The notebook's temperature, fitted on {{n_calibration}} held-out training items, over-softened the probabilities, and its validation ECE was worse than no calibration at all. Refitting on the validation split brought the test ECE to {{m1_ece}} with the labels unchanged, at the price of making validation less independent for the threshold fit that followed.

**Validation can overstate test even without model selection.** The validation split of {{n_val}} issues overstated the test result by {{m1_gap_points}} points of cross-repo macro-F1 although only one model was trained, and the thresholds fitted on it lost precision on test. A bootstrap lower bound protects against noise in the precision estimate on the validation issues, but not against a shift between validation and test. A larger or repeated validation scheme would give a more honest estimate.

**Measure on the target hardware early.** The local latency measurements suggested that the per-issue target would hold; the runner, with one usable thread, missed it by a wide margin. The issues-event cache restriction was also found only on a real runner. Both findings came from the end-to-end sandbox, not from unit tests.

**Generate documents from results.** Typing numbers into prose invites drift and rounding. Generating the README, the deep dive, the usage guides, the model card and this report from the committed result files, with tests that reject typed numbers, turned rule eleven into a check. The tests also caught small errors in the tooling itself: the first test report picked up {{fact:parser_bug}} from a table in `results/phase2.md`, which a regression test now prevents.

**Pin everything, including the training tokenisation.** The items-hash check proved that the remote GPU notebook built exactly the token sequences the local evaluation and the Action use. Pinning the model by revision, the dependencies by a lock from CI and the actions by commit made every number in this report traceable to a fixed artefact.

**Treat issue text as hostile.** Keeping issue content out of shell commands, out of comments and out of logs was cheap when designed in from the start, and static tests keep it that way.
