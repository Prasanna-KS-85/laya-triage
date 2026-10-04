# Future work

The release map of the specification (chapter one) and the roadmap of the deep dive set the order of future work. The roadmap reads:

{{include:DEEP_DIVE#roadmap}}

## v1.1

- **`needs_info`.** A yes/no (`noul`) question, "is the report missing reproduction steps, version or logs?", with explicit criteria and a labels override, because Laya's yes/no questions can follow their option labels rather than the text. It needs labelled data from repositories that use such labels, and a hand-annotated evaluation set, and the specification requires its own mini-specification before any code.
- **A language guard.** Detect the language of the issue and escalate non-English issues instead of classifying them, closing the gap that fixture S17 showed.
- **A same-protocol comparison with a plain classifier.** Fine-tune a plain encoder classifier on the same rows, fit its temperature and thresholds on the same validation split, evaluate it once on the same test split under a pre-declared protocol, and compare accuracy, calibration, precision at the gate and CPU time. Until this exists, the choice of Laya rests on design reasons, not on a measured advantage.
- **A length-matched latency probe.** Synthetic texts whose token lengths follow the validation length distribution, so that the runner latency can be estimated for realistic inputs without using issue text.
- **A hash-pinned dependency lock,** so that the runner installs exactly the files that were locked, not only the same versions.
- **An optional formatting pass.** `ruff check` is clean, but `ruff format --check` would reformat {{fact:format_note}}, including the frozen `preprocess.py`; a formatting pass must leave the frozen files alone.

## v1.2

Export the model to ONNX with INT8 quantisation, using Laya's export script, and compare accuracy, ECE and latency with the torch backend; switch `model.backend` only if the comparison favours it. This is the documented plan for the missed latency target. Quantisation changes the probabilities, so the thresholds would have to be refitted and the gate re-evaluated.

## v2.0

The stretch goals of the release map: a priority score (a `score` question), duplicate suggestions from embedding similarity, and a `documentation` class, which would need the larger NLBSE'23 data set. None of them is committed.

## Extending to more classes

{{include:DEEP_DIVE#extending-to-more-classes}}
