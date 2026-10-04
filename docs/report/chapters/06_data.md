# Data

The data card, `data/DATA_CARD.md`, is the reference for this chapter; its tables are read by the generator.

## Source, licence status and hash verification

{{caption:The NLBSE'24 source files (data/DATA_CARD.md)}}

{{data_source_table}}

The upstream repository's licence file is empty ({{fact:licence_empty}}), so no licence is granted for the data. The project therefore never commits the raw data: `data/raw/` and `data/processed/` are ignored by git, and only the manifests (synthetic identifiers and content hashes) and the scripts are committed. Anyone who rebuilds the data set fetches it with `training/fetch_data.py`, which downloads both files at the pinned upstream commit and checks their SHA-256 before anything else runs. The pre-public audit of the repository history found no raw-data paths; in the same audit, {{fact:audit_titles}}.

## Leakage check and dropped rows

The data set has no issue identifiers, so every row gets a synthetic identifier (`nlbse24-<source>-<row>`, where the row is its zero-based position in the raw file) and a SHA-1 content hash over its raw title and body. Leakage is checked by content, not by identifier. Official-train rows whose content also occurs in the official test file, or more than once in the official train file, are dropped in all copies. On the real data both rules select the same {{n_dropped}} rows:

{{caption:Official-train rows dropped before splitting (data/DATA_CARD.md)}}

{{dropped_table}}

One of the overlapping pairs carries {{fact:conflicting_pair}} (`bug` in train, `feature` in test), an early sign of the label noise discussed at the end of this chapter. An automated check asserts that no identifier and no content hash appears in more than one split, and that train, validation and dropped rows add up to the {{n_official}} official training rows.

## Splits and manifests

The validation split takes {{n_val}} of the kept training rows, stratified by repository and label with a fixed seed, and never takes the {{phase0_n}} rows used for the Phase 0 wording comparison (those are forced into training). The official test file is used whole and untouched.

{{caption:Split sizes by label (data/DATA_CARD.md)}}

{{split_table}}

The split membership is committed as manifests, one line per row with its identifier and content hash. Their SHA-256 values are computed by the generator from the committed files:

{{caption:Committed split manifests}}

{{manifest_table}}

## Preprocessing

One function, `preprocess()` in `src/laya_triage/preprocess.py`, cleans every issue in training, evaluation and at run time. Its steps (specification §8.4) are:

{{preprocess_steps}}

The data card records how often each step changed a body across all raw issues:

{{caption:How often each preprocessing step applied, over all raw issues of both files (data/DATA_CARD.md)}}

{{hit_table}}

After cleaning, the frozen question leaves room for {{state_room_native}} state tokens at the base checkpoint's native budget and {{state_room_prod}} at the training and production budget. Long issues are truncated by the tokenizer:

{{caption:Token lengths of the cleaned states and the share truncated at each budget (results/phase1/token_lengths.json)}}

{{token_table}}

## Manual data-quality review

A random sample of training rows was reviewed by the owner, with an AI first pass on the same view, against the frozen question definitions, using the title and the {{fact:review_window}} of the cleaned body. The outcome was: labels {{fact:review_outcome}}, and cleaning {{fact:review_cleaning}}. Every wrong or doubtful label was in the `feature` or `question` class. In {{fact:review_template}}, the author had picked a bug or feature template although the gold label is question; the cleaned text keeps those template lines, which help on real bug and feature reports and mislead on these. The review estimates the label noise at {{fact:review_noise}}, with wide uncertainty on so small a sample. It predicted that `question` would be the hardest class, which the results confirm.

Other limitations recorded in the data card: the data are balanced by construction and split at random rather than by time; {{fact:empty_bodies}} become empty after cleaning and are classified on the title alone; and long logs pasted without a code fence are cut only by the character cap.
