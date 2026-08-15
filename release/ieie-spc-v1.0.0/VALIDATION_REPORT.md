# Public Release Validation Report

Release ref: `ieie-spc-v1.0.0`
Prepared: 2026-08-12

## Publicly repeatable checks

- Historical aggregate dimensions: PASS (110 method rows; 22 winner rows).
- Historical exact-zero rows: PASS (4 rows at stored precision).
- Pilot compact run coverage: PASS (60/60 COMPLETE rows; unique signatures; five seeds per variant-condition group).
- Full-data compact run coverage: PASS (40/40 COMPLETE rows; unique signatures; five seeds per variant-condition group).
- Public aggregate tables contain no per-example text, predictions, checkpoint paths, completion timestamps, or machine-local runtime dictionaries.

## Local-only check

Per-example predictions for 100 follow-up runs were retained locally and independently recomputed with zero metric discrepancies. Those prediction files are not in this public release, so that prediction-level check cannot be independently repeated from the tagged tree.

## Reproducibility boundary

The tagged code is the maintained reconstruction/re-execution implementation. The recorded executed-suite hash is `3bcdaa9bafe78748aa3e7658008c984c09c0b0349fcd5fdc55d0d34c2469bdcb`, but its exact source snapshot was not preserved. The release therefore supports code inspection, rerunning under documented configurations, and regeneration of compact summaries; it does not claim bit-identical replay of the historical or follow-up executions.
