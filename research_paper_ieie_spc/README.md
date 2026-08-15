# IEIE SPC Integrated Manuscript Package

This directory contains the integrated English manuscript for **IEIE Transactions on Smart Processing and Computing**:

> **Performance, Efficiency, and Collapse-Aware Stability Trade-offs in Parameter-Efficient Fine-Tuning Across Domains**

It restores the original paper's performance--time--parameter--stability comparison and connects it to the later zero-SD audit and 100-run recovery experiment. The two evidence stages remain analytically separate: the post-selected remedy values are never substituted into the original 110-row leaderboard.

## Current build

- Source: `main.tex`
- Inspection PDF: `main.pdf`
- Format: true A4 (210 × 297 mm), two columns, single spacing
- Font: embedded Times New Roman body and Arial display/title text
- Length: 10 pages, within the IEIE SPC Regular Paper range of 9--14 pages
- Abstract: 178 words; keywords: 6
- Main figures: 2 vector figures; 2 additional generated figures retained for reuse/supplement
- Tables: 9, covering the original benchmark, task inference, sensitivities, exact-zero audit, pilot, full paired follow-up, and selection guide
- References: 27, through 2025
- Visual QA: every page rendered and inspected; no clipped body, table, or figure remains
- Automated checks: A4/page range, abstract/keyword limits, embedded fonts, legacy DOI/placeholders, unresolved references, overfull boxes, and public aggregate integrity; the 100 saved follow-up prediction files have a separate local-only validation target

The manuscript is mechanically ready for upload. The items in `SUBMISSION_CHECKLIST.md` that require author authority---especially corresponding-author contact, ORCIDs, declarations, author contributions, and approval of biographies---must be confirmed before submission.

## Evidence structure

1. **Original fixed-policy benchmark:** 13 task blocks, 22 task--model conditions, five methods, five seeds, and a configured 550-run design.
2. **Original multi-criteria results:** Full FT leads 19/22 Macro-F1 means; BitFit is fastest in 22/22; IA3 has the smallest trainable ratio in 22/22.
3. **Task-level inference:** 13-task Friedman and exact Wilcoxon/Holm comparisons, regenerated from the preserved 110-row aggregate.
4. **Robustness:** complete-block analyses excluding Study 1, excluding the two collapse-affected tasks, and excluding both.
5. **Zero-SD audit:** four historical rows match the aggregate constant-class fingerprint; historical predictions are unavailable.
6. **Targeted follow-up:** 60 pilot and 40 full-data runs, with prediction-distribution diagnostics and same-seed baseline/remedy comparisons.

## Directory map

| Path | Purpose |
|---|---|
| `main.tex`, `main.pdf` | Integrated IEIE source and compiled manuscript |
| `SPC.cls`, `picins.sty`, `orcid_mark.jpg` | Unmodified official template assets |
| `figures/` | Vector figures and PNG previews |
| `data/original_aggregate_110_rows.csv` | Self-contained snapshot of the original aggregate |
| `data/original_task_blocks_13.csv` | Reconstructed task blocks used under an explicit exchangeability assumption |
| `data/legacy_friedman_metrics.csv` | Original four-outcome Friedman results |
| `data/pairwise_full_ft_analyses.csv` | Pairwise results for every robustness definition |
| `data/f1_sensitivity_analyses.csv` | Global Macro-F1 robustness analyses |
| `data/table_*.csv`, `data/appendix_*.csv` | Follow-up manuscript tables and per-seed evidence |
| `scripts/generate_figures.py` | Regenerates all benchmark statistics, snapshots, and figures |
| `scripts/validate_followup_predictions.py` | Recomputes metrics/diagnostics from 100 prediction files |
| `scripts/check_submission.py` | Enforces mechanical IEIE PDF checks |
| `CLAIM_SOURCE_MAP.md` | Claim-to-source and caveat ledger |
| `SUBMISSION_CHECKLIST.md` | Author/editorial gates before ScholarOne upload |

## Rebuild and verify

From this directory:

```bash
make all
```

Or run the stages separately:

```bash
make figures
make validate-public
make validate-private  # requires the locally retained prediction files
make release
make paper
make check
```

Python build dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, and `matplotlib`; their current versions are not represented as a complete cross-platform lock. Tectonic may download its LaTeX bundle on first use. The build uses installed Times New Roman, Arial, and Menlo fonts; Tectonic reports their absolute system paths because font files are not redistributed.

## Template and format provenance

The official assets were copied from the IEIE SPC LaTeX archive and retained unchanged. `main.tex` applies review-copy overrides because the legacy class itself emits a 210 × 280 mm page, a `JSTS` DOI, and a 2024 copyright. The overrides enforce current A4 output and suppress author-invented volume, issue, DOI, dates, and copyright metadata.

| File | SHA-256 |
|---|---|
| `SPC.cls` | `d71e16ad4d0f09399a0771bbb11a4f12885193452c4ddcee09fd7ec301feecbd` |
| `picins.sty` | `21b057b49527e0a568913d0ee2782b51be1595be7439f96e2736eec39d564769` |
| `orcid_mark.jpg` | `e8b06724c543e65f81d9f01cb2357cd33492579281ac5999de2cee1e0e67ab29` |

Official pages consulted on 2026-08-12:

- [Information for Authors](https://www.ieiespc.org/ieiespc/AimsAndScope)
- [Paper Submission Guidelines](https://www.ieiespc.org/ieiespc/PaperSubmissionGuidelines)
- [Ethics and Publication Policy](https://ieiespc.org/ieiespc/AuthorGuide)

## Evidence boundaries

- The original design is described as **configured for 550 runs** because its run-level files are absent.
- Historical collapse is described as **aggregate-fingerprint consistent**, not directly proven from unavailable predictions.
- The broad fixed-policy estimand is kept separate from the targeted post-selected remedy.
- The exact two-sided sign-flip value is 0.0625 with five pairs; no conventional significance claim is made.
- The remedy changes schedule, rate, and weighting jointly; no component-level causal claim is made.
- Historical CUDA and follow-up Apple MPS results are not treated as identical-environment replications.
- Study 1's annotator-row split cannot be verified as group-disjoint; complete-block sensitivity excluding it is reported.
- Public commit `539e3d3`, tagged code/compact-evidence snapshot `ieie-spc-v1.0.0`, and the current local manuscript package are distinct provenance states.
- Compact metric-level rows for all 100 follow-up runs are published in the tag. Per-example predictions, histories, events, run JSON files, checkpoints, and caches remain local and are not described as public.
- The original optimizer/scheduler, exact package patch versions, and model/dataset revisions are not backfilled from the later follow-up configuration.
