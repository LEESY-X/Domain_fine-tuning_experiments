# IEIE SPC Final Submission Checklist

`make all` verifies the evidence package and the mechanical PDF requirements. The unchecked items below require author/editorial authority and cannot be inferred safely from experiment files.

## Author confirmation — blocking

- [ ] Confirm the author order: **Yuseok Hong, Seonyeok Lee, Sooyong Lee**.
- [ ] Confirm the English affiliation: **Major in Big Data Convergence, Sangmyung University, Seoul, Republic of Korea**; add a postal code if the journal requires one.
- [ ] Confirm that Yuseok Hong is the corresponding author and approve `hyscodebase@gmail.com` for publication. This address came from public Git commit metadata, not the ORIGIN manuscript.
- [ ] Supply and link an ORCID for every author in ScholarOne.
- [ ] Approve or replace the three conservative biographies; add author photographs only if requested by the journal workflow.
- [ ] Obtain explicit final-manuscript and author-order approval from all three authors.

## Declarations — blocking

- [ ] Confirm funding and insert the exact funding statement or ``no external funding'' declaration.
- [ ] Confirm competing interests and insert the approved declaration.
- [ ] Confirm CRediT author contributions for all three authors.
- [ ] Obtain the institution's determination on IRB/exemption/not-human-subjects status for the public hate-speech and review datasets.
- [ ] Confirm dataset license, deleted-content, PII, and redistribution language; no raw text is included in this paper package.
- [ ] Retain or revise the Codex assistance acknowledgment according to the current IEIE policy and all authors' approval.
- [ ] Disclose any prior conference or journal version. If one exists, follow IEIE's new-content, first-page footnote, and cover-letter requirements.

## Artifact release

- [ ] Verify that public tag `ieie-spc-v1.0.0` resolves to the intended code and compact-evidence commit.
- [x] Keep the manuscript source/PDF separate from the public code tag until author metadata and declarations are approved.
- [ ] Archive the release and add a DOI only if one is actually assigned; the manuscript currently makes no DOI claim.
- [x] Publish compact run metrics only; keep raw per-example predictions and other per-run files local.
- [x] Preserve the executed-source hash caveat unless the exact historical source snapshot is recovered.

## Scientific checks — completed

- [x] Public 110-row aggregate snapshot copied; downstream summaries, statistics, and figures regenerated deterministically from that snapshot.
- [x] Original 13-task Friedman and Full-FT pairwise statistics independently recomputed.
- [x] Study-1, collapse-task, and strict complete-block sensitivities generated.
- [x] Four exact-zero rows checked at full stored precision.
- [x] Pilot 60/60 and full-data 40/40 runs found complete.
- [x] Metrics and collapse diagnostics recomputed from all 100 prediction files with zero discrepancies.
- [x] Historical aggregate and targeted remedy kept in separate tables and estimands.
- [x] Exact `p=0.0625`, combined-treatment, post-selection, and CUDA/MPS limitations retained.
- [x] Study-1 dataset revision, group-split manifest, and comment-group separation limitations retained.
- [x] Historical snapshot, tagged compact release, local run artifacts, and local manuscript distinguished.
- [x] Stale result/checkpoint signatures and incomplete or duplicate seed aggregates rejected by the current code.

## Journal format — completed

- [x] Official `SPC.cls`, `picins.sty`, and ORCID asset retained unmodified.
- [x] Review-copy overrides remove the legacy JSTS DOI, placeholder issue, and 2024 copyright.
- [x] True A4, double-column, single-spaced PDF compiled.
- [x] Times New Roman and Arial are embedded.
- [x] Abstract is 178 words with no citation; six keywords are present.
- [x] PDF is 10 pages, inside the 9--14 page range including biographies.
- [x] Every final PDF page inspected at full resolution.
- [x] No unresolved citation, placeholder, legacy DOI, overfull box, clipping, or broken figure remains.

## Final ScholarOne actions — corresponding author only

- [ ] Run `make all` once after all declarations and author metadata are finalized.
- [ ] Ensure title, abstract, keywords, authors, affiliation, and declarations exactly match ScholarOne fields.
- [ ] Upload `main.tex`, `SPC.cls`, `picins.sty`, figures, and the final PDF; inspect the generated proof.
- [ ] Upload the source-backed supplementary CSVs and validation manifest allowed by the journal.
- [ ] Leave submission, copyright, publication-fee, and consent actions to the confirmed corresponding author.

Official references: [Information for Authors](https://www.ieiespc.org/ieiespc/AimsAndScope), [Paper Submission Guidelines](https://www.ieiespc.org/ieiespc/PaperSubmissionGuidelines), and [Ethics and Publication Policy](https://ieiespc.org/ieiespc/AuthorGuide).
