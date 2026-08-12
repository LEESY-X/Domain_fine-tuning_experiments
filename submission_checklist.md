# Submission Checklist

Use this checklist before submitting this folder as a paper supplement or uploading it to a public reproducibility repository.

## Included

- [x] Paper-facing `README.md`
- [x] Dependency list: `requirements.txt`
- [x] Pinned top-level follow-up dependency versions: `requirements-followup.txt` (not a complete transitive or OS lock)
- [x] Experiment configuration: `config/experiment_config.json`
- [x] Core experiment code: `src/suite.py`
- [x] Utility scripts: `tools/*.py`
- [x] Output-free Colab notebooks: `colab_a100_*.ipynb`
- [x] Final summary tables: `final_tables/*.csv`
- [x] Environment record: `results/environment.json`
- [x] Result summaries and experiment overview docs
- [x] Targeted collapse follow-up compact metric tables and paper-ready report
- [x] Release manifest, run coverage, validation boundary, and SHA-256 checksums

## Excluded on Purpose

- [x] No `cache/` directory
- [x] No model checkpoints
- [x] No raw dataset files
- [x] No original 550-run per-run artifacts
- [x] No public per-example follow-up predictions, histories, events, or per-run JSON files
- [x] No machine-local absolute paths in follow-up artifacts
- [x] No internal audit/process folders
- [x] No `.git/` history copied from the previous upload repository

## Before Public Upload

- [ ] Replace any citation placeholder after the paper title, venue, DOI, or arXiv link is finalized.
- [ ] Confirm dataset licenses and redistribution terms for all referenced datasets.
- [ ] Confirm whether the target venue wants a zip archive, GitHub link, or both.
- [x] Publish only compact follow-up metrics; keep sample-level labels and predictions local.
- [ ] Verify that tag `ieie-spc-v1.0.0` resolves to the intended release commit after push.

## Provenance

This folder was rebuilt as a clean submission copy on 2026-07-04 from the previously curated reproducibility bundle:

`paper_finetuning_strategy_study/local_5method_notebook_suite/github_upload_supplementary_repo`

Original experiment workspaces were not modified during this rebuild.
