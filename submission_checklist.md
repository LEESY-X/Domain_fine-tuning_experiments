# Submission Checklist

Use this checklist before submitting this folder as a paper supplement or uploading it to a public reproducibility repository.

## Included

- [x] Paper-facing `README.md`
- [x] Dependency list: `requirements.txt`
- [x] Experiment configuration: `config/experiment_config.json`
- [x] Core experiment code: `src/suite.py`
- [x] Utility scripts: `tools/*.py`
- [x] Local notebooks: `notebooks/*.ipynb`
- [x] Colab notebooks: `colab_a100_*.ipynb`
- [x] Final summary tables: `final_tables/*.csv`
- [x] Environment record: `results/environment.json`
- [x] Result summaries and experiment overview docs

## Excluded on Purpose

- [x] No `cache/` directory
- [x] No model checkpoints
- [x] No raw dataset files
- [x] No per-run predictions
- [x] No per-run trainer histories
- [x] No per-run event logs
- [x] No generated HTML reports
- [x] No internal audit/process folders
- [x] No `.git/` history copied from the previous upload repository

## Before Public Upload

- [ ] Replace any citation placeholder after the paper title, venue, DOI, or arXiv link is finalized.
- [ ] Confirm dataset licenses and redistribution terms for all referenced datasets.
- [ ] Confirm whether the target venue wants a zip archive, GitHub link, or both.
- [ ] If uploading to GitHub, initialize Git in this folder only after the file list is final.

## Provenance

This folder was rebuilt as a clean submission copy on 2026-07-04 from the previously curated reproducibility bundle:

`paper_finetuning_strategy_study/local_5method_notebook_suite/github_upload_supplementary_repo`

Original experiment workspaces were not modified during this rebuild.
