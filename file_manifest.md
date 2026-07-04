# File Manifest

This manifest describes the intended role of each top-level file or directory in the submission package.

| Path | Role |
|---|---|
| `README.md` | Main paper-facing reproducibility guide. |
| `submission_checklist.md` | Practical checklist for paper supplement or GitHub upload. |
| `file_manifest.md` | Description of the submitted file surface. |
| `.gitignore` | Ignore rules for caches, checkpoints, logs, and local outputs. |
| `requirements.txt` | Minimal Python dependency list. |
| `config/experiment_config.json` | Reproducible experiment settings, seeds, methods, and hyperparameters. |
| `src/suite.py` | Main experiment implementation. |
| `src/__init__.py` | Python package marker. |
| `tools/` | Validation, notebook creation, progress checking, and reporting utilities. |
| `notebooks/` | Local notebook workflow for precheck, studies, aggregation, and monitoring. |
| `colab_a100_full_550_runs.ipynb` | Full Colab execution notebook. |
| `colab_a100_low_drive_550_runs.ipynb` | Low-storage Colab execution notebook. |
| `final_tables/summary_by_task_model_method.csv` | Main aggregate result table. |
| `final_tables/winners_by_metric.csv` | Method winners by performance, efficiency, and stability. |
| `results/environment.json` | Recorded run environment. |
| `results_summary/` | Human-readable summaries for experiment results and interpretation. |
| `docs/experiment_overview.md` | Experiment design overview. |
| `github_assets/README.md` | Optional GitHub-facing asset notes. |

The package is designed to be small and reviewable. Raw checkpoints, caches, and per-run artifacts should remain in the original private experiment workspace unless a venue explicitly requests them.
