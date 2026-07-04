# Fine-Tuning Strategy Reproducibility Package

This folder is the paper-facing reproducibility package for the fine-tuning strategy comparison study. It contains the code, configuration, notebooks, environment record, and final summary tables needed to inspect or rerun the reported experiments without including raw checkpoints, dataset caches, or per-run intermediate artifacts.

## Study Scope

The experiments compare five fine-tuning strategies for pretrained language models on text classification tasks:

- Full Fine-tuning
- LoRA
- Adapter
- IA3
- BitFit

The final experiment set covers 3 studies, 22 task/model combinations, 5 methods, and 5 random seeds, for 550 completed runs in total. The full run-level table is intentionally excluded from this lightweight submission package; the submitted evidence is centered on the final aggregate tables and reproducibility code.

## Repository Contents

```text
.
|-- README.md
|-- submission_checklist.md
|-- file_manifest.md
|-- requirements.txt
|-- config/
|   `-- experiment_config.json
|-- src/
|   |-- __init__.py
|   `-- suite.py
|-- tools/
|   `-- *.py
|-- notebooks/
|   `-- *.ipynb
|-- colab_a100_full_550_runs.ipynb
|-- colab_a100_low_drive_550_runs.ipynb
|-- final_tables/
|   |-- summary_by_task_model_method.csv
|   `-- winners_by_metric.csv
|-- results/
|   `-- environment.json
|-- results_summary/
|   `-- *.md
|-- docs/
|   `-- experiment_overview.md
`-- github_assets/
    `-- README.md
```

## Key Files

- `config/experiment_config.json`: methods, seeds, task groups, model list, hyperparameters, and training settings.
- `src/suite.py`: main experiment suite implementation, including task definitions, preprocessing, model setup, metrics, and training flow.
- `notebooks/`: local notebook workflow for precheck, study execution, aggregation, and progress monitoring.
- `colab_a100_full_550_runs.ipynb`: Colab-oriented full experiment notebook.
- `colab_a100_low_drive_550_runs.ipynb`: Colab-oriented low-Drive-storage variant.
- `final_tables/summary_by_task_model_method.csv`: final aggregated metrics by study, task, model, and method.
- `final_tables/winners_by_metric.csv`: best methods by performance, efficiency, and stability metric.
- `results/environment.json`: recorded software and hardware environment for the completed local run.
- `results_summary/`: paper-facing summaries of results, hyperparameters, and interpretation.

## Environment Snapshot

The recorded local environment in `results/environment.json` includes:

- Python: 3.10.19
- PyTorch: 2.11.0.dev20260119+cu128
- CUDA runtime: 12.8
- GPU: NVIDIA GeForce RTX 5070 Ti, 15.92 GB
- transformers: 5.9.0
- datasets: 4.8.5
- peft: 0.19.1

## Reproduction Outline

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight validation before launching full experiments:

```bash
python tools/validate_suite.py
python tools/smoke_data.py
```

Use the notebooks for execution:

- Local workflow: `notebooks/00_precheck.ipynb` through `notebooks/05_progress_monitor.ipynb`
- Colab workflow: `colab_a100_full_550_runs.ipynb`
- Colab low-storage workflow: `colab_a100_low_drive_550_runs.ipynb`

Full reruns require substantial GPU time and access to the public datasets referenced in `src/suite.py`. Dataset license and redistribution conditions should be checked separately by users before sharing derived artifacts beyond this package.

## What Is Intentionally Excluded

This submission package intentionally does not include:

- dataset caches
- model checkpoints
- per-run prediction files
- per-run trainer histories
- per-run event logs
- per-run final metric files
- internal audit notes
- generated HTML reports
- local cache or notebook checkpoint directories

These exclusions keep the package suitable for a paper supplement or public reproducibility repository while preserving the original experiment workspace untouched.
