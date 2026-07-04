# Fine-Tuning Strategy Reproducibility Package

This repository contains the paper-facing reproducibility package for a text classification fine-tuning strategy study. It is designed as a lightweight supplement: it provides the experiment code, configuration, notebooks, environment record, and final aggregate result tables, while intentionally excluding raw checkpoints, dataset caches, and per-run intermediate artifacts.

The package was prepared from a larger local experiment workspace. The original experiment artifacts were not modified when this public-facing package was assembled.

## What This Repository Is

This repository supports inspection and rerunning of experiments comparing full model fine-tuning with parameter-efficient fine-tuning methods.

The compared methods are:

| Method key | Display name | Implementation source in this repo |
|---|---|---|
| `full_ft` | Full Fine-tuning | `src/suite.py`, `build_model()` |
| `lora` | LoRA | `src/suite.py`, Hugging Face PEFT `LoraConfig` |
| `adapter` | Bottleneck Adapter | `src/suite.py`, local `BottleneckAdapter` / `OutputWithAdapter` implementation |
| `ia3` | IA3 | `src/suite.py`, Hugging Face PEFT `IA3Config` |
| `bitfit` | BitFit | `src/suite.py`, bias parameters plus classification head unfrozen |

The final aggregate tables in this package report:

- 3 studies
- 22 task/model combinations
- 5 fine-tuning methods
- 5 seeds per method
- 550 completed runs in the original experiment workspace

The run-level table `all_runs_550.csv` is not included in this lightweight public package. The included result evidence is the aggregate output in `final_tables/summary_by_task_model_method.csv` and `final_tables/winners_by_metric.csv`.

## Repository Layout

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
|   |-- check_progress.py
|   |-- create_colab_a100_notebook.py
|   |-- create_notebooks.py
|   |-- report_dataset_sizes.py
|   |-- smoke_data.py
|   |-- test_recovery.py
|   |-- validate_suite.py
|   `-- verify_colab_parity.py
|-- notebooks/
|   |-- 00_precheck.ipynb
|   |-- 01_study1_bertweet_hate.ipynb
|   |-- 02_study2_multitask.ipynb
|   |-- 03_study3_korean.ipynb
|   |-- 04_aggregate.ipynb
|   `-- 05_progress_monitor.ipynb
|-- colab_a100_full_550_runs.ipynb
|-- colab_a100_low_drive_550_runs.ipynb
|-- final_tables/
|   |-- summary_by_task_model_method.csv
|   `-- winners_by_metric.csv
|-- results/
|   `-- environment.json
|-- results_summary/
|   |-- experiment_summary.md
|   |-- hyperparameters.md
|   |-- key_insights.md
|   `-- results_table.md
|-- docs/
|   `-- experiment_overview.md
`-- github_assets/
    `-- README.md
```

## Experiment Design

Experiment settings are defined in `config/experiment_config.json`.

| Study | Models | Tasks | Epochs | Runs represented in final aggregation |
|---|---|---|---:|---:|
| `study1` | `vinai/bertweet-base` | `measuring_hate_speech` | 3 | 25 |
| `study2` | `vinai/bertweet-base`, `FacebookAI/roberta-base` | `tweet_sentiment`, `finance_sentiment`, `movie_reviews`, `product_reviews`, `tweet_emotion`, `tweet_hate`, `tweet_offensive`, `tweet_irony`, `news_topic` | 2 | 450 |
| `study3` | `klue/roberta-base` | `news_ynat`, `movie_nsmc`, `comment_kmhas_binary` | 5 | 75 |
| Total | 3 base models | 22 task/model combinations | varies by study | 550 |

Each method is run with the following seeds:

```text
42, 52, 62, 72, 82
```

Core training settings from `config/experiment_config.json`:

| Setting | Value |
|---|---:|
| `max_length` | 128 |
| `batch_size` | 16 |
| `eval_batch_size` | 32 |
| `gradient_accumulation_steps` | 4 |
| `precision` | `fp16` |
| `weight_decay` | 0.01 |
| `warmup_ratio` | 0.06 |
| `early_stopping_patience` | 2 |
| `dataloader_num_workers` | 2 |
| SMOKE split limits | train 128, validation 64, test 64 |

Learning rates:

| Method | Learning rate |
|---|---:|
| Full Fine-tuning | 0.00002 |
| LoRA | 0.0001 |
| Adapter | 0.0001 |
| IA3 | 0.0005 |
| BitFit | 0.0001 |

PEFT-specific settings:

| Method | Setting |
|---|---|
| LoRA | rank `r=8`, alpha `16`, dropout `0.05`, target modules `query`, `value` |
| Adapter | bottleneck `64`, dropout `0.0` |
| IA3 | target modules `key`, `value`, `intermediate.dense`; feedforward module `intermediate.dense` |
| BitFit | freezes model parameters except bias parameters and classifier/score head |

## Datasets and Tasks

Task definitions are in `src/suite.py` under `TASKS`. The repository does not redistribute dataset files. Datasets are loaded from Hugging Face datasets or public GitHub TSV files during execution.

| Task key | Source path | Subset / direct loader | Text column | Label column | Labels |
|---|---|---|---|---|---:|
| `measuring_hate_speech` | `ucberkeley-dlab/measuring-hate-speech` | source split `train` | `comment` | `hatespeech` | 2 |
| `tweet_sentiment` | `cardiffnlp/tweet_eval` | `sentiment` | `text` | `label` | 3 |
| `finance_sentiment` | `lmassaron/FinancialPhraseBank` | default dataset loading | `sentence` | `label` | 3 |
| `movie_reviews` | `stanfordnlp/imdb` | default dataset loading | `text` | `label` | 2 |
| `product_reviews` | `SetFit/amazon_reviews_multi_en` | default dataset loading | `text` | `label` | 5 |
| `tweet_emotion` | `cardiffnlp/tweet_eval` | `emotion` | `text` | `label` | 4 |
| `tweet_hate` | `cardiffnlp/tweet_eval` | `hate` | `text` | `label` | 2 |
| `tweet_offensive` | `cardiffnlp/tweet_eval` | `offensive` | `text` | `label` | 2 |
| `tweet_irony` | `cardiffnlp/tweet_eval` | `irony` | `text` | `label` | 2 |
| `news_topic` | `fancyzhx/ag_news` | default dataset loading | `text` | `label` | 4 |
| `news_ynat` | `klue` | `ynat` | `title` | `label` | 7 |
| `movie_nsmc` | public TSV via `e9t/nsmc` | `direct="nsmc"` | `document` | `label` | 2 |
| `comment_kmhas_binary` | public TSV via `adlnlp/K-MHaS` | `direct="kmhas"` | `text` | `label` | 2 |

Dataset licensing and redistribution rights must be checked against the original dataset providers. This repository only contains code and aggregate outputs.

## Evaluation Metrics

Metrics are computed in `src/suite.py`, `compute_metrics()`.

| Metric | Meaning |
|---|---|
| `accuracy` | Fraction of correct predictions |
| `macro_f1` | Macro-averaged F1 across classes |
| `macro_precision` | Macro-averaged precision across classes |
| `macro_recall` | Macro-averaged recall across classes |
| `train_seconds` | Wall-clock training time measured per run |
| `trainable_parameter_ratio` | Trainable parameters divided by total model parameters |

The final aggregate table uses method-level means and standard deviations across the 5 seeds for each study/task/model/method combination.

## Result Files

### `final_tables/summary_by_task_model_method.csv`

This is the main aggregate results table. It has 110 rows:

```text
22 task/model combinations x 5 methods = 110 rows
```

Important columns include:

| Column | Meaning |
|---|---|
| `study` | Study group: `study1`, `study2`, or `study3` |
| `task` | Task key from `src/suite.py` |
| `model` | Base model name |
| `method` | Method key |
| `method_label` | Human-readable method name |
| `seeds` | Number of seeds included in the aggregation |
| `f1_mean`, `f1_sd` | Mean and standard deviation of Macro-F1 |
| `accuracy_mean` | Mean accuracy |
| `precision_mean`, `recall_mean` | Mean macro precision and macro recall |
| `train_seconds_mean`, `train_seconds_sd` | Mean and standard deviation of training time |
| `trainable_params_mean` | Mean number of trainable parameters |
| `trainable_ratio_mean` | Mean trainable parameter ratio |
| `train_rows`, `validation_rows`, `test_rows` | Split sizes used for the corresponding task/model/method row |
| `epochs` | Requested epoch count for the study |
| `delta_f1_vs_full` | Difference from the Full Fine-tuning Macro-F1 in the same task/model condition |
| `time_saving_vs_full_pct` | Training-time saving percentage against Full Fine-tuning in the same task/model condition |

### `final_tables/winners_by_metric.csv`

This table has 22 rows, one for each task/model combination. It records:

| Column | Meaning |
|---|---|
| `best_performance_method` | Method with the highest Macro-F1 |
| `best_performance_f1` | Macro-F1 for the best performance method |
| `most_efficient_method` | Method with the lowest trainable parameter ratio |
| `efficient_ratio` | Trainable parameter ratio for the most parameter-efficient method |
| `most_stable_method` | Method with the lowest seed-level Macro-F1 standard deviation |
| `stable_f1_sd` | Seed-level Macro-F1 standard deviation for the most stable method |

## Aggregate Findings in the Included Tables

The following summary statistics are computed from the included CSV files.

Performance winners by task/model combination:

| Method | Count |
|---|---:|
| Full Fine-tuning | 19 / 22 |
| IA3 | 2 / 22 |
| Adapter | 1 / 22 |
| LoRA | 0 / 22 |
| BitFit | 0 / 22 |

Mean Macro-F1 across the 22 task/model combinations:

| Method | Mean Macro-F1 |
|---|---:|
| Full Fine-tuning | 0.7588 |
| IA3 | 0.6702 |
| LoRA | 0.6537 |
| Adapter | 0.6506 |
| BitFit | 0.6443 |

Parameter-efficiency winner:

| Method | Count |
|---|---:|
| IA3 | 22 / 22 |

Stability winners by lowest seed-level Macro-F1 standard deviation:

| Method | Count |
|---|---:|
| BitFit | 10 / 22 |
| LoRA | 4 / 22 |
| IA3 | 4 / 22 |
| Full Fine-tuning | 2 / 22 |
| Adapter | 2 / 22 |

These results should be interpreted as task/model-specific comparisons, not as a universal claim that one method is best for every setting.

## Environment Record

The completed local run environment is recorded in `results/environment.json`.

| Field | Recorded value |
|---|---|
| Python | `3.10.19` |
| PyTorch | `2.11.0.dev20260119+cu128` |
| CUDA runtime | `12.8` |
| CUDA available | `true` |
| GPU | `NVIDIA GeForce RTX 5070 Ti` |
| GPU memory | `15.92 GB` |
| transformers | `5.9.0` |
| datasets | `4.8.5` |
| peft | `0.19.1` |

`requirements.txt` contains a minimal dependency list:

```text
torch
transformers>=4.45
datasets>=2.18
peft>=0.13
accelerate
scikit-learn
pandas
numpy
jupyter
```

Exact reproducibility may require matching CUDA, GPU, PyTorch, transformers, datasets, and peft versions more closely than the minimum requirements specify.

## Reproduction Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a lightweight dataset smoke check:

```bash
python tools/smoke_data.py
```

Use the notebooks for execution:

| Notebook | Role |
|---|---|
| `notebooks/00_precheck.ipynb` | Local environment and setup precheck |
| `notebooks/01_study1_bertweet_hate.ipynb` | Study 1 execution |
| `notebooks/02_study2_multitask.ipynb` | Study 2 execution |
| `notebooks/03_study3_korean.ipynb` | Study 3 execution |
| `notebooks/04_aggregate.ipynb` | Aggregation workflow |
| `notebooks/05_progress_monitor.ipynb` | Progress monitoring workflow |
| `colab_a100_full_550_runs.ipynb` | Colab-oriented full 550-run workflow |
| `colab_a100_low_drive_550_runs.ipynb` | Colab workflow variant intended to reduce Drive storage pressure |

Important: full reruns require substantial GPU time and access to the referenced external datasets. The included public package is intended to support reproducibility and inspection, not to store every generated artifact.

## Utility Scripts

| Script | Purpose |
|---|---|
| `tools/smoke_data.py` | Lightweight dataset loading/smoke test utility |
| `tools/check_progress.py` | Progress inspection utility for run outputs |
| `tools/report_dataset_sizes.py` | Dataset size reporting utility |
| `tools/create_notebooks.py` | Regenerates local workflow notebooks |
| `tools/create_colab_a100_notebook.py` | Regenerates the full Colab notebook |
| `tools/verify_colab_parity.py` | Checks parity assumptions for the Colab notebook |
| `tools/test_recovery.py` | Recovery/resume-related utility |
| `tools/validate_suite.py` | Legacy validation script from the fuller workspace |

Known limitation: `tools/validate_suite.py` still contains checks for HTML visualization files that are intentionally excluded from this lightweight package. Therefore, it should be treated as a legacy/full-workspace validation helper unless those HTML files are restored or the script is adjusted for the public package.

## What Is Intentionally Excluded

The following are excluded to keep this repository small and suitable for paper submission or public GitHub use:

- dataset caches
- model checkpoints
- model binary outputs
- per-run `predictions.csv`
- per-run `trainer_history.csv`
- per-run `events.jsonl`
- per-run `final_metrics.json`
- per-run `epoch_metrics.csv`
- the run-level `final_tables/all_runs_550.csv`
- generated HTML reports
- internal audit/process folders
- local notebook checkpoint directories
- Python cache directories

The excluded files are not needed to understand the aggregate results in `final_tables/`, but they may be needed for a complete archival replication package if a venue explicitly requests every run-level artifact.

## Limitations and Notes

- Optimizer and scheduler details should be interpreted through Hugging Face `TrainingArguments` and the installed transformers version unless explicitly set in `src/suite.py`.
- Dataset licenses are not restated in this repository; check the original dataset sources before redistributing data or derived artifacts.
- The included aggregate CSV files are the public evidence surface. They do not contain raw per-example predictions.
- Time comparisons should be interpreted within the same task/model condition. Different studies use different datasets, models, and epoch counts.
- Parameter efficiency and wall-clock training time are separate quantities in this study.

## Citation

If this repository is cited as supplementary material, replace the placeholder below with the final paper metadata.

```bibtex
@misc{fine_tuning_strategy_reproducibility,
  title        = {Fine-Tuning Strategy Reproducibility Package},
  author       = {확인 필요},
  year         = {2026},
  howpublished = {\url{https://github.com/LEESY-X/Domain_fine-tuning_experiments}},
  note         = {Supplementary reproducibility package for a fine-tuning strategy comparison study}
}
```

