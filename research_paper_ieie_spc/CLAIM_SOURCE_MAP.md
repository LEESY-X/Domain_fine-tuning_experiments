# Integrated Manuscript Claim-to-Evidence Map

This ledger distinguishes configured historical scope, calculations regenerated from a preserved aggregate snapshot, direct local follow-up observations, and interpretation. `scripts/generate_figures.py` deterministically recomputes downstream benchmark values from the copied 110-row snapshot; it does not recreate the missing 550 historical run records. `scripts/validate_followup_predictions.py` independently checks the 100 locally saved follow-up prediction files.

## Original broad benchmark

| Manuscript claim | Value | Generated evidence | Boundary |
|---|---:|---|---|
| Historical design | 13 tasks; 22 task--model conditions; 5 methods; 5 seeds; 550 configured | `data/original_aggregate_110_rows.csv`, `../config/experiment_config.json` | Run-level historical artifacts absent; not called 550 independently verified completions |
| Full FT equal-condition mean/wins | 0.7588; 19/22 | `data/method_summary.csv` | Conditions weighted equally, not by sample count |
| LoRA mean/wins | 0.6537; 0/22 | `data/method_summary.csv` | Fixed historical policy |
| Adapter mean/wins | 0.6506; 1/22 | `data/method_summary.csv` | Local win is KLUE YNAT |
| IA3 mean/wins | 0.6702; 2/22 | `data/method_summary.csv` | Local wins are both TweetEval hate conditions |
| BitFit fastest | 22/22; 15.56% mean saving; 1.188× geometric speedup | `data/method_summary.csv` | `trainer.train()` wall time, not loading/prediction/energy |
| IA3 smallest trainable state | 22/22; 0.4869% mean ratio | `data/method_summary.csv` | Trainable ratio, not peak memory or final artifact size |
| Aggregate-implied historical training time | 47.49 h | `../final_tables/summary_by_task_model_method.csv` | Reconstructed as sum of each mean time × five reported seeds; no raw timestamp sum |
| Original task blocks | 13 | `data/original_task_blocks_13.csv` | Study-2 backbones averaged within task; blocks treated as exchangeable, not proven independent |
| Macro-F1 global test | Q=24.80; p=5.52e-5; W=.4769; Iman F=10.94 | `data/legacy_friedman_metrics.csv` | Complete-block task-level inference |
| Time global test | Q=48.31; p=8.14e-10; W=.9290 | `data/legacy_friedman_metrics.csv` | Lower time ranks better |
| Trainable-ratio global test | Q=52; p=1.38e-10; W=1 | `data/legacy_friedman_metrics.csv` | Lower ratio ranks better |
| F1-SD global test | Q=13.23; p=.0102; W=.2544 | `data/legacy_friedman_metrics.csv` | Reported but not interpreted as reliability because of collapse fingerprints |
| Full FT pairwise comparisons | Four Holm p values ≤.0242 in original 13 tasks | `data/pairwise_full_ft_analyses.csv` | Fixed-policy summary retaining all configured condition-level outcomes; collapse-task sensitive |

## Sensitivity and interaction claims

| Manuscript claim | Value | Generated evidence | Boundary |
|---|---:|---|---|
| Excluding Study 1 | n=12; Q=21.67; p=2.33e-4; W=.451; all four Holm decisions retained | `data/f1_sensitivity_analyses.csv`, `data/pairwise_full_ft_analyses.csv` | Addresses unverifiable comment-group separation, not every leakage mechanism |
| Excluding collapse tasks | n=11; Q=22.62; p=1.51e-4; W=.514; only Adapter Holm p<.05 | same files | Removes complete Finance and Emotion task blocks |
| Strict exclusion | n=10; Q=19.12; p=7.44e-4; W=.478; only Adapter Holm p<.05 | same files | Excludes Study 1 and the two collapse tasks |
| Adapter within .02 of Full FT on three Korean tasks | 3/3 | `data/original_aggregate_110_rows.csv` | Descriptive; tasks/model/epochs confounded with language |
| BERTweet vs RoBERTa task direction | BERTweet 5/9, RoBERTa 4/9 after method averaging | `data/study2_model_method_summary.csv`, original snapshot | Descriptive backbone/domain interaction |

## Exact-zero audit

| Manuscript claim | Value | Evidence | Boundary |
|---|---:|---|---|
| Exact-zero SD rows | 4/110 | `data/table_1_original_zero_sd.csv` | Exact in stored aggregate, not display rounding |
| Fingerprint relationship | 4/4 jointly match constant-class equations | same CSV, `../src/result_analysis.py` | Historical predictions absent; aggregate-consistent, not directly observed |
| Raw co-lowest-SD count | BitFit 10/22; 23 co-winner assignments in 22 conditions | `data/stability_winner_sensitivity.csv` | Variance-only definition; Finance/RoBERTa has an Adapter/BitFit tie |
| After four-row exclusion | BitFit 8, IA3 5, Full FT 3, LoRA 3, Adapter 3 | same CSV | Tied fingerprint rows removed; does not certify every remaining row as non-collapsed |

## Follow-up

| Manuscript claim | Value | Evidence | Boundary |
|---|---:|---|---|
| Follow-up scope | pilot 60 + full 40 = 100 complete runs | public compact `../results/followup/collapse_followup_v2_*/aggregate/all_runs.csv`; local status files | Compact metric rows released in tag `ieie-spc-v1.0.0`; per-run JSON and predictions remain local |
| Prediction audit | 100 files; zero numeric discrepancies | `scripts/validate_followup_predictions.py` | Recomputes metrics and prediction diagnostics from locally saved label/prediction pairs |
| Pilot baseline collapse | 20/20 | pilot aggregate and run predictions | Direct under limited-data follow-up settings, not identical historical replay |
| Exploratory pilot result | combined remedy 0.8040±.0063; 0/5 collapse; 5/5 full coverage | `data/table_2_pilot_ablation.csv` | Post hoc comparison; combined configuration began before several single/partial variants; two of three possible two-way combinations untested |
| Finance/RoBERTa/Adapter | +.5727 [.5239,.6215]; collapse 2/5→0/5; 2.34× time | `data/table_3_full_comparison.csv` | Paired-t CI, n=5; Apple MPS |
| Finance/RoBERTa/BitFit | +.5866 [.5760,.5972]; 5/5→0/5; 2.32× | same | Same boundary |
| Emotion/RoBERTa/BitFit | +.6236 [.6167,.6306]; 5/5→0/5; 2.56× | same | Same boundary |
| Emotion/BERTweet/LoRA | +.6430 [.6384,.6475]; 5/5→0/5; 2.52× | same | Same boundary |
| Direction across seeds | remedy higher in 20/20 pairs | `data/appendix_full_per_seed.csv` | Descriptive directional consistency |
| Exact sign-flip | p=.0625 in every condition | `data/table_3_full_comparison.csv` | Minimum two-sided value with five nonzero pairs; no p<.05 claim |
| Baseline prediction failure | 17/20 constant; 18/20 near-collapse | per-seed CSV and prediction files | Near-collapse threshold .98 |
| Remedy prediction behavior | 0/20 constant; 0/20 near-collapse; 20/20 full class coverage | same | Four post-selected conditions only |
| Historical comparability | three baseline aggregates match; Finance/Adapter differs | original snapshot and full comparison | Historical CUDA vs follow-up MPS; numeric comparison only, not replication or a causal contrast |

## Artifact provenance states

| State | Contents | Availability boundary |
|---|---|---|
| Existing public snapshot | Commit `539e3d3`: original 110-row aggregate and 22-row winner table | Public and hash-matched; no historical per-run records |
| Tagged code and compact evidence | Tag `ieie-spc-v1.0.0`: maintained code/configurations, original aggregate, 60+40 compact metric rows, environment/provenance, release manifest and checksums | Public versioned snapshot; not an archival DOI and not the exact executed source snapshot |
| Current local manuscript package | Integrated manuscript, copied aggregate, regenerated downstream tables/figures, follow-up summary tables, and validation scripts | Local worktree pending author metadata and declaration approval |
| Local run-level follow-up evidence | 100 prediction, metric, history, configuration, and status artifacts | Validated locally; intentionally excluded from the public tag |
| Future archival record | Author-approved final source/PDF and any permitted supplement | Add an archive DOI only after one is actually assigned |

## Reproducibility boundaries retained

- Study 1 used annotator-row splitting after discarding the source comment identifier; historical group separation cannot be reconstructed.
- The original run-level predictions, run configurations, checkpoints, exact optimizer/scheduler evidence, package patch versions, model revisions, and dataset revisions are absent.
- The executed follow-up suite hash differs from current `src/suite.py`, and the executed source snapshot was not preserved.
- Compact metric-level rows for the 100 follow-up runs are public in tag `ieie-spc-v1.0.0`; per-example predictions and other detailed run artifacts remain local and must not be described as public.
- The current suite rejects mismatched cached-result/checkpoint signatures and strict final aggregates with missing or duplicate seeds. These safeguards protect future runs; they do not retroactively prove the missing historical run lineage.
- Remedy epochs, learning rate, and class weighting change together.
- Post-selected remedy cells are never inserted into or used to rerank the broad benchmark.
- Macro-F1, trainable ratio, and training-call time are not treated as calibration, latency, memory, energy, or robustness measures.
