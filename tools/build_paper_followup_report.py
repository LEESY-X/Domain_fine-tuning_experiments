from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.result_analysis import diagnose_public_summary, paired_difference_stats


PILOT_ROOT = ROOT / "results" / "followup" / "collapse_followup_v2_pilot"
FULL_ROOT = ROOT / "results" / "followup" / "collapse_followup_v2_full"
OUT = ROOT / "results_summary" / "paper_followup"
SEEDS = [42, 52, 62, 72, 82]
BASELINE = "baseline"
REMEDY = "higher_lr_weighted"

TASK_LABELS = {
    "finance_sentiment": "FinancialPhraseBank sentiment",
    "tweet_emotion": "TweetEval emotion",
}
METHOD_LABELS = {"adapter": "Adapter", "bitfit": "BitFit", "lora": "LoRA"}
VARIANT_LABELS = {
    "baseline": "Baseline (2 ep.)",
    "longer": "Longer (max 5 ep.)",
    "weighted": "Weighted (2 ep.)",
    "longer_weighted": "Longer + weighted (max 5 ep.)",
    "higher_lr": "Higher LR (2 ep.)",
    "higher_lr_weighted": "Higher LR + weighted (max 5 ep.)",
}
VARIANT_ORDER = list(VARIANT_LABELS)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required completed experiment artifact is missing: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def condition_label(row) -> str:
    return f"{TASK_LABELS[row['task']]} / {row['model']} / {METHOD_LABELS[row['method']]}"


def completed_epochs(source_file: str) -> float:
    metrics_path = Path(source_file)
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    history_path = metrics_path.parent / "trainer_history.csv"
    history = pd.read_csv(history_path, encoding="utf-8-sig")
    epochs = pd.to_numeric(history.get("epoch"), errors="coerce").dropna()
    if epochs.empty:
        raise ValueError(f"no epoch evidence in {history_path}")
    return float(epochs.max())


def markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" if index == 0 else "---:" for index in range(len(columns))) + "|"
    rows = [header, separator]
    for row in frame.to_dict("records"):
        rows.append("| " + " | ".join(str(row[field]) for field, _ in columns) + " |")
    return "\n".join(rows)


def build_original_table() -> pd.DataFrame:
    diagnosed = diagnose_public_summary(ROOT / "final_tables" / "summary_by_task_model_method.csv")
    diagnosed = diagnosed[diagnosed["f1_sd_exact_zero"]].copy()
    diagnosed["condition"] = diagnosed.apply(condition_label, axis=1)
    diagnosed["macro_f1_mean_sd"] = diagnosed.apply(lambda row: mean_sd(row["f1_mean"], row["f1_sd"]), axis=1)
    diagnosed["fingerprint"] = diagnosed["constant_prediction_fingerprint"].map({True: "Yes", False: "No"})
    diagnosed["interpretation"] = diagnosed["degenerate_stability"].map(
        {True: "Aggregate consistent with constant-class collapse", False: "Zero variance; cause unresolved"}
    )
    return diagnosed[[
        "condition", "task", "model", "method", "seeds", "num_labels",
        "accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "f1_sd",
        "macro_f1_mean_sd", "fingerprint", "interpretation",
    ]]


def build_pilot_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[
        (summary["task"] == "finance_sentiment")
        & (summary["model"] == "FacebookAI/roberta-base")
        & (summary["method"] == "adapter")
    ].copy()
    table["variant_order"] = table["variant"].map({name: index for index, name in enumerate(VARIANT_ORDER)})
    table = table.sort_values("variant_order")
    table["variant_label"] = table["variant"].map(VARIANT_LABELS)
    table["macro_f1_mean_sd"] = table.apply(lambda row: mean_sd(row["f1_mean"], row["f1_sd"]), axis=1)
    table["collapse_runs_display"] = table.apply(lambda row: f"{int(row['collapsed_runs'])}/{int(row['runs'])}", axis=1)
    table["full_coverage_display"] = table.apply(
        lambda row: f"{int(row['full_class_coverage_runs'])}/{int(row['runs'])}", axis=1
    )
    return table[[
        "variant", "variant_label", "epochs_requested", "f1_mean", "f1_sd", "macro_f1_mean_sd",
        "collapsed_runs", "collapse_runs_display", "full_class_coverage_runs", "full_coverage_display",
        "majority_prediction_rate_mean", "train_rows", "runs",
    ]]


def build_full_tables(all_runs: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_index = summary.set_index(["task", "model", "method", "variant"])
    comparisons = []
    seed_rows = []
    identity = ["task", "model", "method"]
    for keys, group in all_runs.groupby(identity, sort=True):
        pivot = group.pivot(index="seed", columns="variant", values="test_macro_f1").sort_index()
        if list(pivot.index.astype(int)) != SEEDS or set(pivot.columns) != {BASELINE, REMEDY}:
            raise ValueError(f"incomplete paired seed coverage for {keys}: index={list(pivot.index)}, columns={list(pivot.columns)}")
        stats = paired_difference_stats(pivot[BASELINE].to_numpy(), pivot[REMEDY].to_numpy())
        baseline = summary_index.loc[(*keys, BASELINE)]
        remedy = summary_index.loc[(*keys, REMEDY)]
        base_runs = group[group["variant"] == BASELINE].set_index("seed")
        remedy_runs = group[group["variant"] == REMEDY].set_index("seed")
        row = {
            "task": keys[0], "model": keys[1], "method": keys[2],
            "condition": condition_label({"task": keys[0], "model": keys[1], "method": keys[2]}),
            "baseline_mean": float(baseline["f1_mean"]), "baseline_sd": float(baseline["f1_sd"]),
            "baseline_mean_sd": mean_sd(baseline["f1_mean"], baseline["f1_sd"]),
            "baseline_collapse_runs": int(baseline["collapsed_runs"]),
            "baseline_near_constant_runs": int(baseline["near_constant_runs"]),
            "baseline_majority_prediction_rate_mean": float(baseline["majority_prediction_rate_mean"]),
            "baseline_epochs_completed_mean": float(base_runs["epochs_completed_derived"].mean()),
            "baseline_train_seconds_mean": float(baseline["train_seconds_mean"]),
            "remedy_mean": float(remedy["f1_mean"]), "remedy_sd": float(remedy["f1_sd"]),
            "remedy_mean_sd": mean_sd(remedy["f1_mean"], remedy["f1_sd"]),
            "remedy_collapse_runs": int(remedy["collapsed_runs"]),
            "remedy_near_constant_runs": int(remedy["near_constant_runs"]),
            "remedy_majority_prediction_rate_mean": float(remedy["majority_prediction_rate_mean"]),
            "remedy_epochs_completed_mean": float(remedy_runs["epochs_completed_derived"].mean()),
            "remedy_train_seconds_mean": float(remedy["train_seconds_mean"]),
            "remedy_full_class_coverage_runs": int(remedy["full_class_coverage_runs"]),
            "train_rows": int(remedy["train_rows"]), "test_rows": int(remedy["test_rows"]),
            **stats,
        }
        row["delta_ci95"] = f"{row['delta_mean']:+.4f} [{row['delta_ci95_low']:+.4f}, {row['delta_ci95_high']:+.4f}]"
        row["baseline_collapse_display"] = f"{row['baseline_collapse_runs']}/5"
        row["remedy_collapse_display"] = f"{row['remedy_collapse_runs']}/5"
        row["baseline_majority_display"] = f"{row['baseline_majority_prediction_rate_mean']:.3f}"
        row["remedy_majority_display"] = f"{row['remedy_majority_prediction_rate_mean']:.3f}"
        row["remedy_full_coverage_display"] = f"{row['remedy_full_class_coverage_runs']}/5"
        row["better_seeds_display"] = f"{row['treatment_better_seeds']}/5"
        row["sample_size_display"] = f"{row['train_rows']}/{row['test_rows']}"
        row["train_time_multiplier"] = row["remedy_train_seconds_mean"] / row["baseline_train_seconds_mean"]
        row["train_time_multiplier_display"] = f"{row['train_time_multiplier']:.2f}×"
        comparisons.append(row)

        for seed in SEEDS:
            seed_rows.append({
                "condition": row["condition"], "task": keys[0], "model": keys[1], "method": keys[2],
                "seed": seed,
                "baseline_f1": float(pivot.loc[seed, BASELINE]),
                "remedy_f1": float(pivot.loc[seed, REMEDY]),
                "delta_f1": float(pivot.loc[seed, REMEDY] - pivot.loc[seed, BASELINE]),
                "baseline_predicted_classes": int(base_runs.loc[seed, "predicted_class_count"]),
                "remedy_predicted_classes": int(remedy_runs.loc[seed, "predicted_class_count"]),
                "baseline_majority_prediction_rate": float(base_runs.loc[seed, "majority_prediction_rate"]),
                "remedy_majority_prediction_rate": float(remedy_runs.loc[seed, "majority_prediction_rate"]),
                "baseline_collapse": bool(base_runs.loc[seed, "constant_prediction_collapse"]),
                "remedy_collapse": bool(remedy_runs.loc[seed, "constant_prediction_collapse"]),
                "baseline_epochs_completed": float(base_runs.loc[seed, "epochs_completed_derived"]),
                "remedy_epochs_completed": float(remedy_runs.loc[seed, "epochs_completed_derived"]),
            })
    comparison_frame = pd.DataFrame(comparisons).sort_values(["task", "model", "method"]).reset_index(drop=True)
    return comparison_frame, pd.DataFrame(seed_rows)


def latex_escape(value: str) -> str:
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")


def build_latex(original: pd.DataFrame, pilot: pd.DataFrame, full: pd.DataFrame) -> str:
    original_rows = "\n".join(
        f"{latex_escape(row['condition'])} & ${row['f1_mean']:.4f} \\pm {row['f1_sd']:.4f}$ & {row['fingerprint']} \\\\"
        for row in original.to_dict("records")
    )
    pilot_rows = "\n".join(
        f"{latex_escape(row['variant_label'])} & ${row['f1_mean']:.4f} \\pm {row['f1_sd']:.4f}$ & {row['collapse_runs_display']} & {row['full_coverage_display']} \\\\"
        for row in pilot.to_dict("records")
    )
    full_rows = "\n".join(
        f"{latex_escape(row['condition'])} & ${row['baseline_mean']:.4f} \\pm {row['baseline_sd']:.4f}$ & ${row['remedy_mean']:.4f} \\pm {row['remedy_sd']:.4f}$ & {row['delta_mean']:+.4f} & {row['baseline_collapse_display']} & {row['remedy_collapse_display']} \\\\"
        for row in full.to_dict("records")
    )
    return f"""% Requires \\usepackage{{booktabs}}
% Values are Macro-F1 mean $\\pm$ sample SD across seeds 42, 52, 62, 72, 82.
\\begin{{table*}}[t]
\\centering\\small
\\caption{{Original exact-zero standard-deviation rows and collapse diagnosis.}}
\\label{{tab:zero-sd-diagnosis}}
\\begin{{tabular}}{{lcc}}
\\toprule
Condition & Macro-F1 & Constant-class fingerprint \\\\
\\midrule
{original_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}

\\begin{{table}}[t]
\\centering\\small
\\caption{{Pilot ablation for RoBERTa Adapter on FinancialPhraseBank (1,024 training examples).}}
\\label{{tab:pilot-ablation}}
\\begin{{tabular}}{{lccc}}
\\toprule
Variant & Macro-F1 & Collapse & Full coverage \\\\
\\midrule
{pilot_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{table*}}[t]
\\centering\\small
\\caption{{Full-data paired follow-up. Remedy combines up to five epochs with early stopping, a higher method-specific learning rate, and inverse-square-root class weighting.}}
\\label{{tab:full-collapse-followup}}
\\begin{{tabular}}{{lccrcc}}
\\toprule
Condition & Baseline & Remedy & $\\Delta$F1 & Base collapse & Remedy collapse \\\\
\\midrule
{full_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""


def source(source_id: str, label: str, path: str) -> dict:
    return {"id": source_id, "label": label, "path": path}


def query_source(
    source_id: str,
    label: str,
    path: str,
    sql: str,
    description: str,
    tables_used: list[str],
    generated_at: str,
    *,
    filters: list[str] | None = None,
    metric_definitions: list[str] | None = None,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "id": source_id,
            "engine": "sqlite",
            "sql": sql.strip(),
            "description": description,
            "tables_used": tables_used,
            "filters": filters or [],
            "metric_definitions": metric_definitions or [],
            "executed_at": generated_at,
        },
    }


def build_artifact(
    original: pd.DataFrame,
    pilot: pd.DataFrame,
    pilot_summary: pd.DataFrame,
    full: pd.DataFrame,
    generated_at: str,
) -> dict:
    short_condition_labels = {
        ("finance_sentiment", "FacebookAI/roberta-base", "adapter"): "Finance / Adapter",
        ("finance_sentiment", "FacebookAI/roberta-base", "bitfit"): "Finance / BitFit",
        ("tweet_emotion", "FacebookAI/roberta-base", "bitfit"): "Emotion / RoBERTa BitFit",
        ("tweet_emotion", "vinai/bertweet-base", "lora"): "Emotion / BERTweet LoRA",
    }
    full_chart_rows = []
    for row in full.to_dict("records"):
        condition_short = short_condition_labels[(row["task"], row["model"], row["method"])]
        for configuration, prefix in (("Baseline", "baseline"), ("Combined remedy", "remedy")):
            full_chart_rows.append({
                "condition_short": condition_short,
                "condition": row["condition"],
                "task": row["task"],
                "model": row["model"],
                "method": row["method"],
                "configuration": configuration,
                "macro_f1": row[f"{prefix}_mean"],
                "f1_sd": row[f"{prefix}_sd"],
                "collapse_runs": row[f"{prefix}_collapse_runs"],
                "majority_prediction_rate": row[f"{prefix}_majority_prediction_rate_mean"],
                "seed_count": 5,
                "train_rows": row["train_rows"],
                "test_rows": row["test_rows"],
            })
    headline_original_sql = """
SELECT COUNT(*) AS zero_rows
FROM original_zero_sd
"""
    headline_pilot_sql = """
SELECT COUNT(*) AS recovered_conditions
FROM pilot_summary AS baseline
JOIN pilot_summary AS remedy
  ON baseline.task = remedy.task
 AND baseline.model = remedy.model
 AND baseline.method = remedy.method
WHERE baseline.variant = 'baseline'
  AND remedy.variant = 'higher_lr_weighted'
  AND baseline.collapsed_runs > remedy.collapsed_runs
  AND remedy.f1_mean > baseline.f1_mean
"""
    headline_full_sql = """
SELECT
  SUM(CASE WHEN baseline_collapse_runs > remedy_collapse_runs AND delta_mean > 0 THEN 1 ELSE 0 END)
    AS recovered_conditions,
  SUM(CASE WHEN treatment_better_seeds = 5 THEN 1 ELSE 0 END)
    AS all_positive_conditions
FROM full_paired_comparison
"""
    original_table_sql = """
SELECT condition, macro_f1_mean_sd, fingerprint, interpretation
FROM original_zero_sd
ORDER BY condition
"""
    pilot_table_sql = """
SELECT variant_label, macro_f1_mean_sd, collapse_runs_display, full_coverage_display
FROM pilot_adapter_ablation
ORDER BY CASE variant
  WHEN 'baseline' THEN 0
  WHEN 'longer' THEN 1
  WHEN 'weighted' THEN 2
  WHEN 'longer_weighted' THEN 3
  WHEN 'higher_lr' THEN 4
  WHEN 'higher_lr_weighted' THEN 5
END
"""
    full_chart_sql = """
SELECT
  condition_short, condition, task, model, method, configuration,
  macro_f1, f1_sd, collapse_runs, majority_prediction_rate,
  seed_count, train_rows, test_rows
FROM full_chart_comparison
ORDER BY condition_short,
         CASE configuration WHEN 'Baseline' THEN 0 ELSE 1 END
"""
    full_table_sql = """
SELECT
  condition, baseline_mean_sd, remedy_mean_sd, delta_mean, delta_ci95,
  baseline_collapse_display, remedy_collapse_display, better_seeds_display,
  baseline_majority_display, remedy_majority_display, remedy_full_coverage_display,
  baseline_epochs_completed_mean, remedy_epochs_completed_mean,
  train_time_multiplier, train_time_multiplier_display,
  train_rows, test_rows, exact_sign_flip_p_two_sided
FROM full_paired_comparison
ORDER BY delta_mean DESC
"""

    def sql_rows(connection: sqlite3.Connection, query: str) -> list[dict]:
        return pd.read_sql_query(query, connection).to_dict("records")

    with sqlite3.connect(":memory:") as connection:
        original.to_sql("original_zero_sd", connection, index=False)
        pilot.to_sql("pilot_adapter_ablation", connection, index=False)
        pilot_summary.to_sql("pilot_summary", connection, index=False)
        full.to_sql("full_paired_comparison", connection, index=False)
        pd.DataFrame(full_chart_rows).to_sql("full_chart_comparison", connection, index=False)
        headline_original_rows = sql_rows(connection, headline_original_sql)
        headline_pilot_rows = sql_rows(connection, headline_pilot_sql)
        headline_full_rows = sql_rows(connection, headline_full_sql)
        original_rows = sql_rows(connection, original_table_sql)
        pilot_rows = sql_rows(connection, pilot_table_sql)
        full_chart_rows = sql_rows(connection, full_chart_sql)
        full_rows = sql_rows(connection, full_table_sql)

    recovered_conditions = int(headline_full_rows[0]["recovered_conditions"])
    all_positive_conditions = int(headline_full_rows[0]["all_positive_conditions"])
    sources = [
        source("original_summary", "Original 550-run aggregate summary", "final_tables/summary_by_task_model_method.csv"),
        source("pilot_runs", "Targeted pilot run-level results", "results/followup/collapse_followup_v2_pilot/aggregate/all_runs.csv"),
        source("full_runs", "Targeted full-data run-level results", "results/followup/collapse_followup_v2_full/aggregate/all_runs.csv"),
        source("followup_plan", "Follow-up experiment specification", "config/collapse_followup.json"),
        source("analysis_code", "Collapse diagnostics and exact paired statistics", "src/result_analysis.py"),
        source("training_code", "Training and weighted-loss implementation", "src/suite.py"),
        source("original_environment", "Original experiment environment", "results/environment.json"),
        source("followup_environment", "Full-data follow-up environment", "results/followup/collapse_followup_v2_full/environment.json"),
        source("followup_provenance", "Model revisions and dataset split fingerprints", "results/followup/provenance.json"),
        query_source(
            "original_headline_query", "Exact-zero row count query",
            "tools/build_paper_followup_report.py", headline_original_sql,
            "Counts the original aggregate rows diagnosed as exact-zero Macro-F1 SD.",
            ["original_zero_sd"], generated_at,
            metric_definitions=["zero_rows = COUNT(*) over the exact-zero diagnosis table"],
        ),
        query_source(
            "pilot_headline_query", "Pilot collapse-recovery query",
            "tools/build_paper_followup_report.py", headline_pilot_sql,
            "Counts paired pilot conditions where the combined remedy reduced collapsed runs and increased mean Macro-F1.",
            ["pilot_summary"], generated_at,
            filters=["baseline variant = baseline", "remedy variant = higher_lr_weighted"],
            metric_definitions=["recovered_conditions requires fewer collapsed runs and positive mean Macro-F1 change"],
        ),
        query_source(
            "full_headline_query", "Full-data recovery query",
            "tools/build_paper_followup_report.py", headline_full_sql,
            "Counts full-data conditions with collapse reduction and conditions improved in every paired seed.",
            ["full_paired_comparison"], generated_at,
            metric_definitions=[
                "recovered_conditions requires fewer collapsed runs and delta_mean > 0",
                "all_positive_conditions requires treatment_better_seeds = 5",
            ],
        ),
        query_source(
            "original_table_query", "Original zero-SD evidence table query",
            "tools/build_paper_followup_report.py", original_table_sql,
            "Selects the four exact-zero rows and their constant-class fingerprint diagnosis.",
            ["original_zero_sd"], generated_at,
        ),
        query_source(
            "pilot_table_query", "Pilot ablation evidence table query",
            "tools/build_paper_followup_report.py", pilot_table_sql,
            "Selects the six pilot Adapter ablation variants in experimental order.",
            ["pilot_adapter_ablation"], generated_at,
            filters=["task = finance_sentiment", "model = FacebookAI/roberta-base", "method = adapter"],
        ),
        query_source(
            "full_chart_query", "Full-data baseline-remedy chart query",
            "tools/build_paper_followup_report.py", full_chart_sql,
            "Selects long-form baseline and combined-remedy Macro-F1 rows for the four targeted conditions.",
            ["full_chart_comparison"], generated_at,
            metric_definitions=["macro_f1 = mean test Macro-F1 across seeds 42, 52, 62, 72, and 82"],
        ),
        query_source(
            "full_table_query", "Full-data paired comparison table query",
            "tools/build_paper_followup_report.py", full_table_sql,
            "Selects paired effect sizes, uncertainty, collapse diagnostics, seed direction, and compute cost.",
            ["full_paired_comparison"], generated_at,
            metric_definitions=[
                "delta_mean = mean paired remedy-minus-baseline Macro-F1",
                "delta_ci95 = paired t interval across five seed differences",
                "exact_sign_flip_p_two_sided = exact two-sided sign-flip p-value",
            ],
        ),
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "표준편차 0 진단 및 붕괴 후속 재학습 보고서",
            "description": "논문용 원본 진단, 파일럿 ablation, 전체 데이터 재학습 결과와 주장 한계를 정리한 기술 보고서.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "original_zero_rows", "description": "원본 110개 집계 행 중 Macro-F1 SD가 정확히 0인 행.",
                    "dataset": "headline_original", "sourceId": "original_headline_query",
                    "metrics": [{"label": "원본 exact-zero SD 행", "field": "zero_rows", "format": "number"}],
                },
                {
                    "id": "pilot_recovery", "description": "파일럿에서 결합 개선 조건이 constant-class collapse를 제거한 조건 수.",
                    "dataset": "headline_pilot", "sourceId": "pilot_headline_query",
                    "metrics": [{"label": "파일럿 붕괴 제거 조건", "field": "recovered_conditions", "format": "number"}],
                },
                {
                    "id": "full_recovery", "description": "전체 데이터에서 기준선보다 붕괴 run이 감소하고 평균 Macro-F1이 증가한 조건 수.",
                    "dataset": "headline_full", "sourceId": "full_headline_query",
                    "metrics": [
                        {"label": "전체 데이터 개선 조건", "field": "recovered_conditions", "format": "number"},
                        {"label": "5/5 seed 개선 조건", "field": "all_positive_conditions", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "full_f1_comparison",
                    "title": "Baseline and combined-remedy Macro-F1 by condition",
                    "subtitle": "The combined remedy was higher in every five-seed comparison; Table 3 reports exact uncertainty and collapse diagnostics.",
                    "type": "bar",
                    "intent": "comparison",
                    "question": "How did mean Macro-F1 change from the baseline to the combined remedy in each targeted condition?",
                    "rationale": "A grouped bar chart makes the paired configuration gap comparable across the four targeted conditions while preserving exact values in the adjacent table.",
                    "comparisonContext": {
                        "baseline": "Original two-epoch follow-up baseline",
                        "grain": "Task-model-method condition and configuration",
                        "unit": "Mean test Macro-F1 across five seeds",
                    },
                    "dataset": "full_chart_rows",
                    "sourceId": "full_chart_query",
                    "encodings": {
                        "x": {"field": "condition_short", "type": "nominal", "label": "Condition"},
                        "y": {"field": "macro_f1", "type": "quantitative", "label": "Mean Macro-F1", "format": "number"},
                        "color": {"field": "configuration", "type": "nominal", "label": "Configuration"},
                        "tooltip": [
                            {"field": "f1_sd", "type": "quantitative", "label": "Sample SD", "format": "number"},
                            {"field": "collapse_runs", "type": "quantitative", "label": "Collapsed runs", "format": "number"},
                            {"field": "majority_prediction_rate", "type": "quantitative", "label": "Mean majority-prediction rate", "format": "number"},
                            {"field": "seed_count", "type": "quantitative", "label": "Seeds", "format": "number"},
                        ],
                    },
                    "palette": {"kind": "categorical", "name": "baseline-vs-remedy"},
                    "legend": {"position": "bottom", "sort": "spec", "title": "Configuration"},
                    "labels": {"values": "all"},
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "original_table", "title": "원본 exact-zero SD는 수치 잘림이 아니라 붕괴 지문과 일치",
                    "subtitle": "전체 정밀도 값으로 확인한 네 조건. Macro recall=1/K 및 대응 precision/F1 관계를 동시에 확인했다.",
                    "dataset": "original_rows", "sourceId": "original_table_query",
                    "defaultSort": {"field": "condition", "direction": "asc"},
                    "columns": [
                        {"field": "condition", "label": "Condition", "type": "text"},
                        {"field": "macro_f1_mean_sd", "label": "Macro-F1 mean ± SD", "type": "text"},
                        {"field": "fingerprint", "label": "Collapse fingerprint", "type": "text"},
                        {"field": "interpretation", "label": "Interpretation", "type": "text"},
                    ],
                },
                {
                    "id": "pilot_table", "title": "파일럿 ablation에서는 세 요소를 결합한 조건만 Adapter 붕괴를 해소",
                    "subtitle": "FinancialPhraseBank, RoBERTa Adapter, train=1,024, seed 5개. 파일럿 결과는 전체 데이터 결론과 분리한다.",
                    "dataset": "pilot_rows", "sourceId": "pilot_table_query",
                    "defaultSort": {"field": "variant_label", "direction": "asc"},
                    "columns": [
                        {"field": "variant_label", "label": "Variant", "type": "text"},
                        {"field": "macro_f1_mean_sd", "label": "Macro-F1 mean ± SD", "type": "text"},
                        {"field": "collapse_runs_display", "label": "Collapsed runs", "type": "text"},
                        {"field": "full_coverage_display", "label": "Full-class coverage", "type": "text"},
                    ],
                },
                {
                    "id": "full_table", "title": "전체 데이터 paired 재학습 결과",
                    "subtitle": "동일 seed paired 비교. CI는 paired t interval, p는 2-sided exact sign-flip test다.",
                    "dataset": "full_rows", "sourceId": "full_table_query",
                    "defaultSort": {"field": "delta_mean", "direction": "desc"},
                    "columns": [
                        {"field": "condition", "label": "Condition", "type": "text"},
                        {"field": "train_rows", "label": "Train n", "format": "number"},
                        {"field": "test_rows", "label": "Test n", "format": "number"},
                        {"field": "baseline_mean_sd", "label": "Baseline F1", "type": "text"},
                        {"field": "remedy_mean_sd", "label": "Remedy F1", "type": "text"},
                        {"field": "delta_mean", "label": "ΔF1", "format": "number", "movement": True},
                        {"field": "baseline_collapse_display", "label": "Base collapse", "type": "text"},
                        {"field": "remedy_collapse_display", "label": "Remedy collapse", "type": "text"},
                        {"field": "baseline_majority_display", "label": "Base majority rate", "type": "text"},
                        {"field": "remedy_majority_display", "label": "Remedy majority rate", "type": "text"},
                        {"field": "remedy_epochs_completed_mean", "label": "Remedy epochs mean", "format": "number"},
                        {"field": "train_time_multiplier_display", "label": "Train-time multiplier", "type": "text"},
                        {"field": "remedy_full_coverage_display", "label": "Remedy full coverage", "type": "text"},
                        {"field": "better_seeds_display", "label": "Improved seeds", "type": "text"},
                        {"field": "exact_sign_flip_p_two_sided", "label": "Exact p", "format": "number"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 표준편차 0 진단 및 붕괴 후속 재학습 보고서"},
                {
                    "id": "technical_summary", "type": "markdown",
                    "body": "## 핵심 결론\n\n원본의 표준편차 0 네 행은 constant-class prediction 공식과 일치하는 aggregate fingerprint다. 원본 prediction 부재 때문에 과거 run의 실제 예측을 직접 확인한 것은 아니다. 후속 실험은 예측 클래스 수와 최빈 예측 비율을 함께 기록하여 현재 run의 정상적인 seed 안정성과 붕괴를 직접 구분한다.",
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["original_zero_rows", "pilot_recovery", "full_recovery"]},
                {"id": "original_heading", "type": "markdown", "sourceId": "original_summary", "body": "## 원본 집계의 4개 exact-zero 행은 constant-class 지문과 정확히 일치\n\n소수점 표시 문제가 아니다. 저장된 전체 정밀도에서 SD가 정확히 0이고, accuracy·macro precision·macro recall·macro F1의 조합이 단일 클래스 예측 공식과 일치한다."},
                {"id": "original_evidence", "type": "table", "tableId": "original_table", "layout": "full"},
                {"id": "pilot_heading", "type": "markdown", "sourceId": "pilot_runs", "body": "## 제한 데이터 파일럿의 탐색적 비교\n\n네 조건의 baseline 20개 run이 모두 constant-class collapse를 직접 보였다. 이는 원본 과거 실행의 동일 재현이 아니다. Adapter 비교에서는 epoch 연장, class weighting, 높은 learning rate를 단독 또는 일부 결합했을 때 붕괴가 남았고, 세 요소를 함께 적용한 조건에서만 5개 seed 모두 세 클래스를 예측했다. 실행 순서가 prospectively ordered ablation이 아니므로 사후 탐색으로 해석한다."},
                {"id": "pilot_evidence", "type": "table", "tableId": "pilot_table", "layout": "full"},
                {"id": "full_heading", "type": "markdown", "sourceId": "full_runs", "body": "## 전체 데이터 결과는 동일 seed paired 비교로 평가\n\n표의 평균과 표본 표준편차는 다섯 seed에서 계산했다. 개선 조건은 early stopping을 포함한 최대 5 epoch, 더 높은 method-specific learning rate, inverse-square-root-frequency class weighting을 동시에 사용한다."},
                {"id": "full_chart", "type": "chart", "chartId": "full_f1_comparison", "layout": "full"},
                {"id": "full_evidence", "type": "table", "tableId": "full_table", "layout": "full"},
                {"id": "design", "type": "markdown", "sourceId": "followup_plan", "body": "## 실험 설계와 metric 정의\n\n기준선은 원본 Study 2의 2-epoch, unweighted loss, method-specific learning rate 설정이다. 개선 조건은 최대 5 epoch와 기존 early stopping을 함께 사용한다. 붕괴는 테스트 예측 클래스 수가 1인 경우, near-collapse는 최빈 예측 비율이 0.98 이상인 경우로 operationally 정의했다. 주 지표는 테스트 Macro-F1이다."},
                {"id": "weighting", "type": "markdown", "sourceId": "training_code", "body": "### 결합 개선 조건 구현\n\nClass weight는 각 train label count n_c에 대해 n_c^{-1/2}를 계산한 뒤 평균이 1이 되도록 정규화하여 cross-entropy에 적용했다. Optimizer와 scheduler는 각각 adamw_torch와 linear로 명시했다."},
                {"id": "limitations_scope", "type": "markdown", "sourceId": "followup_plan", "body": "## 논문 해석 한계\n\n이번 후속 실험은 네 개의 문제 조건과 다섯 seed에 한정된 진단 실험이다. 결합 개선 조건은 세 개의 변경을 동시에 포함하므로, 전체 데이터 비교만으로 개별 요소의 인과 효과를 분리할 수 없다."},
                {"id": "limitations_inference", "type": "markdown", "sourceId": "analysis_code", "body": "### 소표본 추론 한계\n\n95% CI는 paired seed 차이의 t interval이다. n=5 exact sign-flip 검정의 가능한 최소 양측 p-value는 0.0625이므로 관례적 0.05 유의성을 주장하지 않는다. 네 조건의 p-value는 탐색적이며 multiplicity adjustment를 적용하지 않았다. 결과는 효과 크기, seed 방향 일관성, 붕괴율과 함께 기술해야 한다."},
                {"id": "original_runtime", "type": "markdown", "sourceId": "original_environment", "body": "### 원본 실행 환경과 증거 한계\n\n원본 집계는 CUDA GPU와 별도의 PyTorch build에서 생성되었다. 원본 run-level predictions가 공개 패키지에 없으므로 원본 붕괴 판정은 aggregate metric fingerprint에 근거하며, 현재 baseline과 비트 단위 동일성도 검증할 수 없다."},
                {"id": "followup_runtime", "type": "markdown", "sourceId": "followup_environment", "body": "### 후속 실행 환경\n\n후속 실험은 Apple MPS와 별도의 PyTorch build에서 실행했다. 따라서 원본 aggregate와 현재 baseline의 차이는 환경 차이를 포함하며, 개선 효과는 현재 환경 내부의 동일 seed paired 비교를 우선 해석한다."},
                {"id": "provenance", "type": "markdown", "sourceId": "followup_provenance", "body": "### 재현 provenance\n\n후속 결과에는 사용한 두 base model의 Hugging Face commit과 pilot/full dataset split fingerprint를 별도 저장했다. Split seed는 42로 고정했다."},
                {"id": "next_steps", "type": "markdown", "body": "## 권장 다음 실험\n\n전체 데이터에서 learning rate × class weighting × epoch의 factorial ablation을 수행하고, validation Macro-F1·prediction entropy·per-class recall을 checkpoint별로 저장한다. 그 결과가 확보되기 전까지는 결합 개선 조건이 붕괴를 완화했다는 기술적 사실까지만 주장한다."},
                {"id": "questions", "type": "markdown", "body": "## 남은 질문\n\n- CUDA 원 실행 환경에서 동일 후속 조건이 재현되는가?\n- BERTweet LoRA의 높은 seed 변동은 initialization, class imbalance, 또는 early stopping 중 무엇이 주도하는가?\n- 예측 붕괴를 validation 단계에서 자동 중단·재시작하는 기준은 어떤 threshold가 적절한가?"},
            ],
        },
        "snapshot": {
            "version": 1, "generatedAt": generated_at, "status": "ready",
            "datasets": {
                "headline_original": headline_original_rows,
                "headline_pilot": headline_pilot_rows,
                "headline_full": headline_full_rows,
                "original_rows": original_rows,
                "pilot_rows": pilot_rows,
                "full_chart_rows": full_chart_rows,
                "full_rows": full_rows,
            },
        },
        "sources": sources,
    }


def main() -> None:
    pilot_summary = read_csv(PILOT_ROOT / "aggregate" / "summary.csv")
    pilot_runs = read_csv(PILOT_ROOT / "aggregate" / "all_runs.csv")
    full_summary = read_csv(FULL_ROOT / "aggregate" / "summary.csv")
    full_runs = read_csv(FULL_ROOT / "aggregate" / "all_runs.csv")
    if len(pilot_runs) != 60:
        raise ValueError(f"expected 60 pilot runs, found {len(pilot_runs)}")
    if len(full_runs) != 40:
        raise ValueError(f"expected 40 full-data runs, found {len(full_runs)}")
    pilot_baseline = pilot_summary[pilot_summary["variant"] == BASELINE]
    pilot_remedy = pilot_summary[pilot_summary["variant"] == REMEDY]
    if len(pilot_baseline) != 4 or int(pilot_baseline["collapsed_runs"].sum()) != 20:
        raise ValueError("pilot baseline must contain four conditions and 20/20 collapsed runs")
    if len(pilot_remedy) != 4 or int(pilot_remedy["collapsed_runs"].sum()) != 0:
        raise ValueError("pilot remedy must contain four conditions and 0/20 collapsed runs")
    full_runs["epochs_completed_derived"] = (
        pd.to_numeric(full_runs["epochs_completed"], errors="raise")
        if "epochs_completed" in full_runs.columns
        else full_runs["source_file"].map(completed_epochs)
    )

    original = build_original_table()
    pilot = build_pilot_table(pilot_summary)
    full, per_seed = build_full_tables(full_runs, full_summary)
    OUT.mkdir(parents=True, exist_ok=True)
    original.to_csv(OUT / "table_1_original_zero_sd.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    pilot.to_csv(OUT / "table_2_pilot_ablation.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    full.to_csv(OUT / "table_3_full_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    per_seed.to_csv(OUT / "appendix_full_per_seed.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    (OUT / "paper_tables.tex").write_text(build_latex(original, pilot, full), encoding="utf-8")

    original_md = markdown_table(original, [
        ("condition", "Condition"), ("macro_f1_mean_sd", "Macro-F1 mean ± SD"),
        ("fingerprint", "Collapse fingerprint"), ("interpretation", "Interpretation"),
    ])
    pilot_md = markdown_table(pilot, [
        ("variant_label", "Variant"), ("macro_f1_mean_sd", "Macro-F1 mean ± SD"),
        ("collapse_runs_display", "Collapse"), ("full_coverage_display", "Full-class coverage"),
    ])
    full_md = markdown_table(full, [
        ("condition", "Condition"), ("sample_size_display", "Train/Test n"),
        ("baseline_mean_sd", "Baseline F1"), ("remedy_mean_sd", "Remedy F1"),
        ("delta_ci95", "Paired ΔF1 [95% CI]"), ("baseline_collapse_display", "Base collapse"),
        ("remedy_collapse_display", "Remedy collapse"), ("baseline_majority_display", "Base majority rate"),
        ("remedy_majority_display", "Remedy majority rate"), ("better_seeds_display", "Improved seeds"),
        ("remedy_full_coverage_display", "Remedy full coverage"),
        ("train_time_multiplier_display", "Train-time ×"),
        ("exact_sign_flip_p_two_sided", "Exact p"),
    ])
    recovered = int(((full["baseline_collapse_runs"] > full["remedy_collapse_runs"]) & (full["delta_mean"] > 0)).sum())
    trainer_devices = sorted(set(str(value) for value in full_runs["trainer_device"].dropna()))
    time_multiplier_min = float(full["train_time_multiplier"].min())
    time_multiplier_max = float(full["train_time_multiplier"].max())
    lora_full = full[(full["task"] == "tweet_emotion") & (full["method"] == "lora")].iloc[0]
    report_md = f"""# 표준편차 0 진단 및 붕괴 후속 재학습: 논문용 결과

## 결론

원본 결과의 Macro-F1 표준편차 0은 소수점 잘림이 아니다. 전체 정밀도 aggregate에서 네 행 모두 단일 클래스 예측의 accuracy·macro precision·macro recall·macro F1 관계와 정확히 일치한다. 따라서 이 네 행은 **constant-class collapse와 일치하는 aggregate 수준의 퇴화 안정성 증거**로 해석한다. 원본 run-level predictions가 없어 각 과거 run의 실제 예측을 직접 확인한 것은 아니다.

## Table 1. 원본 exact-zero SD 진단

{original_md}

## Table 2. 파일럿 ablation

FinancialPhraseBank/RoBERTa/Adapter, train=1,024. 표의 SD는 5개 seed의 표본 표준편차다.

{pilot_md}

## Table 3. 전체 데이터 paired 재학습

개선 조건은 early stopping을 포함한 최대 5 epoch + 높은 method-specific learning rate + inverse-square-root-frequency class weighting을 결합했다. 동일 seed끼리 paired 비교했다.
Class weight는 train label count를 n_c라 할 때 n_c^(-1/2)를 계산한 뒤 class 간 평균이 1이 되도록 정규화하여 cross-entropy에 적용했다.
Constant-class collapse는 테스트 예측 클래스 수가 1인 run, near-collapse는 최빈 예측 비율이 0.98 이상인 run으로 정의했다. 다만 threshold 주변 정보 손실을 피하기 위해 표와 appendix에는 최빈 예측 비율의 연속값도 함께 보존했다.
95% CI는 동일 seed의 Macro-F1 차이에 대한 paired t interval이며, p-value는 2-sided exact sign-flip test로 계산했다.

{full_md}

## 논문에서 명확히 말할 수 있는 사실

1. 원본 110개 집계 행 중 4개 행의 Macro-F1 SD는 저장된 전체 정밀도에서도 정확히 0이었다.
2. 네 행 모두 단일 클래스 예측의 metric fingerprint와 일치하므로, 해당 0은 반올림 결과가 아니며 seed 전반의 constant-class collapse와 일치하는 aggregate 증거다.
3. 1,024-example 후속 baseline 20/20 run은 constant-class collapse를 직접 보였다. 이는 제한 데이터 후속 관찰이며 원본 실행의 동일 재현은 아니다.
4. 파일럿의 결합 개선 조건에서는 20/20 run에서 constant-class collapse가 사라졌지만, BERTweet-LoRA emotion 조건은 일부 seed가 모든 클래스를 예측하지 않았고 SD도 상대적으로 컸다.
5. BERTweet-LoRA emotion의 파일럿 불안정성은 전체 데이터에서 다시 관찰되지 않았다. 전체 데이터 개선 조건은 `{lora_full['remedy_mean_sd']}`, full-class coverage {lora_full['remedy_full_coverage_display']}였다. 파일럿은 사후 탐색적 비교이며 최종 효과 추정치가 아니다.
6. 전체 데이터 paired 비교에서 붕괴 run 수가 줄고 평균 Macro-F1이 증가한 조건은 {recovered}/4개였다. 세부 효과 크기와 seed 일관성은 Table 3에 제시했다.
7. n=5에서는 모든 paired 차이가 같은 방향이어도 exact two-sided sign-flip p의 최소값이 0.0625이므로, 이번 후속 결과만으로 p<0.05의 통계적 유의성을 주장할 수 없다.
8. 원본은 CUDA/RTX 환경, 후속은 Apple MPS 환경이므로 원본 aggregate와 현재 baseline의 차이는 cross-environment replication 차이를 포함한다. 개선 조건의 효과는 후속 환경 내부 paired 비교로 해석해야 한다.
9. 네 조건의 p-value는 탐색적이며 multiple-comparison adjustment를 적용하지 않았다.
10. 결합 개선 조건의 평균 학습시간은 기준선의 {time_multiplier_min:.2f}–{time_multiplier_max:.2f}배로, 붕괴 완화에는 명확한 계산 비용 증가가 동반됐다.

## 논문에 그대로 사용할 수 있는 한국어 문장

“초기 집계 결과에서 네 개의 task–model–method 조합은 Macro-F1 표준편차가 정확히 0이었다. 전체 정밀도 지표를 재검토한 결과, 이 값들은 단일 클래스 예측에서 유도되는 macro precision, recall, F1 관계와 정확히 일치하였다. 따라서 네 행은 반올림에 의한 0이나 바람직한 seed 안정성보다 constant-class prediction과 일치하는 aggregate 증거로 해석한다. 원본 run-level prediction이 없으므로 모든 과거 seed의 실제 예측을 직접 확인한 것은 아니다.”

“후속 실험에서는 예측 클래스 수, 최빈 예측 비율, normalized prediction entropy를 run별로 저장하였다. 또한 동일한 5개 seed에서 원본 2-epoch 설정과 early stopping을 포함한 최대 5 epochs, 상향 learning rate, inverse-square-root-frequency class weighting을 결합한 설정을 paired 비교하였다. Table 3은 평균 Macro-F1의 변화와 함께 붕괴 run 수를 보고하므로, 성능 안정성과 퇴화 안정성을 구분한다.”

## English paper-ready wording

“Four task–model–method combinations in the original aggregate exhibited an exactly zero standard deviation of Macro-F1 across five reported seeds. Re-evaluation at full stored precision showed that their accuracy, macro-precision, macro-recall, and Macro-F1 jointly matched the analytical fingerprint of constant-class prediction. We therefore treat these rows as aggregate evidence consistent with degenerate constant-class stability, rather than as direct proof of the historical predictions.”

“The follow-up recorded the number of predicted classes, majority-prediction rate, and normalized prediction entropy for every run. We paired the original two-epoch configuration with a combined remedy using up to five epochs with early stopping, a higher method-specific learning rate, and inverse-square-root-frequency class weighting under the same five seeds. Because five pairs permit a minimum two-sided exact sign-flip p-value of 0.0625, we report effect sizes, seed-wise direction, and collapse counts without claiming conventional statistical significance.”

## 주장하면 안 되는 내용

- “표준편차가 0이므로 가장 안정적이다.” 붕괴 안정성을 정상 안정성으로 오해한다.
- “class weighting 하나가 붕괴의 원인이자 해결책이다.” 전체 데이터 개선 조건은 세 변경을 결합했으며 개별 인과 효과가 분리되지 않았다.
- “모든 PEFT와 모든 task에 일반화된다.” 후속 범위는 원본에서 문제가 발견된 네 조건뿐이다.
- “통계적으로 유의하다(p<0.05).” n=5 exact paired test로는 그 결론에 도달할 수 없다.

## 재현 정보

- Seeds: `{SEEDS}`
- Full-data Trainer devices: `{trainer_devices}`
- Pinned top-level follow-up dependency versions: `requirements-followup.txt` (not a complete transitive or OS lock)
- Model revisions and dataset fingerprints: `results/followup/provenance.json`
- 원본 진단: `tools/analyze_result_variance.py`
- 후속 실행: `tools/run_collapse_followup.py`
- 논문 표 생성: `tools/build_paper_followup_report.py`
- 원시 run 결과: `results/followup/collapse_followup_v2_*/**/final_metrics.json`, `predictions.csv`
"""
    (OUT / "paper_followup_report.md").write_text(report_md, encoding="utf-8")

    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    artifact = build_artifact(original, pilot, pilot_summary, full, generated_at)
    (OUT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PAPER REPORT SOURCES PASS original={len(original)} pilot={len(pilot_runs)} full={len(full_runs)}")
    print(OUT)


if __name__ == "__main__":
    main()
