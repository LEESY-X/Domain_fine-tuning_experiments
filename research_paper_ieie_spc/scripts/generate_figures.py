#!/usr/bin/env python3
"""Regenerate every numeric manuscript artifact from preserved result tables.

This script performs no training.  It copies the immutable inputs needed by the
paper, reconstructs the original task-level complete blocks, runs the reported
non-parametric analyses, and creates the publication figures.  The regenerated
tables intentionally keep the broad benchmark and the targeted follow-up
separate: the post-selected remedy cells are never substituted into the
original 110-row comparison.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PAPER_DIR.parent
FIGURE_DIR = PAPER_DIR / "figures"
DATA_DIR = PAPER_DIR / "data"

METHOD_SOURCE = REPO_DIR / "final_tables" / "summary_by_task_model_method.csv"
FOLLOWUP_DIR = REPO_DIR / "results_summary" / "paper_followup"

SOURCE_SNAPSHOTS = (
    "table_1_original_zero_sd.csv",
    "table_2_pilot_ablation.csv",
    "table_3_full_comparison.csv",
    "appendix_full_per_seed.csv",
)

ANALYSIS_ORDER = ["full_ft", "lora", "adapter", "ia3", "bitfit"]
PLOT_ORDER = ["full_ft", "ia3", "lora", "adapter", "bitfit"]
METHOD_LABELS = {
    "full_ft": "Full FT",
    "ia3": "IA3",
    "lora": "LoRA",
    "adapter": "Adapter",
    "bitfit": "BitFit",
}

VARIANTS = {
    "legacy_13_tasks": lambda x: x,
    "exclude_group_split_task_12": lambda x: x[x["study"] != "study1"],
    "exclude_collapse_tasks_11": lambda x: x[
        ~x["task"].isin(["finance_sentiment", "tweet_emotion"])
    ],
    "strict_10_tasks": lambda x: x[
        (x["study"] != "study1")
        & ~x["task"].isin(["finance_sentiment", "tweet_emotion"])
    ],
}

CONDITION_LABELS = {
    "FinancialPhraseBank sentiment / FacebookAI/roberta-base / Adapter": "Finance / RoBERTa\nAdapter",
    "FinancialPhraseBank sentiment / FacebookAI/roberta-base / BitFit": "Finance / RoBERTa\nBitFit",
    "TweetEval emotion / FacebookAI/roberta-base / BitFit": "Emotion / RoBERTa\nBitFit",
    "TweetEval emotion / vinai/bertweet-base / LoRA": "Emotion / BERTweet\nLoRA",
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#6F6F6F"
LIGHT_GRAY = "#D9D9D9"
METHOD_COLORS = {
    "full_ft": BLUE,
    "ia3": GREEN,
    "lora": "#56B4E9",
    "adapter": PURPLE,
    "bitfit": "#CC79A7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def task_blocks(frame: pd.DataFrame) -> pd.DataFrame:
    """Average the two Study-2 backbones within task before inference."""
    return (
        frame.groupby(["study", "task", "method"], as_index=False)
        .agg(
            f1_mean=("f1_mean", "mean"),
            f1_sd=("f1_sd", "mean"),
            train_seconds_mean=("train_seconds_mean", "mean"),
            trainable_ratio_mean=("trainable_ratio_mean", "mean"),
        )
        .sort_values(["study", "task", "method"])
    )


def holm_adjust(p_values: list[float]) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def hodges_lehmann(differences: np.ndarray) -> float:
    walsh = [
        (differences[i] + differences[j]) / 2
        for i in range(len(differences))
        for j in range(i, len(differences))
    ]
    return float(np.median(walsh))


def matched_rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def percentile_bootstrap_mean(
    differences: np.ndarray, rng: np.random.Generator, draws: int = 50_000
) -> tuple[float, float]:
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    means = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def friedman_summary(blocks: pd.DataFrame, metric: str) -> dict[str, float]:
    pivot = blocks.pivot(index=["study", "task"], columns="method", values=metric)
    pivot = pivot[ANALYSIS_ORDER].dropna()
    arrays = [pivot[method].to_numpy() for method in ANALYSIS_ORDER]
    q_value, p_value = stats.friedmanchisquare(*arrays)
    n_blocks = len(pivot)
    n_methods = len(ANALYSIS_ORDER)
    kendall_w = q_value / (n_blocks * (n_methods - 1))
    denominator = n_blocks * (n_methods - 1) - q_value
    iman_f = math.inf if denominator == 0 else (n_blocks - 1) * q_value / denominator
    iman_p = 0.0 if math.isinf(iman_f) else float(
        stats.f.sf(iman_f, n_methods - 1, (n_methods - 1) * (n_blocks - 1))
    )
    ranks = pivot.rank(axis=1, ascending=(metric != "f1_mean"), method="average")
    output: dict[str, float] = {
        "task_blocks": n_blocks,
        "friedman_q": float(q_value),
        "friedman_p": float(p_value),
        "kendall_w": float(kendall_w),
        "iman_davenport_f": float(iman_f),
        "iman_davenport_p": iman_p,
    }
    for method in ANALYSIS_ORDER:
        output[f"mean_rank_{method}"] = float(ranks[method].mean())
    return output


def generate_benchmark_tables(frame: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(METHOD_SOURCE, DATA_DIR / "original_aggregate_110_rows.csv")

    key = ["study", "task", "model"]
    full_time = (
        frame[frame["method"] == "full_ft"][key + ["train_seconds_mean"]]
        .rename(columns={"train_seconds_mean": "full_train_seconds"})
    )
    augmented = frame.merge(full_time, on=key, validate="many_to_one")
    augmented["speedup_vs_full"] = (
        augmented["full_train_seconds"] / augmented["train_seconds_mean"]
    )

    winner_rows = frame.loc[frame.groupby(key)["f1_mean"].idxmax()]
    fastest_rows = frame.loc[frame.groupby(key)["train_seconds_mean"].idxmin()]
    smallest_rows = frame.loc[frame.groupby(key)["trainable_ratio_mean"].idxmin()]
    wins = winner_rows.groupby("method").size()
    fastest = fastest_rows.groupby("method").size()
    smallest = smallest_rows.groupby("method").size()

    summaries = []
    for method in PLOT_ORDER:
        subset = augmented[augmented["method"] == method]
        summaries.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "task_model_conditions": len(subset),
                "macro_f1_mean": subset["f1_mean"].mean(),
                "macro_f1_sd_mean": subset["f1_sd"].mean(),
                "macro_f1_wins": int(wins.get(method, 0)),
                "time_saving_vs_full_pct_mean": subset[
                    "time_saving_vs_full_pct"
                ].mean(),
                "geometric_speedup_vs_full": float(
                    stats.gmean(subset["speedup_vs_full"])
                ),
                "fastest_conditions": int(fastest.get(method, 0)),
                "trainable_ratio_mean": subset["trainable_ratio_mean"].mean(),
                "smallest_ratio_conditions": int(smallest.get(method, 0)),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(DATA_DIR / "method_summary.csv", index=False)

    blocks_all = task_blocks(frame)
    blocks_all.to_csv(DATA_DIR / "original_task_blocks_13.csv", index=False)

    metric_rows = []
    for metric in (
        "f1_mean",
        "train_seconds_mean",
        "trainable_ratio_mean",
        "f1_sd",
    ):
        row = {"analysis": "legacy_13_tasks", "metric": metric}
        row.update(friedman_summary(blocks_all, metric))
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(
        DATA_DIR / "legacy_friedman_metrics.csv", index=False
    )

    sensitivity_rows = []
    pairwise_rows = []
    for variant_name, selector in VARIANTS.items():
        selected = selector(frame.copy())
        blocks = task_blocks(selected)
        result = {"analysis": variant_name, "metric": "f1_mean"}
        result.update(friedman_summary(blocks, "f1_mean"))
        sensitivity_rows.append(result)

        pivot = blocks.pivot(
            index=["study", "task"], columns="method", values="f1_mean"
        )[ANALYSIS_ORDER].dropna()
        raw_p = []
        pending = []
        for method in ANALYSIS_ORDER[1:]:
            differences = (pivot[method] - pivot["full_ft"]).to_numpy()
            wilcoxon = stats.wilcoxon(
                differences,
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="exact",
            )
            # Match the preserved analysis convention: the documented seed is
            # restarted for each method/analysis rather than consumed globally.
            rng = np.random.default_rng(20_260_718)
            ci_low, ci_high = percentile_bootstrap_mean(differences, rng)
            raw_p.append(float(wilcoxon.pvalue))
            pending.append(
                {
                    "analysis": variant_name,
                    "method": method,
                    "task_blocks": len(differences),
                    "mean_delta_f1_vs_full": differences.mean(),
                    "bootstrap_95_ci_low": ci_low,
                    "bootstrap_95_ci_high": ci_high,
                    "median_delta_f1_vs_full": np.median(differences),
                    "hodges_lehmann": hodges_lehmann(differences),
                    "wilcoxon_w": float(wilcoxon.statistic),
                    "wilcoxon_exact_p": float(wilcoxon.pvalue),
                    "matched_rank_biserial": matched_rank_biserial(differences),
                    "wins": int((differences > 0).sum()),
                    "losses": int((differences < 0).sum()),
                    "ties": int((differences == 0).sum()),
                }
            )
        for row, adjusted in zip(pending, holm_adjust(raw_p)):
            row["holm_adjusted_p"] = adjusted
            pairwise_rows.append(row)

    pd.DataFrame(sensitivity_rows).to_csv(
        DATA_DIR / "f1_sensitivity_analyses.csv", index=False
    )
    pd.DataFrame(pairwise_rows).to_csv(
        DATA_DIR / "pairwise_full_ft_analyses.csv", index=False
    )

    study_summary = (
        frame.groupby(["study", "method"], as_index=False)
        .agg(
            task_model_conditions=("f1_mean", "size"),
            macro_f1_mean=("f1_mean", "mean"),
            time_saving_vs_full_pct_mean=("time_saving_vs_full_pct", "mean"),
            trainable_ratio_mean=("trainable_ratio_mean", "mean"),
        )
    )
    study_summary["status"] = np.where(
        study_summary["study"] == "study1",
        "provisional_group_split_unverified",
        "primary",
    )
    study_summary.to_csv(DATA_DIR / "study_method_summary.csv", index=False)

    study2_model = (
        frame[frame["study"] == "study2"]
        .groupby(["model", "method"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            macro_f1_mean=("f1_mean", "mean"),
            train_seconds_mean=("train_seconds_mean", "mean"),
            trainable_ratio_mean=("trainable_ratio_mean", "mean"),
        )
    )
    study2_model.to_csv(DATA_DIR / "study2_model_method_summary.csv", index=False)

    fingerprint_keys = {
        ("study2", "finance_sentiment", "FacebookAI/roberta-base", "adapter"),
        ("study2", "finance_sentiment", "FacebookAI/roberta-base", "bitfit"),
        ("study2", "tweet_emotion", "FacebookAI/roberta-base", "bitfit"),
        ("study2", "tweet_emotion", "vinai/bertweet-base", "lora"),
    }
    keep_mask = [
        (row.study, row.task, row.model, row.method) not in fingerprint_keys
        for row in frame.itertuples(index=False)
    ]
    filtered = frame.loc[keep_mask]
    raw_minimum = frame.groupby(key)["f1_sd"].transform("min")
    filtered_minimum = filtered.groupby(key)["f1_sd"].transform("min")
    raw_stability = frame.loc[np.isclose(frame["f1_sd"], raw_minimum, rtol=0.0, atol=1e-15)]
    filtered_stability = filtered.loc[
        np.isclose(filtered["f1_sd"], filtered_minimum, rtol=0.0, atol=1e-15)
    ]
    stability_rows = []
    for method in ANALYSIS_ORDER:
        stability_rows.append(
            {
                "method": method,
                "raw_co_lowest_sd_conditions": int(
                    (raw_stability["method"] == method).sum()
                ),
                "co_lowest_sd_after_fingerprint_exclusion": int(
                    (filtered_stability["method"] == method).sum()
                ),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        DATA_DIR / "stability_winner_sensitivity.csv", index=False
    )
    return summary


def snapshot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_SNAPSHOTS:
        shutil.copy2(FOLLOWUP_DIR / filename, DATA_DIR / filename)
    frame = pd.read_csv(METHOD_SOURCE, encoding="utf-8-sig")
    return generate_benchmark_tables(frame), frame


def plot_method_overview(summary: pd.DataFrame) -> None:
    plot_frame = summary.set_index("method").loc[PLOT_ORDER[::-1]].reset_index()
    colors = [METHOD_COLORS[method] for method in plot_frame["method"]]

    fig, ax = plt.subplots(figsize=(6.9, 2.85))
    bars = ax.barh(
        plot_frame["method_label"],
        plot_frame["macro_f1_mean"],
        color=colors,
        edgecolor="white",
        height=0.68,
    )
    # A zero baseline avoids visually exaggerating the relatively small gaps.
    ax.set_xlim(0.0, 0.83)
    ax.set_xlabel("Macro-F1 averaged equally over 22 task-model conditions")
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.7)
    ax.set_axisbelow(True)

    for bar, row in zip(bars, plot_frame.itertuples(index=False)):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row.macro_f1_mean:.4f}  ({row.macro_f1_wins}/22 wins)",
            va="center",
            ha="left",
            fontsize=8.2,
        )

    fig.tight_layout()
    save_figure(fig, "method_overview")


def plot_performance_efficiency(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.45))
    for row in summary.itertuples(index=False):
        ax.scatter(
            row.time_saving_vs_full_pct_mean,
            row.macro_f1_mean,
            s=82,
            color=METHOD_COLORS[row.method],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        x_offset, y_offset = {
            "full_ft": (0.5, -0.006),
            "ia3": (-0.7, 0.010),
            "lora": (0.7, -0.012),
            "adapter": (-2.8, -0.011),
            "bitfit": (-2.8, 0.010),
        }[row.method]
        ax.annotate(
            f"{row.method_label}\n{100 * row.trainable_ratio_mean:.3g}% trainable",
            (row.time_saving_vs_full_pct_mean, row.macro_f1_mean),
            xytext=(row.time_saving_vs_full_pct_mean + x_offset, row.macro_f1_mean + y_offset),
            fontsize=7.7,
        )
    ax.set_xlim(-1.0, 18.5)
    ax.set_ylim(0.625, 0.775)
    ax.set_xlabel("Mean wall-clock saving relative to Full FT (%)")
    ax.set_ylabel("Equal-condition mean Macro-F1")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, "performance_efficiency_tradeoff")


def load_per_seed() -> pd.DataFrame:
    return pd.read_csv(
        FOLLOWUP_DIR / "appendix_full_per_seed.csv", encoding="utf-8-sig"
    )


def plot_collapse_diagnostics(per_seed: pd.DataFrame) -> None:
    baseline = per_seed[
        ["condition", "seed", "baseline_f1", "baseline_majority_prediction_rate"]
    ].rename(
        columns={
            "baseline_f1": "macro_f1",
            "baseline_majority_prediction_rate": "majority_rate",
        }
    )
    baseline["configuration"] = "Baseline"
    remedy = per_seed[
        ["condition", "seed", "remedy_f1", "remedy_majority_prediction_rate"]
    ].rename(
        columns={
            "remedy_f1": "macro_f1",
            "remedy_majority_prediction_rate": "majority_rate",
        }
    )
    remedy["configuration"] = "Combined remedy"
    long_frame = pd.concat([baseline, remedy], ignore_index=True)

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for configuration, color, marker in (
        ("Baseline", ORANGE, "o"),
        ("Combined remedy", BLUE, "s"),
    ):
        subset = long_frame[long_frame["configuration"] == configuration]
        ax.scatter(
            subset["majority_rate"],
            subset["macro_f1"],
            label=configuration,
            color=color,
            marker=marker,
            s=42,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.axvspan(0.98, 1.005, color="#F4CCCC", alpha=0.7, label="Near-collapse region")
    ax.axvline(0.98, color=GRAY, linestyle="--", linewidth=1)
    ax.set_xlim(0.34, 1.01)
    ax.set_ylim(0.10, 0.89)
    ax.set_xlabel("Majority-prediction rate on the test set")
    ax.set_ylabel("Test Macro-F1")
    ax.grid(color=LIGHT_GRAY, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=1)
    fig.tight_layout()
    save_figure(fig, "collapse_diagnostics")


def plot_paired_followup(per_seed: pd.DataFrame) -> None:
    conditions = list(CONDITION_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), sharex=True, sharey=True)

    for ax, condition in zip(axes.ravel(), conditions):
        subset = per_seed[per_seed["condition"] == condition].sort_values("seed")
        offsets = [-0.06, -0.03, 0.0, 0.03, 0.06]
        for offset, row in zip(offsets, subset.itertuples(index=False)):
            ax.plot(
                [offset, 1 + offset],
                [row.baseline_f1, row.remedy_f1],
                color=GRAY,
                linewidth=1.05,
                alpha=0.7,
                zorder=1,
            )
            ax.scatter(offset, row.baseline_f1, color=ORANGE, s=27, zorder=2)
            ax.scatter(1 + offset, row.remedy_f1, color=BLUE, marker="s", s=27, zorder=2)
        ax.set_title(CONDITION_LABELS[condition])
        ax.set_xticks([0, 1], ["Baseline", "Remedy"])
        ax.set_xlim(-0.20, 1.20)
        ax.set_ylim(0.10, 0.90)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.7)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Test Macro-F1")
    axes[1, 0].set_ylabel("Test Macro-F1")
    fig.tight_layout(w_pad=1.4, h_pad=1.5)
    save_figure(fig, "paired_seed_followup")


def main() -> None:
    setup_style()
    summary, _ = snapshot_data()
    per_seed = load_per_seed()
    plot_method_overview(summary)
    plot_performance_efficiency(summary)
    plot_collapse_diagnostics(per_seed)
    plot_paired_followup(per_seed)


if __name__ == "__main__":
    main()
