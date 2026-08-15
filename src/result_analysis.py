from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "full_ft": "Full Fine-tuning",
    "lora": "LoRA",
    "adapter": "Adapter",
    "ia3": "IA3",
    "bitfit": "BitFit",
}

TASK_NUM_LABELS = {
    "measuring_hate_speech": 2,
    "tweet_sentiment": 3,
    "finance_sentiment": 3,
    "movie_reviews": 2,
    "product_reviews": 5,
    "tweet_emotion": 4,
    "tweet_hate": 2,
    "tweet_offensive": 2,
    "tweet_irony": 2,
    "news_topic": 4,
    "news_ynat": 7,
    "movie_nsmc": 2,
    "comment_kmhas_binary": 2,
}

T_CRITICAL_975 = {4: 2.7764451051977987}

# Compact, publication-safe run fields.  Per-example predictions, local paths,
# checkpoint names, timestamps, and machine-specific runtime dictionaries stay
# in the ignored per-run artifacts rather than public aggregate CSVs.
PUBLIC_RUN_COLUMNS = (
    "status",
    "study",
    "task",
    "model",
    "method",
    "seed",
    "run_mode",
    "experiment_id",
    "variant",
    "run_signature",
    "train_rows",
    "validation_rows",
    "test_rows",
    "epochs_requested",
    "learning_rate",
    "class_weighting",
    "trainer_device",
    "train_seconds",
    "trainable_params",
    "total_params",
    "trainable_parameter_ratio",
    "test_loss",
    "test_accuracy",
    "test_macro_f1",
    "test_macro_precision",
    "test_macro_recall",
    "true_label_counts",
    "predicted_label_counts",
    "predicted_class_count",
    "predicted_class_coverage",
    "majority_prediction_rate",
    "normalized_prediction_entropy",
    "constant_prediction_collapse",
    "near_constant_prediction",
    "epochs_completed",
    "global_step",
    "best_validation_macro_f1",
)


def public_run_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return compact run metrics suitable for the public release surface."""
    columns = [name for name in PUBLIC_RUN_COLUMNS if name in frame.columns]
    public = frame.loc[:, columns].copy()
    identity = [
        name
        for name in ("experiment_id", "variant", "study", "task", "model", "method", "seed")
        if name in public.columns
    ]
    return public.sort_values(identity, kind="stable").reset_index(drop=True)


def require_matching_run_signature(
    payload: dict,
    current_signature: str,
    artifact_path: Path | str,
) -> None:
    """Reject cached or resumable artifacts from a different run definition."""
    existing_signature = payload.get("run_signature")
    if existing_signature != current_signature:
        raise RuntimeError(
            f"stale run artifact at {artifact_path}; "
            "the saved run signature does not match the current code/configuration. "
            "Use force=True or a new experiment_id."
        )


def validate_run_frame_integrity(
    frame: pd.DataFrame,
    *,
    expected_seeds: list[int] | None,
    expected_run_mode: str,
    expected_keys: set[tuple[str, str, str, str, int]] | None = None,
) -> None:
    """Fail before aggregation when status, identity, or seed coverage is unsafe."""
    identity_columns = ["study", "task", "model", "method", "seed"]
    required = set(identity_columns) | {"status", "run_mode"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"run table is missing integrity columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("run table is empty")

    incomplete = frame[frame["status"] != "COMPLETE"]
    if not incomplete.empty:
        raise ValueError(f"run table contains {len(incomplete)} non-COMPLETE rows")

    wrong_mode = frame[frame["run_mode"] != expected_run_mode]
    if not wrong_mode.empty:
        found = sorted(set(str(value) for value in wrong_mode["run_mode"]))
        raise ValueError(
            f"run table contains modes other than {expected_run_mode}: {found}"
        )

    duplicate_mask = frame.duplicated(identity_columns, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(f"run table contains {duplicate_count} rows with duplicate run keys")

    if expected_seeds is not None:
        expected_seed_set = sorted(int(seed) for seed in expected_seeds)
        bad_groups = []
        for keys, group in frame.groupby(identity_columns[:-1], sort=True, dropna=False):
            seeds = sorted(int(seed) for seed in group["seed"])
            if seeds != expected_seed_set:
                bad_groups.append((keys, seeds))
        if bad_groups:
            preview = "; ".join(f"{keys}: {seeds}" for keys, seeds in bad_groups[:3])
            raise ValueError(
                f"run table has {len(bad_groups)} groups with incomplete seed coverage; {preview}"
            )

    if expected_keys is not None:
        actual_keys = {
            (
                str(row.study),
                str(row.task),
                str(row.model),
                str(row.method),
                int(row.seed),
            )
            for row in frame[identity_columns].itertuples(index=False)
        }
        missing_keys = expected_keys - actual_keys
        unexpected_keys = actual_keys - expected_keys
        if missing_keys or unexpected_keys:
            raise ValueError(
                "run table does not match the configured design: "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
            )


def constant_prediction_expected(accuracy: float, num_labels: int) -> dict[str, float]:
    """Metrics produced when every sample is assigned to one class.

    If the predicted class has prevalence ``accuracy``, macro precision is
    accuracy/K, macro recall is 1/K, and macro F1 is
    2*accuracy/(1+accuracy)/K. This fingerprint can be checked from an
    aggregate table even when per-example predictions are unavailable.
    """
    return {
        "precision": accuracy / num_labels,
        "recall": 1.0 / num_labels,
        "f1": (2.0 * accuracy / (1.0 + accuracy)) / num_labels,
    }


def has_constant_prediction_fingerprint(
    *,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    num_labels: int,
    atol: float = 1e-12,
) -> bool:
    expected = constant_prediction_expected(accuracy, num_labels)
    observed = {"precision": precision, "recall": recall, "f1": f1}
    return all(math.isclose(observed[key], expected[key], rel_tol=0.0, abs_tol=atol) for key in expected)


def prediction_diagnostics(labels, predictions, num_labels: int) -> dict:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    true_counts = np.bincount(labels, minlength=num_labels)
    predicted_counts = np.bincount(predictions, minlength=num_labels)
    total = int(predicted_counts.sum())
    nonzero = predicted_counts[predicted_counts > 0]
    probabilities = nonzero / total if total else np.asarray([], dtype=float)
    entropy = float(-(probabilities * np.log(probabilities)).sum()) if probabilities.size else 0.0
    normalized_entropy = entropy / math.log(num_labels) if num_labels > 1 else 0.0
    predicted_class_count = int(np.count_nonzero(predicted_counts))
    majority_prediction_rate = float(predicted_counts.max() / total) if total else 0.0
    return {
        "true_label_counts": true_counts.astype(int).tolist(),
        "predicted_label_counts": predicted_counts.astype(int).tolist(),
        "predicted_class_count": predicted_class_count,
        "predicted_class_coverage": predicted_class_count / num_labels,
        "majority_prediction_rate": majority_prediction_rate,
        "normalized_prediction_entropy": normalized_entropy,
        "constant_prediction_collapse": predicted_class_count == 1,
        "near_constant_prediction": majority_prediction_rate >= 0.98,
    }


def paired_difference_stats(baseline, treatment) -> dict[str, float | int]:
    """Return paired effect estimates and an exact two-sided sign-flip test."""
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if baseline.shape != treatment.shape or baseline.ndim != 1 or baseline.size < 2:
        raise ValueError("baseline and treatment must be paired one-dimensional arrays with n >= 2")
    if baseline.size > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    differences = treatment - baseline
    observed = float(differences.mean())
    permuted = [
        float(np.mean(differences * np.asarray(signs)))
        for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
    ]
    p_value = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-15))
    delta_sd = float(np.std(differences, ddof=1))
    critical = T_CRITICAL_975.get(int(differences.size - 1), math.nan)
    margin = critical * delta_sd / math.sqrt(differences.size) if not math.isnan(critical) else math.nan
    return {
        "paired_n": int(differences.size),
        "delta_mean": observed,
        "delta_sd": delta_sd,
        "delta_ci95_low": observed - margin,
        "delta_ci95_high": observed + margin,
        "treatment_better_seeds": int(np.sum(differences > 0)),
        "treatment_equal_seeds": int(np.sum(differences == 0)),
        "exact_sign_flip_p_two_sided": p_value,
    }


def diagnose_public_summary(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "task", "model", "method", "seeds", "f1_mean", "f1_sd",
        "accuracy_mean", "precision_mean", "recall_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"summary table is missing columns: {sorted(missing)}")

    rows = []
    for row in frame.to_dict("records"):
        num_labels = TASK_NUM_LABELS.get(row["task"])
        if num_labels is None:
            raise ValueError(f"unknown task label count: {row['task']}")
        exact_zero = float(row["f1_sd"]) == 0.0
        fingerprint = has_constant_prediction_fingerprint(
            accuracy=float(row["accuracy_mean"]),
            precision=float(row["precision_mean"]),
            recall=float(row["recall_mean"]),
            f1=float(row["f1_mean"]),
            num_labels=num_labels,
        )
        rows.append({
            "study": row.get("study", ""),
            "task": row["task"],
            "model": row["model"],
            "method": row["method"],
            "seeds": int(row["seeds"]),
            "num_labels": num_labels,
            "accuracy_mean": float(row["accuracy_mean"]),
            "precision_mean": float(row["precision_mean"]),
            "recall_mean": float(row["recall_mean"]),
            "f1_mean": float(row["f1_mean"]),
            "f1_sd": float(row["f1_sd"]),
            "f1_sd_exact_zero": exact_zero,
            "constant_prediction_fingerprint": fingerprint,
            "degenerate_stability": exact_zero and fingerprint,
            "diagnosis": (
                "aggregate fingerprint consistent with constant-class collapse"
                if exact_zero and fingerprint
                else "zero variance without constant-class fingerprint"
                if exact_zero
                else "non-zero seed variance"
            ),
        })
    return pd.DataFrame(rows)


def summarize_run_frame(frame: pd.DataFrame, expected_seeds: list[int] | None = None) -> pd.DataFrame:
    required = {
        "study", "task", "model", "method", "seed", "test_macro_f1",
        "test_accuracy", "test_macro_precision", "test_macro_recall",
        "train_seconds", "trainable_params", "trainable_parameter_ratio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"run table is missing columns: {sorted(missing)}")

    optional_group_columns = [name for name in ("experiment_id", "variant", "run_mode") if name in frame.columns]
    group_columns = optional_group_columns + ["study", "task", "model", "method"]
    rows = []
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys))
        seeds = sorted(int(value) for value in group["seed"].tolist())
        unique_seeds = sorted(set(seeds))
        f1_values = [float(value) for value in group.sort_values("seed")["test_macro_f1"].tolist()]
        seed_counts = group["seed"].value_counts()
        expected = sorted(expected_seeds) if expected_seeds is not None else unique_seeds
        collapsed = (
            group["constant_prediction_collapse"].fillna(False).astype(bool)
            if "constant_prediction_collapse" in group.columns
            else pd.Series(False, index=group.index)
        )
        num_labels = TASK_NUM_LABELS.get(identity["task"])
        predicted_class_counts = (
            pd.to_numeric(group["predicted_class_count"], errors="coerce")
            if "predicted_class_count" in group.columns
            else pd.Series(math.nan, index=group.index)
        )
        majority_rates = (
            pd.to_numeric(group["majority_prediction_rate"], errors="coerce")
            if "majority_prediction_rate" in group.columns
            else pd.Series(math.nan, index=group.index)
        )
        near_constant = (
            group["near_constant_prediction"].fillna(False).astype(bool)
            if "near_constant_prediction" in group.columns
            else pd.Series(False, index=group.index)
        )
        rows.append({
            **identity,
            "method_label": METHOD_LABELS.get(identity["method"], identity["method"]),
            "runs": len(group),
            "unique_seed_count": len(unique_seeds),
            "seeds": json.dumps(unique_seeds, separators=(",", ":")),
            "missing_seeds": json.dumps(sorted(set(expected) - set(unique_seeds)), separators=(",", ":")),
            "duplicate_seed_count": int((seed_counts - 1).clip(lower=0).sum()),
            "f1_values_by_seed": json.dumps(f1_values, separators=(",", ":")),
            "f1_mean": float(np.mean(f1_values)),
            "f1_sd": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else math.nan,
            "f1_min": float(np.min(f1_values)),
            "f1_max": float(np.max(f1_values)),
            "f1_range": float(np.max(f1_values) - np.min(f1_values)),
            "f1_all_identical": len(set(f1_values)) == 1,
            "accuracy_mean": float(group["test_accuracy"].mean()),
            "precision_mean": float(group["test_macro_precision"].mean()),
            "recall_mean": float(group["test_macro_recall"].mean()),
            "train_seconds_mean": float(group["train_seconds"].mean()),
            "train_seconds_sd": float(group["train_seconds"].std(ddof=1)) if len(group) > 1 else math.nan,
            "trainable_params_mean": float(group["trainable_params"].mean()),
            "trainable_ratio_mean": float(group["trainable_parameter_ratio"].mean()),
            "collapsed_runs": int(collapsed.sum()),
            "collapse_rate": float(collapsed.mean()),
            "near_constant_runs": int(near_constant.sum()),
            "predicted_class_count_min": float(predicted_class_counts.min()),
            "predicted_class_count_mean": float(predicted_class_counts.mean()),
            "full_class_coverage_runs": int((predicted_class_counts == num_labels).sum()) if num_labels else 0,
            "majority_prediction_rate_mean": float(majority_rates.mean()),
            "majority_prediction_rate_max": float(majority_rates.max()),
            "train_rows": int(group["train_rows"].iloc[0]) if "train_rows" in group.columns else None,
            "validation_rows": int(group["validation_rows"].iloc[0]) if "validation_rows" in group.columns else None,
            "test_rows": int(group["test_rows"].iloc[0]) if "test_rows" in group.columns else None,
            "epochs_requested": int(group["epochs_requested"].iloc[0]) if "epochs_requested" in group.columns else None,
            "trainer_devices": json.dumps(sorted(set(str(value) for value in group.get("trainer_device", pd.Series(dtype=str)).dropna())), separators=(",", ":")),
            "seed_coverage_ok": unique_seeds == expected and len(seeds) == len(unique_seeds),
        })
    return pd.DataFrame(rows)


def read_final_metrics(root: Path | str, relative_to: Path | str | None = None) -> pd.DataFrame:
    rows = []
    relative_base = Path(relative_to).resolve() if relative_to is not None else None
    for path in Path(root).glob("**/final_metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            continue
        resolved_path = path.resolve()
        payload["source_file"] = str(resolved_path.relative_to(relative_base)) if relative_base else str(resolved_path)
        rows.append(payload)
    return pd.DataFrame(rows)
