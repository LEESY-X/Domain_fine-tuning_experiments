#!/usr/bin/env python3
"""Recompute the 100 follow-up runs from saved labels and predictions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PAPER_DIR.parent
FOLLOWUP_DIR = REPO_DIR / "results" / "followup"
TOLERANCE = 1e-12


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=TOLERANCE)


def recompute(prediction_path: Path) -> dict[str, object]:
    frame = pd.read_csv(prediction_path, encoding="utf-8-sig")
    labels = frame["label"].to_numpy(dtype=int)
    predictions = frame["prediction"].to_numpy(dtype=int)
    metrics_path = prediction_path.with_name("final_metrics.json")
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    num_labels = len(saved["true_label_counts"])
    predicted_counts = np.bincount(predictions, minlength=num_labels)
    probabilities = predicted_counts[predicted_counts > 0] / len(predictions)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    normalized_entropy = entropy / math.log(num_labels) if num_labels > 1 else 0.0
    return {
        "saved": saved,
        "rows": len(frame),
        "test_accuracy": accuracy_score(labels, predictions),
        "test_macro_precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "predicted_label_counts": predicted_counts.tolist(),
        "predicted_class_count": int((predicted_counts > 0).sum()),
        "predicted_class_coverage": float((predicted_counts > 0).sum() / num_labels),
        "majority_prediction_rate": float(predicted_counts.max() / len(predictions)),
        "normalized_prediction_entropy": normalized_entropy,
        "constant_prediction_collapse": bool((predicted_counts > 0).sum() == 1),
        "near_constant_prediction": bool(predicted_counts.max() / len(predictions) >= 0.98),
    }


def main() -> None:
    errors: list[str] = []
    counts = {"collapse_followup_v2_pilot": 0, "collapse_followup_v2_full": 0}
    metric_keys = (
        "test_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "predicted_class_coverage",
        "majority_prediction_rate",
        "normalized_prediction_entropy",
    )
    exact_keys = (
        "predicted_label_counts",
        "predicted_class_count",
        "constant_prediction_collapse",
        "near_constant_prediction",
    )

    prediction_paths = sorted(FOLLOWUP_DIR.glob("collapse_followup_v2_*/**/predictions.csv"))
    for prediction_path in prediction_paths:
        result = recompute(prediction_path)
        saved = result["saved"]
        experiment_id = str(saved["experiment_id"])
        counts[experiment_id] = counts.get(experiment_id, 0) + 1
        relative = prediction_path.relative_to(REPO_DIR)
        if saved.get("status") != "COMPLETE":
            errors.append(f"{relative}: status={saved.get('status')}")
        if result["rows"] != saved.get("test_rows"):
            errors.append(f"{relative}: prediction row count differs from test_rows")
        for key in metric_keys:
            if not close(result[key], saved[key]):
                errors.append(f"{relative}: {key} differs")
        for key in exact_keys:
            if result[key] != saved[key]:
                errors.append(f"{relative}: {key} differs")

    expected = {"collapse_followup_v2_pilot": 60, "collapse_followup_v2_full": 40}
    if counts != expected:
        errors.append(f"run counts differ: expected {expected}, found {counts}")
    if errors:
        print("FOLLOW-UP PREDICTION VALIDATION FAILED")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("FOLLOW-UP PREDICTION VALIDATION PASS")
    print("pilot=60, full=40, numeric_discrepancies=0")


if __name__ == "__main__":
    main()
