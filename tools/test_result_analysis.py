from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.result_analysis import (
    constant_prediction_expected,
    diagnose_public_summary,
    paired_difference_stats,
    prediction_diagnostics,
    public_run_frame,
    require_matching_run_signature,
    summarize_run_frame,
    validate_run_frame_integrity,
)


finance = constant_prediction_expected(0.5929752066115702, 3)
assert math.isclose(finance["precision"], 0.19765840220385675, abs_tol=1e-15)
assert math.isclose(finance["recall"], 1 / 3, abs_tol=1e-15)
assert math.isclose(finance["f1"], 0.24816255944660615, abs_tol=1e-15)

paired = paired_difference_stats([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])
assert paired["paired_n"] == 5
assert paired["treatment_better_seeds"] == 5
assert paired["exact_sign_flip_p_two_sided"] == 0.0625

diagnostics = diagnose_public_summary(ROOT / "final_tables" / "summary_by_task_model_method.csv")
collapsed = diagnostics[diagnostics["degenerate_stability"]]
assert len(diagnostics) == 110
assert len(collapsed) == 4
assert set(zip(collapsed["task"], collapsed["model"], collapsed["method"])) == {
    ("finance_sentiment", "FacebookAI/roberta-base", "adapter"),
    ("finance_sentiment", "FacebookAI/roberta-base", "bitfit"),
    ("tweet_emotion", "FacebookAI/roberta-base", "bitfit"),
    ("tweet_emotion", "vinai/bertweet-base", "lora"),
}

prediction = prediction_diagnostics([0, 1, 1, 1], [1, 1, 1, 1], 2)
assert prediction["constant_prediction_collapse"] is True
assert prediction["predicted_label_counts"] == [0, 4]
assert prediction["true_label_counts"] == [1, 3]

rows = []
for seed, f1 in ((42, 0.4), (52, 0.5), (62, 0.6)):
    rows.append({
        "experiment_id": "test", "variant": "baseline", "run_mode": "FOLLOWUP",
        "study": "study2", "task": "finance_sentiment", "model": "model", "method": "bitfit",
        "seed": seed, "test_macro_f1": f1, "test_accuracy": f1,
        "test_macro_precision": f1, "test_macro_recall": f1, "train_seconds": 1.0,
        "trainable_params": 10, "trainable_parameter_ratio": 0.1,
        "constant_prediction_collapse": False,
    })
summary = summarize_run_frame(pd.DataFrame(rows), expected_seeds=[42, 52, 62])
assert len(summary) == 1
assert np.isclose(summary.iloc[0]["f1_mean"], 0.5)
assert np.isclose(summary.iloc[0]["f1_sd"], 0.1)
assert bool(summary.iloc[0]["seed_coverage_ok"]) is True

integrity_frame = pd.DataFrame([{**row, "status": "COMPLETE"} for row in rows])
expected_keys = {
    ("study2", "finance_sentiment", "model", "bitfit", seed)
    for seed in (42, 52, 62)
}
validate_run_frame_integrity(
    integrity_frame,
    expected_seeds=[42, 52, 62],
    expected_run_mode="FOLLOWUP",
    expected_keys=expected_keys,
)

duplicate_seed_frame = pd.concat(
    [integrity_frame.iloc[[0]], integrity_frame.iloc[[0]], integrity_frame.iloc[[1]]],
    ignore_index=True,
)
try:
    validate_run_frame_integrity(
        duplicate_seed_frame,
        expected_seeds=[42, 52, 62],
        expected_run_mode="FOLLOWUP",
    )
except ValueError as exc:
    assert "duplicate run keys" in str(exc)
else:
    raise AssertionError("duplicate seeds must fail integrity validation")

try:
    validate_run_frame_integrity(
        integrity_frame.iloc[:2].copy(),
        expected_seeds=[42, 52, 62],
        expected_run_mode="FOLLOWUP",
    )
except ValueError as exc:
    assert "incomplete seed coverage" in str(exc)
else:
    raise AssertionError("missing seeds must fail integrity validation")

try:
    require_matching_run_signature(
        {"run_signature": "saved"},
        "current",
        "synthetic/final_metrics.json",
    )
except RuntimeError as exc:
    assert "stale run artifact" in str(exc)
else:
    raise AssertionError("mismatched run signatures must be rejected")

require_matching_run_signature(
    {"run_signature": "current"},
    "current",
    "synthetic/final_metrics.json",
)

private_frame = integrity_frame.assign(
    completed_at="2026-08-07T20:00:00+09:00",
    source_file="private/final_metrics.json",
    best_checkpoint="checkpoint-10",
    runtime='{"gpu":"local"}',
    run_signature="signature",
)
released_frame = public_run_frame(private_frame)
assert len(released_frame) == len(private_frame)
assert "run_signature" in released_frame
assert not {"completed_at", "source_file", "best_checkpoint", "runtime"} & set(released_frame.columns)

print("RESULT ANALYSIS TEST PASS")
