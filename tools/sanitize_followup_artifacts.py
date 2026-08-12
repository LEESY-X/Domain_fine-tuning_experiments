from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.result_analysis import public_run_frame, read_final_metrics, summarize_run_frame


def sanitize_value(value, key=None):
    if isinstance(value, dict):
        sanitized = {item_key: sanitize_value(item_value, item_key) for item_key, item_value in value.items()}
        trainer_device = sanitized.get("trainer_device")
        runtime = sanitized.get("runtime")
        if trainer_device == "mps" and isinstance(runtime, dict):
            runtime.update({
                "mps_available": True,
                "accelerator": "mps",
                "gpu": "Apple MPS",
            })
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        if key == "best_checkpoint" and value:
            return Path(value).name
        if value.startswith(str(ROOT)):
            return str(Path(value).relative_to(ROOT))
    return value


def atomic_json(path: Path, payload) -> None:
    descriptor, temp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove machine-local paths and rebuild follow-up aggregates.")
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "followup")
    args = parser.parse_args()
    changed = 0
    for path in args.root.glob("collapse_followup_v2_*/**/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.name in {"final_metrics.json", "status.json"} and payload.get("status") == "COMPLETE":
            history_path = path.parent / "trainer_history.csv"
            if history_path.exists():
                history = pd.read_csv(history_path, encoding="utf-8-sig")
                epochs = pd.to_numeric(history.get("epoch"), errors="coerce").dropna()
                steps = pd.to_numeric(history.get("step"), errors="coerce").dropna()
                validation_f1 = pd.to_numeric(history.get("eval_macro_f1"), errors="coerce").dropna()
                payload["epochs_completed"] = float(epochs.max()) if not epochs.empty else None
                payload["global_step"] = int(steps.max()) if not steps.empty else None
                payload["best_validation_macro_f1"] = float(validation_f1.max()) if not validation_f1.empty else None
        sanitized = sanitize_value(payload)
        if sanitized != payload:
            atomic_json(path, sanitized)
            changed += 1

    for experiment_root in sorted(args.root.glob("collapse_followup_v2_*")):
        frame = read_final_metrics(experiment_root, relative_to=ROOT)
        if frame.empty:
            continue
        aggregate_root = experiment_root / "aggregate"
        aggregate_root.mkdir(parents=True, exist_ok=True)
        public_run_frame(frame).to_csv(
            aggregate_root / "all_runs.csv",
            index=False,
            encoding="utf-8-sig",
            float_format="%.17g",
        )
        summary = summarize_run_frame(frame, expected_seeds=[42, 52, 62, 72, 82])
        summary.to_csv(aggregate_root / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        print(experiment_root.name, f"runs={len(frame)} groups={len(summary)}")
    print(f"sanitized_json_files={changed}")


if __name__ == "__main__":
    main()
