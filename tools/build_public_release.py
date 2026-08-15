from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.result_analysis import public_run_frame, summarize_run_frame, validate_run_frame_integrity


TAG = "ieie-spc-v1.0.0"
RELEASE_DIR = ROOT / "release" / TAG
EXPECTED_SEEDS = [42, 52, 62, 72, 82]
RUN_INPUTS = {
    "pilot": ROOT / "results/followup/collapse_followup_v2_pilot/aggregate/all_runs.csv",
    "full": ROOT / "results/followup/collapse_followup_v2_full/aggregate/all_runs.csv",
}
SUMMARY_OUTPUTS = {
    "pilot": ROOT / "results/followup/collapse_followup_v2_pilot/aggregate/summary.csv",
    "full": ROOT / "results/followup/collapse_followup_v2_full/aggregate/summary.csv",
}
CANONICAL_PATHS = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements-followup.txt",
    "config/experiment_config.json",
    "config/collapse_followup.json",
    "src/__init__.py",
    "src/suite.py",
    "src/result_analysis.py",
    "tools/run_collapse_followup.py",
    "tools/analyze_result_variance.py",
    "tools/build_paper_followup_report.py",
    "tools/sanitize_followup_artifacts.py",
    "tools/test_followup_plan.py",
    "tools/test_result_analysis.py",
    "tools/validate_suite.py",
    "tools/verify_colab_parity.py",
    "notebooks/00_precheck.ipynb",
    "notebooks/01_study1_bertweet_hate.ipynb",
    "notebooks/02_study2_multitask.ipynb",
    "notebooks/03_study3_korean.ipynb",
    "notebooks/04_aggregate.ipynb",
    "notebooks/05_progress_monitor.ipynb",
    "colab_a100_full_550_runs.ipynb",
    "colab_a100_low_drive_550_runs.ipynb",
    "final_tables/summary_by_task_model_method.csv",
    "final_tables/winners_by_metric.csv",
    "results/environment.json",
    "results/followup/provenance.json",
    "results/followup/collapse_followup_v2_pilot/environment.json",
    "results/followup/collapse_followup_v2_pilot/aggregate/all_runs.csv",
    "results/followup/collapse_followup_v2_pilot/aggregate/summary.csv",
    "results/followup/collapse_followup_v2_full/environment.json",
    "results/followup/collapse_followup_v2_full/aggregate/all_runs.csv",
    "results/followup/collapse_followup_v2_full/aggregate/summary.csv",
    "results_summary/variance_diagnostics.csv",
    "results_summary/variance_diagnostics.md",
    "results_summary/paper_followup/table_1_original_zero_sd.csv",
    "results_summary/paper_followup/table_2_pilot_ablation.csv",
    "results_summary/paper_followup/table_3_full_comparison.csv",
    "results_summary/paper_followup/appendix_full_per_seed.csv",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_and_sanitize_runs(label: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    public = public_run_frame(frame)
    groups: list[dict] = []
    for (experiment_id, variant), group in public.groupby(["experiment_id", "variant"], sort=True):
        validate_run_frame_integrity(
            group,
            expected_seeds=EXPECTED_SEEDS,
            expected_run_mode="FOLLOWUP",
        )
        groups.append(
            {
                "experiment_id": str(experiment_id),
                "variant": str(variant),
                "runs": int(len(group)),
                "conditions": int(group[["study", "task", "model", "method"]].drop_duplicates().shape[0]),
                "seeds": sorted(int(value) for value in group["seed"].unique()),
            }
        )
    if public["run_signature"].nunique() != len(public):
        raise ValueError(f"{label}: run signatures are not unique")
    summary = summarize_run_frame(public, expected_seeds=EXPECTED_SEEDS)
    if not bool(summary["seed_coverage_ok"].all()):
        raise ValueError(f"{label}: seed coverage failed")
    public.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")
    summary.to_csv(SUMMARY_OUTPUTS[label], index=False, encoding="utf-8-sig", float_format="%.17g")
    return public, summary, groups


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / "final_tables/summary_by_task_model_method.csv", encoding="utf-8-sig")
    winners = pd.read_csv(ROOT / "final_tables/winners_by_metric.csv", encoding="utf-8-sig")
    if len(original) != 110 or len(winners) != 22:
        raise ValueError("historical aggregate must contain 110 method rows and 22 winner rows")
    if int((original["f1_sd"] == 0).sum()) != 4:
        raise ValueError("historical aggregate must contain four exact-zero F1-SD rows")

    coverage: dict[str, object] = {
        "release_tag": TAG,
        "expected_seeds": EXPECTED_SEEDS,
        "run_mode": "FOLLOWUP",
        "suites": {},
    }
    run_totals = {}
    for label, path in RUN_INPUTS.items():
        public, summary, groups = validate_and_sanitize_runs(label, path)
        expected = 60 if label == "pilot" else 40
        if len(public) != expected:
            raise ValueError(f"{label}: expected {expected} runs, found {len(public)}")
        run_totals[label] = len(public)
        coverage["suites"][label] = {
            "expected_runs": expected,
            "observed_runs": int(len(public)),
            "complete_runs": int((public["status"] == "COMPLETE").sum()),
            "unique_run_signatures": int(public["run_signature"].nunique()),
            "summary_groups": int(len(summary)),
            "missing_or_duplicate_seeds": False,
            "variant_groups": groups,
        }
    write_json(RELEASE_DIR / "FOLLOWUP_RUN_COVERAGE.json", coverage)

    provenance = json.loads((ROOT / "results/followup/provenance.json").read_text(encoding="utf-8"))
    files = []
    for relative in CANONICAL_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "release_tag": TAG,
        "release_url": f"https://github.com/LEESY-X/Domain_fine-tuning_experiments/tree/{TAG}",
        "prepared_on": str(date.today()),
        "scope": "code, configurations, historical aggregate tables, and compact follow-up run metrics",
        "historical_evidence": {"aggregate_method_rows": 110, "winner_rows": 22, "historical_run_level_files_available": False},
        "followup_evidence": {"pilot_runs": run_totals["pilot"], "full_data_runs": run_totals["full"], "per_example_predictions_public": False},
        "contains_raw_dataset_text": False,
        "contains_model_checkpoints": False,
        "contains_per_example_predictions": False,
        "executed_suite_sha256": provenance["executed_suite_sha256"],
        "released_current_suite_sha256": sha256(ROOT / "src/suite.py"),
        "executed_source_snapshot_available": bool(provenance["executed_source_snapshot_available"]),
        "dataset_revisions": provenance["dataset_revisions"],
        "model_revisions": provenance["model_revisions"],
        "limitations": [
            "The released suite is current reconstruction and re-execution code, not the exact executed follow-up source snapshot.",
            "Compact run tables support regeneration of summaries and paired statistics, not per-example metric recomputation.",
            "The historical 550-run design is represented by aggregate tables; its run-level files are unavailable.",
        ],
        "files": files,
    }
    write_json(RELEASE_DIR / "RELEASE_MANIFEST.json", manifest)

    validation = f"""# Public Release Validation Report

Release ref: `{TAG}`
Prepared: {date.today()}

## Publicly repeatable checks

- Historical aggregate dimensions: PASS (110 method rows; 22 winner rows).
- Historical exact-zero rows: PASS (4 rows at stored precision).
- Pilot compact run coverage: PASS ({run_totals['pilot']}/60 COMPLETE rows; unique signatures; five seeds per variant-condition group).
- Full-data compact run coverage: PASS ({run_totals['full']}/40 COMPLETE rows; unique signatures; five seeds per variant-condition group).
- Public aggregate tables contain no per-example text, predictions, checkpoint paths, completion timestamps, or machine-local runtime dictionaries.

## Local-only check

Per-example predictions for 100 follow-up runs were retained locally and independently recomputed with zero metric discrepancies. Those prediction files are not in this public release, so that prediction-level check cannot be independently repeated from the tagged tree.

## Reproducibility boundary

The tagged code is the maintained reconstruction/re-execution implementation. The recorded executed-suite hash is `{provenance['executed_suite_sha256']}`, but its exact source snapshot was not preserved. The release therefore supports code inspection, rerunning under documented configurations, and regeneration of compact summaries; it does not claim bit-identical replay of the historical or follow-up executions.
"""
    (RELEASE_DIR / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")

    readme = f"""# IEIE SPC Code and Compact Evidence Snapshot

This directory documents the versioned public artifact `{TAG}` for the accompanying IEIE SPC manuscript. The tagged repository contains the maintained experiment code, configurations, the preserved 110-row historical aggregate, and compact metric-level records for all 60 pilot and 40 full-data follow-up runs.

The public snapshot intentionally excludes dataset text and caches, model checkpoints, per-example predictions, trainer histories, events, and per-run JSON files. See `RELEASE_MANIFEST.json` for file hashes and evidence boundaries, `FOLLOWUP_RUN_COVERAGE.json` for the run matrix, and `VALIDATION_REPORT.md` for checks that are public versus local-only.

Versioned repository: https://github.com/LEESY-X/Domain_fine-tuning_experiments/tree/{TAG}
"""
    (RELEASE_DIR / "README.md").write_text(readme, encoding="utf-8")

    checksum_paths = [ROOT / item["path"] for item in files]
    checksum_paths.extend(
        [
            RELEASE_DIR / "README.md",
            RELEASE_DIR / "FOLLOWUP_RUN_COVERAGE.json",
            RELEASE_DIR / "VALIDATION_REPORT.md",
        ]
    )
    # RELEASE_MANIFEST.json cannot list its own digest without a circular
    # dependency.  The manifest hashes the canonical source/evidence surface;
    # SHA256SUMS additionally covers the manifest itself.
    checksum_paths.append(RELEASE_DIR / "RELEASE_MANIFEST.json")
    lines = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in sorted(checksum_paths)]
    (RELEASE_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PUBLIC RELEASE EVIDENCE PASS: tag={TAG}, pilot=60, full=40")


if __name__ == "__main__":
    main()
