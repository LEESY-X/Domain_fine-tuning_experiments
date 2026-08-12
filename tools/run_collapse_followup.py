from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "collapse_followup.json"


def load_plan(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_followup_jobs(plan, variants=None, conditions=None, full_data=False):
    selected_variants = variants or list(plan["variants"])
    selected_conditions = conditions or [row["id"] for row in plan["conditions"]]
    unknown_variants = set(selected_variants) - set(plan["variants"])
    known_conditions = {row["id"] for row in plan["conditions"]}
    unknown_conditions = set(selected_conditions) - known_conditions
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {sorted(unknown_conditions)}")

    experiment_id = plan["experiment_id"] + ("_full" if full_data else "_pilot")
    limits = None if full_data else plan["pilot_limits"]
    jobs = []
    for variant_name in selected_variants:
        variant = plan["variants"][variant_name]
        for condition in plan["conditions"]:
            if condition["id"] not in selected_conditions:
                continue
            for seed in plan["seeds"]:
                training_overrides = {"epochs": variant["epochs"]}
                if "learning_rates" in variant:
                    training_overrides["learning_rate"] = variant["learning_rates"][condition["method"]]
                jobs.append({
                    "experiment_id": experiment_id,
                    "variant": variant_name,
                    "study": condition["study"],
                    "task_key": condition["task"],
                    "model_name": condition["model"],
                    "method": condition["method"],
                    "seed": seed,
                    "run_mode": "FOLLOWUP",
                    "epochs": variant["epochs"],
                    "limits": limits,
                    "training_overrides": training_overrides,
                    "class_weighting": variant["class_weighting"],
                    "keep_checkpoint": plan.get("keep_checkpoints", False),
                })
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run the targeted constant-class-collapse follow-up experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--condition", action="append", dest="conditions")
    parser.add_argument("--full-data", action="store_true", help="Use full dataset splits instead of the safe local pilot limits.")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    plan = load_plan(args.config)
    jobs = build_followup_jobs(plan, args.variants, args.conditions, args.full_data)
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    print(f"experiment={jobs[0]['experiment_id'] if jobs else 'EMPTY'} jobs={len(jobs)} full_data={args.full_data}")
    for index, job in enumerate(jobs, 1):
        print(
            f"{index:03d} {job['variant']} {job['task_key']} {job['model_name']} "
            f"{job['method']} seed={job['seed']} weighting={job['class_weighting']}"
        )
    if args.dry_run or not jobs:
        return

    sys.path.insert(0, str(ROOT))
    from src.result_analysis import read_final_metrics, summarize_run_frame
    from src.suite import precheck, run_one

    environment_path = ROOT / "results" / "followup" / jobs[0]["experiment_id"] / "environment.json"
    precheck(require_cuda=args.require_cuda, output_path=environment_path)
    failures = []
    for index, job in enumerate(jobs, 1):
        print(f"RUN {index}/{len(jobs)}: {job['variant']} {job['task_key']} {job['method']} seed={job['seed']}")
        try:
            result = run_one(force=args.force, **job)
            print(
                "RESULT",
                f"f1={result['test_macro_f1']:.8f}",
                f"predicted_classes={result['predicted_class_count']}",
                f"majority_rate={result['majority_prediction_rate']:.6f}",
                f"collapse={result['constant_prediction_collapse']}",
            )
        except Exception as exc:
            failures.append((job, exc))
            print("FAILED", type(exc).__name__, str(exc))
            if not args.continue_on_error:
                raise

    experiment_root = ROOT / "results" / "followup" / jobs[0]["experiment_id"]
    frame = read_final_metrics(experiment_root, relative_to=ROOT)
    aggregate_root = experiment_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(aggregate_root / "all_runs.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    summary = summarize_run_frame(frame, expected_seeds=plan["seeds"])
    summary.to_csv(aggregate_root / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    print(f"aggregate_runs={len(frame)} aggregate_groups={len(summary)} failures={len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
