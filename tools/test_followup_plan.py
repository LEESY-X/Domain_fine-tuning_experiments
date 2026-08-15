from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_collapse_followup import build_followup_jobs, load_plan


plan = load_plan(ROOT / "config" / "collapse_followup.json")
jobs = build_followup_jobs(plan)
assert len(jobs) == 6 * 4 * 5
assert {job["seed"] for job in jobs} == {42, 52, 62, 72, 82}
assert {job["variant"] for job in jobs} == {
    "baseline", "longer", "weighted", "longer_weighted", "higher_lr", "higher_lr_weighted"
}
assert all(job["experiment_id"] == "collapse_followup_v2_pilot" for job in jobs)
assert all(job["limits"] == {"train": 1024, "validation": 512, "test": 2048} for job in jobs)

full = build_followup_jobs(
    plan,
    variants=["baseline"],
    conditions=["finance_roberta_adapter"],
    full_data=True,
)
assert len(full) == 5
assert all(job["limits"] is None for job in full)
assert all(job["experiment_id"] == "collapse_followup_v2_full" for job in full)

print("FOLLOW-UP PLAN TEST PASS")
