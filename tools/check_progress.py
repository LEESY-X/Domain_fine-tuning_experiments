import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--study", choices=["study1", "study2", "study3"], required=True)
parser.add_argument("--mode", choices=["SMOKE", "PAPER"], default="PAPER")
args = parser.parse_args()

root = ROOT / "results" / args.study / args.mode
statuses = []
for path in root.glob("**/status.json") if root.exists() else []:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["path"] = str(path.parent.relative_to(ROOT))
        statuses.append(row)
    except Exception:
        statuses.append({"status": "UNREADABLE", "path": str(path.parent.relative_to(ROOT))})

counts = Counter(row.get("status", "UNKNOWN") for row in statuses)
print(f"{args.study} / {args.mode}")
print("status:", dict(counts))
active = [x for x in statuses if x.get("status") in {"RUNNING", "FAILED"}]
for row in active[-10:]:
    print(row.get("status"), row.get("task"), row.get("model"), row.get("method"), row.get("seed"), row.get("resumed_from_checkpoint"), row["path"])

progress = root / "progress.json"
if progress.exists():
    print("\nprogress.json")
    print(progress.read_text(encoding="utf-8"))

