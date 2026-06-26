import ast
import json
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []

cfg = json.loads((ROOT / "config" / "experiment_config.json").read_text(encoding="utf-8"))
expected = ["full_ft", "lora", "adapter", "ia3", "bitfit"]
if cfg["methods"] != expected: errors.append(f"methods mismatch: {cfg['methods']}")
if cfg["seeds"] != [42, 52, 62, 72, 82]: errors.append(f"seeds mismatch: {cfg['seeds']}")
if cfg["study2"].get("limits") is not None: errors.append("study2 PAPER must use full original splits (limits=null)")

ast.parse((ROOT / "src" / "suite.py").read_text(encoding="utf-8"))
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
        if nb.get("nbformat") != 4: errors.append(f"bad nbformat: {path.name}")
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code": ast.parse("".join(c.get("source", [])))
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")

expected_notebooks = {
    "00_PRECHECK.ipynb", "01_STUDY1_BERTWEET_HATE.ipynb", "02_STUDY2_MULTITASK.ipynb",
    "03_STUDY3_KOREAN.ipynb", "04_AGGREGATE.ipynb", "05_PROGRESS_MONITOR.ipynb",
}
actual_notebooks = {x.name for x in (ROOT / "notebooks").glob("*.ipynb")}
if not expected_notebooks.issubset(actual_notebooks): errors.append(f"missing notebooks: {sorted(expected_notebooks - actual_notebooks)}")

html_files = [
    ROOT / "experiment_design_visualization.html",
    ROOT / "paper_experiment_methodology_visualization.html",
    ROOT / "project_experiment_overview.html",
]
for path in html_files:
    try:
        source = path.read_text(encoding="utf-8")
        HTMLParser().feed(source)
        for marker in ("550", "Full FT", "LoRA", "Adapter", "IA³", "BitFit"):
            if marker not in source: errors.append(f"{path.name}: missing {marker}")
        if any(stale in source for stale in ("Total: 625 runs", "Size sensitivity: 75 runs", "최대 50,000 train")):
            errors.append(f"{path.name}: stale experiment scope")
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")

for study, count in (("study1", 25), ("study2", 450), ("study3", 75)):
    section = cfg[study]
    actual = len(section["tasks"]) * len(section["models"]) * len(cfg["methods"]) * len(cfg["seeds"])
    if actual != count: errors.append(f"{study}: expected {count}, got {actual}")

if errors:
    print("VALIDATION FAILED")
    print("\n".join(f"- {x}" for x in errors))
    sys.exit(1)
print("VALIDATION PASS")
print("study1=25, study2=450, study3=75 PAPER runs")
