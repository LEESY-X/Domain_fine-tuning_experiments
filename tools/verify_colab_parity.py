import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "COLAB_A100_FULL_550_RUNS.ipynb"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
embedded = {}
all_source = ""
for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    all_source += source
    if cell.get("cell_type") != "code":
        continue
    if source.startswith("SUITE_SOURCE = ") or source.startswith("CONFIG_SOURCE = "):
        tree = ast.parse(source)
        assignment = tree.body[0]
        embedded[assignment.targets[0].id] = ast.literal_eval(assignment.value)

local_suite = (ROOT / "src" / "suite.py").read_text(encoding="utf-8")
local_config = (ROOT / "config" / "experiment_config.json").read_text(encoding="utf-8")
assert embedded["SUITE_SOURCE"] == local_suite
assert embedded["CONFIG_SOURCE"] == local_config

config = json.loads(local_config)
counts = {
    study: len(config[study]["tasks"]) * len(config[study]["models"]) * len(config["methods"]) * len(config["seeds"])
    for study in ("study1", "study2", "study3")
}
assert counts == {"study1": 25, "study2": 450, "study3": 75}
for marker in ("run_study(STUDY", "aggregate('PAPER')", "A100", "DRIVE_ROOT"):
    assert marker in all_source

print("COLAB PARITY PASS")
print("suite.py sha256:", sha(local_suite))
print("config sha256:", sha(local_config))
print("jobs:", counts, "total=", sum(counts.values()))
