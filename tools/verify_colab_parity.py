import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "colab_a100_full_550_runs.ipynb",
    ROOT / "colab_a100_low_drive_550_runs.ipynb",
]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


local_suite = (ROOT / "src" / "suite.py").read_text(encoding="utf-8")
local_result_analysis = (ROOT / "src" / "result_analysis.py").read_text(encoding="utf-8")
local_config = (ROOT / "config" / "experiment_config.json").read_text(encoding="utf-8")

for notebook_path in NOTEBOOKS:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    embedded = {}
    all_source = ""
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        all_source += source
        if cell.get("cell_type") != "code":
            continue
        if source.startswith(("SUITE_SOURCE = ", "RESULT_ANALYSIS_SOURCE = ", "CONFIG_SOURCE = ")):
            tree = ast.parse(source)
            assignment = tree.body[0]
            embedded[assignment.targets[0].id] = ast.literal_eval(assignment.value)

    assert embedded["SUITE_SOURCE"] == local_suite
    assert embedded["RESULT_ANALYSIS_SOURCE"] == local_result_analysis
    assert embedded["CONFIG_SOURCE"] == local_config
    run_marker = "run_study_low_drive(STUDY" if "low_drive" in notebook_path.name else "run_study(STUDY"
    for marker in (run_marker, "aggregate('PAPER')", "A100", "DRIVE_ROOT"):
        assert marker in all_source
    print(notebook_path.name, "embedded sources PASS")

config = json.loads(local_config)
counts = {
    study: len(config[study]["tasks"]) * len(config[study]["models"]) * len(config["methods"]) * len(config["seeds"])
    for study in ("study1", "study2", "study3")
}
assert counts == {"study1": 25, "study2": 450, "study3": 75}
print("COLAB PARITY PASS")
print("suite.py sha256:", sha(local_suite))
print("result_analysis.py sha256:", sha(local_result_analysis))
print("config sha256:", sha(local_config))
print("jobs:", counts, "total=", sum(counts.values()))
