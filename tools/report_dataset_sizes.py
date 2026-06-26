import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.suite import TASKS, _ensure_three_splits, _raw_dataset, _standardize, load_config

config = load_config()
keys = config["study1"]["tasks"] + config["study2"]["tasks"] + config["study3"]["tasks"]
for key in dict.fromkeys(keys):
    spec = TASKS[key]
    ds = _ensure_three_splits(_standardize(_raw_dataset(spec), spec))
    print(key, *(f"{name}={len(split)}" for name, split in ds.items()))
