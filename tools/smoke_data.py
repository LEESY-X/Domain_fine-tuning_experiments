import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.suite import load_config, load_task

config = load_config()
keys = config["study1"]["tasks"] + config["study2"]["tasks"] + config["study3"]["tasks"]
for key in dict.fromkeys(keys):
    dataset = load_task(key, "SMOKE")
    print(key, {split: len(rows) for split, rows in dataset.items()})
