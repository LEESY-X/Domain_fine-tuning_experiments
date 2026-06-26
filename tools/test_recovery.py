import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.suite import AtomicEpochCallback, _latest_checkpoint

with tempfile.TemporaryDirectory(dir=ROOT) as temp_name:
    temp = Path(temp_name)
    good = temp / "checkpoints" / "checkpoint-10"
    bad = temp / "checkpoints" / "checkpoint-20"
    good.mkdir(parents=True); bad.mkdir(parents=True)
    (good / "trainer_state.json").write_text(json.dumps({"global_step": 10}), encoding="utf-8")
    assert _latest_checkpoint(temp) == str(good)

    epoch_path = temp / "epoch_metrics.csv"
    pd.DataFrame([{"time": "before", "event": "checkpoint", "epoch": 1, "global_step": 10}]).to_csv(epoch_path, index=False, encoding="utf-8-sig")
    callback = AtomicEpochCallback(epoch_path)
    callback.on_log(None, SimpleNamespace(epoch=2, global_step=20), None, logs={"loss": 0.5})
    restored = pd.read_csv(epoch_path)
    assert len(restored) == 2
    assert set(restored["global_step"].astype(int)) == {10, 20}

print("RECOVERY TEST PASS: valid checkpoint selected and prior epoch log preserved")
