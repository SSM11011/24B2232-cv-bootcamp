"""Writing prediction files and filling in results.json as we go."""
import csv
import json

import numpy as np

from data import OUTPUT_ROOT, PREDICTIONS_DIR, RESULTS_PATH, ROOT

STUDENT = {"name": "Shibam Sundar Mahakud", "roll_no": "24B2232"}


def write_predictions(filename, preds):
    """Write predictions/<filename> in the id,pred format (10,000 test images)."""
    preds = np.asarray(preds)
    if preds.shape != (10_000,):
        raise ValueError(f"Expected 10,000 test predictions, got shape {preds.shape}")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "pred"])
        writer.writerows(enumerate(preds.tolist()))
    print(f"wrote {path.relative_to(OUTPUT_ROOT)}")


def update_results(updates):
    """Merge `updates` into results.json, creating it from the template if needed."""
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text())
    else:
        results = {**json.loads((ROOT / "results_template.json").read_text()), **STUDENT}
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _deep_merge(results, updates)
    tmp = RESULTS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n")
    tmp.replace(RESULTS_PATH)  # atomic, so a crash never leaves a half-written file


def _deep_merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
