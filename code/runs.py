"""Reading and writing the per-run logs that train() leaves in runs/<run_name>/.

Kept free of torch so analysis scripts can run anywhere numpy does.
"""
import csv
import json

import numpy as np

from data import RUNS_DIR, val_indices


def run_dir(run_name):
    return RUNS_DIR / run_name


def num_epochs(run_dir):
    return len(list((run_dir / "epochs").glob("epoch_*.npz")))


def load_epoch(run_dir, epoch):
    """Logits saved after one epoch: train_logits (train_idx order), val_logits, test_logits."""
    return np.load(run_dir / "epochs" / f"epoch_{epoch:02d}.npz")


def test_predictions(run_dir, epoch):
    return load_epoch(run_dir, epoch)["test_logits"].argmax(1)


def val_accuracy_per_epoch(run_dir, labels):
    """Validation accuracy of every saved epoch against `labels` (an array over all 50k images)."""
    val_labels = np.asarray(labels)[val_indices()]
    return np.array([
        (load_epoch(run_dir, epoch)["val_logits"].argmax(1) == val_labels).mean()
        for epoch in range(1, num_epochs(run_dir) + 1)
    ])


def best_epoch(run_dir, labels):
    """(epoch, accuracy) with the highest validation accuracy; ties go to the earliest epoch."""
    accuracies = val_accuracy_per_epoch(run_dir, labels)
    epoch = int(np.argmax(accuracies)) + 1  # epochs are 1-based
    return epoch, float(accuracies[epoch - 1])


def load_summary(run_dir):
    return json.loads((run_dir / "summary.json").read_text())


def load_history(run_dir):
    with open(run_dir / "history.csv", newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_history(path, history):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
