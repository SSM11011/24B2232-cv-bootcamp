"""Paths, images, split indices and labels.

Clean labels are behind a guard. The rules allow them only for training the
Task 1 reference model and for measurement (confusion matrix, Task 2 curves,
checking Task 3 scores). Any other file that asks for them gets an error, so
they cannot leak into Tasks 2-5 by accident.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# Outputs go next to data/ unless CV_OUTPUT_ROOT points elsewhere (used for smoke tests).
OUTPUT_ROOT = Path(os.environ.get("CV_OUTPUT_ROOT", ROOT))
RUNS_DIR = OUTPUT_ROOT / "runs"                # per-epoch logs, large, not submitted
ANALYSIS_DIR = OUTPUT_ROOT / "analysis"        # figures and tables for the report
PREDICTIONS_DIR = OUTPUT_ROOT / "predictions"
RESULTS_PATH = OUTPUT_ROOT / "results.json"

NUM_CLASSES = 10
NUM_FOLDS = 5
NOISY, SYNTHETIC, CLEAN = "noisy_label", "synthetic_label", "clean_label"

# The only scripts allowed to read clean labels. Add a file here only if it
# trains the Task 1 reference model or purely measures things.
CLEAN_LABEL_ALLOWED_FILES = {
    "task1_noise.py",   # noise rate and confusion matrix
    "task1_train.py",   # Task 1 reference model
    "measure.py",       # Task 2 curves
    "task3_check.py",   # AUROC / precision@k of the Task 3 scores, failure examples
    "task5_check.py",   # real vs synthetic noise comparison for Q5
}


class CleanLabelAccessError(RuntimeError):
    pass


def class_names():
    return (DATA_DIR / "classes.txt").read_text().split()


def train_indices():
    return np.loadtxt(DATA_DIR / "train_idx.txt", dtype=np.int64)


def val_indices():
    return np.loadtxt(DATA_DIR / "val_idx.txt", dtype=np.int64)


def cross_validation_folds(seed=0):
    """Fold number 0..NUM_FOLDS-1 for every position in train_idx (a random, equal-sized split)."""
    num_images = len(train_indices())
    folds = np.empty(num_images, dtype=np.int64)
    folds[np.random.default_rng(seed).permutation(num_images)] = np.arange(num_images) % NUM_FOLDS
    return folds


def train_images():
    """All 50k training-set images, (50000, 32, 32, 3) uint8. Row i is image index i."""
    return np.load(DATA_DIR / "train_images.npy")


def test_images():
    """(10000, 32, 32, 3) uint8, no labels."""
    return np.load(DATA_DIR / "test_images.npy")


def load_labels(column):
    """One label column for all 50k images, indexed by image index.

    Use load_clean_labels() for the clean column; this function refuses it.
    """
    if column == CLEAN:
        raise CleanLabelAccessError("Use load_clean_labels() for clean labels.")
    if column not in (NOISY, SYNTHETIC):
        raise ValueError(f"Unknown label column: {column}")
    return _read_label_column(column)


def load_clean_labels():
    """Clean labels for all 50k images. Fails unless called from an allowed file."""
    caller = Path(sys._getframe(1).f_code.co_filename).name
    if caller not in CLEAN_LABEL_ALLOWED_FILES:
        raise CleanLabelAccessError(
            f"{caller} tried to read clean labels. They may only be used for the "
            f"Task 1 reference model and for measurement "
            f"(allowed files: {sorted(CLEAN_LABEL_ALLOWED_FILES)})."
        )
    return _read_label_column(CLEAN)


def _read_label_column(column):
    table = np.genfromtxt(DATA_DIR / "labels.csv", delimiter=",", names=True, dtype=np.int64)
    if not np.array_equal(table["index"], np.arange(len(table))):
        raise ValueError("labels.csv rows are expected to be in index order 0..N-1")
    labels = np.ascontiguousarray(table[column])
    labels.setflags(write=False)  # callers that relabel must copy first
    return labels
