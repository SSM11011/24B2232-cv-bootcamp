"""Clean-label measurements of a finished run (the Task 2 curves).

This file is on the clean-label allow list in data.py. It only reports
numbers about a finished run; nothing here may feed back into training,
filtering, relabeling or model selection.
"""
import numpy as np

from data import NOISY, load_clean_labels, load_labels, train_indices, val_indices
from runs import load_epoch, num_epochs


def training_dynamics(run_dir, given=NOISY):
    """Per-epoch accuracies of a run trained on the `given` label column, from saved eval-mode logits.

    train_acc_correct     prediction == label, on training images whose given label is correct
    train_acc_mislabeled  prediction == the (wrong) given label, on mislabeled training images,
                          i.e. the fraction of wrong labels the network has memorized
    true_label_recovery   prediction == the clean label, on mislabeled training images
    val_acc_noisy         validation accuracy against the given (noisy or synthetic) labels
    val_acc_clean         validation accuracy against the clean labels
    """
    train_idx, val_idx = train_indices(), val_indices()
    given_labels, clean = load_labels(given), load_clean_labels()
    train_given, train_clean = given_labels[train_idx], clean[train_idx]
    mislabeled = train_given != train_clean

    rows = []
    for epoch in range(1, num_epochs(run_dir) + 1):
        logits = load_epoch(run_dir, epoch)
        train_pred = logits["train_logits"].argmax(1)
        val_pred = logits["val_logits"].argmax(1)
        rows.append({
            "epoch": epoch,
            "train_acc_correct": _mean(train_pred[~mislabeled] == train_given[~mislabeled]),
            "train_acc_mislabeled": _mean(train_pred[mislabeled] == train_given[mislabeled]),
            "true_label_recovery": _mean(train_pred[mislabeled] == train_clean[mislabeled]),
            "val_acc_noisy": _mean(val_pred == given_labels[val_idx]),
            "val_acc_clean": _mean(val_pred == clean[val_idx]),
        })
    return rows


def _mean(matches):
    return round(float(np.mean(matches)), 4)
