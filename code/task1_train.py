"""Task 1: train the clean-label reference and the noisy-label cross-entropy baseline.

Both runs use the shared train() and report accuracy on the noisy validation
labels. Final-epoch test predictions go to
  predictions/t1_clean_ref.csv and predictions/t1_noisy_ce.csv.
The noisy run's per-epoch logs are reused by Task 2 (dynamics, early stopping).
"""
import argparse

from data import NOISY, load_clean_labels, load_labels
from outputs import update_results, write_predictions
from runs import load_summary, test_predictions
from train import EPOCHS, train

RESULTS_KEYS = {
    "t1_clean_ref": "val_acc_noisy_labels__clean_ref_model",
    "t1_noisy_ce": "val_acc_noisy_labels__noisy_ce_model",
}


def main():
    parser = argparse.ArgumentParser(description="Task 1 baselines")
    parser.add_argument("--runs", nargs="+", choices=list(RESULTS_KEYS), default=list(RESULTS_KEYS))
    args = parser.parse_args()

    noisy = load_labels(NOISY)
    for name in args.runs:
        # The reference model is the only training run allowed to see clean labels.
        labels = load_clean_labels() if name == "t1_clean_ref" else noisy
        run_dir = train(name, labels, val_labels=noisy)
        write_predictions(f"{name}.csv", test_predictions(run_dir, EPOCHS))
        update_results({"task1": {RESULTS_KEYS[name]: load_summary(run_dir)["final_val_acc"]}})


if __name__ == "__main__":
    main()
