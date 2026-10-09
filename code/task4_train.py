"""Task 4: three robust-training methods, each covering a different strategy.

  gce          robust loss: generalized cross-entropy, q = 0.7 (one network)
  coteaching   sample selection: two networks teach each other with small-loss
               samples; forget rate ramps from 0 to the estimated noise rate
               over the first 10 epochs
  cv_filter    data cleaning: drop the top-X% of images by the Task 3 score and
               retrain plain cross-entropy, with X = the estimated noise rate

The estimated noise rate comes from task3_scores.py (Gaussian mixture on the
cross-validated margin), so no method sees clean labels. Everything else is
the shared train() setup, same as the Task 1 baseline.

Like the Task 2 early stop, each method submits the test predictions of its
best epoch by noisy validation accuracy: predictions/t4_<method>.csv.
"""
import argparse
import json

import numpy as np

from data import NOISY, load_labels
from losses import co_teaching, generalized_cross_entropy
from outputs import update_results, write_predictions
from runs import best_epoch, run_dir, test_predictions
from task3_scores import NOISE_ESTIMATE_PATH, SIGNALS_PATH
from train import train

METHODS = ["gce", "coteaching", "cv_filter"]


def estimated_noise_rate():
    return json.loads(NOISE_ESTIMATE_PATH.read_text())["estimated_noise_rate"]


def images_to_keep(noise_rate):
    """Positions in train_idx, minus the top noise_rate fraction by Task 3 score."""
    score = np.load(SIGNALS_PATH)["score"]
    num_drop = round(noise_rate * len(score))
    return np.sort(np.argsort(-score, kind="mergesort")[num_drop:])


def train_method(method, labels):
    name = f"t4_{method}"
    if method == "gce":
        return train(name, labels, loss_fn=generalized_cross_entropy(q=0.7))
    if method == "coteaching":
        return train(name, labels, loss_fn=co_teaching(final_forget_rate=estimated_noise_rate()), num_networks=2)
    if method == "cv_filter":
        return train(name, labels, train_on=images_to_keep(estimated_noise_rate()))
    raise ValueError(f"Unknown method: {method}")


def record_results(noisy):
    """List every finished Task 4 run in results.json and name the best one."""
    entries = []
    for method in METHODS:
        run = run_dir(f"t4_{method}")
        if not (run / "summary.json").exists():
            continue
        epoch, val_acc = best_epoch(run, noisy)
        entries.append({"name": method, "file": f"t4_{method}.csv",
                        "val_acc_noisy_labels": round(val_acc, 4), "epoch": epoch})
    if entries:
        best = max(entries, key=lambda e: e["val_acc_noisy_labels"])
        update_results({"task4": {"methods": entries, "best_method": best["name"]}})
    for e in entries:
        print(f"{e['name']:11s} noisy val acc {e['val_acc_noisy_labels']:.4f} (epoch {e['epoch']})")


def main():
    parser = argparse.ArgumentParser(description="Task 4 robust training")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()

    noisy = load_labels(NOISY)
    for method in args.methods:
        run = train_method(method, noisy)
        epoch, _ = best_epoch(run, noisy)
        write_predictions(f"t4_{method}.csv", test_predictions(run, epoch))
    record_results(noisy)


if __name__ == "__main__":
    main()
