"""Task 3, step 2: a noise score for each of the 45k training images (higher = label more likely wrong).

Uses only the noisy labels and the five cross-validation runs. Two signals:
  cv_margin  held-out confidence: max prob of another class - prob of the given
             label, from the run that never trained on the image, with
             softmax probabilities averaged over all 60 epochs
  aum        Area Under the Margin (Pleiss et al., 2020), negated: the given
             label's logit minus the largest other logit, averaged over all
             epochs of the four runs that trained on the image
The submitted score is the rank average of the two. This combination was
fixed in advance; clean labels never chose it (task3_check.py only measures it).

Also estimates the noise rate without clean labels, by fitting a
two-component Gaussian mixture to cv_margin. Task 4 uses this estimate as its
filtering fraction and co-teaching forget rate.

Outputs: predictions/t3_noise_scores.csv, analysis/task3_signals.npz,
analysis/task3_noise_estimate.json. Needs only numpy.
"""
import csv
import json

import numpy as np

from data import ANALYSIS_DIR, NOISY, NUM_FOLDS, PREDICTIONS_DIR, load_labels, train_indices
from outputs import update_results
from runs import load_epoch, num_epochs, run_dir
from stats import fit_two_gaussians, rank_average, softmax

NOISE_ESTIMATE_PATH = ANALYSIS_DIR / "task3_noise_estimate.json"
SIGNALS_PATH = ANALYSIS_DIR / "task3_signals.npz"


def given_label_margin(values, labels):
    """values[given label] - max over the other classes, per row."""
    rows = np.arange(len(labels))
    given = values[rows, labels]
    others = values.copy()
    others[rows, labels] = -np.inf
    return given - others.max(axis=1)


def compute_signals(labels):
    """Held-out probabilities and in-fold margins from the cross-validation runs."""
    num_images = len(labels)
    held_out_probs = np.zeros((num_images, 10))
    margin_sum = np.zeros(num_images)
    margin_count = np.zeros(num_images)

    for k in range(NUM_FOLDS):
        run = run_dir(f"t3_fold{k}")
        in_fold = np.zeros(num_images, dtype=bool)
        in_fold[np.load(run / "train_on.npy")] = True
        epochs = num_epochs(run)
        for epoch in range(1, epochs + 1):
            logits = load_epoch(run, epoch)["train_logits"].astype(np.float64)
            held_out_probs[~in_fold] += softmax(logits[~in_fold]) / epochs
            margin_sum[in_fold] += given_label_margin(logits[in_fold], labels[in_fold])
            margin_count[in_fold] += 1

    return {
        "cv_loss": 1 - held_out_probs[np.arange(num_images), labels],   # 1 - p(given label)
        "cv_margin": -given_label_margin(held_out_probs, labels),
        "aum": -(margin_sum / margin_count),
        "held_out_probs": held_out_probs,
    }


def area_under_margin(run, labels):
    """Negated AUM of every training image over all epochs of one run (higher = more suspicious)."""
    epochs = num_epochs(run)
    margin_sum = np.zeros(len(labels))
    for epoch in range(1, epochs + 1):
        margin_sum += given_label_margin(load_epoch(run, epoch)["train_logits"].astype(np.float64), labels)
    return -(margin_sum / epochs)


def estimate_noise_rate(score, fitted_on):
    """Fraction of images the upper (likely mislabeled) mixture component claims."""
    weights, means, stds, upper_posterior = fit_two_gaussians(score)
    return {
        "estimated_noise_rate": round(float((upper_posterior > 0.5).mean()), 4),
        "mixture_weights": weights.round(4).tolist(),
        "mixture_means": means.round(4).tolist(),
        "mixture_stds": stds.round(4).tolist(),
        "fitted_on": fitted_on,
    }


def main():
    train_idx = train_indices()
    labels = load_labels(NOISY)[train_idx]
    signals = compute_signals(labels)
    score = rank_average(signals["cv_margin"], signals["aum"])
    estimate = estimate_noise_rate(signals["cv_margin"], fitted_on="cv_margin")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(SIGNALS_PATH, train_idx=train_idx, score=score, **signals)
    NOISE_ESTIMATE_PATH.write_text(json.dumps(estimate, indent=2) + "\n")
    with open(PREDICTIONS_DIR / "t3_noise_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "score"])
        writer.writerows(zip(train_idx.tolist(), np.round(score, 6).tolist()))

    update_results({"task3": {
        "method": "rank average of 5-fold cross-validated margin (held-out softmax averaged over "
                  "60 epochs) and negated Area Under the Margin from the in-fold runs",
        "estimated_noise_rate_no_clean_labels": estimate["estimated_noise_rate"],
    }})
    print(f"wrote {len(score)} scores; estimated noise rate (no clean labels): "
          f"{estimate['estimated_noise_rate']:.2%}")


if __name__ == "__main__":
    main()
