"""Task 3, step 3: measure the noise scores against clean labels (allowed: this is our own Task 3 check).

Nothing here feeds back into the scores, the noise-rate estimate or Task 4.
Outputs analysis/task3_check.json and analysis/task3_failures.png (worst
misses and worst false alarms, for Q3). Needs numpy and matplotlib.
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import ANALYSIS_DIR, NOISY, class_names, load_clean_labels, load_labels, train_images
from stats import auroc, precision_at_k
from task3_scores import NOISE_ESTIMATE_PATH, SIGNALS_PATH

SIGNAL_NAMES = {
    "cv_loss": "cross-validated 1 - p(given label)",
    "cv_margin": "cross-validated margin",
    "aum": "AUM (in-fold training dynamics)",
    "score": "submitted: rank average of cv_margin and AUM",
}
EXAMPLES_PER_ROW = 8


def show_examples(images, captions, title, ax_row):
    ax_row[0].set_title(title, loc="left", fontsize=10)
    for ax, image, caption in zip(ax_row, images, captions):
        ax.imshow(image)
        ax.set_xlabel(caption, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    signals = np.load(SIGNALS_PATH)
    train_idx = signals["train_idx"]
    noisy = load_labels(NOISY)[train_idx]
    clean = load_clean_labels()[train_idx]
    mislabeled = noisy != clean
    k = int(mislabeled.sum())
    names = class_names()

    quality = {
        name: {"description": description,
               "auroc": round(auroc(signals[name], mislabeled), 4),
               "precision_at_k": round(precision_at_k(signals[name], mislabeled, k), 4)}
        for name, description in SIGNAL_NAMES.items()
    }
    estimate = json.loads(NOISE_ESTIMATE_PATH.read_text())["estimated_noise_rate"]

    # Failure modes of the submitted score.
    score = signals["score"]
    held_out_pred = signals["held_out_probs"].argmax(1)
    flagged = np.zeros(len(score), dtype=bool)
    flagged[np.argsort(-score, kind="mergesort")[:k]] = True
    missed = mislabeled & ~flagged
    false_alarms = ~mislabeled & flagged
    missed_pairs = Counter(f"{names[c]} labeled {names[n]}" for c, n in zip(clean[missed], noisy[missed]))
    failures = {
        "k": k,
        "missed_errors": int(missed.sum()),
        "missed_where_held_out_model_agrees_with_wrong_label": round(
            float((held_out_pred[missed] == noisy[missed]).mean()), 4),
        "most_common_missed": missed_pairs.most_common(8),
        "false_alarms": int(false_alarms.sum()),
        "false_alarms_where_held_out_model_disagrees_with_correct_label": round(
            float((held_out_pred[false_alarms] != noisy[false_alarms]).mean()), 4),
    }

    # The scores at the cutoff Task 4 actually used (top estimated-noise-rate fraction dropped).
    num_dropped = round(estimate * len(score))
    dropped = np.zeros(len(score), dtype=bool)
    dropped[np.argsort(-score, kind="mergesort")[:num_dropped]] = True
    at_task4_cutoff = {
        "dropped": num_dropped,
        "kept": int((~dropped).sum()),
        "noise_rate_of_kept_images": round(float(mislabeled[~dropped].mean()), 4),
        "share_of_errors_removed": round(float(dropped[mislabeled].mean()), 4),
        "share_of_correct_labels_removed": round(float(dropped[~mislabeled].mean()), 4),
    }

    report = {
        "true_noise_rate": round(float(mislabeled.mean()), 4),
        "estimated_noise_rate_no_clean_labels": estimate,
        "signals": quality,
        "failures_at_k": failures,
        "at_task4_cutoff": at_task4_cutoff,
    }
    (ANALYSIS_DIR / "task3_check.json").write_text(json.dumps(report, indent=2) + "\n")

    # Worst misses: mislabeled images with the lowest scores. Worst false alarms: correct ones with the highest.
    images = train_images()[train_idx]
    worst_missed = np.flatnonzero(mislabeled)[np.argsort(score[mislabeled])[:EXAMPLES_PER_ROW]]
    worst_alarms = np.flatnonzero(~mislabeled)[np.argsort(-score[~mislabeled])[:EXAMPLES_PER_ROW]]
    fig, axes = plt.subplots(2, EXAMPLES_PER_ROW, figsize=(14, 4.4))
    show_examples(images[worst_missed],
                  [f"label {names[noisy[i]]}\ntrue {names[clean[i]]}\nscore {score[i]:.3f}" for i in worst_missed],
                  "Missed: wrong label, lowest scores", axes[0])
    show_examples(images[worst_alarms],
                  [f"label {names[noisy[i]]} (correct)\nmodel {names[held_out_pred[i]]}\nscore {score[i]:.3f}"
                   for i in worst_alarms],
                  "False alarms: correct label, highest scores", axes[1])
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "task3_failures.png", dpi=150)
    plt.close(fig)

    print(f"true noise rate {report['true_noise_rate']:.2%}, estimated without clean labels {estimate:.2%}\n")
    print(f"{'signal':10s} {'AUROC':>7s} {'P@k':>7s}   (k = {k})")
    for name, q in quality.items():
        print(f"{name:10s} {q['auroc']:7.4f} {q['precision_at_k']:7.4f}   {q['description']}")
    print(f"\nat k: {failures['missed_errors']} missed errors, "
          f"{failures['missed_where_held_out_model_agrees_with_wrong_label']:.0%} of them with the held-out "
          f"model agreeing with the wrong label")
    print("most common missed:", failures["most_common_missed"][:5])
    print("at the Task 4 cutoff:", at_task4_cutoff)


if __name__ == "__main__":
    main()
