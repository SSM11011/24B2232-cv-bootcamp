"""Task 1: noise rate, noisy-vs-clean confusion matrix, and the class pairs behind most of the noise.

Writes analysis/task1_confusion.png, analysis/task1_noise.json and fills in
task1.noise_rate in results.json. Needs only numpy and matplotlib.
"""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import ANALYSIS_DIR, NOISY, NUM_CLASSES, class_names, load_clean_labels, load_labels, train_indices
from outputs import update_results


def confusion_matrix(true, observed):
    """counts[i, j] = number of images whose true class is i and whose noisy label is j."""
    counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(counts, (true, observed), 1)
    return counts


def rank_confused_pairs(counts):
    """Off-diagonal errors summed over unordered pairs (cat->dog plus dog->cat), largest first."""
    pairs = [(i, j, int(counts[i, j] + counts[j, i]))
             for i, j in itertools.combinations(range(NUM_CLASSES), 2)]
    return sorted(pairs, key=lambda p: p[2], reverse=True)


def plot_confusion(row_normalized, names, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(row_normalized, cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(NUM_CLASSES), repeat=2):
        value = row_normalized[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.5 else "black")
    ax.set_xticks(range(NUM_CLASSES), names, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES), names)
    ax.set_xlabel("noisy label")
    ax.set_ylabel("true (clean) label")
    ax.set_title("CIFAR-10N labels on the 45k training split (row-normalized)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    idx = train_indices()
    noisy = load_labels(NOISY)[idx]
    clean = load_clean_labels()[idx]
    names = class_names()

    is_wrong = noisy != clean
    noise_rate = float(is_wrong.mean())
    counts = confusion_matrix(clean, noisy)
    row_normalized = counts / counts.sum(axis=1, keepdims=True)
    pairs = rank_confused_pairs(counts)
    total_errors = int(is_wrong.sum())

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(row_normalized, names, ANALYSIS_DIR / "task1_confusion.png")
    report = {
        "noise_rate": round(noise_rate, 4),
        "num_mislabeled": total_errors,
        "num_train": len(idx),
        "per_class_noise_rate": {
            names[c]: round(1 - float(row_normalized[c, c]), 4) for c in range(NUM_CLASSES)
        },
        "confusion_counts": counts.tolist(),
        "top_confused_pairs": [
            {"pair": f"{names[i]}<->{names[j]}", "count": n, "share_of_errors": round(n / total_errors, 4)}
            for i, j, n in pairs[:10]
        ],
    }
    (ANALYSIS_DIR / "task1_noise.json").write_text(json.dumps(report, indent=2) + "\n")
    update_results({"task1": {"noise_rate": round(noise_rate, 4)}})

    print(f"noise rate: {noise_rate:.2%} ({total_errors} of {len(idx)} training labels are wrong)")
    print("\nnoise rate per true class:")
    for name, rate in report["per_class_noise_rate"].items():
        print(f"  {name:11s} {rate:.2%}")
    print("\nmost confused unordered pairs:")
    cumulative = 0
    for rank, entry in enumerate(report["top_confused_pairs"], 1):
        cumulative += entry["share_of_errors"]
        print(f"  {rank:2d}. {entry['pair']:22s} {entry['count']:5d}  "
              f"{entry['share_of_errors']:6.2%} of errors  (cumulative {cumulative:6.2%})")


if __name__ == "__main__":
    main()
