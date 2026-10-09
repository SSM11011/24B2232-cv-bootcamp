"""Task 5 measurement: how real (human) and synthetic (symmetric) noise differ, for Q5.

Compares the two cross-entropy baselines, t1_noisy_ce (real) and t5_ce
(synthetic), on equal terms:
  - memorization rate and clean validation accuracy per epoch
  - how well the AUM of that single run separates mislabeled images (AUROC)
Uses clean labels for measurement only. Nothing here feeds back into training
or selection. Outputs analysis/task5_check.json and analysis/task5_real_vs_synthetic.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import ANALYSIS_DIR, NOISY, SYNTHETIC, load_clean_labels, load_labels, train_indices
from measure import training_dynamics
from runs import run_dir
from stats import auroc, precision_at_k
from task3_scores import area_under_margin

BLUE, ORANGE = "#2a78d6", "#eb6834"      # real vs synthetic noise
AQUA, VIOLET = "#1baf7a", "#4a3aa7"      # correct vs wrong label
MUTED, GRID = "#52514e", "#e4e3df"
NOISE_TYPES = {"real": ("t1_noisy_ce", NOISY), "synthetic": ("t5_ce", SYNTHETIC)}


def summarize(run_name, column, clean):
    run = run_dir(run_name)
    given = load_labels(column)[train_indices()]
    mislabeled = given != clean
    score = area_under_margin(run, given)
    dynamics = training_dynamics(run, given=column)
    peak = max(dynamics, key=lambda r: r["val_acc_clean"])
    return {
        "noise_rate": round(float(mislabeled.mean()), 4),
        "aum_auroc": round(auroc(score, mislabeled), 4),
        "aum_precision_at_k": round(precision_at_k(score, mislabeled, int(mislabeled.sum())), 4),
        "peak_clean_val_epoch": peak["epoch"],
        "peak_clean_val_acc": peak["val_acc_clean"],
        "final_clean_val_acc": dynamics[-1]["val_acc_clean"],
        "final_memorization_rate": dynamics[-1]["train_acc_mislabeled"],
    }, dynamics, score, mislabeled


def main():
    clean = load_clean_labels()[train_indices()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    report = {}

    for noise_type, color in [("real", BLUE), ("synthetic", ORANGE)]:
        run_name, column = NOISE_TYPES[noise_type]
        summary, dynamics, score, mislabeled = summarize(run_name, column, clean)
        report[noise_type] = {"run": run_name, **summary}

        epochs = [r["epoch"] for r in dynamics]
        axes[0].plot(epochs, [r["train_acc_mislabeled"] for r in dynamics], color=color, lw=2,
                     label=f"{noise_type}: memorized wrong labels")
        axes[0].plot(epochs, [r["val_acc_clean"] for r in dynamics], color=color, lw=2, ls="--",
                     label=f"{noise_type}: clean val acc")

        ax = axes[1] if noise_type == "real" else axes[2]
        bins = np.linspace(np.percentile(score, 0.5), np.percentile(score, 99.5), 60)
        ax.hist(score[~mislabeled], bins=bins, color=AQUA, alpha=0.6, label="correct label")
        ax.hist(score[mislabeled], bins=bins, color=VIOLET, alpha=0.6, label="wrong label")
        ax.set_title(f"{noise_type} noise: negated AUM (AUROC {summary['aum_auroc']:.3f})", loc="left")
        ax.set_xlabel("negated AUM (higher = more suspicious)")
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_title("CE baseline: memorization vs clean val acc", loc="left")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "task5_real_vs_synthetic.png", dpi=150)
    plt.close(fig)

    (ANALYSIS_DIR / "task5_check.json").write_text(json.dumps(report, indent=2) + "\n")
    for noise_type, r in report.items():
        print(f"{noise_type:9s} noise {r['noise_rate']:.2%}  AUM AUROC {r['aum_auroc']:.4f}  P@k {r['aum_precision_at_k']:.4f}  "
              f"clean val peak {r['peak_clean_val_acc']:.3f} @ {r['peak_clean_val_epoch']}, final {r['final_clean_val_acc']:.3f}  "
              f"memorized at 60: {r['final_memorization_rate']:.3f}")


if __name__ == "__main__":
    main()
