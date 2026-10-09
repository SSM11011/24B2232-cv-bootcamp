"""Task 2: training dynamics of the Task 1 noisy-label run, and early stopping.

Reads the per-epoch logits saved by train(), so nothing is retrained.
  predictions/t2_dynamics.csv            the required per-epoch curves
  predictions/t2_early_stop.csv          test predictions from the early-stopping epoch
  analysis/task2_extra_dynamics.csv      learning rate and true-label recovery per epoch
  analysis/task2_dynamics.png            the curves, for Q2
Needs only numpy and matplotlib.

The early-stopping epoch is the argmax of noisy validation accuracy, computed
here from the saved logits and the noisy labels only. Clean-label curves come
from measure.py and are used only to describe when memorization starts.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import ANALYSIS_DIR, NOISY, PREDICTIONS_DIR, load_labels
from measure import training_dynamics
from outputs import update_results, write_predictions
from runs import best_epoch, load_history, run_dir, test_predictions

NOISY_RUN = "t1_noisy_ce"
DYNAMICS_COLUMNS = ["epoch", "train_acc_correct", "train_acc_mislabeled", "val_acc_noisy", "val_acc_clean"]

# Reference palette, light mode (blue, orange, aqua), plus neutral ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_dynamics(rows, learning_rates, memorization_start, early_stop, path):
    epochs = [r["epoch"] for r in rows]
    fig, (ax_train, ax_val, ax_lr) = plt.subplots(
        3, 1, figsize=(8, 9), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.2]})

    train_curves = [
        ("train_acc_correct", "correctly labeled: predicts label", BLUE),
        ("train_acc_mislabeled", "mislabeled: predicts wrong label (memorized)", ORANGE),
        ("true_label_recovery", "mislabeled: predicts true label", AQUA),
    ]
    for key, label, color in train_curves:
        ax_train.plot(epochs, [r[key] for r in rows], color=color, lw=2, label=label)
    ax_train.set_title("Training set (eval mode, no augmentation)", loc="left", color=INK)

    val_curves = [("val_acc_noisy", "noisy labels", BLUE), ("val_acc_clean", "clean labels", ORANGE)]
    for key, label, color in val_curves:
        ax_val.plot(epochs, [r[key] for r in rows], color=color, lw=2, label=label)
    ax_val.set_title("Validation accuracy", loc="left", color=INK)

    ax_lr.plot(epochs, learning_rates, color=MUTED, lw=2)
    ax_lr.set_title("Learning rate (cosine)", loc="left", color=INK)
    ax_lr.set_xlabel("epoch")

    for ax in (ax_train, ax_val, ax_lr):
        ax.axvline(memorization_start, color=MUTED, lw=1, ls="--")
        ax.axvline(early_stop, color=MUTED, lw=1, ls=":")
        ax.grid(color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED)
    for ax in (ax_train, ax_val):
        ax.set_ylim(0, 1)
        ax.set_ylabel("accuracy")
    # The band between the memorized and true-label curves is empty early on.
    ax_train.legend(loc="lower left", bbox_to_anchor=(0.08, 0.17), frameon=False, fontsize=9)
    ax_val.legend(loc="upper left", frameon=False, fontsize=9)

    ax_train.text(memorization_start + 0.5, 0.03, f"memorization starts ({memorization_start})",
                  color=MUTED, fontsize=8)
    ax_train.text(early_stop - 0.5, 0.03, f"early stop ({early_stop})", color=MUTED, fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    run = run_dir(NOISY_RUN)

    # Early stopping: noisy validation labels only.
    early_stop, early_stop_val_acc = best_epoch(run, load_labels(NOISY))
    write_predictions("t2_early_stop.csv", test_predictions(run, early_stop))

    # Measurement: clean-label curves, for describing memorization only.
    rows = training_dynamics(run)
    memorization_start = max(rows, key=lambda r: r["val_acc_clean"])["epoch"]
    learning_rates = [h["lr"] for h in load_history(run)]

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(PREDICTIONS_DIR / "t2_dynamics.csv", rows, DYNAMICS_COLUMNS)
    extra = [{"epoch": r["epoch"], "lr": round(lr, 6), "true_label_recovery": r["true_label_recovery"]}
             for r, lr in zip(rows, learning_rates)]
    write_csv(ANALYSIS_DIR / "task2_extra_dynamics.csv", extra, ["epoch", "lr", "true_label_recovery"])
    plot_dynamics(rows, learning_rates, memorization_start, early_stop, ANALYSIS_DIR / "task2_dynamics.png")

    update_results({"task2": {
        "memorization_start_epoch": memorization_start,
        "early_stop_epoch": early_stop,
        "val_acc_noisy_labels_at_early_stop": round(early_stop_val_acc, 4),
    }})

    print(f"early stop (argmax noisy val acc): epoch {early_stop}, "
          f"noisy val acc {early_stop_val_acc:.4f}")
    print(f"memorization start (peak clean val acc): epoch {memorization_start}\n")
    print("epoch      lr  correct  memorized  true-label  val_noisy  val_clean")
    for r, lr in zip(rows, learning_rates):
        print(f"{r['epoch']:5d}  {lr:.4f}  {r['train_acc_correct']:7.3f}  {r['train_acc_mislabeled']:9.3f}  "
              f"{r['true_label_recovery']:10.3f}  {r['val_acc_noisy']:9.3f}  {r['val_acc_clean']:9.3f}")


if __name__ == "__main__":
    main()
