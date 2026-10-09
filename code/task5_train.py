"""Task 5: the cross-entropy baseline and the best Task 4 method, retrained on synthetic (symmetric) noise.

Runs, at most three networks in total:
  t5_ce          the Task 1 cross-entropy baseline
  t5_<best>      the best Task 4 method, read from results.json
  t5_<other>     if the budget allows, the next-best Task 4 method that fits,
                 so Q5 can compare the method ranking

Methods that need a noise score or a noise rate cannot reuse the Task 3
cross-validation (five more runs would exceed the budget). They use the negated
AUM recorded during the t5_ce run, the strongest single Task 3 signal, and the
same no-clean-label noise-rate estimator as Task 4 (a two-component Gaussian
mixture) fitted to that score.

Every run validates on, and is selected by, the synthetic validation labels
only, and submits its best epoch by that accuracy:
  predictions/t5_baseline_synth.csv  from t5_ce
  predictions/t5_best_synth.csv      from t5_<best>
"""
import json

import numpy as np

from data import ANALYSIS_DIR, RESULTS_PATH, SYNTHETIC, load_labels, train_indices
from losses import co_teaching, generalized_cross_entropy
from outputs import update_results, write_predictions
from runs import best_epoch, run_dir, test_predictions
from task3_scores import area_under_margin, estimate_noise_rate
from train import train

NETWORK_BUDGET = 3
NETWORKS_PER_METHOD = {"gce": 1, "cv_filter": 1, "coteaching": 2}
SCORES_PATH = ANALYSIS_DIR / "task5_scores.npz"
NOISE_ESTIMATE_PATH = ANALYSIS_DIR / "task5_noise_estimate.json"


def plan_methods():
    """The best Task 4 method, plus the next-best one if it still fits in the network budget."""
    task4 = json.loads(RESULTS_PATH.read_text())["task4"]
    ranking = [m["name"] for m in sorted(task4["methods"], key=lambda m: -m["val_acc_noisy_labels"])]
    best = task4["best_method"]
    plan = [best]
    networks_used = 1 + NETWORKS_PER_METHOD[best]  # the baseline is one network
    for method in ranking:
        if method != best and networks_used + NETWORKS_PER_METHOD[method] <= NETWORK_BUDGET:
            plan.append(method)
            break
    return best, plan


def baseline_noise_score(synthetic):
    """Negated AUM from the t5_ce run, and the noise rate estimated from it without clean labels."""
    score = area_under_margin(run_dir("t5_ce"), synthetic[train_indices()])
    estimate = estimate_noise_rate(score, fitted_on="negated AUM of the t5_ce run")
    np.savez(SCORES_PATH, train_idx=train_indices(), score=score)
    NOISE_ESTIMATE_PATH.write_text(json.dumps(estimate, indent=2) + "\n")
    print(f"estimated synthetic noise rate (no clean labels): {estimate['estimated_noise_rate']:.2%}")
    return score, estimate["estimated_noise_rate"]


def train_method(method, synthetic):
    name = f"t5_{method}"
    if method == "gce":
        return train(name, synthetic, loss_fn=generalized_cross_entropy(q=0.7), val_labels=synthetic)
    score, noise_rate = baseline_noise_score(synthetic)
    if method == "cv_filter":
        keep = np.sort(np.argsort(-score, kind="mergesort")[round(noise_rate * len(score)):])
        return train(name, synthetic, train_on=keep, val_labels=synthetic)
    if method == "coteaching":
        return train(name, synthetic, loss_fn=co_teaching(final_forget_rate=noise_rate), num_networks=2,
                     val_labels=synthetic)
    raise ValueError(f"Unknown method: {method}")


def main():
    synthetic = load_labels(SYNTHETIC)
    best, plan = plan_methods()
    print(f"Task 5 plan: baseline + {plan} (best Task 4 method: {best})")

    train("t5_ce", synthetic, val_labels=synthetic)
    for method in plan:
        train_method(method, synthetic)

    selected = {}
    for name in ["t5_ce"] + [f"t5_{m}" for m in plan]:
        epoch, val_acc = best_epoch(run_dir(name), synthetic)
        selected[name] = {"epoch": epoch, "val_acc_synthetic_labels": round(val_acc, 4)}
        print(f"{name:14s} synthetic val acc {val_acc:.4f} (epoch {epoch})")

    baseline, best_run = selected["t5_ce"], selected[f"t5_{best}"]
    write_predictions("t5_baseline_synth.csv", test_predictions(run_dir("t5_ce"), baseline["epoch"]))
    write_predictions("t5_best_synth.csv", test_predictions(run_dir(f"t5_{best}"), best_run["epoch"]))
    update_results({"task5": {
        "val_acc_synthetic_labels__baseline": baseline["val_acc_synthetic_labels"],
        "val_acc_synthetic_labels__best_method": best_run["val_acc_synthetic_labels"],
        "best_method": best,
        "runs": selected,
    }})


if __name__ == "__main__":
    main()
