"""Small numpy statistics helpers (no scipy or sklearn needed)."""
import numpy as np


def softmax(logits):
    z = logits.astype(np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def average_ranks(x):
    """Ranks 1..n, with tied values sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    _, first, counts = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)
    return ranks


def rank_average(*scores):
    """Average of each score's normalized rank, in [0, 1]. Puts signals on different scales on equal footing."""
    return np.mean([average_ranks(s) / len(s) for s in scores], axis=0)


def auroc(scores, positives):
    """Area under the ROC curve via the rank-sum formula. Same value as sklearn.metrics.roc_auc_score."""
    positives = np.asarray(positives, dtype=bool)
    num_pos, num_neg = positives.sum(), (~positives).sum()
    rank_sum = average_ranks(scores)[positives].sum()
    return float((rank_sum - num_pos * (num_pos + 1) / 2) / (num_pos * num_neg))


def precision_at_k(scores, positives, k):
    top_k = np.argsort(-scores, kind="mergesort")[:k]
    return float(np.asarray(positives)[top_k].mean())


def fit_two_gaussians(x, iterations=200):
    """1-D two-component Gaussian mixture fitted with EM.

    Returns (weights, means, stds, posterior_of_upper_component), with the
    components sorted so that index 1 has the larger mean.
    """
    x = np.asarray(x, dtype=np.float64)
    means = np.percentile(x, [25, 75])
    stds = np.full(2, x.std())
    weights = np.full(2, 0.5)
    for _ in range(iterations):
        density = weights * np.exp(-0.5 * ((x[:, None] - means) / stds) ** 2) / stds
        responsibility = density / density.sum(axis=1, keepdims=True)
        weights = responsibility.mean(axis=0)
        means = (responsibility * x[:, None]).sum(axis=0) / responsibility.sum(axis=0)
        variances = (responsibility * (x[:, None] - means) ** 2).sum(axis=0) / responsibility.sum(axis=0)
        stds = np.sqrt(np.maximum(variances, 1e-8))
    upper = np.argsort(means)
    return weights[upper], means[upper], stds[upper], responsibility[:, upper[1]]
