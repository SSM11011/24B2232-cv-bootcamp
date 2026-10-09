"""Noise-robust losses for Task 4. All follow train()'s loss_fn(logits, targets, idx=, epoch=) interface."""
import torch
import torch.nn.functional as F


def generalized_cross_entropy(q=0.7):
    """GCE (Zhang & Sabuncu, 2018): (1 - p_y^q) / q. q -> 0 gives CE, q = 1 gives MAE."""
    def loss_fn(logits, targets, **_):
        p_true = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        return ((1 - p_true.clamp_min(1e-7) ** q) / q).mean()
    return loss_fn


def co_teaching(final_forget_rate, ramp_epochs=10):
    """Co-teaching (Han et al., 2018) for two networks.

    In every batch each network keeps the (1 - forget_rate) fraction of samples
    with its smallest loss, and its peer is updated on those samples. The forget
    rate ramps linearly from 0 at epoch 1 to final_forget_rate at epoch ramp_epochs + 1.
    """
    def forget_rate(epoch):
        return final_forget_rate * min(1.0, (epoch - 1) / ramp_epochs)

    def loss_fn(logits, targets, epoch, **_):
        logits_a, logits_b = logits
        loss_a = F.cross_entropy(logits_a, targets, reduction="none")
        loss_b = F.cross_entropy(logits_b, targets, reduction="none")
        num_keep = max(1, round((1 - forget_rate(epoch)) * len(targets)))
        with torch.no_grad():
            picked_by_a = loss_a.argsort()[:num_keep]
            picked_by_b = loss_b.argsort()[:num_keep]
        # Each network learns from the samples its peer believes are clean.
        return loss_a[picked_by_b].mean() + loss_b[picked_by_a].mean()
    return loss_fn
