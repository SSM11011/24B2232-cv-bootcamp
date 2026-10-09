"""The one training function used by every task.

Every run (baseline and all methods) goes through train(), so the optimizer,
schedule, augmentation and budget are identical by construction. A method can
only change the labels, which images it trains on, the loss, and (for
co-teaching) use a second network.

Each epoch we save, under runs/<run_name>/:
  epochs/epoch_XX.npz  logits of all 45k training images (no augmentation,
                       in train_idx order), all 5k validation images and all
                       10k test images
  history.csv          loss, learning rate and accuracies per epoch
so later tasks (early stopping, noise scores, dynamics) never need to retrain.
"""
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import NOISY, RUNS_DIR, load_labels, test_images, train_images, train_indices, val_indices
from gpu_loader import GpuImages
from model import ResNet18
from outputs import update_results
from runs import write_history

SEED = 0
EPOCHS = int(os.environ.get("CV_EPOCHS", 60))  # override only for smoke tests
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EVAL_BATCH_SIZE = 1000

TRAINING_SETUP = {
    "optimizer": f"SGD, momentum {MOMENTUM}, weight decay {WEIGHT_DECAY}",
    "learning_rate_schedule": f"cosine from {LEARNING_RATE} to 0 over {EPOCHS} epochs, updated every step",
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "augmentation": "random crop 32 with zero padding 4, random horizontal flip",
}


def cross_entropy(logits, targets, **_):
    """Standard CE. Loss functions get (logits, targets, idx=, epoch=) so methods can track per-sample state."""
    return F.cross_entropy(logits, targets)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Dataset:
    """Train, validation and test images on the GPU, loaded once per process."""

    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            cls._cache = cls()
        return cls._cache

    def __init__(self):
        self.train_idx = train_indices()
        self.val_idx = val_indices()
        all_images = train_images()
        train = all_images[self.train_idx]
        # Normalization statistics come from the training split only.
        mean = (train / 255).mean(axis=(0, 1, 2)).tolist()
        std = (train / 255).std(axis=(0, 1, 2)).tolist()
        self.train = GpuImages(train, mean, std)
        self.val = GpuImages(all_images[self.val_idx], mean, std)
        self.test = GpuImages(test_images(), mean, std)


def train(run_name, labels, loss_fn=cross_entropy, val_labels=None, train_on=None, num_networks=1,
          epochs=EPOCHS, seed=SEED):
    """Train fresh ResNet-18(s) and log everything every epoch.

    labels:       int array over all 50k images (index = image index). Entries at
                  train_idx are the training targets.
    val_labels:   labels used for the validation accuracy that model selection
                  looks at. Defaults to the noisy labels, as the rules require
                  for Tasks 1-4. Pass the synthetic labels for Task 5.
    train_on:     positions in train_idx to train on (cross-validation folds,
                  filtering). Defaults to all 45k. Logits are still logged for
                  all 45k, so held-out images get predictions too.
    num_networks: 1, or 2 for methods like co-teaching. With 2 networks
                  loss_fn receives a tuple of logits, one per network, and the
                  logged predictions come from the first network.
    Returns the path of the run directory.
    """
    if val_labels is None:
        val_labels = load_labels(NOISY)
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    data = Dataset.get()
    device = "cuda"
    if train_on is None:
        train_on = np.arange(len(data.train))

    run_dir = RUNS_DIR / run_name
    shutil.rmtree(run_dir, ignore_errors=True)  # never mix epochs from an earlier run
    (run_dir / "epochs").mkdir(parents=True)
    np.save(run_dir / "train_idx.npy", data.train_idx)
    np.save(run_dir / "val_idx.npy", data.val_idx)
    np.save(run_dir / "train_on.npy", train_on)
    update_results({
        "seed": SEED,
        "training_setup": {**TRAINING_SETUP, "gpu_used": torch.cuda.get_device_name()},
        "runs": {run_name: {"seed": seed, "epochs": epochs, "status": "running"}},
    })

    train_targets_np = np.asarray(labels)[data.train_idx]
    train_targets = torch.as_tensor(train_targets_np, device=device)
    val_targets = np.asarray(val_labels)[data.val_idx]
    train_positions = torch.as_tensor(train_on, device=device)

    models = [ResNet18(num_classes=10).to(device).to(memory_format=torch.channels_last)
              for _ in range(num_networks)]
    parameters = [p for model in models for p in model.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = -(-len(train_on) // BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
    generator = torch.Generator(device=device).manual_seed(seed)

    history = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        for model in models:
            model.train()
        total_loss = torch.zeros((), device=device)
        shuffled = train_positions[torch.randperm(len(train_positions), device=device, generator=generator)]
        for idx in shuffled.split(BATCH_SIZE):
            images = data.train.augmented_batch(idx, generator)
            with torch.autocast(device, dtype=torch.bfloat16):
                logits = tuple(model(images).float() for model in models)
            if num_networks == 1:
                logits = logits[0]
            loss = loss_fn(logits, train_targets[idx], idx=idx, epoch=epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.detach() * len(idx)

        train_logits = predict(models[0], data.train)
        val_logits = predict(models[0], data.val)
        test_logits = predict(models[0], data.test)
        np.savez(run_dir / "epochs" / f"epoch_{epoch:02d}.npz",
                 train_logits=train_logits, val_logits=val_logits, test_logits=test_logits)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss.item() / len(train_on),
            "train_acc": accuracy(train_logits[train_on], train_targets_np[train_on]),
            "val_acc": accuracy(val_logits, val_targets),
            "minutes": (time.time() - start_time) / 60,
        }
        history.append(row)
        write_history(run_dir / "history.csv", history)
        print(f"[{run_name}] epoch {epoch:2d}/{epochs}  loss {row['train_loss']:.3f}  "
              f"train_acc {row['train_acc']:.4f}  val_acc {row['val_acc']:.4f}  "
              f"{row['minutes']:.1f} min", flush=True)

    for k, model in enumerate(models):
        torch.save(model.state_dict(), run_dir / f"final_model_{k}.pt")
    best = max(history, key=lambda r: r["val_acc"])
    summary = {
        "seed": seed,
        "epochs": epochs,
        "num_networks": num_networks,
        "num_train_images": len(train_on),
        "status": "done",
        "final_val_acc": round(history[-1]["val_acc"], 4),
        "best_val_acc": round(best["val_acc"], 4),
        "best_val_epoch": best["epoch"],
        "minutes": round(history[-1]["minutes"], 1),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    update_results({"runs": {run_name: summary}})
    return run_dir


@torch.no_grad()
def predict(model, images):
    """Logits for every image in order, without augmentation, as float16 numpy."""
    model.eval()
    chunks = []
    for batch in images.plain_batches(EVAL_BATCH_SIZE):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            chunks.append(model(batch).float())
    return torch.cat(chunks).cpu().numpy().astype(np.float16)


def accuracy(logits, targets):
    return float((logits.argmax(1) == targets).mean())

