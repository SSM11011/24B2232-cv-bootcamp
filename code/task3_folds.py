"""Task 3, step 1: five cross-validation runs on the noisy labels (the full Task 3 budget).

Run k trains on the other four folds of the 45k training images. Because
train() logs logits for all 45k images every epoch, each image gets held-out
predictions from the one run that never trained on it, plus its training
dynamics from the four runs that did. The fold split is stored in each run as
runs/t3_fold<k>/train_on.npy.
"""
import argparse

import numpy as np

from data import NOISY, NUM_FOLDS, cross_validation_folds, load_labels
from train import SEED, train


def main():
    parser = argparse.ArgumentParser(description="Task 3 cross-validation runs")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(NUM_FOLDS)))
    args = parser.parse_args()

    noisy = load_labels(NOISY)
    folds = cross_validation_folds(seed=SEED)
    for k in args.folds:
        train(f"t3_fold{k}", noisy, train_on=np.flatnonzero(folds != k))


if __name__ == "__main__":
    main()
