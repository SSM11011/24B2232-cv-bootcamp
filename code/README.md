# CIFAR-10N: learning from noisy human labels

Shibam Sundar Mahakud (24B2232). Every command runs from this `code/` folder. Data is read from `../data/`, and
outputs go to `../runs/` (per-epoch logs, not submitted), `../analysis/`, `../predictions/` and `../results.json`.

## Reproduce everything
```
bash run_all.sh            # Tasks 1-5 in order, 14 networks, about 3 hours on an NVIDIA Thor
bash make_submission.sh    # builds ../24B2232_cv_bootcamp.zip and runs check_format.py on it
```
`run_all.sh` runs each step with the right interpreter and stops at the first failure. Each step's output goes to
`../logs/<script>.log`.
- **Training scripts** need PyTorch with CUDA (`TRAIN_PY`, default `~/IE643/.venv/bin/python`, torch 2.14).
- **Analysis scripts** need only numpy and matplotlib (`ANALYSIS_PY`, default `python3`). They read the saved logs and
  never train.

To test the whole pipeline in a few minutes without touching the real outputs:
```
CV_OUTPUT_ROOT=/tmp/cv_smoke CV_EPOCHS=2 bash run_all.sh
```

| Step | Script | Interpreter | Networks | Outputs |
|---|---|---|---|---|
| 1 | `task1_noise.py` | analysis | – | `analysis/task1_confusion.png`, `analysis/task1_noise.json` |
| 1 | `task1_train.py` | training | 2 | `predictions/t1_clean_ref.csv`, `predictions/t1_noisy_ce.csv` |
| 2 | `task2.py` | analysis | – | `predictions/t2_dynamics.csv`, `predictions/t2_early_stop.csv`, `analysis/task2_*` |
| 3 | `task3_folds.py` | training | 5 | `runs/t3_fold{0..4}/` |
| 3 | `task3_scores.py` | analysis | – | `predictions/t3_noise_scores.csv`, `analysis/task3_signals.npz`, `analysis/task3_noise_estimate.json` |
| 3 | `task3_check.py` | analysis | – | `analysis/task3_check.json`, `analysis/task3_failures.png` |
| 4 | `task4_train.py` | training | 4 | `predictions/t4_<method>.csv` |
| 5 | `task5_train.py` | training | 3 | `predictions/t5_baseline_synth.csv`, `predictions/t5_best_synth.csv` |
| 5 | `task5_check.py` | analysis | – | `analysis/task5_check.json`, `analysis/task5_real_vs_synthetic.png` |

That is 14 networks in total, all with seed 0.

## Code layout
| File | Purpose |
|---|---|
| `model.py` | The provided ResNet-18, copied unchanged |
| `data.py` | Paths, images, splits, cross-validation folds, labels, and the clean-label guard |
| `gpu_loader.py` | All images on the GPU as uint8; random crop (pad 4) and flip done on the GPU |
| `train.py` | `train()`, the single training function behind every network |
| `losses.py` | GCE and co-teaching losses |
| `runs.py` | Reading the per-run logs; best epoch by validation accuracy |
| `outputs.py` | Writing `predictions/*.csv`; merging values into `results.json` as runs progress |
| `measure.py` | Clean-label training curves of a finished run (measurement only) |
| `stats.py` | AUROC, precision@k, rank averaging, 1-D two-component Gaussian mixture |

## Shared training setup (`train.py`)
Every network uses the same settings:
- ResNet-18 trained from scratch
- SGD, momentum 0.9, weight decay 5e-4
- Learning rate 0.1 with a per-step cosine decay to 0 over 60 epochs
- Batch size 128
- Random crop with 4-pixel zero padding, plus horizontal flip
- bf16 autocast and channels_last

A method can change only four things passed to `train()`: the labels, which training images it uses, the loss, and the
number of networks (two for co-teaching).

After every epoch, `runs/<run>/epochs/epoch_XX.npz` stores logits for all 45k training images (eval mode, no
augmentation, in `train_idx` order), the 5k validation images and the 10k test images. Early-stopped submissions,
training dynamics and noise scores are all read from these files, so nothing is retrained. `cudnn.benchmark` is on for
speed, so reruns match closely but not bit for bit.

## Clean-label guard
- `data.load_labels()` refuses the `clean_label` column.
- `data.load_clean_labels()` raises `CleanLabelAccessError` unless it is called from a file in
  `CLEAN_LABEL_ALLOWED_FILES`. Those files are `task1_train.py` (the reference model) and the measurement code:
  `task1_noise.py`, `measure.py`, `task3_check.py` and `task5_check.py`.
- No training, scoring, filtering or model-selection code can read clean labels.
- All label arrays are read-only.

## Definitions and decisions
**Task 2 columns.** All values are fractions in [0, 1], and epochs are numbered from 1.

| Column | Definition |
|---|---|
| `train_acc_correct` | On training images whose noisy label equals the clean label: fraction where prediction == label |
| `train_acc_mislabeled` | On training images whose noisy label is wrong: fraction where prediction == **the wrong noisy label**, i.e. the memorization rate |
| `val_acc_noisy` | Validation accuracy against the noisy labels |
| `val_acc_clean` | Validation accuracy against the clean labels |
| `true_label_recovery` (in `analysis/task2_extra_dynamics.csv`) | On mislabeled training images: fraction where prediction == the clean label |

**Model selection.** All selection uses noisy validation labels, or synthetic ones in Task 5. Tasks 2, 4 and 5 submit
each run's best epoch by that accuracy, with ties going to the earliest epoch. Task 1 submits the final epoch, as the task
requires.

**Memorization start (Q2).** The epoch where clean validation accuracy peaks. This is measurement only and never chooses
a model.

**Task 3 scores.**
- **Folds:** the 45k images are split at random into 5 equal folds. Run k trains on the other four folds.
- **`cv_margin`:** held-out confidence. For each image, softmax probabilities from the one run that never trained on it
  are averaged over all 60 epochs. The signal is the highest probability among the other classes minus the probability of
  the given label.
- **`aum`:** negated Area Under the Margin. The given label's logit minus the largest other logit, averaged over every
  epoch of the four runs that did train on the image.
- **Submitted score:** the rank average of `cv_margin` and `aum`. This was fixed before any clean-label check;
  `task3_check.py` only measures it, with AUROC (the rank-sum formula, equal to `sklearn.metrics.roc_auc_score`) and
  precision@k, where k is the true number of mislabeled images.

**Noise-rate estimate (no clean labels).** Fit a two-component Gaussian mixture to `cv_margin`, and take the fraction of
images in the upper component. This sets the Task 4 filter fraction and the co-teaching forget rate. The estimator was
fixed in advance and is never tuned against clean labels, even where its fit is poor.

**Task 4 methods.**
- **`gce`:** generalized cross-entropy, q = 0.7.
- **`coteaching`:** two networks. In every batch, each keeps its smallest-loss samples and updates its peer on them. The
  forget rate ramps from 0 to the estimated noise rate over epochs 1 to 11. Predictions come from the first network.
- **`cv_filter`:** drop the top X% of images by the Task 3 score, with X = the estimated noise rate, and retrain with
  plain cross-entropy.

The best method is the one with the highest noisy validation accuracy.

**Task 5.**
- Retrains cross-entropy and the best Task 4 method (read from `results.json`) on `synthetic_label`.
- If the 3-network budget allows, it also trains the next-best Task 4 method that fits, so the ranking can be compared.
- Methods that need scores or a noise rate can't afford a new 5-fold cross-validation. They use the negated AUM recorded
  during the Task 5 cross-entropy run, and the same Gaussian-mixture estimator fitted to it.
