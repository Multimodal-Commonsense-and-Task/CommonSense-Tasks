# IDI and COLA: independent paper implementation

Paper: **An Information Theoretic Evaluation Metric for Strong Unlearning**, AAAI 2026, arXiv v3.
Source: https://arxiv.org/html/2405.17878v3

Unofficial, newly written implementation. Implements the paper's COLA method and IDI evaluation; no other unlearning baseline. Original and retain-only Retrain are necessary reference models for the IDI denominator.

## Core experiment: CIFAR-10 single-class forgetting

```powershell
pip install -r requirements.txt
python experiment.py --smoke
python experiment.py --download --data data --device cuda --forget-class 0 --output outputs/cifar10
```

The real experiment downloads CIFAR-10 only with --download, trains Original and Retrain for 300 epochs, runs COLA collapse and alignment for 10+10 epochs, and trains separate MI critics for the last two residual stages. This is a long GPU experiment. To reuse compatible checkpoints from this implementation:

```powershell
python experiment.py --data data --device cuda --original outputs/cifar10/original.pt --retrain outputs/cifar10/retrain.pt --output outputs/reuse
```

Only state dictionaries are loaded, with weights_only=True. --epochs, --collapse-epochs, --align-epochs and --critic-epochs allow shorter exploratory runs. Smoke mode uses random 16x16 RGB images, a small convolutional encoder and one epoch per phase; its metrics are not meaningful unlearning results.

## Implementation

- COLA Algorithm 1: supervised contrastive learning of encoder using retain samples, new random head, cross-entropy alignment of all weights on retain samples.
- IDI: separate randomly initialized suffix-network critic and trainable binary label embeddings per layer/model; maximize the batch InfoNCE bound for forget-membership labels.
- Signed difference of MI curves divided by Original-minus-Retrain difference (Eq. 2). Negative IDI is preserved. A denominator below 1e-6 returns JSON null rather than an invented score.
- Saves Original, Retrain and COLA state dictionaries, per-layer MI, denominator, UA, RA, all-class test accuracy and full configuration.

## Reproduction scope and assumptions

This is runnable research code, not a validated reproduction of the paper's numbers. CIFAR ResNet-18 uses a 3x3 stride-1 stem without maxpool. Defaults follow the paper: Original/Retrain SGD lr 0.1, momentum 0.9, weight decay 5e-4, cosine schedule, 300 epochs, batch 128; COLA Adam, batch 64, collapse lr 2e-4 and align lr 5e-6. Retain training uses random crops and horizontal flips.

Choices not fully specified in the paper: SupCon temperature 0.07 and normalized features, 128-dimensional critic embedding, Adam critic optimizer, 10 critic epochs. Critic learning rates are 2e-5 for features and 5e-4 for labels. Critic fitting/reporting uses the clean training set at its natural class proportions, as a finite-batch lower-bound estimate; it is not a certified MI value and can be noisy. Independent shuffled minibatches create estimator variance. Report several seeds and convergence curves before interpreting small IDI changes. TA_all_classes explicitly includes the forgotten test class; it may differ from table-specific retain-only accuracy conventions.

Only class-wise COLA is implemented; COLA+ random-subset forgetting is outside this selected experiment. The initial smoke model and real ResNet checkpoints are not interchangeable.

## Acknowledgements

The original research belongs to the cited authors. This repository is maintained under the CommonSense-Tasks project (IITP/MSIT 2022-0-00077); this independent implementation is not official author code.

