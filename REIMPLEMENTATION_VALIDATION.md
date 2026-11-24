# Independent implementations: verification scope

These three folders contain new paper-based implementations, not official author code.
They are part of CommonSense-Tasks and deliberately have no nested Git repository.

## Checked locally (2026-11-24)

- Python 3.12, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu.
- All three `python experiment.py --smoke` commands completed training, evaluation,
  checkpoint saving and JSON output using synthetic data.
- Four unit tests check MA-VR gradient flow from relevance into boundary heads,
  valid-span masking, exclusion of same-video negatives, ESTimator prefix invariance
  to future changes and OBD queue insertion, finite SupCon gradients, IDI reference
  identities/negative values/zero denominator, and the independent InfoNCE case.
- MA-VR's Figure 3 was visually checked against the code, including residual addition.

Run the tests from the repository root:

```powershell
python -m unittest test_paper_implementations -v
```

## Not established by these checks

No full TVR, Kinetics-GEBD/TAPOS or CIFAR-10 training was performed. No result here
establishes the paper's reported accuracy. Synthetic smoke metrics are explicitly
labeled in JSON and must not be entered as research benchmark results.
Real-data commands and schema are in each README. Training datasets, downloaded
weights, generated features, checkpoints, caches and run outputs are Git-ignored.

MA-VR uses fixed-length resampled features. ESTimator's minimal scorer uses a single
annotator and video-length-normalized temporal distance; use the official evaluator
for benchmark comparisons. IDI uses finite-batch neural lower-bound estimates and
requires adequately trained reference models and critics. Unspecified architecture
and optimization choices are recorded in the respective READMEs.
