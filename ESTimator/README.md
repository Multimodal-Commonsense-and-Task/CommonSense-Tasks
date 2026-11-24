# ESTimator: independent paper implementation

Paper: **Online Generic Event Boundary Detection**, ICCV 2025.
Source: https://arxiv.org/html/2510.06855v1

Unofficial implementation written from the paper. Includes only CEA, EST/REST and OBD; no competing baseline. The target experiment is online boundary detection using cached ResNet-50 features (Kinetics-GEBD, or TAPOS).

## Run

```powershell
pip install -r requirements.txt
python experiment.py --smoke
python prepare.py data/train.json data/train.pt
python prepare.py data/val.json data/val.pt
python experiment.py --train data/train.pt --val data/val.pt --device cuda --output outputs/gebd
```

A 2048-dimensional, three-layer model can require substantial GPU memory; reduce --batch-size from the paper's 512 when needed. Smoke mode uses 16-dimensional features and two epochs. It is an execution check, not a GEBD benchmark result.

For raw videos, first extract zero-padded JPEG frames with ffmpeg at 24 FPS for Kinetics-GEBD, or 6 FPS for TAPOS. For example:

```powershell
ffmpeg -i video.mp4 -vf fps=24 data/frames/%06d.jpg
python extract_frames.py data/frames data/features/video.npy --device cuda
```

Create the frames directory beforehand. extract_frames.py uses ImageNet ResNet-50 V1 global-average-pool features, standard torchvision resize/crop/normalization, and downloads weights on first use. Already cached features can be used directly. Dataset/annotations: https://github.com/StanLei52/GEBD .

Manifest format (paths relative to the JSON, timestamps in seconds):

```json
[
  {"id": "video_1", "features": "features/video_1.npy", "fps": 24, "boundaries": [2.0, 5.5]}
]
```

Use video-disjoint training and validation manifests. Features are float32 [T,2048] NumPy arrays. Pick and document one annotator per video for this minimal protocol; preparation creates nearest-frame binary targets. A prepared .pt is a list of dictionaries with features [T,D], labels [T], and floating frame-coordinate boundaries.

## Equation mapping and assumptions

model.py implements a learned prediction token after L past features, three causal self-attention blocks, scaled cosine error, EST BCE, REST BCE, and the FIFO Gaussian z-score test. Every current prediction uses only the previous L frames; stream returns decisions immediately. OBD tests before insertion and includes detected outliers in its queue.

Defaults from the paper/appendix: L=8, K=9, queue=21, threshold=1.5, alpha=0.5, AdamW lr=1e-4, batch=512. Eight heads, feed-forward width 4D, learned positions, dropout 0.1 and 20 epochs are implementation assumptions. The decoder-only architecture is expressed with PyTorch TransformerEncoder blocks plus a causal mask (no cross-attention memory).

Eq. 4 prints a K denominator with endpoints containing K+1 values. This implementation interprets K as the **number of samples**, using exactly K past-to-present errors. Positive BCE terms use negative/positive count within the batch; all-negative batches retain normal negative BCE. No label smoothing is added. Initial L frames have no prediction; OBD needs two prior errors and uses population standard deviation with epsilon 1e-6.

## Evaluation limits

Outputs contain raw online flags and errors in predictions.json, metrics/configuration in metrics.json, and model.pt. For scoring only, contiguous positive flags are grouped at their midpoint, as described in the paper; this grouping is not part of the live decision stream.

The provided minimal evaluator uses one-to-one timestamp matching with tolerance = relative threshold times video length, and aggregates counts before F1, for thresholds 0.05..0.50. This is a **single-annotator protocol**. It does not implement the official benchmark's multi-annotator aggregation or its full split/filtering conventions. Use the official GEBD evaluator on exported predictions for published-score comparisons. No published-score reproduction is claimed.

## Acknowledgements

The original research belongs to the cited authors. This repository is maintained under the CommonSense-Tasks project (IITP/MSIT 2022-0-00077); the independently written code is not an official author release.

