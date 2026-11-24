# MA-VR: independent paper implementation

Paper: **Moment-Aware Video Retrieval for Video Corpus Moment Retrieval**, Yura Choi, Daechul Ahn, Jonghyun Choi, IEEE Access 2025. DOI: https://doi.org/10.1109/ACCESS.2025.3542720

This is a newly written, unofficial implementation of the proposed method, not author-released code. No competing baseline is included. The PDF was used to identify equations and architecture; its prose, figures and assets are not redistributed.

## Implemented experiment

Train MA-VR and measure corpus moment R@1/5/10 at temporal IoU 0.5 and 0.7 on a held-out feature corpus. The intended benchmark is TVR validation. The executable protocol uses uniformly resampled feature bins and is **not an exact reproduction of the published TVR scores**.

- Single-layer video/query encoders and cross-modal fusion; temporal start/end heads.
- Differentiable moment-aware filter from boundary logits, query and fused video features (Eqs. 4-7).
- Corpus-wide top-k non-ground-truth videos refreshed each epoch (Eq. 8).
- Joint video/time boundary normalization, MA-VR cross-entropy and bidirectional margin ranking (Eqs. 9-12).
- Corpus span ranking with invalid end-before-start spans excluded (Eq. 13).

## Run

Python 3.12 recommended. Install a GPU-compatible PyTorch build separately if using CUDA.

```powershell
pip install -r requirements.txt
python experiment.py --smoke
python prepare.py data/train.json data/train.pt
python prepare.py data/val.json data/val.pt
python experiment.py --train data/train.pt --val data/val.pt --device cuda --epochs 20 --output outputs/tvr
```

Smoke mode uses separate synthetic training/evaluation corpora, a 32-dimensional model and two epochs. It checks execution, not published accuracy. Results and configuration are saved to outputs/run/metrics.json and weights to model.pt.

## Feature data

Obtain TVR data and precomputed features through the dataset project: https://tvr.cs.unc.edu/ . Keep its official train/validation split. Supply cached video and token-level text embeddings; no raw-video encoder or text foundation model is trained here. The preparation script consumes this JSON (paths relative to the manifest):

```json
{
  "videos": [
    {"id": "video_1", "features": "v1.npy", "subtitles": "sub1.npy", "duration": 60.0},
    {"id": "video_2", "features": "v2.npy", "subtitles": "sub2.npy", "duration": 45.0}
  ],
  "queries": [
    {"video_id": "video_1", "features": "q1.npy", "start": 10.0, "end": 20.0}
  ]
}
```

Each feature is a float32 [time-or-token, channels] NumPy array. Subtitles are optional, but use them consistently across the corpus. Include **all evaluation videos**, including distractors. A query's video_id must exist in that corpus. Seconds are converted to inclusive uniform-bin endpoints. Default resampling uses 128 video bins and 32 query bins; flags --frames and --tokens change these. Prepared .pt contains videos [V,T,Dv], queries [Q,L,Dq], video_ids [Q] and spans [Q,2]. No padding masks are necessary because sequences are resampled to equal length.

## Explicit implementation choices

The paper leaves hidden size and optimizer schedule unspecified. Here: hidden dimension 256, 8 attention heads, feed-forward width 4D, PyTorch default dropout 0.1, learned positions, scalar-to-vector boundary MLPs, width-3 convolutions, AdamW at 1e-4 for 20 epochs. These are configurable engineering choices, not recovered author settings. Cross-modal fusion uses a TransformerDecoderLayer with video targets and query memory. Subtitle features are concatenated with visual features before projection. Figure 3 specifies residual addition and average pooling: the scorer receives mean(v* + v* times filter); Eq. 7 abbreviates away this residual. We follow the figure.

The paper's specified negative count 15, batch size 16, margin 0.1 and loss coefficients 0.05/0.01/0.005 are used. Batch ranking excludes same-video pairs to avoid treating another query about the same video as a negative. Validation ranks all valid spans without temporal NMS. Uniform bin IoU is a discretized approximation of timestamp IoU. Dense feature-bank storage and all-corpus ranking are intended for a manageable feature corpus; large TVR runs need sufficient host/GPU memory. There is no claimed benchmark score or pretrained checkpoint.

## Acknowledgements

The source paper acknowledges IITP/MSIT grant RS-2022-II220077 (Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data), among other grants listed in the paper. This statement attributes the original research, not authorship or funding of this independent implementation.

