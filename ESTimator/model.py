"""CEA, EST/REST training objectives and strictly causal OBD."""
from collections import deque
import statistics
import torch
from torch import nn
from torch.nn import functional as F


class CEA(nn.Module):
    def __init__(self, dim=2048, history=10, layers=3, heads=8):
        super().__init__()
        self.history = history
        self.token = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.position = nn.Parameter(torch.randn(1, history+1, dim) * .02)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim, heads, dim*4, batch_first=True), layers)

    def forward(self, past):
        if past.shape[1] != self.history:
            raise ValueError('CEA expects exactly history past features, excluding current target')
        x = torch.cat([past, self.token.expand(len(past), -1, -1)], 1) + self.position
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        return self.decoder(x, mask=causal)[:, -1]


def error(predicted, actual):
    return ((1 - F.cosine_similarity(predicted, actual, dim=-1)) / 2).clamp(1e-6, 1-1e-6)


def est_rest(errors, labels, alpha=.5):
    # [B,K], K consecutive past-to-present prediction errors.
    positives = labels.sum()
    ratio = (labels.numel()-positives) / positives.clamp_min(1)
    weights = torch.where(labels.bool(), ratio, 1.)
    est = (F.binary_cross_entropy(errors, labels, reduction='none') * weights).sum(1)
    rest = F.binary_cross_entropy(errors.mean(1), labels[:, -1], reduction='none')
    rest = rest * torch.where(labels[:, -1].bool(), ratio, 1.)
    return (est + alpha*rest).mean()


class OBD:
    def __init__(self, size=10, threshold=1.5):
        if size < 2: raise ValueError('queue size must be >=2')
        self.queue = deque(maxlen=size)
        self.threshold = threshold

    def update(self, value):
        # Test before insertion; include outliers in subsequent history (Table 7).
        boundary = False
        if len(self.queue) >= 2:
            z = (value-statistics.mean(self.queue)) / max(statistics.pstdev(self.queue), 1e-6)
            boundary = z > self.threshold
        self.queue.append(float(value))
        return boundary


@torch.no_grad()
def stream(model, features, queue_size=10, threshold=1.5):
    model.eval(); detector = OBD(queue_size, threshold)
    flags = [False]*min(model.history, len(features))
    errors = [0.]*min(model.history, len(features))
    for t in range(model.history, len(features)):
        prediction = model(features[t-model.history:t][None])
        value = error(prediction, features[t][None]).item()
        errors.append(value); flags.append(detector.update(value))
    return flags, errors
