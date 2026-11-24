"""Independent implementation of MA-VR, paper Eqs. (1), (4)-(13)."""
import torch
from torch import nn
from torch.nn import functional as F


class MA_VR(nn.Module):
    def __init__(self, video_dim, query_dim, dim=256, heads=8, max_len=2048):
        super().__init__()
        self.vproj, self.qproj = nn.Linear(video_dim, dim), nn.Linear(query_dim, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.vid = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.text = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.cross = nn.TransformerDecoderLayer(dim, heads, dim * 4, batch_first=True)
        self.pool = nn.Linear(dim, 1)
        self.boundary = nn.Conv1d(dim, 2, 3, padding=1)
        self.start = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.end = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.filter = nn.Sequential(nn.Conv1d(4 * dim, dim, 3, padding=1), nn.ReLU(), nn.Conv1d(dim, 1, 3, padding=1))
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def encode(self, video, query):
        v = self.vid(self.vproj(video) + self.pos(torch.arange(video.shape[1], device=video.device)))
        q = self.text(self.qproj(query) + self.pos(torch.arange(query.shape[1], device=query.device)))
        pooled_q = (self.pool(q).softmax(1) * q).sum(1)
        return v, q, v.mean(1), pooled_q

    def forward(self, video, query):
        v, q, _, pq = self.encode(video, query)
        fused = self.cross(v, q)
        s, e = self.boundary(fused.transpose(1, 2)).unbind(1)
        combined = torch.cat([fused, self.start(s[..., None]), self.end(e[..., None]), pq[:, None].expand_as(fused)], -1)
        gate = self.filter(combined.transpose(1, 2)).sigmoid().transpose(1, 2)
        # Figure 3 includes a residual addition omitted from the shorthand Eq. 7.
        relevance = self.score((fused + fused * gate).mean(1)).squeeze(-1)
        return s, e, relevance


def ranking_loss(q, v, ids, margin=0.1):
    scores = q @ v.T
    valid = ids[:, None] != ids[None, :]
    if not valid.any():
        return scores.sum() * 0
    diagonal = scores.diag()
    return (F.relu(margin + scores - diagonal[:, None])[valid].mean() +
            F.relu(margin + scores - diagonal[None, :])[valid].mean()) / 2


def objective(start, end, relevance, spans, negative_loss):
    # Shapes [batch, positive+negatives, time]; positive is always index 0.
    mr = (F.cross_entropy(start.flatten(1), spans[:, 0], reduction='sum') +
          F.cross_entropy(end.flatten(1), spans[:, 1], reduction='sum')) / 2
    vr = F.cross_entropy(relevance, torch.zeros(len(spans), dtype=torch.long, device=spans.device), reduction='sum')
    return 0.05 * negative_loss + 0.01 * mr + 0.005 * vr


def corpus_scores(start, end, relevance):
    # Normalize jointly across videos and time; retain only valid inclusive spans.
    s = start.flatten().log_softmax(0).view_as(start)
    e = end.flatten().log_softmax(0).view_as(end)
    score = s[:, :, None] + e[:, None, :] + relevance.log_softmax(0)[:, None, None]
    invalid = torch.ones(start.shape[1], start.shape[1], device=start.device, dtype=torch.bool).tril(-1)
    return score.masked_fill(invalid, -torch.inf)
