"""Feature-based corpus moment retrieval experiment; no baseline implementations."""
import argparse
import json
from pathlib import Path
import torch
from model import MA_VR, ranking_loss, objective, corpus_scores


def fixture(seed):
    g = torch.Generator().manual_seed(seed)
    videos = torch.randn(8, 12, 16, generator=g)
    ids = torch.arange(8).repeat_interleave(2)
    spans = torch.tensor([[2, 5], [6, 9]] * 8)
    queries = torch.stack([videos[v, s:e+1].mean(0).repeat(3, 1) for v, (s, e) in zip(ids, spans)])
    return dict(videos=videos, queries=queries, video_ids=ids, spans=spans)


def validate(d):
    v, q, ids, spans = (d[k] for k in ('videos', 'queries', 'video_ids', 'spans'))
    assert v.ndim == q.ndim == 3 and len(v) >= 2
    assert len(q) == len(ids) == len(spans) and spans.shape[1] == 2
    assert ids.dtype == spans.dtype == torch.long
    assert (ids >= 0).all() and (ids < len(v)).all()
    assert (spans[:, 0] >= 0).all() and (spans[:, 0] <= spans[:, 1]).all() and (spans[:, 1] < v.shape[1]).all()
    assert torch.isfinite(v).all() and torch.isfinite(q).all()


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()
    hits = {f'R@{k}/IoU={threshold}': 0 for k in (1, 5, 10) for threshold in (0.5, 0.7)}
    videos = data['videos'].to(device)
    for query, target, span in zip(data['queries'], data['video_ids'], data['spans']):
        parts = [model(v, query.to(device)[None].expand(len(v), -1, -1)) for v in videos.split(32)]
        s, e, r = [torch.cat([p[j] for p in parts]) for j in range(3)]
        scores = corpus_scores(s, e, r)
        t = videos.shape[1]
        top = scores.flatten().topk(min(10, len(videos)*t*(t+1)//2)).indices.cpu()
        vi, si, ei = top // (t*t), (top // t) % t, top % t
        overlap = (torch.minimum(ei, span[1]) - torch.maximum(si, span[0]) + 1).clamp_min(0)
        iou = overlap / (ei-si+1 + span[1]-span[0]+1-overlap)
        for k in (1, 5, 10):
            for threshold in (0.5, 0.7):
                hits[f'R@{k}/IoU={threshold}'] += int(((vi[:k] == target) & (iou[:k] >= threshold)).any())
    return {k: v / len(data['queries']) for k, v in hits.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train'); p.add_argument('--val'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--dim', type=int, default=256); p.add_argument('--negatives', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu'); p.add_argument('--output', default='outputs/run')
    a = p.parse_args(); torch.manual_seed(a.seed); torch.set_num_threads(2)
    if a.smoke:
        train, val = fixture(a.seed), fixture(a.seed+100)
        a.epochs, a.dim, a.batch_size = 2, 32, 4
    else:
        if not a.train or not a.val: p.error('--train and --val are required')
        train = torch.load(a.train, weights_only=True); val = torch.load(a.val, weights_only=True)
    validate(train); validate(val)
    model = MA_VR(train['videos'].shape[-1], train['queries'].shape[-1], a.dim).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    videos, queries, ids, spans = [train[k].to(a.device) for k in ('videos', 'queries', 'video_ids', 'spans')]
    for epoch in range(a.epochs):
        # Refresh corpus-wide unimodal hard negatives using latest weights every epoch.
        model.eval()
        with torch.no_grad():
            vbank = torch.cat([model.encode(v, queries[:1].expand(len(v), -1, -1))[2] for v in videos.split(32)])
            qbank = torch.cat([model.encode(videos[:1].expand(len(q), -1, -1), q)[3] for q in queries.split(32)])
            sim = qbank @ vbank.T
            sim[torch.arange(len(ids), device=a.device), ids] = -torch.inf
            negatives = sim.topk(min(a.negatives, len(videos)-1), dim=1).indices
        model.train(); losses = []
        for ix in torch.randperm(len(ids), device=a.device).split(a.batch_size):
            candidates = torch.cat([ids[ix, None], negatives[ix]], 1)
            b, c = candidates.shape
            q = queries[ix, None].expand(-1, c, -1, -1).flatten(0, 1)
            s, e, r = model(videos[candidates.flatten()], q)
            _, _, pv, pq = model.encode(videos[ids[ix]], queries[ix])
            loss = objective(s.view(b,c,-1), e.view(b,c,-1), r.view(b,c), spans[ix], ranking_loss(pq, pv, ids[ix]))
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(loss.item())
        print(json.dumps({'epoch': epoch+1, 'loss': sum(losses)/len(losses)}))
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    metrics = {'synthetic_smoke': a.smoke, 'config': vars(a), 'metrics': evaluate(model, val, a.device)}
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf8')
    torch.save({'state_dict': model.state_dict(), 'config': vars(a)}, output/'model.pt')
    print(json.dumps(metrics))


if __name__ == '__main__': main()
