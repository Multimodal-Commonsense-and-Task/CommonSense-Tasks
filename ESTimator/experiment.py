import argparse
import json
from pathlib import Path
import torch
from model import CEA, error, est_rest, stream


def fixture(seed):
    g = torch.Generator().manual_seed(seed)
    records = []
    for _ in range(4):
        centers = torch.randn(4, 16, generator=g)
        x = centers.repeat_interleave(20, 0) + .05*torch.randn(80, 16, generator=g)
        y = torch.zeros(80); y[[20, 40, 60]] = 1
        records.append({'features': x, 'labels': y, 'boundaries': [20.,40.,60.]})
    return records


def centers(flags):
    """Evaluation only: center consecutive positive predictions (paper Sec. 5.1)."""
    result, start = [], None
    for i, flag in enumerate(list(flags)+[False]):
        if flag and start is None: start = i
        if not flag and start is not None:
            result.append((start+i-1)/2); start = None
    return result


def matches(pred, truth, tolerance):
    # Maximum cardinality matching of sorted timestamps under a fixed tolerance.
    pred, truth = sorted(pred), sorted(truth)
    i = j = count = 0
    while i < len(pred) and j < len(truth):
        if abs(pred[i]-truth[j]) <= tolerance: count += 1; i += 1; j += 1
        elif pred[i] < truth[j]: i += 1
        else: j += 1
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train'); p.add_argument('--val'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--epochs', type=int, default=20); p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--history', type=int, default=8); p.add_argument('--region', type=int, default=9)
    p.add_argument('--queue', type=int, default=21); p.add_argument('--threshold', type=float, default=1.5)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu'); p.add_argument('--output', default='outputs/run')
    a = p.parse_args(); torch.manual_seed(a.seed); torch.set_num_threads(2)
    if a.smoke:
        train, val = fixture(a.seed), fixture(a.seed+100); a.epochs=2; a.batch_size=16
    else:
        if not a.train or not a.val: p.error('--train and --val required')
        train, val = [torch.load(f, weights_only=True) for f in (a.train, a.val)]
    if a.history < 1 or a.region < 1: p.error('history and region must be positive')
    dim = train[0]['features'].shape[-1]
    for row in train+val:
        assert row['features'].ndim == 2 and row['features'].shape[-1] == dim
        assert len(row['features']) == len(row['labels']) and len(row['features']) >= a.history+a.region
        assert torch.isfinite(row['features']).all()
        assert ((row['labels']==0) | (row['labels']==1)).all()
    windows = [(vi,t) for vi,row in enumerate(train) for t in range(a.history+a.region-1,len(row['features']))]
    model = CEA(dim, a.history).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    for epoch in range(a.epochs):
        model.train(); values=[]
        for indexes in torch.randperm(len(windows)).split(a.batch_size):
            past, targets, labels = [], [], []
            for index in indexes.tolist():
                vi,t = windows[index]; row=train[vi]
                for j in range(t-a.region+1,t+1):
                    past.append(row['features'][j-a.history:j]); targets.append(row['features'][j]); labels.append(row['labels'][j])
            predictions = model(torch.stack(past).to(a.device))
            errors = error(predictions, torch.stack(targets).to(a.device)).view(-1,a.region)
            loss = est_rest(errors, torch.stack(labels).to(a.device).view(-1,a.region))
            opt.zero_grad(); loss.backward(); opt.step(); values.append(loss.item())
        print(json.dumps({'epoch': epoch+1, 'loss': sum(values)/len(values)}))
    counts = [[0,0,0] for _ in range(10)]; predictions=[]
    for row in val:
        flags, errors = stream(model, row['features'].to(a.device), a.queue, a.threshold)
        pred = centers(flags); truth = row.get('boundaries', row['labels'].nonzero().flatten().tolist())
        predictions.append({'flags': flags, 'errors': errors, 'centers': pred})
        for i in range(10):
            counts[i][0] += matches(pred, truth, .05*(i+1)*len(flags))
            counts[i][1] += len(pred); counts[i][2] += len(truth)
    metrics={f'F1@{.05*(i+1):.2f}': 2*tp/max(np+ng,1) for i,(tp,np,ng) in enumerate(counts)}
    metrics['mean_F1']=sum(metrics.values())/10
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    result={'synthetic_smoke':a.smoke,'config':vars(a),'metrics':metrics}
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    (output/'predictions.json').write_text(json.dumps(predictions),encoding='utf8')
    torch.save({'state_dict':model.state_dict(),'config':vars(a),'dim':dim},output/'model.pt')
    print(json.dumps(result))


if __name__=='__main__': main()
