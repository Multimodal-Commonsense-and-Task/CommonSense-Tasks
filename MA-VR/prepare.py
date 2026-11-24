"""Pack cached visual/text features and second-based annotations into a tensor corpus."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def resample(x, length):
    x=torch.as_tensor(x,dtype=torch.float32)
    if x.ndim!=2 or not len(x): raise ValueError('features must be nonempty [time,channels]')
    return F.interpolate(x.T[None],size=length,mode='linear',align_corners=False)[0].T


def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('output')
    p.add_argument('--frames',type=int,default=128); p.add_argument('--tokens',type=int,default=32)
    a=p.parse_args(); source=Path(a.manifest); data=json.loads(source.read_text(encoding='utf8'))
    videos=[]; durations=[]; names=[]
    for row in data['videos']:
        v=resample(np.load(source.parent/row['features'],allow_pickle=False),a.frames)
        if 'subtitles' in row:
            v=torch.cat([v,resample(np.load(source.parent/row['subtitles'],allow_pickle=False),a.frames)],-1)
        videos.append(v); names.append(row['id']); durations.append(row['duration'])
    if len(set(names))!=len(names): raise ValueError('duplicate video id')
    queries=[]; ids=[]; spans=[]
    for row in data['queries']:
        vi=names.index(row['video_id']); start,end=row['start'],row['end']; duration=durations[vi]
        if not 0<=start<end<=duration: raise ValueError('invalid seconds boundary')
        queries.append(resample(np.load(source.parent/row['features'],allow_pickle=False),a.tokens)); ids.append(vi)
        s=min(int(start/duration*a.frames),a.frames-1)
        e=min(max(s,int(np.ceil(end/duration*a.frames))-1),a.frames-1); spans.append([s,e])
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'videos':torch.stack(videos),'queries':torch.stack(queries),'video_ids':torch.tensor(ids),
                'spans':torch.tensor(spans),'corpus_ids':names,'durations':durations},output)


if __name__=='__main__': main()
