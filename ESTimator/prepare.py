"""Pack per-video cached ResNet features and boundary timestamps in seconds."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(); p.add_argument('manifest'); p.add_argument('output')
    a=p.parse_args(); source=Path(a.manifest); records=[]
    for row in json.loads(source.read_text(encoding='utf8')):
        x=torch.tensor(np.load(source.parent/row['features'],allow_pickle=False),dtype=torch.float32)
        fps=float(row['fps']); boundaries=[float(t)*fps for t in row['boundaries']]
        if x.ndim!=2 or fps<=0 or any(t<0 or t>=len(x) for t in boundaries): raise ValueError('invalid feature or timestamp')
        labels=torch.zeros(len(x))
        for t in boundaries: labels[min(round(t),len(x)-1)]=1
        records.append({'features':x,'labels':labels,'boundaries':boundaries,'id':row['id'],'fps':fps})
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); torch.save(records,output)


if __name__=='__main__': main()
