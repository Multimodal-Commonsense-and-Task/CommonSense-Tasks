"""Extract ImageNet ResNet-50 pool features from an already sampled frame directory."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision.models import resnet50,ResNet50_Weights


def main():
    p=argparse.ArgumentParser(); p.add_argument('frames'); p.add_argument('output')
    p.add_argument('--device',default='cpu'); p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args(); weights=ResNet50_Weights.IMAGENET1K_V1
    model=resnet50(weights=weights); model.fc=torch.nn.Identity(); model.to(a.device).eval()
    paths=sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not paths: raise ValueError('no frames; use zero-padded filenames in temporal order')
    features=[]
    with torch.no_grad():
        for start in range(0,len(paths),a.batch_size):
            batch=[]
            for path in paths[start:start+a.batch_size]:
                with Image.open(path) as im: batch.append(weights.transforms()(im.convert('RGB')))
            features.append(model(torch.stack(batch).to(a.device)).cpu().numpy())
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); np.save(output,np.concatenate(features))


if __name__=='__main__': main()
