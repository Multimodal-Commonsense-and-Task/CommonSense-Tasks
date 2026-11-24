"""COLA (Algorithm 1), neural InfoNCE critics, and signed IDI (Eq. 2)."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class Network(nn.Module):
    def __init__(self, smoke=False):
        super().__init__()
        if smoke:
            self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU()),
                nn.Sequential(nn.Conv2d(8,16,3,stride=2,padding=1),nn.ReLU()),
                nn.Sequential(nn.Conv2d(16,32,3,stride=2,padding=1),nn.ReLU())])
            width=32
        else:
            from torchvision.models import resnet18
            r=resnet18(weights=None)
            r.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
            self.blocks=nn.ModuleList([nn.Sequential(r.conv1,r.bn1,r.relu),r.layer1,r.layer2,r.layer3,r.layer4])
            width=512
        self.width=width
        self.head=nn.Linear(width,10)

    def features(self,x):
        values=[]
        for block in self.blocks: x=block(x); values.append(x)
        return values

    def forward(self,x):
        return self.head(self.features(x)[-1].mean((2,3)))


def supcon(features, labels, temperature=.07):
    z=F.normalize(features,dim=1)
    scores=z@z.T/temperature
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positives=(labels[:,None]==labels[None,:]) & ~eye
    valid=positives.any(1)
    if not valid.any(): return features.sum()*0
    logp=scores-scores.masked_fill(eye,-torch.inf).logsumexp(1,keepdim=True)
    return -(logp.masked_fill(~positives,0).sum(1)[valid]/positives.sum(1)[valid]).mean()


def cola(original, loader, device, collapse_epochs=10, align_epochs=10, collapse_lr=2e-4, align_lr=5e-6):
    model=copy.deepcopy(original).to(device)
    optimizer=torch.optim.Adam(model.blocks.parameters(),lr=collapse_lr)
    for epoch in range(collapse_epochs):
        model.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            loss=supcon(model.features(x)[-1].mean((2,3)),y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'COLA collapse {epoch+1}/{collapse_epochs}')
    model.head.reset_parameters()
    optimizer=torch.optim.Adam(model.parameters(),lr=align_lr)
    for epoch in range(align_epochs):
        model.train()
        for x,y in loader:
            loss=F.cross_entropy(model(x.to(device)),y.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f'COLA align {epoch+1}/{align_epochs}')
    return model


class Critic(nn.Module):
    def __init__(self, model, layer, embedding=128):
        super().__init__()
        self.tail=copy.deepcopy(nn.Sequential(*list(model.blocks)[layer+1:]))
        for module in self.tail.modules():
            if hasattr(module,'reset_parameters'): module.reset_parameters()
        self.projection=nn.Linear(model.width,embedding)
        self.label=nn.Embedding(2,embedding)

    def forward(self,z,y):
        f=self.projection(self.tail(z).mean((2,3)))
        return f@self.label(y).T


def nce_bound(scores):
    return math.log(len(scores))-F.cross_entropy(scores,torch.arange(len(scores),device=scores.device))


def information_curve(model, loader, device, forget_class=0, epochs=10, f_lr=2e-5, g_lr=5e-4):
    model.eval(); result=[]
    for layer in range(len(model.blocks)-2,len(model.blocks)):
        critic=Critic(model,layer).to(device)
        opt=torch.optim.Adam([{'params':list(critic.tail.parameters())+list(critic.projection.parameters()),'lr':f_lr},
                              {'params':critic.label.parameters(),'lr':g_lr}])
        for _ in range(epochs):
            critic.train()
            for x,y in loader:
                with torch.no_grad(): z=model.features(x.to(device))[layer]
                bound=nce_bound(critic(z,(y.to(device)==forget_class).long()))
                opt.zero_grad(); (-bound).backward(); opt.step()
        critic.eval(); total=count=0
        with torch.no_grad():
            for x,y in loader:
                z=model.features(x.to(device))[layer]
                total+=nce_bound(critic(z,(y.to(device)==forget_class).long())).item()*len(y)
                count+=len(y)
        result.append(total/count)
    return result


def idi(original, retrain, unlearned, tolerance=1e-6):
    o,r,u=[torch.as_tensor(v,dtype=torch.float64) for v in (original,retrain,unlearned)]
    if o.shape != r.shape or r.shape != u.shape: raise ValueError('MI curves must align')
    denominator=(o-r).sum().item()
    if abs(denominator)<tolerance: return None
    return (u-r).sum().item()/denominator
