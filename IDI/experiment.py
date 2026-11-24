"""CIFAR-10 single-class COLA unlearning with original/retrain IDI references."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset,TensorDataset
from model import Network,cola,information_curve,idi


def pretrain(model,loader,epochs,device):
    opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    for epoch in range(epochs):
        model.train()
        for x,y in loader:
            loss=F.cross_entropy(model(x.to(device)),y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step(); print(f'pretrain {epoch+1}/{epochs}')
    return model


@torch.no_grad()
def accuracy(model,loader,device):
    model.eval(); correct=count=0
    for x,y in loader:
        correct+=(model(x.to(device)).argmax(1).cpu()==y).sum().item(); count+=len(y)
    return correct/count if count else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true'); p.add_argument('--download',action='store_true')
    p.add_argument('--data',default='data'); p.add_argument('--output',default='outputs/run')
    p.add_argument('--epochs',type=int,default=300); p.add_argument('--collapse-epochs',type=int,default=10)
    p.add_argument('--align-epochs',type=int,default=10); p.add_argument('--critic-epochs',type=int,default=10)
    p.add_argument('--forget-class',type=int,default=0); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cpu'); p.add_argument('--original'); p.add_argument('--retrain')
    a=p.parse_args(); torch.manual_seed(a.seed); torch.set_num_threads(2)
    if not 0<=a.forget_class<10: p.error('forget-class must be 0..9')
    if a.smoke:
        labels=torch.arange(10).repeat(8)
        train=TensorDataset(torch.randn(80,3,16,16),labels); clean=train
        test=TensorDataset(torch.randn(40,3,16,16),torch.arange(10).repeat(4))
        a.epochs=a.collapse_epochs=a.align_epochs=a.critic_epochs=1
    else:
        from torchvision import datasets,transforms
        norm=transforms.Normalize((.4914,.4822,.4465),(.247,.243,.261))
        aug=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),norm])
        plain=transforms.Compose([transforms.ToTensor(),norm])
        train=datasets.CIFAR10(a.data,train=True,download=a.download,transform=aug)
        clean=datasets.CIFAR10(a.data,train=True,transform=plain)
        test=datasets.CIFAR10(a.data,train=False,download=a.download,transform=plain)
        labels=torch.tensor(train.targets)
    retain=(labels!=a.forget_class).nonzero().flatten().tolist()
    forget=(labels==a.forget_class).nonzero().flatten().tolist()
    def loader(ds,shuffle=False,batch=128): return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0)
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    models={}
    for name,ds,checkpoint in [('original',train,a.original),('retrain',Subset(train,retain),a.retrain)]:
        model=Network(a.smoke).to(a.device)
        if checkpoint: model.load_state_dict(torch.load(checkpoint,map_location=a.device,weights_only=True))
        else: pretrain(model,loader(ds,True),a.epochs,a.device)
        models[name]=model
        torch.save(model.state_dict(),output/f'{name}.pt')
    models['cola']=cola(models['original'],loader(Subset(train,retain),True,64),a.device,a.collapse_epochs,a.align_epochs)
    torch.save(models['cola'].state_dict(),output/'cola.pt')
    curves={}; metrics={}
    for name,model in models.items():
        # Preserve natural class proportions: binary forget-membership target.
        curves[name]=information_curve(model,loader(clean,True),a.device,a.forget_class,a.critic_epochs)
        metrics[name]={'UA':1-accuracy(model,loader(Subset(clean,forget)),a.device),
                       'RA':accuracy(model,loader(Subset(clean,retain)),a.device),
                       'TA_all_classes':accuracy(model,loader(test),a.device)}
    value=idi(curves['original'],curves['retrain'],curves['cola'])
    result={'synthetic_smoke':a.smoke,'config':vars(a),'MI':curves,'IDI':value,'metrics':metrics,
            'denominator':sum(o-r for o,r in zip(curves['original'],curves['retrain']))}
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__': main()
