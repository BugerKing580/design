import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 120
EPS = 8 / 255
PGD_STEP = 2 / 255
DATASET = 'CIFAR100'
SAVE_PATH = f"model_black_{DATASET}_adaptive.pth"

def get_loader(ds='CIFAR10'):
    train_tf = transforms.Compose([transforms.RandomCrop(32,4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_tf = transforms.ToTensor()
    if ds=='CIFAR10':
        tr = datasets.CIFAR10('./data',train=True,transform=train_tf)
        te = datasets.CIFAR10('./data',train=False,transform=test_tf)
        nc=10
    else:
        tr = datasets.CIFAR100('./data',train=True,transform=train_tf)
        te = datasets.CIFAR100('./data',train=False,transform=test_tf)
        nc=100
    return DataLoader(tr,BATCH_SIZE,shuffle=True,num_workers=2), DataLoader(te,BATCH_SIZE,shuffle=False,num_workers=2), nc

class PreActBlock(nn.Module):
    expansion=1
    def __init__(self,inp,out,stride=1):
        super().__init__()
        self.bn1=nn.BatchNorm2d(inp)
        self.conv1=nn.Conv2d(inp,out,3,stride,1,bias=False)
        self.bn2=nn.BatchNorm2d(out)
        self.conv2=nn.Conv2d(out,out,3,1,1,bias=False)
        self.shortcut=nn.Sequential()
        if stride!=1 or inp!=out:
            self.shortcut=nn.Conv2d(inp,out,1,stride,bias=False)
    def forward(self,x):
        out=F.relu(self.bn1(x))
        sc=self.shortcut(out)
        out=self.conv1(out)
        out=self.conv2(F.relu(self.bn2(out)))
        return out+sc

class PreActResNet18(nn.Module):
    def __init__(self,nc=10):
        super().__init__()
        self.inp=64
        self.conv1=nn.Conv2d(3,64,3,1,1,bias=False)
        self.layer1=self._make(PreActBlock,64,2,1)
        self.layer2=self._make(PreActBlock,128,2,2)
        self.layer3=self._make(PreActBlock,256,2,2)
        self.layer4=self._make(PreActBlock,512,2,2)
        self.bn=nn.BatchNorm2d(512)
        self.linear=nn.Linear(512,nc)
    def _make(self,block,out,blocks,stride):
        layers=[block(self.inp,out,stride)]
        self.inp=out
        for _ in range(blocks-1):layers.append(block(out,out))
        return nn.Sequential(*layers)
    def forward(self,x):
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=F.relu(self.bn(out))
        out=F.avg_pool2d(out,4)
        return self.linear(out.flatten(1))

def fgsm(model,x,y):
    xadv=x.clone().detach()
    xadv.requires_grad=True
    loss=F.cross_entropy(model(xadv),y)
    g=torch.autograd.grad(loss,xadv)[0]
    return (xadv+PGD_STEP*g.sign()).clamp(x-EPS,x+EPS).clamp(0,1).detach()

def pgd(model,x,y,s=10):
    xadv=x.clone().detach()
    for _ in range(s):
        xadv.requires_grad=True
        loss=F.cross_entropy(model(xadv),y)
        g=torch.autograd.grad(loss,xadv)[0]
        xadv=(xadv+PGD_STEP*g.sign()).clamp(x-EPS,x+EPS).clamp(0,1).detach()
    return xadv

def ila(model,x,y,s=5):
    for _ in range(s):
        x.requires_grad=True
        loss=F.cross_entropy(model(x),y)
        g=torch.autograd.grad(loss,x)[0]
        x=(x+PGD_STEP*g.sign()).clamp(x-EPS,x+EPS).clamp(0,1).detach()
    return x

# ====================== 难度感知 ======================
def entropy_sort(model,x):
    model.eval()
    with torch.no_grad():
        p=F.softmax(model(x),1)
        ent=-torch.sum(p*torch.log(p+1e-8),1)
    if ent.numel()>1:
        emin,emax=ent.min(),ent.max()
        en=(ent-emin)/(emax-emin+1e-8)
    else:
        en=torch.zeros_like(ent)
    return en

def adaptive_mixup(x,y,en):
    B,C,H,W=x.size()
    dev=x.device
    ae,ah=0.5,2.0
    mx=x.clone()
    ya,yb,ls=y.clone(),y.clone(),[]
    for i in range(B):
        a=ae+(ah-ae)*en[i].item()
        l=np.random.beta(max(a,0.1),max(a,0.1))
        idx=torch.randint(0,B,(1,)).item()
        mx[i]=l*x[i]+(1-l)*x[idx]
        ya[i]=y[i]
        yb[i]=y[idx]
        ls.append(l)
    return mx,ya,yb,torch.tensor(ls,device=dev).float()

def adaptive_cutmix(x,y,en):
    B,C,H,W=x.size()
    dev=x.device
    re,rh=0.15,0.65
    mx=x.clone()
    ya,yb,ls=y.clone(),y.clone(),[]
    for i in range(B):
        r=re+(rh-re)*en[i].item()
        A=r*H*W
        ch=int(np.sqrt(A))
        cx=np.random.randint(W)
        cy=np.random.randint(H)
        x1=np.clip(cx-ch//2,0,W)
        x2=np.clip(cx+ch//2,0,W)
        y1=np.clip(cy-ch//2,0,H)
        y2=np.clip(cy+ch//2,0,H)
        idx=np.random.randint(B)
        mx[i,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
        ya[i]=y[i]
        yb[i]=y[idx]
        l=1-((x2-x1)*(y2-y1))/(H*W)
        ls.append(l)
    return mx,ya,yb,torch.tensor(ls,device=dev).float()

def mix2aug_loss(model,x,y):
    en=entropy_sort(model,x)
    xm,ya1,yb1,l1=adaptive_mixup(x,y,en)
    xc,ya2,yb2,l2=adaptive_cutmix(x,y,en)
    model.train()
    om=model(xm)
    oc=model(xc)
    lm=(l1*F.cross_entropy(om,ya1)+(1-l1)*F.cross_entropy(om,yb1)).mean()
    lc=(l2*F.cross_entropy(oc,ya2)+(1-l2)*F.cross_entropy(oc,yb2)).mean()
    pm=F.softmax(om,1)
    pc=F.softmax(oc,1)
    m=(pm+pc)/2
    js=0.5*(F.kl_div(torch.log(m+1e-8),pm,reduction='batchmean')+F.kl_div(torch.log(m+1e-8),pc,reduction='batchmean'))
    return (lm+lc)/2 + 2.0*js

def train(model,loader,opt):
    model.train()
    s=0
    for x,y in tqdm(loader):
        x,y=x.to(DEVICE),y.to(DEVICE)
        adv=pgd(model,x,y)
        loss=mix2aug_loss(model,adv,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        s+=loss.item()
    return s/len(loader)

@torch.no_grad()
def eval_black(model,loader,proxy):
    model.eval()
    fg=p10=p20=fi=pi=ci=t=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        xf=fgsm(proxy,x,y)
        xp10=pgd(proxy,x,y,10)
        xp20=pgd(proxy,x,y,20)
        xfi=ila(model,xf,y)
        xpi=ila(model,xp10,y)
        xci=ila(model,xp20,y)
        fg+=(model(xf).argmax(1)==y).sum()
        p10+=(model(xp10).argmax(1)==y).sum()
        p20+=(model(xp20).argmax(1)==y).sum()
        fi+=(model(xfi).argmax(1)==y).sum()
        pi+=(model(xpi).argmax(1)==y).sum()
        ci+=(model(xci).argmax(1)==y).sum()
        t+=y.size(0)
    return {"FGSM":fg/t,"PGD10":p10/t,"PGD20":p20/t,"FGSM_ILA":fi/t,"PGD_ILA":pi/t,"CW_ILA":ci/t}

if __name__=="__main__":
    tr,te,nc=get_loader(DATASET)
    model=PreActResNet18(nc).to(DEVICE)
    proxy=PreActResNet18(nc).to(DEVICE)
    opt=optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    sch=optim.lr_scheduler.MultiStepLR(opt,milestones=[40,80],gamma=0.1)
    for e in range(EPOCHS):
        loss=train(model,tr,opt)
        print(f"Epoch {e+1:3d} Loss {loss:.3f}")
        sch.step()
    torch.save(model.state_dict(),SAVE_PATH)
    print("Saved to",SAVE_PATH)
    res=eval_black(model,te,proxy)
    for k,v in res.items():print(f"{k:10s} {v:.4f}")