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
SAVE_PATH = f"model_black_{DATASET}.pth"

def get_loader(ds='CIFAR10'):
    train_tf = transforms.Compose([transforms.RandomCrop(32,4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    test_tf = transforms.ToTensor()
    if ds=='CIFAR10':
        tr = datasets.CIFAR10('./data',train=True,transform=train_tf)
        te = datasets.CIFAR10('./data',train=False,transform=test_tf)
        nc=10
    else:
        tr = datasets.CIFAR100('./data',train=True,transform=train_tf)
        te = datasets.CIFAR100('./data',train=False,transform=test_tf)
        nc=100
    return DataLoader(tr,BATCH_SIZE,shuffle=True,num_workers=2),DataLoader(te,BATCH_SIZE,shuffle=False,num_workers=2),nc

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

# 攻击
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

# 训练逻辑（复用mix2aug）
def mixup(x,y):
    lam=np.random.beta(1,1)
    idx=torch.randperm(x.size(0)).to(DEVICE)
    return lam*x+(1-lam)*x[idx],y,y[idx],lam

def cutmix(x,y):
    lam=np.random.beta(1,1)
    B,_,H,W=x.shape
    idx=torch.randperm(B).to(DEVICE)
    w=int(W*np.sqrt(1-lam))
    h=int(H*np.sqrt(1-lam))
    cx,cy=np.random.randint(W),np.random.randint(H)
    x1=max(cx-w//2,0)
    x2=min(cx+w//2,W)
    y1=max(cy-h//2,0)
    y2=min(cy+h//2,H)
    x[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    return x,y,y[idx],1-(x2-x1)*(y2-y1)/(H*W)

def ent(model,x):
    model.eval()
    with torch.no_grad():
        p=F.softmax(model(x),1)
        e=-torch.sum(p*torch.log(p+1e-8),1)
    return e.argsort(),e.argsort(descending=True)

def js(p,q):
    m=(p+q)/2
    return 0.5*F.kl_div(torch.log(m+1e-8),p,'batchmean')+0.5*F.kl_div(torch.log(m+1e-8),q,'batchmean')

def lossfn(model,x,y):
    a,b=ent(model,x)
    x=torch.cat([x[a],x[b]])
    y=torch.cat([y[a],y[b]])
    x1,ya1,yb1,l1=mixup(x,y)
    x2,ya2,yb2,l2=cutmix(x,y)
    model.train()
    o1,o2=model(x1),model(x2)
    loss = F.cross_entropy(o1,ya1)*l1 + F.cross_entropy(o1,yb1)*(1-l1)
    loss2= F.cross_entropy(o2,ya2)*l2 + F.cross_entropy(o2,yb2)*(1-l2)
    return (loss+loss2)/2 + 2*js(F.softmax(o1,1),F.softmax(o2,1))

def train(model,loader,opt):
    model.train()
    s=0
    for x,y in tqdm(loader):
        x,y=x.to(DEVICE),y.to(DEVICE)
        adv=pgd(model,x,y)
        l=lossfn(model,adv,y)
        opt.zero_grad()
        l.backward()
        opt.step()
        s+=l.item()
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
    sch=optim.lr_scheduler.MultiStepLR(opt,[40,80],0.1)

    print("训练开始...")
    for e in range(EPOCHS):
        loss=train(model,tr,opt)
        print(f"Epoch {e+1:03d} Loss {loss:.3f}")
        sch.step()

    torch.save(model.state_dict(),SAVE_PATH)
    print(f"模型保存至 {SAVE_PATH}")

    res=eval_black(model,te,proxy)
    print("\n===== 黑盒攻击结果 =====")
    for k,v in res.items():print(f"{k:10s}: {v:.4f}")