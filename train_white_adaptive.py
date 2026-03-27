import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
from pyautoattack import AutoAttack

# ====================== 超参数 ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 120
EPS = 8 / 255
PGD_STEP = 2 / 255
CONS_LAMBDA = 2.0
DATASET = 'CIFAR100'
SAVE_PATH = f"model_white_{DATASET}_adaptive.pth"

# ====================== 数据集 ======================
def get_loader(dataset='CIFAR10'):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    if dataset == 'CIFAR10':
        train_ds = datasets.CIFAR10('./data', train=True, transform=train_tf)
        test_ds = datasets.CIFAR10('./data', train=False, transform=test_tf)
        nc = 10
    else:
        train_ds = datasets.CIFAR100('./data', train=True, transform=train_tf)
        test_ds = datasets.CIFAR100('./data', train=False, transform=test_tf)
        nc = 100
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader, nc

# ====================== PreActResNet18 ======================
class PreActBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inp)
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or inp != out:
            self.shortcut = nn.Conv2d(inp, out, 1, stride, bias=False)
    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + sc

class PreActResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.inp = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make(PreActBlock, 64, 2, 1)
        self.layer2 = self._make(PreActBlock, 128, 2, 2)
        self.layer3 = self._make(PreActBlock, 256, 2, 2)
        self.layer4 = self._make(PreActBlock, 512, 2, 2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, nc)
    def _make(self, block, out, blocks, stride):
        layers = [block(self.inp, out, stride)]
        self.inp = out
        for _ in range(blocks-1):
            layers.append(block(out, out))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        return self.linear(out.flatten(1))

# ====================== 攻击函数 ======================
def pgd_attack(model, x, y, steps=10):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad = True
        loss = F.cross_entropy(model(x_adv), y)
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + PGD_STEP * g.sign()).clamp(x - EPS, x + EPS).clamp(0, 1).detach()
    return x_adv

def cw_attack(model, x, y, steps=20):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = (logits.max(1)[0] - logits[range(len(y)), y]).sum()
        g = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + PGD_STEP * g.sign()).clamp(x - EPS, x + EPS).clamp(0, 1).detach()
    return x_adv

# ====================== 难度感知核心 ======================
def entropy_sort(model, x):
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(x), dim=1)
        ent = -torch.sum(p * torch.log(p + 1e-8), dim=1)
    if ent.numel() > 1:
        ent_min, ent_max = ent.min(), ent.max()
        ent_norm = (ent - ent_min) / (ent_max - ent_min + 1e-8)
    else:
        ent_norm = torch.zeros_like(ent)
    return ent_norm, ent.argsort(), ent.argsort(descending=True)

def adaptive_mixup(x, y, ent_norm):
    B, C, H, W = x.size()
    dev = x.device
    alpha_easy = 0.5
    alpha_hard = 2.0
    mixed_x = x.clone()
    ya, yb, lams = y.clone(), y.clone(), []
    for i in range(B):
        alpha = alpha_easy + (alpha_hard - alpha_easy) * ent_norm[i].item()
        lam = np.random.beta(max(alpha, 0.1), max(alpha, 0.1))
        idx = torch.randint(0, B, (1,)).item()
        mixed_x[i] = lam * x[i] + (1-lam) * x[idx]
        ya[i] = y[i]
        yb[i] = y[idx]
        lams.append(lam)
    return mixed_x, ya, yb, torch.tensor(lams, device=dev).float()

def adaptive_cutmix(x, y, ent_norm):
    B, C, H, W = x.size()
    dev = x.device
    r_easy, r_hard = 0.15, 0.65
    mixed_x = x.clone()
    ya, yb, lams = y.clone(), y.clone(), []
    for i in range(B):
        r = r_easy + (r_hard - r_easy) * ent_norm[i].item()
        area = r * H * W
        ch = int(np.sqrt(area))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - ch//2, 0, W)
        x2 = np.clip(cx + ch//2, 0, W)
        y1 = np.clip(cy - ch//2, 0, H)
        y2 = np.clip(cy + ch//2, 0, H)
        idx = np.random.randint(B)
        mixed_x[i,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
        ya[i] = y[i]
        yb[i] = y[idx]
        lam = 1 - ((x2-x1)*(y2-y1))/(H*W)
        lams.append(lam)
    return mixed_x, ya, yb, torch.tensor(lams, device=dev).float()

# ====================== 自适应 Mix2Aug 损失 ======================
def mix2aug_loss(model, x, y):
    ent_norm, _, _ = entropy_sort(model, x)
    x_mu, ya1, yb1, lam1 = adaptive_mixup(x, y, ent_norm)
    x_cm, ya2, yb2, lam2 = adaptive_cutmix(x, y, ent_norm)

    model.train()
    o_mu = model(x_mu)
    o_cm = model(x_cm)

    loss_mu = (lam1 * F.cross_entropy(o_mu, ya1) + (1-lam1) * F.cross_entropy(o_mu, yb1)).mean()
    loss_cm = (lam2 * F.cross_entropy(o_cm, ya2) + (1-lam2) * F.cross_entropy(o_cm, yb2)).mean()

    p_mu = F.softmax(o_mu, dim=1)
    p_cm = F.softmax(o_cm, dim=1)
    m = (p_mu + p_cm) / 2
    js = 0.5 * (F.kl_div(torch.log(m+1e-8), p_mu, reduction='batchmean') +
                F.kl_div(torch.log(m+1e-8), p_cm, reduction='batchmean'))
    return (loss_mu + loss_cm) / 2 + CONS_LAMBDA * js

# ====================== 训练 & 评估 ======================
def train_epoch(model, loader, opt):
    model.train()
    total = 0
    for x,y in tqdm(loader):
        x,y = x.to(DEVICE), y.to(DEVICE)
        x_adv = pgd_attack(model, x, y)
        loss = mix2aug_loss(model, x_adv, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_white(model, loader):
    model.eval()
    clean = p20 = p100 = cw = total = 0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        clean += (model(x).argmax(1)==y).sum().item()
        p20 += (model(pgd_attack(model,x,y,20)).argmax(1)==y).sum().item()
        p100 += (model(pgd_attack(model,x,y,100)).argmax(1)==y).sum().item()
        cw += (model(cw_attack(model,x,y)).argmax(1)==y).sum().item()
        total += y.size(0)

    adversary = AutoAttack(model, norm='Linf', eps=EPS, version='standard', device=DEVICE)
    xs, ys = [], []
    for x,y in loader:
        xs.append(x)
        ys.append(y)
    xs, ys = torch.cat(xs).to(DEVICE), torch.cat(ys).to(DEVICE)
    aa = (adversary.run_standard(xs, ys) == ys).float().mean().item()
    return {'Clean':clean/total, 'PGD20':p20/total, 'PGD100':p100/total, 'CW':cw/total, 'AA':aa}

# ====================== 主程序 ======================
if __name__ == '__main__':
    tr, te, nc = get_loader(DATASET)
    model = PreActResNet18(nc).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[40,80], gamma=0.1)

    for e in range(EPOCHS):
        loss = train_epoch(model, tr, opt)
        print(f"Epoch {e+1:3d} | Loss {loss:.3f}")
        sch.step()

    torch.save(model.state_dict(), SAVE_PATH)
    print("Saved:", SAVE_PATH)
    res = eval_white(model, te)
    for k,v in res.items():
        print(f"{k:8s} {v:.4f}")