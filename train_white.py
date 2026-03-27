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
MIX_ALPHA = 1.0
CONS_LAMBDA = 2.0
DATASET = 'CIFAR100'  # 改 CIFAR10 / CIFAR100
SAVE_PATH = f"model_white_{DATASET}.pth"

# ====================== 数据集 ======================
def get_loader(dataset='CIFAR10'):
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
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

# ====================== 模型 ======================
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
        for _ in range(blocks - 1): layers.append(block(out, out))
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

# ====================== 攻击 ======================
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

# ====================== Loss ======================
def mixup(x, y):
    lam = np.random.beta(MIX_ALPHA, MIX_ALPHA)
    idx = torch.randperm(x.size(0)).to(DEVICE)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(x, y):
    lam = np.random.beta(MIX_ALPHA, MIX_ALPHA)
    B, _, H, W = x.shape
    idx = torch.randperm(B).to(DEVICE)
    w = int(W * np.sqrt(1 - lam))
    h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - w // 2, 0), min(cx + w // 2, W)
    y1, y2 = max(cy - h // 2, 0), min(cy + h // 2, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    return x, y, y[idx], 1 - (x2 - x1) * (y2 - y1) / (H * W)

def entropy_sort(model, x):
    model.eval()
    with torch.no_grad():
        p = F.softmax(model(x), 1)
        e = -torch.sum(p * torch.log(p + 1e-8), 1)
    return e.argsort(), e.argsort(descending=True)

def js_div(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(m + 1e-8), p, reduction='batchmean') + 0.5 * F.kl_div(torch.log(m + 1e-8), q, reduction='batchmean')

def mix2aug_loss(model, x, y):
    a, b = entropy_sort(model, x)
    x = torch.cat([x[a], x[b]])
    y = torch.cat([y[a], y[b]])
    x1, ya1, yb1, l1 = mixup(x, y)
    x2, ya2, yb2, l2 = cutmix(x, y)
    model.train()
    o1, o2 = model(x1), model(x2)
    loss = F.cross_entropy(o1, ya1) * l1 + F.cross_entropy(o1, yb1) * (1 - l1)
    loss2 = F.cross_entropy(o2, ya2) * l2 + F.cross_entropy(o2, yb2) * (1 - l2)
    return (loss + loss2) / 2 + CONS_LAMBDA * js_div(F.softmax(o1, 1), F.softmax(o2, 1))

# ====================== 训练 & 评估 ======================
def train(model, loader, opt):
    model.train()
    s = 0
    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        adv = pgd_attack(model, x, y)
        loss = mix2aug_loss(model, adv, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        s += loss.item()
    return s / len(loader)

@torch.no_grad()
def eval_white(model, loader):
    model.eval()
    clean = p20 = p100 = cw = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        clean += (model(x).argmax(1) == y).sum().item()
        p20 += (model(pgd_attack(model, x, y, 20)).argmax(1) == y).sum().item()
        p100 += (model(pgd_attack(model, x, y, 100)).argmax(1) == y).sum().item()
        cw += (model(cw_attack(model, x, y)).argmax(1) == y).sum().item()
        total += y.size(0)

    adversary = AutoAttack(model, norm='Linf', eps=EPS, version='standard', device=DEVICE)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    xs, ys = torch.cat(xs).to(DEVICE), torch.cat(ys).to(DEVICE)
    aa = (adversary.run_standard(xs, ys) == ys).float().mean().item()

    return {
        'Clean': clean / total, 'PGD-20': p20 / total,
        'PGD-100': p100 / total, 'CW': cw / total, 'AA': aa
    }

# ====================== 主函数 ======================
if __name__ == '__main__':
    train_loader, test_loader, nc = get_loader(DATASET)
    model = PreActResNet18(nc).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sch = optim.lr_scheduler.MultiStepLR(opt, [40, 80], 0.1)

    # 训练
    print("训练开始...")
    for e in range(EPOCHS):
        loss = train(model, train_loader, opt)
        print(f"Epoch {e + 1:03d} | Loss {loss:.3f}")
        sch.step()

    # 保存模型
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"模型已保存至 {SAVE_PATH}")

    # 测试输出
    res = eval_white(model, test_loader)
    print("\n===== 白盒攻击结果 =====")
    for k, v in res.items():
        print(f"{k:8s}: {v:.4f}")