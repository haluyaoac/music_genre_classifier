import os
# 关键：避免 torch 导入/compile 扫描卡顿（你之前遇到过）
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SimpleCNN
from split_dataset import SplitDataset


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def save_curves(log_csv_path, out_png_path):
    # 读 csv
    epochs, train_loss, val_acc = [], [], []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            e, tl, va, sec = line.strip().split(",")
            epochs.append(int(e))
            train_loss.append(float(tl))
            val_acc.append(float(va))

    # 画两张图（同一张画布里用双 y 轴，简单直观）
    fig = plt.figure()
    ax1 = plt.gca()
    ax1.plot(epochs, train_loss)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_acc)
    ax2.set_ylabel("Val Acc")
    ax2.set_ylim(0, 1)

    plt.title("Training Curves")
    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close(fig)


def main():
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 路径
    split_json = "data/splits/split.json"
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # 读取 label_map（保证类别顺序一致）
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    genres = [label_map[str(i)] for i in range(len(label_map))]

    # 数据集：train 用 random_clip=True，val 用 False（更稳定、可复现）
    train_ds = SplitDataset(split_json, split="train", random_clip=True)
    val_ds   = SplitDataset(split_json, split="val", random_clip=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # 模型/优化器
    model = SimpleCNN(n_classes=len(genres)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    # 日志
    log_csv = "runs/train_log.csv"
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_acc,seconds\n")

    best_acc = -1.0
    best_path = "models/cnn_melspec.pth"

    EPOCHS = 20
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_acc = eval_acc(model, val_loader, device)
        sec = time.time() - t0

        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} val_acc={val_acc:.4f} time={sec:.1f}s")

        # 记录日志
        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_acc:.6f},{sec:.3f}\n")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # 训练结束：画曲线
    curve_png = "runs/train_curves.png"
    save_curves(log_csv, curve_png)
    print("saved:", best_path)
    print("saved:", log_csv)
    print("saved:", curve_png)


if __name__ == "__main__":
    main()
