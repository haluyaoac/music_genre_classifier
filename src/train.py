import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json
import time
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from exp_cfg import ExpCFG
from split_dataset import SplitDataset
from models_zoo import build_model


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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_curves(log_csv_path, out_png_path):
    epochs, train_loss, val_acc = [], [], []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            e, tl, va, sec = line.strip().split(",")
            epochs.append(int(e))
            train_loss.append(float(tl))
            val_acc.append(float(va))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=str, default="E0_baseline")
    ap.add_argument("--model", type=str, default=None, help="small_cnn | small_cnn_v2 | resnet18")
    ap.add_argument("--specaug", action="store_true")
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--ls", type=float, default=None, help="label_smoothing")
    ap.add_argument("--wd", type=float, default=None, help="weight_decay")
    args = ap.parse_args()

    cfg = ExpCFG(exp_name=args.exp)

    if args.model is not None:
        cfg.model_type = args.model
    if args.specaug:
        cfg.use_specaug = True
    if args.cosine:
        cfg.use_cosine = True
    if args.ls is not None:
        cfg.label_smoothing = float(args.ls)
    if args.wd is not None:
        cfg.weight_decay = float(args.wd)

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读 label_map（确保 n_classes 对齐）
    with open("models/label_map.json", "r", encoding="utf-8") as f:
        label_map = json.load(f)
    genres = [label_map[str(i)] for i in range(len(label_map))]
    cfg.n_classes = len(genres)

    out_dir = os.path.join(cfg.out_root, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    # 保存本次 cfg
    with open(os.path.join(out_dir, "exp_cfg.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, ensure_ascii=False, indent=2)

    # 数据
    train_ds = SplitDataset(
        cfg.split_json, split="train",
        sr=cfg.sr,
        clip_seconds=cfg.clip_seconds,
        hop_seconds=cfg.hop_seconds,
        random_clip=True,
        use_specaug=cfg.use_specaug,
        freq_mask_param=cfg.freq_mask_param,
        time_mask_param=cfg.time_mask_param,
        num_masks=cfg.num_masks,
    )
    val_ds = SplitDataset(
        cfg.split_json, split="val",
        sr=cfg.sr,
        clip_seconds=cfg.clip_seconds,
        hop_seconds=cfg.hop_seconds,
        random_clip=False,
        use_specaug=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # 模型
    model = build_model(cfg.model_type, cfg.n_classes).to(device)
    n_params = count_params(model)

    # 训练策略
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scheduler = None
    if cfg.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_acc = -1.0
    bad_epochs = 0
    best_path = os.path.join(out_dir, "best.pt")

    log_csv = os.path.join(out_dir, "train_log.csv")
    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_acc,seconds\n")

    for epoch in range(1, cfg.epochs + 1):
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

        if scheduler is not None:
            scheduler.step()

        print(f"[{cfg.exp_name}] epoch={epoch:02d} loss={train_loss:.4f} val_acc={val_acc:.4f} time={sec:.1f}s")

        with open(log_csv, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_acc:.6f},{sec:.3f}\n")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if cfg.early_stop_patience > 0 and bad_epochs >= cfg.early_stop_patience:
                print(f"Early stop at epoch {epoch}, best_val_acc={best_acc:.4f}")
                break

    # 曲线
    curve_png = os.path.join(out_dir, "train_curves.png")
    save_curves(log_csv, curve_png)

    # 写入 summary.csv（追加一行）
    summary_path = os.path.join(cfg.out_root, "summary.csv")
    header = "exp_name,model_type,use_specaug,use_cosine,label_smoothing,weight_decay,params,best_val_acc\n"
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(header)

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"{cfg.exp_name},{cfg.model_type},{int(cfg.use_specaug)},{int(cfg.use_cosine)},"
                f"{cfg.label_smoothing},{cfg.weight_decay},{n_params},{best_acc:.6f}\n")

    print("saved:", best_path)
    print("saved:", log_csv)
    print("saved:", curve_png)
    print("updated:", summary_path)


if __name__ == "__main__":
    main()
