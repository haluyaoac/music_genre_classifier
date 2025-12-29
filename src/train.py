import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CFG
from .dataset import LogMelDataset
from .model import SimpleCNN

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total

def main():
    cfg = CFG()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_csv = os.path.join(cfg.split_dir, "train.csv")
    val_csv = os.path.join(cfg.split_dir, "val.csv")
    if not (os.path.exists(train_csv) and os.path.exists(val_csv)):
        raise FileNotFoundError("Missing splits. Run: python -m src.preprocess")

    train_ds = LogMelDataset(train_csv)
    label2id = train_ds.label2id
    val_ds = LogMelDataset(val_csv, label2id=label2id)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = SimpleCNN(num_classes=len(label2id)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    os.makedirs("models/checkpoints", exist_ok=True)
    with open(cfg.label2id_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    best_acc = 0.0
    for epoch in range(1, 21):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.ckpt_path)
            print(f"Saved: {cfg.ckpt_path}")

    print("Done. best_acc =", best_acc)

if __name__ == "__main__":
    main()
