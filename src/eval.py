import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from model import SimpleCNN
from utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel


SPLIT_JSON = "data/splits/split.json"
MODEL_PATH = "models/cnn_melspec.pth"
MAP_PATH   = "models/label_map.json"

SR = 22050
CLIP_SECONDS = 3.0
HOP_SECONDS = 1.5


def load_label_map():
    m = json.load(open(MAP_PATH, "r", encoding="utf-8"))
    genres = [m[str(i)] for i in range(len(m))]
    return genres


def load_test_items():
    items = json.load(open(SPLIT_JSON, "r", encoding="utf-8"))
    test_items = [it for it in items if it["split"] == "test"]
    if len(test_items) == 0:
        raise RuntimeError("No test items found in split.json")
    return test_items


def load_model(n_classes, device):
    model = SimpleCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_one_file_meanprob(model, device, path):
    """
    对单个音频：切片 -> 每片 softmax -> 平均融合
    返回 proba: np.ndarray [C]
    """
    y = load_audio(path, sr=SR)
    clips = split_fixed(y, SR, clip_seconds=CLIP_SECONDS, hop_seconds=HOP_SECONDS)

    probs = []
    for seg in clips:
        m = normalize_mel(mel_spectrogram(seg, sr=SR))
        x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,128,T]
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        probs.append(p)

    if len(probs) == 0:
        # 极端情况兜底
        return None
    return np.mean(probs, axis=0)


def save_confusion_matrix(cm, labels, out_path):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, aspect="auto", origin="upper")
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs("runs", exist_ok=True)

    genres = load_label_map()
    test_items = load_test_items()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(n_classes=len(genres), device=device)

    y_true, y_pred = [], []
    skipped = 0

    for it in test_items:
        fp = it["path"]
        label = int(it["label"])
        try:
            proba = predict_one_file_meanprob(model, device, fp)
            if proba is None:
                skipped += 1
                continue
            pred = int(np.argmax(proba))
            y_true.append(label)
            y_pred.append(pred)
        except Exception:
            skipped += 1

    report = classification_report(y_true, y_pred, target_names=genres, digits=4)
    print(report)

    report_path = "runs/test_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\n\nskipped_files={skipped}\n")
    print("saved:", report_path)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = "runs/confusion_matrix_test.png"
    save_confusion_matrix(cm, genres, cm_path)
    print("saved:", cm_path)

    acc = (np.array(y_true) == np.array(y_pred)).mean() if len(y_true) else 0.0
    print(f"test_acc={acc:.4f}  skipped={skipped}  total_test={len(test_items)}")


if __name__ == "__main__":
    main()
