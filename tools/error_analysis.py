import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import json
import csv
import numpy as np
import torch

from src.model import SimpleCNN
from src.utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel

SPLIT_JSON = "data/splits/split.json"
MODEL_PATH = "models/cnn_melspec.pth"
MAP_PATH   = "models/label_map.json"

SR = 22050
CLIP_SECONDS = 3.0
HOP_SECONDS = 1.5


def load_label_map():
    m = json.load(open(MAP_PATH, "r", encoding="utf-8"))
    return [m[str(i)] for i in range(len(m))]


def load_test_items():
    items = json.load(open(SPLIT_JSON, "r", encoding="utf-8"))
    return [it for it in items if it["split"] == "test"]


@torch.no_grad()
def mean_prob(model, device, path):
    y = load_audio(path, sr=SR)
    clips = split_fixed(y, SR, clip_seconds=CLIP_SECONDS, hop_seconds=HOP_SECONDS)

    probs = []
    for seg in clips:
        m = normalize_mel(mel_spectrogram(seg, sr=SR))
        x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        probs.append(p)
    return np.mean(probs, axis=0)


def main():
    os.makedirs("runs", exist_ok=True)

    genres = load_label_map()
    test_items = load_test_items()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN(n_classes=len(genres))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device).eval()

    rows = []
    for it in test_items:
        fp = it["path"]
        true_id = int(it["label"])
        proba = mean_prob(model, device, fp)
        pred_id = int(np.argmax(proba))
        pred_p = float(proba[pred_id])
        true_p = float(proba[true_id])

        rows.append({
            "path": fp,
            "true_id": true_id,
            "true_genre": genres[true_id],
            "pred_id": pred_id,
            "pred_genre": genres[pred_id],
            "pred_prob": pred_p,
            "true_prob": true_p,
            "correct": int(pred_id == true_id),
        })

    out_all = "runs/test_predictions.csv"
    with open(out_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # “最自信但错”的 Top20
    wrong = [r for r in rows if r["correct"] == 0]
    wrong.sort(key=lambda r: r["pred_prob"], reverse=True)
    out_hard = "runs/hard_cases.csv"
    with open(out_hard, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(wrong[:20])

    print("saved:", out_all)
    print("saved:", out_hard)
    print("wrong_total:", len(wrong), " / test_total:", len(rows))


if __name__ == "__main__":
    main()
