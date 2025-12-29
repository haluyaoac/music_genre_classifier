import os, json
import numpy as np
import torch

from .config import CFG
from .model import SimpleCNN
from .preprocess import load_fix_length, wav_to_logmel

def load_artifacts():
    cfg = CFG()
    if not (os.path.exists(cfg.label2id_path) and os.path.exists(cfg.ckpt_path)):
        return None, None, None

    with open(cfg.label2id_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    model = SimpleCNN(num_classes=len(label2id))
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu"))
    model.eval()
    return model, cfg, id2label

def predict(audio_path: str, topk: int = 5):
    model, cfg, id2label = load_artifacts()
    if model is None:
        raise RuntimeError("Model not found. Train first: python -m src.preprocess && python -m src.train")

    y = load_fix_length(audio_path, cfg.sr, cfg.duration)
    feat = wav_to_logmel(y, cfg)
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)  # (1,1,mel,time)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze(0).numpy()

    idx = prob.argsort()[::-1][:topk]
    return [(id2label[int(i)], float(prob[int(i)])) for i in idx]

if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))
