import json, torch
import numpy as np
from src.model import SimpleCNN
from src.utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel


def predict_file(path, model_path="models/cnn_melspec.pth", map_path="models/label_map.json"):
    # path: 音频文件路径；model_path: 模型权重；map_path: 标签映射。
    # 返回 Top-3 (流派, 概率)。
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取标签映射，并按索引顺序构建类别列表。
    label_map = json.load(open(map_path, "r", encoding="utf-8"))
    genres = [label_map[str(i)] for i in range(len(label_map))]

    # 构建模型并加载权重。
    model = SimpleCNN(n_classes=len(genres))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()

    # 读取音频并切成多个片段。
    y = load_audio(path, sr=22050)
    clips = split_fixed(y, 22050, clip_seconds=3.0, hop_seconds=1.5)

    probs = []
    with torch.no_grad():
        for seg in clips:
            # 计算每个片段的预测概率。
            m = normalize_mel(mel_spectrogram(seg, sr=22050))
            x = torch.tensor(m).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 128, T]
            p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            probs.append(p)

    # 对片段概率取平均，得到整体预测。
    mean_p = np.mean(probs, axis=0)
    top = mean_p.argsort()[::-1][:3]
    return [(genres[i], float(mean_p[i])) for i in top]
