# 评估/混淆矩阵脚本（待实现）。
# 预期流程：
# 1) 加载训练好的模型与标签映射；
# 2) 构建测试集 DataLoader；
# 3) 逐批推理并累计真实/预测标签；
# 4) 计算准确率与混淆矩阵并输出。
import json, torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from dataset import GenreDataset
from model import SimpleCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = json.load(open("models/label_map.json", "r", encoding="utf-8"))
    genres = [label_map[str(i)] for i in range(len(label_map))]

    ds = GenreDataset("data/raw", genres)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = SimpleCNN(n_classes=len(genres))
    model.load_state_dict(torch.load("models/cnn_melspec.pth", map_location="cpu"))
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)

    print(classification_report(y_true, y_pred, target_names=genres))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()