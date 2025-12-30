import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix

MAP_PATH = "models/label_map.json"
PRED_CSV = "runs/test_predictions.csv"

def load_genres():
    m = json.load(open(MAP_PATH, "r", encoding="utf-8"))
    return [m[str(i)] for i in range(len(m))]

def main():
    genres = load_genres()

    # 读预测 CSV
    import csv
    y_true, y_pred = [], []
    with open(PRED_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            y_true.append(int(row["true_id"]))
            y_pred.append(int(row["pred_id"]))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(genres))))

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, aspect="auto", origin="upper")
    plt.xticks(range(len(genres)), genres, rotation=30, ha="right")
    plt.yticks(range(len(genres)), genres)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")

    # 数字标注
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v != 0:
                plt.text(j, i, str(v), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    out = "runs/confusion_matrix_test_numbered.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)

if __name__ == "__main__":
    main()
