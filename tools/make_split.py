import os, glob, json, random

ROOT = "data/raw"
OUT = "data/splits/split.json"
SEED = 42

# 读取 label_map 得到 genres 顺序（保持一致）
with open("models/label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)
genres = [label_map[str(i)] for i in range(len(label_map))]

items = []
for gi, g in enumerate(genres):
    files = glob.glob(os.path.join(ROOT, g, "*"))
    for fp in files:
        items.append({"path": fp, "label": gi, "genre": g})

random.Random(SEED).shuffle(items)

n = len(items)
n_train = int(n * 0.8)
n_val = int(n * 0.1)

for i, it in enumerate(items):
    if i < n_train:
        it["split"] = "train"
    elif i < n_train + n_val:
        it["split"] = "val"
    else:
        it["split"] = "test"

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print("saved:", OUT, "total:", n)
