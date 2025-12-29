import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class LogMelDataset(Dataset):
    def __init__(self, csv_path: str, label2id=None):
        self.df = pd.read_csv(csv_path)
        if label2id is None:
            labels = sorted(self.df["label"].unique())
            self.label2id = {l: i for i, l in enumerate(labels)}
        else:
            self.label2id = label2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = np.load(row["path"])              # (n_mels, time)
        x = torch.from_numpy(x).unsqueeze(0)  # (1, n_mels, time)
        y = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return x, y
