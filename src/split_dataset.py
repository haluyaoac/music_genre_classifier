import json
import random
import torch
from torch.utils.data import Dataset

from utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel


class SplitDataset(Dataset):
    """
    从 data/splits/split.json 读取固定划分的样本列表
    每次 __getitem__ 会尝试读取音频并抽取一个切片，读失败则换样本重试（容错）
    """
    def __init__(
        self,
        split_json_path="data/splits/split.json",
        split="train",
        sr=22050,
        clip_seconds=3.0,
        hop_seconds=1.5,
        max_retry=20,
        random_clip=True,
    ):
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.hop_seconds = hop_seconds
        self.max_retry = max_retry
        self.random_clip = random_clip

        with open(split_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.items = [it for it in items if it["split"] == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items found for split='{split}' in {split_json_path}")

    def __len__(self):
        return len(self.items)

    def _make_one(self, fp, label):
        y = load_audio(fp, sr=self.sr)
        clips = split_fixed(y, self.sr, self.clip_seconds, self.hop_seconds)
        seg = random.choice(clips) if self.random_clip else clips[0]

        m = normalize_mel(mel_spectrogram(seg, sr=self.sr))
        x = torch.tensor(m).unsqueeze(0)          # [1, n_mels, time]
        y_t = torch.tensor(int(label)).long()
        return x, y_t

    def __getitem__(self, idx):
        # 容错：遇到坏音频就随机换别的样本
        for _ in range(self.max_retry):
            it = self.items[idx]
            fp, label = it["path"], it["label"]
            try:
                return self._make_one(fp, label)
            except Exception:
                idx = random.randrange(len(self.items))

        raise RuntimeError("Too many unreadable audio files. Please clean dataset further.")
