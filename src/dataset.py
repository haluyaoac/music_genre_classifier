import os, glob, json
import random
import torch
from torch.utils.data import Dataset
from utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel

# 音频流派数据集：按目录结构读取音频文件。
class GenreDataset(Dataset):
    def __init__(self, root_dir, genres, sr=22050, clip_seconds=3.0, hop_seconds=1.5):
        # root_dir: 数据根目录（每个子目录为一个类别）。
        # genres: 类别名称列表（顺序即标签索引）。
        # sr/clip_seconds/hop_seconds: 音频处理参数。
        self.items = []
        self.genres = genres
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.hop_seconds = hop_seconds

        # 遍历各类别目录，收集文件路径与标签索引。
        for gi, g in enumerate(genres):
            for fp in glob.glob(os.path.join(root_dir, g, "*")):
                self.items.append((fp, gi))

    def __len__(self):
        # 数据集样本数。
        return len(self.items)

    def __getitem__(self, idx):
        # 遇到坏音频时，最多重试 20 次换别的样本
        for _ in range(20):
            fp, label = self.items[idx]
            try:
                y = load_audio(fp, sr=self.sr)

                clips = split_fixed(y, self.sr, self.clip_seconds, self.hop_seconds)
                seg = random.choice(clips)

                m = mel_spectrogram(seg, sr=self.sr)
                m = normalize_mel(m)

                x = torch.tensor(m).unsqueeze(0)  # [1, n_mels, time]
                y_t = torch.tensor(label).long()
                return x, y_t

            except Exception:
                idx = random.randrange(len(self.items))

        raise RuntimeError("Too many unreadable audio files. Install ffmpeg and/or clean dataset.")
