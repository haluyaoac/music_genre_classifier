import json
import random
import torch
from torch.utils.data import Dataset

from utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel, spec_augment



class SplitDataset(Dataset):
    """
    从 data/splits/split.json 读取固定划分的样本列表
    每次 __getitem__ 会尝试读取音频并抽取一个切片，读失败则换样本重试（容错）
    """
    def __init__(
        self,
        split_json_path="data/splits/split.json",
        split="train",    # "train" | "val" | "test"
        sr=22050,         # 采样率
        clip_seconds=3.0, # 切片长度
        hop_seconds=1.5,  # 切片间隔
        max_retry=20,     # 容错时最大重试次数
        random_clip=True, # 是否随机抽取切片，否则总是第一个切片
        use_specaug=False, freq_mask_param=20, time_mask_param=30, num_masks=2
    ):
        self.sr = sr
        self.clip_seconds = clip_seconds
        self.hop_seconds = hop_seconds
        self.max_retry = max_retry
        self.random_clip = random_clip
        self.use_specaug = use_specaug
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

        with open(split_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        
        # 过滤出指定 split 的样本
        self.items = [it for it in items if it["split"] == split]
        if len(self.items) == 0:
            raise RuntimeError(f"No items found for split='{split}' in {split_json_path}")

    def __len__(self):
        return len(self.items)


    def _make_one(self, fp, label):
        """读取音频并抽取一个切片，返回模型输入张量和标签张量"""
        y = load_audio(fp, sr=self.sr)
        clips = split_fixed(y, self.sr, self.clip_seconds, self.hop_seconds)
        seg = random.choice(clips) if self.random_clip else clips[0]

        m = normalize_mel(mel_spectrogram(seg, sr=self.sr))
        if self.use_specaug and self.random_clip:
            m = spec_augment(m,
                            freq_mask_param=self.freq_mask_param,
                            time_mask_param=self.time_mask_param,
                            num_masks=self.num_masks)
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
