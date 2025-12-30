import numpy as np
import librosa

# 音频读取、切片与 mel 特征计算的工具函数。
def load_audio(path, sr=22050, mono=True):
    # path: 音频文件路径；sr: 目标采样率；mono: 是否转为单声道。
    # 返回值 y 为一维波形数组（numpy）。
    y, _sr = librosa.load(path, sr=sr, mono=mono)
    return y


def split_fixed(y, sr, clip_seconds=3.0, hop_seconds=1.5):
    # y: 波形数组；sr: 采样率；clip_seconds: 片段时长；hop_seconds: 片段步长。
    # 输出为若干固定长度片段（可能重叠）。
    clip_len = int(sr * clip_seconds)
    hop_len = int(sr * hop_seconds)
    clips = []
    for start in range(0, max(1, len(y) - clip_len + 1), hop_len):
        seg = y[start:start + clip_len]
        if len(seg) < clip_len:
            # 末尾不足长度时补零到固定长度。
            seg = np.pad(seg, (0, clip_len - len(seg)))
        clips.append(seg)
    return clips


def mel_spectrogram(y, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    # y: 波形数组；n_mels/n_fft/hop_length 控制频谱分辨率。
    # 返回 log-mel 频谱矩阵 (n_mels, time)。
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_mel(m):
    # 按样本归一化到 0 均值、1 方差（也可替换为全局 mean/std）。
    return (m - m.mean()) / (m.std() + 1e-6)
