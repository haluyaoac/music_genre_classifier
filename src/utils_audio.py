# 音频读取、切片与 mel 特征计算的工具函数。
import numpy as np
import librosa


# 把文件解码成波形数组 y（float32），并重采样到 sr。
def load_audio(path, sr=22050, mono=True):
    '''
    load_audio 的 Docstring
    :param path: 音频文件路径
    :param sr: 目标采样率，每秒样本数
    :param mono: 是否转为单声道
    '''     
    # 返回值 y 为一维波形数组（numpy）。
    y, _sr = librosa.load(path, sr=sr, mono=mono)
    return y

# 把波形切成多个片段
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

def spec_augment(m, freq_mask_param=20, time_mask_param=30, num_masks=2):
    m = m.copy()
    n_mels, n_steps = m.shape
    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, n_mels - f))
            m[f0:f0+f, :] = 0
        t = np.random.randint(0, time_mask_param + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, n_steps - t))
            m[:, t0:t0+t] = 0
    return m