from dataclasses import dataclass

# 配置集中管理：预处理参数与路径。
@dataclass
class CFG:
    # 音频预处理参数。
    sr: int = 22050          # 采样率（Hz）
    duration: float = 30.0   # 读取的目标时长（秒）
    n_fft: int = 2048        # FFT 窗口大小
    hop_length: int = 512    # 帧移
    n_mels: int = 128        # mel 频带数量
    fmin: int = 20           # mel 最低频率
    fmax: int = 8000         # mel 最高频率

    # 数据集与特征路径（raw_root 按你的数据集调整）。
    raw_root: str = "data/raw/gtzan/genres_original"  # 原始音频目录
    feature_dir: str = "data/features/gtzan_logmel"   # 预处理特征保存目录
    split_dir: str = "data/splits"                    # 训练/验证/测试划分文件目录

    # 模型产物（权重与标签映射）。
    label2id_path: str = "models/label2id.json"       # 标签到索引映射
    ckpt_path: str = "models/checkpoints/best.pt"     # 最佳模型权重