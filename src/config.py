from dataclasses import dataclass

@dataclass
class CFG:
    # audio
    sr: int = 22050
    duration: float = 30.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 8000

    # paths (you will align raw_root to your dataset later)
    raw_root: str = "data/raw/gtzan/genres_original"
    feature_dir: str = "data/features/gtzan_logmel"
    split_dir: str = "data/splits"

    # model artifacts
    label2id_path: str = "models/label2id.json"
    ckpt_path: str = "models/checkpoints/best.pt"
