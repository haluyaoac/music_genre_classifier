import os, glob
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from .config import CFG

def load_fix_length(path: str, sr: int, duration: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y

def wav_to_logmel(y: np.ndarray, cfg: CFG) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)

def main():
    cfg = CFG()
    os.makedirs(cfg.feature_dir, exist_ok=True)
    os.makedirs(cfg.split_dir, exist_ok=True)

    if not os.path.exists(cfg.raw_root):
        raise FileNotFoundError(
            f"raw_root not found: {cfg.raw_root}\n"
            f"Please download/extract dataset and update CFG.raw_root."
        )

    genres = sorted([d for d in os.listdir(cfg.raw_root) if os.path.isdir(os.path.join(cfg.raw_root, d))])
    rows = []

    for g in genres:
        files = glob.glob(os.path.join(cfg.raw_root, g, "*.wav"))
        for fp in tqdm(files, desc=f"Extract {g}", leave=False):
            base = os.path.splitext(os.path.basename(fp))[0]
            out = os.path.join(cfg.feature_dir, f"{g}__{base}.npy")
            if not os.path.exists(out):
                y = load_fix_length(fp, cfg.sr, cfg.duration)
                feat = wav_to_logmel(y, cfg)
                np.save(out, feat)
            rows.append({"path": out, "label": g})

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(os.path.join(cfg.split_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(cfg.split_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.split_dir, "test.csv"), index=False)

    print("Done.")
    print("Features:", cfg.feature_dir)
    print("Splits:", cfg.split_dir)

if __name__ == "__main__":
    main()
