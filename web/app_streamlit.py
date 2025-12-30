# web/app_streamlit.py
import os
import sys
from pathlib import Path
# 关键：Streamlit 启动时也要禁用 torch 的 autoload/compile（否则可能又卡）
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

# Ensure project root is on sys.path when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import tempfile
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.infer import predict_proba_file, topk_from_proba
from src.utils_audio import load_audio, split_fixed, mel_spectrogram, normalize_mel

# 可选：让页面干净一点（不影响功能）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed*")


st.set_page_config(page_title="Music Genre Classifier", layout="wide")
st.title("🎵 音乐风格识别 Demo")
st.caption("上传 MP3/WAV → 提取 Log-Mel 频谱 → CNN 预测风格（Top-K）")


with st.sidebar:
    st.header("推理参数")
    model_path = st.text_input("模型权重路径", "models/cnn_melspec.pth")
    map_path = st.text_input("类别映射路径", "models/label_map.json")

    topk = st.slider("Top-K 展示", 1, 10, 5)
    clip_seconds = st.slider("切片长度（秒）", 1.0, 10.0, 3.0, 0.5)
    hop_seconds = st.slider("切片步长（秒）", 0.5, 10.0, 1.5, 0.5)

    st.header("频谱显示")
    preview_seconds = st.slider("频谱预览音频长度（秒）", 1.0, 20.0, 6.0, 1.0)


up = st.file_uploader("上传音频文件（MP3/WAV/FLAC/OGG 等）", type=["wav", "mp3", "flac", "ogg", "m4a", "aac"])

if up is None:
    st.info("把音频拖进来或点击上传。你也可以先用 samples/ 里的音频来测试。")
    st.stop()
    raise SystemExit

# 播放音频
st.audio(up)

# 保存临时文件（很多音频库更喜欢路径而不是 bytes）
suffix = "." + up.name.split(".")[-1].lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
    f.write(up.getbuffer())
    tmp_path = f.name

try:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📈 Log-Mel 频谱图（预览）")
        # 只取前 preview_seconds 秒，避免超长音频渲染慢
        y = load_audio(tmp_path, sr=22050)
        max_len = int(22050 * preview_seconds)
        y_preview = y[:max_len] if len(y) > max_len else y

        # 再从预览里取一段切片做展示
        clips = split_fixed(y_preview, 22050, clip_seconds=float(min(clip_seconds, preview_seconds)), hop_seconds=float(min(hop_seconds, preview_seconds)))
        seg = clips[0]
        m = mel_spectrogram(seg, sr=22050)
        m = normalize_mel(m)

        fig = plt.figure()
        plt.imshow(m, aspect="auto", origin="lower")
        plt.xlabel("Time")
        plt.ylabel("Mel bins")
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("🤖 预测结果")
        t0 = time.time()
        genres, proba, clip_cnt = predict_proba_file(
            tmp_path,
            model_path=model_path,
            map_path=map_path,
            sr=22050,
            clip_seconds=float(clip_seconds),
            hop_seconds=float(hop_seconds),
        )
        dt = time.time() - t0

        top = topk_from_proba(genres, proba, k=topk)
        df = pd.DataFrame(top, columns=["genre", "prob"])
        st.write(f"切片数：**{clip_cnt}**  |  推理耗时：**{dt:.2f}s**")
        st.dataframe(df, use_container_width=True)

        st.subheader("Top-K 概率条形图")
        fig2 = plt.figure()
        plt.bar(df["genre"], df["prob"])
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig2, clear_figure=True)

        with st.expander("查看全量类别概率"):
            full_df = pd.DataFrame({"genre": genres, "prob": proba})
            full_df = full_df.sort_values("prob", ascending=False)
            st.dataframe(full_df, use_container_width=True)

except Exception as e:
    st.error(f"推理失败：{e}")
finally:
    try:
        os.remove(tmp_path)
    except Exception:
        pass
